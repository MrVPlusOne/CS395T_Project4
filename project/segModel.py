from pathlib import Path

from project.common import *
import torch.nn as nn
from torchvision.models import vgg


def normalize(x, mean, std):
    dtype = x.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=device)
    std = torch.as_tensor(std, dtype=dtype, device=device)
    return (x - (mean[None, :, None, None])) / (std[None, :, None, None])


def makeModel(useVggWeights):
    m = SegModel(useVggWeights)
    toDevice(m)
    return m


class SegModel(nn.Module):
    def __init__(self, useVggWeights: bool):
        super(SegModel, self).__init__()
        model = toDevice(vgg.vgg16(pretrained=useVggWeights))
        self._features = model.eval().features
        self.vgg_features = self._features._modules.items()

        self.layers = {
            # '16': 'pool3',
            '23': 'pool4',
            '30': 'pool5',
        }

        self.scoreLayer1 = nn.Sequential(
            nn.Conv2d(512, 2048, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(2048, 2048, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.ReLU(inplace=True),
            nn.Conv2d(2048, nClass, 1),
        )

        self.scoreLayer2 = nn.Sequential(
            nn.Conv2d(512, 2048, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(2048, nClass, 1)
        )

        self.upLayer2 = makeUpSampler(nClass, 2)
        self.upLayer16 = makeUpSampler(nClass, 16)

    def selectedLayers(self, x):
        outputs = dict()
        x = normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        for name, module in self.vgg_features:
            x = module(x)
            if name in self.layers:
                outputs[self.layers[name]] = x
        return outputs

    def forward(self, x0: Tensor) -> Tensor:
        assert (x0.shape[0] == 1)  # only works for a single image (of arbitrary shape)
        # printShape(x0, "x0")
        x, padding = padInput(x0)
        # printShape(x, "padded")
        # print("padding: " + str(padding))
        layers = self.selectedLayers(x)
        pool5 = layers['pool5']
        pool4 = layers['pool4']
        # print("pool4: {}, pool5: {}".format(pool4.shape, pool5.shape))
        x1 = self.scoreLayer1(pool5)
        # print("score x1: {}".format(x1.shape))
        x1 = self.upLayer2(x1)
        # print("up x1: {}".format(x1.shape))
        x2 = self.scoreLayer2(pool4)
        # print("x2: {}".format(x2.shape))
        assert x1.shape == x2.shape, "x1 shape: {}, x2 shape: {}".format(x1.shape, x2.shape)
        output = self.upLayer16(x1 + x2)

        # assertEq(output.shape[2], x.shape[2])
        # assertEq(output.shape[3], x.shape[3])

        output = dePadOutput(output, padding)
        # assertEq(output.shape[2], x0.shape[2])
        # assertEq(output.shape[3], x0.shape[3])
        return output


def padInput(x: Tensor) -> (Tensor, tuple):
    """pad the image such that its dimensions are multiplies of 32"""

    def computePadding(n: int, base: int):
        diff = n % base
        diff = base - diff if diff != 0 else 0
        left = diff // 2
        right = diff - left
        return left, right

    [_, _, w, h] = list(x.shape)
    w1, w2 = computePadding(w, 32)
    h1, h2 = computePadding(h, 32)

    x = nn.ZeroPad2d((h1, h2, w1, w2)).forward(x)
    return x, (w1, w2, h1, h2)


def dePadOutput(x: Tensor, padding) -> Tensor:
    (w1, w2, h1, h2) = padding

    return x[:, :, w1: (x.shape[2] - w2), h1: (x.shape[3] - h2)]


def makeUpSampler(channels, factor) -> nn.ConvTranspose2d:
    conv = BilinearConvTranspose2d(channels, factor, factor - 1)
    conv.reset_parameters()
    return conv


# taken from https://gist.github.com/mjstevens777/9d6771c45f444843f9e3dce6a401b183
class BilinearConvTranspose2d(nn.ConvTranspose2d):
    """A conv transpose initialized to bilinear interpolation."""

    def __init__(self, channels, stride, outputPadding, groups=1):
        """Set up the layer.
        Parameters
        ----------
        channels: int
            The number of input and output channels
        stride: int or tuple
            The amount of upsampling to do
        groups: int
            Set to 1 for a standard convolution. Set equal to channels to
            make sure there is no cross-talk between channels.
        """
        if isinstance(stride, int):
            stride = (stride, stride)

        assert groups in (1, channels), "Must use no grouping, " + \
                                        "or one group per channel"

        kernel_size = (2 * stride[0] - 1, 2 * stride[1] - 1)
        padding = (stride[0] - 1, stride[1] - 1)
        super().__init__(
            channels, channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            output_padding=outputPadding
        )

    def reset_parameters(self):
        """Reset the weight and bias."""
        nn.init.constant_(self.bias, 0)
        nn.init.constant_(self.weight, 0)
        bilinear_kernel = self.bilinear_kernel(self.stride)
        for i in range(self.in_channels):
            if self.groups == 1:
                j = i
            else:
                j = 0
            self.weight.data[i, j] = bilinear_kernel

    @staticmethod
    def bilinear_kernel(stride):
        """Generate a bilinear upsampling kernel."""
        num_dims = len(stride)

        shape = (1,) * num_dims
        bilinear_kernel = torch.ones(*shape)

        # The bilinear kernel is separable in its spatial dimensions
        # Build up the kernel channel by channel
        for channel in range(num_dims):
            channel_stride = stride[channel]
            kernel_size = 2 * channel_stride - 1
            # e.g. with stride = 4
            # delta = [-3, -2, -1, 0, 1, 2, 3]
            # channel_filter = [0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25]
            delta = torch.arange(1 - channel_stride, channel_stride)
            channel_filter = (1 - torch.abs(delta / channel_stride))
            # Apply the channel filter to the current channel
            shape = [1] * num_dims
            shape[channel] = kernel_size
            bilinear_kernel = bilinear_kernel * channel_filter.view(shape)
        return bilinear_kernel


def loadModelFromFile(model: nn.Module, file: Path):
    print("load model from file: " + str(file))
    model.load_state_dict(torch.load(file, map_location=device))
    model.to(device)
    print("model loaded")
