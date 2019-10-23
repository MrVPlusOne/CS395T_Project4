from project import *
import torch.nn as nn
from torchvision.models import vgg
from torch.nn.functional import relu

def normalize(x, mean, std):
    dtype = x.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=device)
    std = torch.as_tensor(std, dtype=dtype, device=device)
    return (x - (mean[None, :, None, None])) / (std[None, :, None, None])


def makeModel():
    m = SegModel()
    toDevice(m)
    return m


class SegModel(nn.Module):
    def __init__(self):
        super(SegModel, self).__init__()
        model = toDevice(vgg.vgg16(pretrained=True))
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
            nn.Conv2d(2048, nClass, 1),
            makeUpSampler(nClass, nClass, 64, stride=32)
        )

        self.upLayer2 = makeUpSampler(nClass, nClass, 4, stride=2)
        self.upLayer16 = makeUpSampler(nClass, nClass, 32, stride=16)



    def selectedLayers(self, x):
        outputs = dict()
        x = normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        for name, module in self.vgg_features:
            x = module(x)
            if name in self.layers:
                outputs[self.layers[name]] = x
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        [pool4, pool5] = self.selectedLayers(x)
        x1 = self.upLayer2self.scoreLayer1(pool5)
        x2 = x1 + self.scoreLayer2(pool4)
        return self.upLayer16(x2)


def makeUpSampler(nIn, nOut, kernel, stride) -> nn.ConvTranspose2d:
    m = nn.ConvTranspose2d(nIn, nOut, kernel, stride=stride, bias=False)
    m.weight.data.copy_(get_upsampling_weight(nIn, nOut, kernel))
    return m

# https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()