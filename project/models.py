from pathlib import Path

import torch
from torch import Tensor
import torch.nn as nn
from typing import *
import numpy as np
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as transforms
import project.models_res as res


# print("use params: {}".format(params.paramsName))
# %% Data Transformations
testMode = False
imageSize = 256
# device = torch.device("cpu")
device = torch.device("cuda:0")

transform = transforms.Compose([
    transforms.Resize(imageSize),
    transforms.RandomCrop(imageSize),
    transforms.ToTensor()
])

invTransform = transforms.Compose([
    transforms.ToPILImage()
])

# %% model definitions

T = TypeVar('T')


def toDevice(m: T) -> T:
    return m.to(device, torch.float)


def toNumpy(t: Tensor) -> np.array:
    return t.detach().cpu().numpy()


def testOutputSize(params):
    enc: Encoder = Encoder(params)
    x = toDevice(torch.randn([1, 3, imageSize, imageSize]))
    embed: Tensor = enc(x)
    print(
        "output shape {} has {} elements. Required limit = {}".format(embed.shape, embed.numel(),
                                                                      params.maxOutputPatches * 4))
    assert torch.Size([1] + params.bottleShape) == embed.shape
    patchingTest(embed)
    qt: Quantization = Quantization(params)
    logits = qt.centerLogits(embed)
    print("quantized logits shape: {}".format(logits.shape))
    r1R = qt.reconstructFromLogits(logits)
    print("de-quantized shape: {}".format(r1R.shape))

    rec = reconstructFromPatches(divideIntoPatches(embed).squeeze(dim=-1))
    assert torch.all(torch.eq(embed, rec)), "patching test failed"

    symbols = logitsToBytes(logits)
    fromBytes = qt.embeddingFromBytes(symbols)
    assert fromBytes.shape == embed.shape, "fromBytes: {}, embedding: {}".format(fromBytes.shape, embed.shape)
    assert toNumpy(symbols).nbytes == params.maxOutputBytes

    dec: Decoder = Decoder(params)
    r2 = dec(r1R)
    print("reconstructed shape: {}".format(r2.shape))
    assert r2.shape == x.shape


def encodeLayers(params) -> nn.Sequential:
    channels = params.channels
    kernelSizes = params.kernelSizes
    strides = params.strides
    paddings = params.paddings
    bottleChannels = params.bottleChannels
    layers = [
        layer
        for i in range(len(channels) - 1)
        for layer in [
            nn.Conv2d(channels[i], channels[i + 1], kernelSizes[i], strides[i], paddings[i]),
            nn.ReLU(True)]]
    layers.append(nn.Conv2d(channels[-1], bottleChannels, kernelSizes[-1], strides[-1], paddings[-1]))
    return nn.Sequential(*layers)


def decodeLayers(params) -> nn.Sequential:
    channels = params.channels
    kernelSizes = params.kernelSizes
    strides = params.strides
    paddings = params.paddings
    bottleChannels = params.bottleChannels

    def computeOutputPadding(stride: int) -> int:
        return stride - 1

    layers = [nn.ConvTranspose2d(bottleChannels, channels[-1], kernelSizes[-1], strides[-1],
                                 padding=paddings[-1], output_padding=computeOutputPadding(strides[-1]))]

    layers.extend([
        layer
        for i in range(len(channels) - 2, -1, -1)
        for layer in [
            nn.ReLU(True),
            nn.ConvTranspose2d(channels[i + 1], channels[i], kernelSizes[i], strides[i],
                               padding=paddings[i], output_padding=computeOutputPadding(strides[i])),
        ]])
    return nn.Sequential(*layers)


def divideIntoPatches(x: Tensor) -> Tensor:
    """input shape: [N,C,W,H], output shape: [N,C,W/2,H/2,4,1]"""
    assert len(x.shape) == 4
    x = x.unfold(2, 2, 2)
    x = x.unfold(3, 2, 2)
    shape = list(x.shape)
    shape[-2] = 4
    shape[-1] = 1
    return x.contiguous().view(shape)  # the last dimension of x are patch vectors of length 4


def reconstructFromPatches(x: Tensor) -> Tensor:
    """input shape: [N,C,W/2,H/2,4], output shape: [N,C,W,H]"""
    assert len(x.shape) == 5
    [n, c, w2, h2, last] = list(x.shape)
    assert last == 4
    x = x.view([n, c, w2, h2, 2, 2])
    x = x.permute([0, 1, 2, 4, 3, 5]).contiguous()
    x = x.view([n, c, w2 * 2, h2 * 2])
    return x


def patchingTest(x: Tensor):
    patches = torch.squeeze(divideIntoPatches(x), dim=-1)
    y = reconstructFromPatches(patches)
    assert torch.all(torch.eq(x, y)), "x = {}\ny={}".format(x, y)


def squaredDistance(x: Tensor, y: Tensor, dim: int):
    z = x - y
    return torch.sum(z * z, dim)


def cosSimilarity(x: Tensor, y: Tensor, dim: int):
    return F.cosine_similarity(x, y, dim)


def logitsToBytes(logits: Tensor) -> Tensor:
    return logits.argmax(dim=4).byte()


numCenter = 2 ** 8


class Quantization(torch.nn.Module):
    def __init__(self, params):
        super(Quantization, self).__init__()
        centers0 = torch.randn(1, params.centerChannels, 1, 1, 4, numCenter, device=device) * 0.01
        self.centers = torch.nn.Parameter(centers0)
        self.eye = torch.eye(numCenter, device=device)

    def centerLogits(self, x: Tensor) -> Tensor:
        x = divideIntoPatches(x)
        # return -squaredDistance(x, self.centers, dim=4)
        return cosSimilarity(x, self.centers, dim=4)

    def toBytes(self, x: Tensor) -> Tensor:
        return logitsToBytes(self.centerLogits(x))

    def reconstructFromLogits(self, x: Tensor) -> Tensor:
        sim = torch.softmax(x, dim=4)
        sim = sim.unsqueeze(dim=4)
        y = torch.sum(self.centers * sim, dim=5, keepdim=False)
        return reconstructFromPatches(y)

    def embeddingFromBytes(self, bts: Tensor) -> Tensor:
        oneHot = self.eye[bts.to(torch.long), :]
        sim = oneHot.unsqueeze(dim=4)
        y = torch.sum(self.centers * sim, dim=5, keepdim=False)
        return reconstructFromPatches(y)

    def forward(self, x: Tensor, sigma: Tensor) -> Tensor:
        logits = self.centerLogits(x)
        return self.reconstructFromLogits(logits * sigma)


class Encoder(torch.nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.model = encodeLayers(params)

    def forward(self, img) -> Tensor:
        return self.model.forward(img)

    def encodeImage(self, img, device, transform) -> Tensor:
        t = torch.tensor(np.array(img), dtype=torch.float, device=device)
        t = t.permute(2, 0, 1)[None, :, :, :]
        t = transform(t)
        return self.forward(t)


def doubleRelu(x, slop=0.01):
    return F.relu(x) - F.relu(x - 1.0) - F.relu(-x) * slop + F.relu(x - 1.0) * slop


class Decoder(torch.nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.model = decodeLayers(params)

    def forward(self, img) -> Tensor:
        return doubleRelu(self.model(img))


class EncoderDecoder(torch.nn.Module):
    def __init__(self, params):
        super(EncoderDecoder, self).__init__()
        self.encoder: Encoder = Encoder(params)
        self.decoder: Decoder = Decoder(params)
        self.quantization: Quantization = Quantization(params)

    def forward(self, img: Tensor, sigma: Tensor) -> Tensor:
        return self.decoder(self.quantization(self.encoder(img), sigma))

    def hardEncodeDecode(self, img: Tensor) -> Tensor:
        bottleneck: Tensor = self.encoder(img)
        qt: Quantization = self.quantization
        bts = qt.toBytes(bottleneck)
        bottleneck1 = qt.embeddingFromBytes(bts)
        outHard = self.decoder(bottleneck1)
        return outHard


def makeModel(params) -> EncoderDecoder:
    ed = EncoderDecoder(params)
    toDevice(ed)
    return ed


def loadModelFromFile(model: nn.Module, file: Path):
    print("load model from file: " + str(file))
    model.load_state_dict(torch.load(file), strict=False)
    print("model loaded")


if __name__ == '__main__':
    testOutputSize(params)
