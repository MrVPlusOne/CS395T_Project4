import numpy as np
import torch
from torch import Tensor
from typing import *
import torchvision.transforms as transforms

testMode = False

T = TypeVar('T')

# device = torch.device("cpu")
device = torch.device("cuda:0")

nClass = 22

transform = transforms.Compose([
    transforms.ToTensor()
])

invTransform = transforms.Compose([
    transforms.ToPILImage()
])


def toDevice(m: T) -> T:
    return m.to(device)


def toNumpy(t: Tensor) -> np.array:
    return t.detach().cpu().numpy()


def assertEq(x, y):
    assert x == y, "x: {} != y: {}".format(x, y)


def printShape(t: Tensor, name: str):
    print("{} shape: {}".format(name, t.shape))