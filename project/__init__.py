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

def segment(img: np.ndarray) -> np.ndarray:
    """
    Semantically segment an image
    img: an uint8 numpy of size (w,h,3)
    return: a numpy integer array of size (w,h), where the each entry represent the class id
    please refer to data/color_map.json for the id <-> class mapping
    """

    raise NotImplementedError("segment")
