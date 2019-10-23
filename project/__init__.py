import numpy as np
import torch
from torch import Tensor
from typing import *
import torchvision.transforms as transforms

testMode = False

T = TypeVar('T')

imageSize = 256
# device = torch.device("cpu")
device = torch.device("cuda:0")

nClass = 21

transform = transforms.Compose([
    transforms.Resize(imageSize),
    transforms.RandomCrop(imageSize),
    transforms.ToTensor()
])

invTransform = transforms.Compose([
    transforms.ToPILImage()
])

def toDevice(m: T) -> T:
    return m.to(device, torch.float)


def toNumpy(t: Tensor) -> np.array:
    return t.detach().cpu().numpy()

def segment(img: np.ndarray) -> np.ndarray:
    """
    Semantically segment an image
    img: an uint8 numpy of size (w,h,3)
    return: a numpy integer array of size (w,h), where the each entry represent the class id
    please refer to data/color_map.json for the id <-> class mapping
    """
    
    raise NotImplementedError("segment")
