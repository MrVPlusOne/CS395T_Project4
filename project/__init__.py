import numpy as np
from project.segModel import *
from project.data import skImageToTensor, makeColorMap

model = makeModel(useVggWeights=False)
loadModelFromFile(model, Path("trained/state_dict.pth"))
colorMap = makeColorMap(Path("data"))


def segment(img: np.ndarray) -> np.ndarray:
    """
    Semantically segment an image
    img: an uint8 numpy of size (w,h,3)
    return: a numpy integer array of size (w,h), where the each entry represent the class id
    please refer to data/color_map.json for the id <-> class mapping
    """
    img = torch.tensor(img, dtype=torch.float).permute([2, 0, 1]) / 255.0
    img = toDevice(img[None, :, :, :])
    logits = model(img)
    indices = logits.argmax(dim=1) - 1
    indices = toNumpy(indices.squeeze(dim=0))
    return indices
