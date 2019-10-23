from torch.utils.data import Dataset
import json
from pathlib import Path
from skimage.io import imread
from project.common import *

def skImageToTensor(img):
    return torch.tensor(np.asarray(img), dtype=torch.float).permute([2, 0, 1]) / 255.0


class ImagePromise(object):
    def __init__(self, colorMap, path, file_path):
        self.colorMap = colorMap
        self.path = path
        self.file_path = file_path
        self.cache = None

    @staticmethod
    def maskName(name: str):
        return name[:-4] + "_GT.bmp"

    def get(self):
        if self.cache is not None:
            return self.cache

        path = self.path
        file_path = self.file_path
        color_map = self.colorMap
        img = imread(str(path / 'images' / file_path))
        raw_mask = imread(str(path / 'masks' / (ImagePromise.maskName(file_path))))

        mask = - np.ones(raw_mask.shape[:2], dtype=int)

        for line in color_map:
            idx = np.all(raw_mask == line['rgb_values'], axis=2)
            mask[idx] = line['id']

        img = skImageToTensor(img)
        mask = torch.tensor(np.asarray(mask), dtype=torch.long) + 1  # no negative indices
        assert img.shape[1] == mask.shape[0]
        assert img.shape[2] == mask.shape[1]
        result = img, mask
        self.cache = result
        return result


def makeColorMap(mapDir: Path):
    with open(str(mapDir / 'color_map.json'), 'r') as f:
        colorMap = json.load(f)
    out = torch.zeros([nClass, 3])
    for line in colorMap:
        out[line['id'] + 1, :] = torch.tensor(line['rgb_values'], dtype=torch.float)/255.0
    return toDevice(out)


def makeDataSetThunks(path: Path, indexFile: Path):
    """:returns a list of functions whose return values are the image-mask pairs"""

    with open(str(path / 'color_map.json'), 'r') as f:
        color_map = json.load(f)

    with open(str(indexFile), 'r') as f:
        files = f.read().split('\n')

    def _loader():
        for file_path in files:
            if file_path != "":
                yield ImagePromise(color_map, path, file_path)

    return list(_loader())


class SegmentationDataSet(Dataset):
    def __init__(self, path: Path, indexFile: Path):
        super(SegmentationDataSet, self).__init__()
        self.thunks = makeDataSetThunks(path, indexFile)

    def __getitem__(self, index):
        img, label = self.thunks[index].get()
        return img, label

    def __len__(self):
        return len(self.thunks)
