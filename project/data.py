from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
import json
from pathlib import Path
from skimage.io import imread

class ImagePromise(object):
    def __init__(self, colorMap, path, file_path):
        self.colorMap = colorMap
        self.path = path
        self.file_path = file_path
        self.cache = None

    def get(self):
        if self.cache is not None:
            return self.cache

        path = self.path
        file_path = self.file_path
        color_map = self.colorMap
        img = imread(str(path / 'images' / file_path))
        raw_mask = imread(str(path / 'masks' / file_path))

        mask = - np.ones(raw_mask.shape[:2], dtype=int)

        for line in color_map:
            idx = np.all(raw_mask == line['rgb_values'], axis=2)
            mask[idx] = line['id']

        result = img, mask
        self.cache = result
        return result


def makeDataSetThunks(path: Path, indexFile: Path):
    """:returns a list of functions whose return values are the image-mask pairs"""

    with open(str(path / 'color_map.json'), 'r') as f:
        color_map = json.load(f)

    with open(str(indexFile), 'r') as f:
        files = f.read().split('\n')

    def _loader():
        for file_path in files:
            yield ImagePromise(color_map, path, file_path)

    return list(_loader())

class SegmentationDataSet(Dataset):
    def __init__(self, path: Path, indexFile: Path, transform):
        super(SegmentationDataSet, self).__init__()
        self.thunks = makeDataSetThunks(path, indexFile)
        self.transform = transform

    def __getitem__(self, index):
        return self.thunks[index].get()

    def __len__(self):
        return len(self.thunks)
