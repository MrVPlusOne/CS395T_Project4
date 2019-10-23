from pathlib import Path
import numpy as np
from torch.utils.data import Dataset


def makeDataSetThunks(path: Path, indexFile: Path):
    """:returns a list of functions whose return values are the image-mask pairs"""
    import json
    from pathlib import Path
    from skimage.io import imread

    with open(str(path / 'color_map.json'), 'r') as f:
        color_map = json.load(f)

    with open(str(indexFile), 'r') as f:
        files = f.read().split('\n')

    def _loader():
        for file_path in files:
            def thunk():
                img = imread(str(path / 'images' / file_path))
                raw_mask = imread(str(path / 'masks' / file_path))

                mask = - np.ones(raw_mask.shape[:2], dtype=int)

                for line in color_map:
                    idx = np.all(raw_mask == line['rgb_values'], axis=2)
                    mask[idx] = line['id']

                return img, mask
            yield thunk

    return list(_loader())

class SegmentationDataSet(Dataset):
    def __init__(self, path: Path, indexFile: Path, transform):
        super(SegmentationDataSet, self).__init__()
        self.thunks = makeDataSetThunks(path, indexFile)
        self.transform = transform
        self.cache = dict()

    def __getitem__(self, index):
        if index in self.cache:
            return self.cache[index]

        value = self.thunks[index]()
        self.cache[index] = value
        return value

    def __len__(self):
        return len(self.thunks)
