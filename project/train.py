import random
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from itertools import chain, islice
from torch import Tensor
from project.segModel import *
from project.data import *

# %% load data
batchSize = 12

dataFolder = "../ImageNet"


def getDataSet() -> (DataLoader, DataLoader):
    dataSetPath = Path("../MSRC")
    trainIndex = dataSetPath / "Train.txt"
    validIndex = dataSetPath / "Validation.txt"
    trainSet = SegmentationDataSet(dataSetPath, trainIndex, transform)
    validSet = SegmentationDataSet(dataSetPath, validIndex, transform)
    train = DataLoader(trainSet, batch_size=batchSize, shuffle=True, num_workers=4)
    valid = DataLoader(validSet, batch_size=batchSize, shuffle=True, num_workers=4)
    return [train, valid]


trainSet: DataLoader
devSet: DataLoader
trainSet, devSet = getDataSet()

# %% initialize model
model = makeModel()
lossModel = torch.nn.CrossEntropyLoss()
allParams = chain(model.parameters())
optimizer = torch.optim.Adam(allParams, lr=1e-4, weight_decay=1e-5)

# %% training loop
import datetime
from torch.utils.tensorboard import SummaryWriter


def trainOnBatch(img: Tensor, label: Tensor) -> np.array:
    img = toDevice(img)
    out = model.forward(img)
    loss = lossModel(out, label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return toNumpy(loss)


def testOnBatch(img: Tensor, label: Tensor) -> np.array:
    with torch.no_grad():
        img = toDevice(img)
        out = model.forward(img)
        loss = lossModel(out, label)
        return toNumpy(loss)


trainWriter = SummaryWriter(comment="train", flush_secs=30)
validWriter = SummaryWriter(comment="valid", flush_secs=30)

trainBatches = 2 if testMode else min(len(trainSet), 200)
testBatches = 1 if testMode else min(len(devSet), 50)
print("train/dev size: {}/{}".format(trainBatches, testBatches))

# %% test on sample images
from PIL import Image
from pathlib import Path
from val_grader.grader import *

colorMap: any

def logitsToColor(logits: Tensor) -> Tensor:
    indicies = logits.argmax(dim=1)
    return colorMap[indicies]  # todo

def testOnSample(fromDir: Path, toDir: Path, epoch: int):
    with torch.no_grad():
        for img_path in fromDir.glob('*.jpg'):
            img = Image.open(img_path)
            img = toDevice(transform(img)[None, :, :, :])
            output = logitsToColor(model(img)).squeeze(dim=0).cpu()

            validWriter.add_image(img_path.name, output, epoch)
            output = invTransform(output)
            output.save("{}/{}".format(toDir, img_path.name), "JPEG")



def showAtSamePlace(content):
    import sys
    sys.stdout.write(content + "   \r")
    sys.stdout.flush()


def trainingLoop():
    startTime = datetime.datetime.now().ctime()
    step = 0
    for epoch in range(1, 5001):
        print("===epoch {}===".format(epoch))
        progress = 0
        for inputs, labels in islice(trainSet, trainBatches):
            loss = trainOnBatch(inputs, labels)
            trainWriter.add_scalar("Loss", loss, step)
            progress += 1
            showAtSamePlace("progress: {}/{}".format(progress, trainBatches))
            step += 1
        print()

        print("start testing")
        lossCollection = []
        progress = 0
        for inputs, labels in islice(devSet, testBatches):
            lossCollection.append(testOnBatch(inputs, labels).reshape(1, -1))
            progress += 1
            showAtSamePlace("progress: {}/{}".format(progress, testBatches))
        print()
        avgLoss = np.mean(np.concatenate(lossCollection, axis=0))
        validWriter.add_scalar("Loss", avgLoss, step)

        if epoch != 0 and epoch % 5 == 0:
            saveDir = Path("saves/{}/epoch{}".format(startTime, epoch))
            saveDir.mkdir(parents=True)
            testOnSample(Path("data"), saveDir, epoch)
            torch.save(model.state_dict(), '{}/state_dict.pth'.format(str(saveDir)))
            # grade()


if __name__ == '__main__':
    trainingLoop()
