import random
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from itertools import chain, islice
from torch import Tensor
from project.segModel import *
from project.data import *

# %% load data

dataFolder = "../ImageNet"


def getDataSet() -> (DataLoader, DataLoader):
    dataSetPath = Path("../MSRC")
    trainIndex = dataSetPath / "Train.txt"
    validIndex = dataSetPath / "Validation.txt"
    trainSet = SegmentationDataSet(dataSetPath, trainIndex)
    validSet = SegmentationDataSet(dataSetPath, validIndex)
    train = DataLoader(trainSet, batch_size=1, shuffle=True, num_workers=1)
    valid = DataLoader(validSet, batch_size=1, shuffle=True, num_workers=1)
    return [train, valid]


trainSet: DataLoader
devSet: DataLoader
trainSet, devSet = getDataSet()

# %% initialize model
model = makeModel()
lossModel = torch.nn.CrossEntropyLoss()
allParams = chain(model.parameters())
optimizer = torch.optim.Adam(allParams, lr=1e-5, weight_decay=1e-6)

# loadModelFromFile(model, Path("saves/Wed-Oct-23-17_16_10-2019/epoch0/state_dict.pth"))

# %% training loop
import datetime
from torch.utils.tensorboard import SummaryWriter


def trainOnBatch(img: Tensor, label: Tensor) -> np.array:
    img = toDevice(img)
    label = toDevice(label)
    out = model.forward(img)
    loss = lossModel(out, label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return toNumpy(loss)


def testOnBatch(img: Tensor, label: Tensor) -> np.array:
    with torch.no_grad():
        img = toDevice(img)
        label = toDevice(label)
        out = model.forward(img)
        loss = lossModel(out, label)
        return toNumpy(loss)


trainWriter = SummaryWriter(comment="train", flush_secs=30)
validWriter = SummaryWriter(comment="valid", flush_secs=30)

trainBatches = 2 if testMode else len(trainSet)
testBatches = 1 if testMode else len(devSet)
print("train/dev size: {}/{}".format(trainBatches, testBatches))

# %% test on sample images
from PIL import Image
from pathlib import Path
from val_grader.grader import *

colorMap = makeColorMap(Path("../MSRC"))


def logitsToColor(logits: Tensor) -> Tensor:
    indicies = logits.argmax(dim=1)
    return colorMap[indicies, :].permute([0, 3, 1, 2])


def testOnSample(fromDir: Path, toDir: Path, epoch: int):
    with torch.no_grad():
        for img_path in fromDir.glob('*.bmp'):
            img = Image.open(img_path)
            img = toDevice(transform(img)[None, :, :, :])
            output = logitsToColor(model(img)).squeeze(dim=0).cpu()

            printShape(output, "output of color")
            validWriter.add_image(img_path.name, output, epoch)
            output = invTransform(output)
            output.save("{}/{}".format(toDir, img_path.name), "JPEG")


def showAtSamePlace(content):
    import sys
    sys.stdout.write(content + "   \r")
    sys.stdout.flush()


def formatDate(date):
    return str(date).replace(" ", "-").replace(":", "_")


def trainingLoop():
    startTime = datetime.datetime.now().ctime()
    step = 0
    for epoch in range(0, 5001):
        print("===epoch {}===".format(epoch))
        print("test")
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

        if epoch % 50 == 0:
            saveDir = Path("saves/{}/epoch{}".format(formatDate(startTime), epoch))
            saveDir.mkdir(parents=True)
            testOnSample(Path("data/images"), saveDir, epoch)
            torch.save(model.state_dict(), '{}/state_dict.pth'.format(str(saveDir)))
            # grade()


if __name__ == '__main__':
    trainingLoop()
