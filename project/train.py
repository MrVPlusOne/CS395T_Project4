from itertools import chain, islice
from torch.utils.data import DataLoader
from project.data import *
from project.segModel import *

# %% load data
print("Training initialization...")

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
model: SegModel = makeModel(useVggWeights=True)
lossModel = torch.nn.CrossEntropyLoss()
allParams = chain(model.parameters())
optimizer = torch.optim.Adam(allParams, lr=1e-5, weight_decay=1e-6)


# %% training loop
import datetime
from torch.utils.tensorboard import SummaryWriter


def trainOnBatch(img: Tensor, label: Tensor) -> np.array:
    img = toDevice(img)
    label = toDevice(label)
    model.train(True)
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
        model.train(False)
        out = model.forward(img)
        loss = lossModel(out, label)
        return toNumpy(loss)


trainWriter = SummaryWriter(comment="train", flush_secs=30)
validWriter = SummaryWriter(comment="valid", flush_secs=30)

trainBatches = 2 if testMode else len(trainSet)
testBatches = 1 if testMode else len(devSet)
# print("train/dev size: {}/{}".format(trainBatches, testBatches))

# %% test on sample images
from PIL import Image
from pathlib import Path

colorMap = makeColorMap(Path("../MSRC"))

def testOnSample(fromDir: Path, epoch: int):
    model.train(False)
    with torch.no_grad():
        for img_path in fromDir.glob('*.bmp'):
            img = Image.open(img_path)
            img = toDevice(transform(img)[None, :, :, :])
            output = logitsToColor(model(img)).squeeze(dim=0).cpu()

            validWriter.add_image(img_path.name, output, epoch)
            # output = invTransform(output)
            # output.save("{}/{}".format(toDir, img_path.name.replace(".bmp", ".jpg")), "JPEG")


def showAtSamePlace(content):
    import sys
    sys.stdout.write(content + "   \r")
    sys.stdout.flush()


def formatDate(date):
    return str(date).replace(" ", "-").replace(":", "_")


def trainingLoop():
    startTime = datetime.datetime.now().ctime()
    for epoch in range(0, 5001):
        print("===epoch {}===".format(epoch))
        progress = 0
        lossCollection = []
        for inputs, labels in islice(trainSet, trainBatches):
            loss = trainOnBatch(inputs, labels)
            lossCollection.append(loss.reshape(1, -1))
            progress += 1
            showAtSamePlace("progress: {}/{}".format(progress, trainBatches))
        print()
        avgLoss = np.mean(np.concatenate(lossCollection, axis=0))
        trainWriter.add_scalar("Loss", avgLoss, epoch)

        print("start testing")
        lossCollection = []
        progress = 0
        for inputs, labels in islice(devSet, testBatches):
            lossCollection.append(testOnBatch(inputs, labels).reshape(1, -1))
            progress += 1
            showAtSamePlace("progress: {}/{}".format(progress, testBatches))
        print()
        avgLoss = np.mean(np.concatenate(lossCollection, axis=0))
        validWriter.add_scalar("Loss", avgLoss, epoch)

        testOnSample(Path("data/images"), epoch)

        if epoch % 40 == 0:
            saveDir = Path("saves/{}/epoch{}".format(formatDate(startTime), epoch))
            saveDir.mkdir(parents=True)
            torch.save(model.state_dict(), '{}/state_dict.pth'.format(str(saveDir)))
            # grade()


if __name__ == '__main__':
    trainingLoop()
