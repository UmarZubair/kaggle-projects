import argparse
import glob
import numpy as np
import pandas as pd
import torch.optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import random_split
from utils import EthiopicDataset, make_csv, TestEthioipic

TRAIN_DATA = 'dataset/train'
TEST_DATA = 'dataset/test'
MODEL_PATH = 'model/model.pth'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


def train(epoch, lr, device=DEVICE):
    df = make_csv(TRAIN_DATA)
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = EthiopicDataset(df, train_transforms)
    train_dataset, val_dataset = random_split(dataset, [55000, 5000])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    net = Net()
    net.to(device).train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    for epoch in range(epoch):
        print(f"Epoch {epoch}")
        print('-' * 10)
        for batch_idx, (data, target) in enumerate(train_loader):
            if device != 'cpu':
                data = data.to(device)
                target = target.to(device)

            optimizer.zero_grad()
            pred = net(data)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            if (batch_idx + 1) % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                           100. * (batch_idx + 1) / len(train_loader), loss.item()))
        print('\nTrain loss: {}'.format(loss.item()))
        validate(val_loader, net, criterion)
    torch.save(net.state_dict(), MODEL_PATH)


def validate(val_loader, net, criterion, device=DEVICE):
    net.eval()
    loss = 0
    correct = 0

    for data, target in val_loader:
        if device != 'cpu':
            data = data.cuda()
            target = target.cuda()

        pred = net(data)
        loss += criterion(pred, target).item()
        correct += int(sum(target == torch.argmax(pred, dim=1)))

    loss /= len(val_loader.dataset)

    print('Validation loss: {:.4f}, Validation Accuracy: {}/{} ({:.3f}%)\n'.format(
        loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))


def predict(device=DEVICE, save=False):
    net = Net()
    net.load_state_dict(torch.load(MODEL_PATH))
    net.to(device)
    net.eval()
    test_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    test_dataset = TestEthioipic(sorted(glob.glob(TEST_DATA + '/*')), test_transforms)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    predictions = []
    for data in test_dataloader:
        data = data.to(device)
        with torch.no_grad():
            predicts = net(data)
            predicts = predicts.argmax(axis=1)
            predicts = predicts.cpu().numpy()

            for pred in predicts:
                predictions.append(pred)

    predictions = np.array(predictions) + 1
    if save == 'submission':
        with open("submission.csv", "w") as fp:
            fp.write("Id,Category\n")
            for idx in range(10000):
                fp.write(f"{idx:05},{(predictions[idx]) + 1}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', default='train', help='Set mode: train/inference')
    parser.add_argument('-e', '--epoch', default=15, help='Set epochs')
    parser.add_argument('-lr', '--lr', default=0.001, help='Set learning rate')
    parser.add_argument('-save', '--save', default="NA", help='Save either model or submission csv')
    argumentParser = parser.parse_args()

    if argumentParser.mode == 'train':
        print('Device: {}'.format(DEVICE))
        train(int(argumentParser.epoch), float(argumentParser.lr))
    elif argumentParser.mode == 'infer':
        predict(save=str(argumentParser.save))
