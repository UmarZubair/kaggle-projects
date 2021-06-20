import argparse
import pandas as pd
from utils import preprocess_df, TitanicDataset, TestDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

TRAIN_DATA = "dataset/train.csv"
TEST_DATA = "dataset/test.csv"
SUBMISSION = "dataset/gender_submission.csv"


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(7, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.dropout(x, p=0.1)
        x = F.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x


def load_data():
    df_train = pd.read_csv(TRAIN_DATA)
    targets = df_train['Survived'].to_numpy().reshape(-1, 1)
    df_train = preprocess_df(df_train)
    dataset = TitanicDataset(df=df_train, targets=targets)
    return dataset


def train(batch_size, num_epochs, lr):
    dataset = load_data()
    net = Net()
    train_dataloader = DataLoader(dataset, batch_size, shuffle=True)
    criterion = nn.BCELoss()
    inputs = dataset[:][0]
    target = dataset[:][1]
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    tepoch = tqdm(range(num_epochs))
    for epoch in tepoch:
        for xb, yb in train_dataloader:
            tepoch.set_description(f"Epoch {epoch + 1}")
            pred = net(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if epoch % 10 == 0:
                tepoch.set_postfix(loss=criterion(net(inputs), target).item())

    df_test = preprocess_df(pd.read_csv(TEST_DATA))
    test_dataset = TestDataset(df_test)
    predictions = np.rint(net(test_dataset[:]).cpu().detach().numpy())
    sub = pd.read_csv(SUBMISSION)
    sub['Survived'] = np.int8(predictions)
    sub.to_csv('sub.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-lr", "--lr", default=0.001, help="Set learning rate")
    parser.add_argument("-b", "--batch", default=10, help="Set batch size")
    parser.add_argument("-e", "--epochs", default=150, help="Set epochs")
    result = parser.parse_args()
    train(int(result.batch), int(result.epochs), float(result.lr))
