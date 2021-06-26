import argparse
import pandas as pd
from utils import preprocess_df

TRAIN_DATA = 'dataset/train.csv'
TEST_DATA = 'dataset/test.csv'
SAMPLE_SUB = 'dataset/sample_submission.csv'


def train(epoch, lr, batch_size):
    df = preprocess_df(pd.read_csv(TRAIN_DATA))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', help='NLP project')
    parser.add_argument('-e', '--epoch', default=50, help='Set epoch')
    parser.add_argument('-lr', '--lr', default=0.001, help='Set learning rate')
    parser.add_argument('-b', '--batch', default=16, help='Set batch size')
    argsParser = parser.parse_args()
    train(argsParser.epoch, argsParser.lr, argsParser.batch)
