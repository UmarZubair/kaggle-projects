import argparse
from utils import NetflixDataset

DATA_PATH = 'dataset'

def train():
    dataset = NetflixDataset(DATA_PATH)
    print(dataset[0].head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(help='Netflix recommendations')
    parser.add_argument('-m','--mode',help='Mode to either train/visualize/infer')
    argumentParser = parser.parse_args()
    if argumentParser.mode == 'train':
        train()