import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns


class NetflixDataset:
    def __init__(self, data_path):
        super(NetflixDataset, self).__init__()
        names = ['user_id', 'item_id', 'rating', 'timestamp']
        self.data = pd.read_csv(os.path.join(data_path, 'u.data'), '\t', names=names, engine='python')

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, item):
        data = self.data.iloc[[item]]
        return data

    def create_matrix(self):
        matrix = self.data.pivot(index='user_id',columns='item_id',values='rating')
        #matrix = matrix[(1682 - matrix.isnull().sum(axis=1)) > 40]
        return matrix

    def explore_dataset(self):
        print('='*40)
        print('Fist 5 rows:\n{}'.format(self.data.head()))
        print('=' * 40)
        print('Total number of rows: {}'.format(len(self.data)))
        print('Number of unique users: {}'.format(self.data['user_id'].unique().shape[0]))
        print('Number of unique items: {}'.format(self.data['item_id'].unique().shape[0]))
        print('=' * 40)
        print('Count of individual rating: \n{}'.format(self.data['rating'].value_counts()))
        print('=' * 40)
        matrix = self.create_matrix()
        print(f'First 5 rows of matrix:\n{matrix.head()}')
        print('=' * 40)
        #sns.set_theme()
        #sns.histplot(data=self.data['rating'], bins=5)
        #plt.show()


