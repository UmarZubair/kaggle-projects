import torch


class TitanicDataset:
    def __init__(self, df, targets):
        self.data = df.to_numpy()
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data = torch.from_numpy(self.data[item].astype('float32'))
        target = torch.from_numpy(self.targets[item].astype('float32'))
        return data, target


class TestDataset:
    def __init__(self, df):
        self.data = df.to_numpy()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data = torch.from_numpy(self.data[item].astype('float32'))
        return data


def preprocess_df(df):
    df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
    if 'Survived' in df:
        df = df.drop(columns='Survived')
    df = df.replace('male', 1)
    df = df.replace('female', 0)
    df = df.fillna(0)
    df['Embarked'] = df['Embarked'].replace('S', 1)
    df['Embarked'] = df['Embarked'].replace('C', 2)
    df['Embarked'] = df['Embarked'].replace('Q', 3)
    df['Embarked'] = df['Embarked'].fillna(0)

    return df
