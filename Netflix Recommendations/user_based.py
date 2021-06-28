import argparse
import numpy as np
from utils import NetflixDataset
from scipy.stats import pearsonr
import pandas as pd

DATA_PATH = 'dataset'


def user_based_filtering(user_id, info):
    dataset = NetflixDataset(DATA_PATH)
    if info == 'yes':
        dataset.explore_dataset()
    data_matrix = dataset.create_matrix()
    selected_user = data_matrix[user_id]
    movie_idx = []
    movie_pred = []
    for item in range(len(selected_user)):
        if np.isnan(selected_user[item]):
            for user in data_matrix:
                movie_pred.append(pred(selected_user, user, item))
                movie_idx.append(item)

    print("=" * 8)


def pred(a, b, item_p):
    return 0


def sim(a, b):
    distance = pearsonr(a, b)[0]
    return distance


def rated_by_both(a, b):
    """
    Returns the items that both the users have rated.
    Example:
     user a rated items [4, 3, nan, 3]
     user b rated items [nan, 2, 3, 4]
     function will return [3,3],[2,4]

    """
    idx = []
    for i in range(len(a)):
        if np.isfinite(a[i]) and np.isfinite(b[i]):
            idx.append(i)
    return a[idx], b[idx]


def get_rating(a, b):
    print(a)
    print(b)
    ratings = []
    item_ids = []
    for i in range(6):
        if np.isnan(a[i]):
            if np.isnan(b[i]):
                continue
            else:
                ratings.append(b[i])
                item_ids.append(i)
    return ratings[np.argmax(ratings)], item_ids[np.argmax(ratings)]


def check():
    a = [2, 3]
    b = [2, 4]
    print(sim(a, b))


def test():
    # names = ['user_id', 'item_id', 'rating', 'timestamp']
    # check = pd.read_csv(os.path.join(DATA_PATH, 'u.data'), '\t', names=names, engine='python')
    # print(check.iloc[[10]])

    users = []
    users.append([5, 3, 1, 2, 5, 3])
    users.append([2, 4, 4, 1, 2, 5])
    users.append([4, 3, 1, np.nan, 4, 3])
    users.append([1, 5, 4, np.nan, np.nan, np.nan])
    users = np.array(users)
    df = pd.DataFrame(data=users)
    print(df.head())
    r_user = users[3]
    simi = []
    for user in users:
        simi.append(sim(user, r_user))
    print(simi)
    # df = df.fillna(df.mean(axis=1),axis=1)
    recommended_rating, movie_id = get_rating(r_user, users[1])
    print('Recommended movie:{}'.format(movie_id + 1))
    print('Rating:{}'.format(recommended_rating))
    # sim = []
    # for user in a:
    #    sim.append(pearsonr(user, b)[0])

    # print(sim)
    # print(sim(a,c))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='User-based Collaborative Filtering Recommendation')
    parser.add_argument('-m', '--mode', help='Mode to either get user based or item based')
    parser.add_argument('-i', '--info', default='yes', help='Display data info or not. Set either yes/no. Default yes.')
    parser.add_argument('-u', '--user_id', default=1, help='Enter user from database for comparison')
    argumentParser = parser.parse_args()
    if argumentParser.mode == 'user':
        user_based_filtering(int(argumentParser.user_id) - 1, str(argumentParser.info))
    else:
        test()
