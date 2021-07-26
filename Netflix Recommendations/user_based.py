import argparse
import numpy as np
from utils import NetflixDataset
from scipy.stats import pearsonr
DATA_PATH = 'dataset'


def user_based_filtering(user_id, info):
    dataset = NetflixDataset(DATA_PATH)
    if info == 'yes':
        dataset.explore_dataset()
    data_matrix = dataset.create_matrix()
    selected_user = data_matrix.iloc[user_id]
    movie_idx = []
    movie_pred = []
    similarities = []
    for item in range(len(selected_user)):
        if np.isnan(selected_user.iloc[item]):
            for i in range(len(data_matrix)):
                user = data_matrix.iloc[i]
                if np.isfinite(user.iloc[item]):
                    prediction = pred(selected_user, user, item)
                    movie_pred.append(prediction)
                    movie_idx.append(item)
                    # similarities.append(similarity)
    # similarities = np.array(similarities)
    movie_pred = np.array(movie_pred)
    movie_idx = np.array(movie_idx)
    print('User selected: {}'.format(user_id))
    print("=" * 20)
    # print('10 most similar users: \n{}'.format(similarities[similarities.sort()[-10:][::-1]]))
    print("=" * 20)
    print('20 most relevant movies: {}'.format(np.sort(movie_idx)[-20:]))


def most_similar_users(target_user, no_users, data):
    similarities = []
    idx = []
    for user in range(len(data)):
        a , b, index= rated_by_both_(target_user,data.iloc[user])
        similarities.append(sim(a, b))
        idx.append(index)

    simi = np.argsort(similarities)[-no_users:]
    similarities = np.sort(np.array(similarities))[-no_users:]

    return similarities,simi


def pred(a, b, item_p):
    rbp = b.iloc[item_p]
    a, b = rated_by_both(a, b)
    sim_ = sim(a, b)
    rb = np.mean(b)
    ra = np.mean(a)
    pred_ = (ra + (sim_ * (rbp - rb))) / sim_
    return pred_


def sim(a, b):
    if len(a) < 2 or np.isnan(pearsonr(a, b)[0]):
        distance = 0
    else:
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
    a = np.array(a)
    b = np.array(b)
    idx = []
    for i in range(len(a)):
        if np.isfinite(a[i]) and np.isfinite(b[i]):
            idx.append(i)
    return a[idx], b[idx]


def rated_by_both_(a, b):
    """
    Returns the items that both the users have rated.
    Example:
     user a rated items [4, 3, nan, 3]
     user b rated items [nan, 2, 3, 4]
     function will return [3,3],[2,4]

    """
    a = np.array(a)
    b = np.array(b)
    a_ = np.where(np.isfinite(a))
    b_ = np.where(np.isfinite(b))
    idx = np.intersect1d(a_, b_)
    return a[idx], b[idx], idx


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
    dataset = NetflixDataset(DATA_PATH)
    data_matrix = dataset.create_matrix()
    user = data_matrix.iloc[0]
    a, b = rated_by_both_(user,data_matrix.iloc[38])
    print(sim(a,b))
    most_similar, index = most_similar_users(user, 20, data_matrix)
    print(sim())
    print(index)


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
