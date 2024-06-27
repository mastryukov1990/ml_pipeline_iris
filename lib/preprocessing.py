import yaml
from sklearn import datasets
from sklearn.model_selection import train_test_split
import json

IRIS_FEATURES_MAPPING = {
                         'sepal length (cm)': 0,
                         'sepal width (cm)': 1,
                         'petal length (cm)': 2,
                         'petal width (cm)': 3 
                         }


def save_dict(data: dict, filename: str):
    with open(filename, 'w') as f:
        json.dump(data, f)


def load_dict(filename: str):
    with open(filename, 'r') as f:
        return json.load(f)
    

def get_data():

    with open('params.yaml', 'rb') as f:
        params_data = yaml.safe_load(f)

    config = params_data['train']


    iris = datasets.load_iris()

    features = config['features']
    idxs = [IRIS_FEATURES_MAPPING[f] for f in features]

    x = iris['data']
    x = x[:, idxs]
    x = x.tolist()
    y = iris['target'].tolist()


    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=config['test_size'])

    return train_x, test_x, train_y, test_y