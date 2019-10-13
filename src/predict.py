import argparse
import pickle
from modules import *


def get_args():
    '''
    command-line arguments
    '''
    parser = argparse.ArgumentParser(description='classify documtent category')
    parser.add_argument('-m', '--model', required='True',
                        help='path for trained classifier model')
    return parser.parse_args()


def main():
    model = load_model(args.model)
    while 1:
        print(model.predict(input('>> ')))


def load_model(path):
    '''
    load trained model
    :param path: model path
    '''
    with open(path, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    args = get_args()
    main()
