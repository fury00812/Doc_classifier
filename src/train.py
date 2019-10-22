from pathlib import Path
import argparse
import pickle
import json
import tqdm
from modules import *


def get_args():
    '''
    command-line arguments
    '''
    parser = argparse.ArgumentParser(description='train classifier model')
    parser.add_argument('-m', '--model', required='True',
                        help='type of classifier model')
    parser.add_argument('-d', '--train_data', required='True',
                        help='path of train data')
    parser.add_argument('-o', '--save_path', required='True',
                        help='path for saving model')
    return parser.parse_args()


def main():
    all_data = load_json_from_file(args.train_data) 
    model = classifier.Classifier()

    if args.model == 'naive_bayes':
        model = naive_bayes.NaiveBayes()
        train_model_online(model, all_data)
    elif args.model == 'mlp_bow':
        model = mlp.MLP()
        train_model(model, all_data)
    elif args.model == 'mlp_tfidf':
        model = mlp.MLP_Tfidf()
        train_model(model, all_data)

def train_model_online(model, train_data):
    for data in train_data.keys():
        category = train_data[data]['category']
        document = '\n'.join([train_data[data]['title'], train_data[data]['content']])
        model.train(document, category)
    save_model(args.save_path, model)

def train_model(model, train_data):
    categories = []
    documents = []
    for data in train_data.keys():
        categories.append(train_data[data]['category'])
        documents.append('\n'.join([train_data[data]['title'], train_data[data]['content']]))
    model.train(documents, categories)
    save_model(args.save_path, model)

def load_json_from_file(path):
    '''
    load json data from file
    :param path: data path
    '''
    with open(path, 'r') as f:
        all_data = json.load(f)
    return all_data


def save_model(path, model):
    '''
    save trained model
    :param path: save path
    :param model: model to save
    '''
    with open(path, 'wb') as f:
        pickle.dump(model,f)


if __name__ == '__main__':
    args = get_args()
    main()
