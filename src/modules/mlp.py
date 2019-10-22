'''
MLP classifier
'''
from . import classifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import json
import math
import MeCab
import pickle
from sklearn.neural_network import MLPClassifier


class MLP(classifier.Classifier):
    '''
    MLP classifier with bag-of-words
    '''
    def __init__(self):
        self.model = MLPClassifier(max_iter=10, hidden_layer_sizes=(100,),verbose=10,)
        self.classes = ''
        self.vectorizer = CountVectorizer(analyzer=self.tokenize) 

    def tokenize(self, sentence):
        '''
        Tokenize sentence
        :param sentence: input sentence
        :type sentence: String
        :return: word list of sentence
        :rtype: list
        '''
        tagger = MeCab.Tagger('-Owakati')
        mecab_result = tagger.parse(sentence).strip()
        return mecab_result.split(' ') 

    def __get_vector(self, documents):
        '''
        Vectorize sentences using bag-of-words
        :param documents: sentence list;train data
        :return: sentence vector list
        '''
        bow = self.vectorizer.fit_transform(documents)
        X = bow.todense()
        return X 

    def train(self, documents, categories):
        '''
        Train the model
        :param documents: a list of sentences
        :param categories: a list of categories 
        '''
        X = self.__get_vector(documents)

        le = LabelEncoder()
        le.fit(categories)
        Y = le.transform(categories)
        self.model.fit(X, Y)
        self.classes = le.classes_.tolist()

    def predict(self, document):
        '''
        Predict a most likely category
        :param document: input sentence
        '''
        X = self.vectorizer.transform([document])
        key = self.model.predict(X)
        return self.classes[key[0]]

    def get_score(self, document):
        '''
        Get scores (e.g. Probability) for each category
        :param document: input sentence
        '''
        word_list = {}
        X = self.vectorizer.transform([document])
        key = self.model.predict_proba(X)
        for i, category in enumerate(self.classes):
            word_list[category] = key[0][i]
        return word_list

#if __name__ == '__main__':
#    sents = ['皆おはよう今日もいい天気だね','皆皆生きているんだ友達なんだ']
#    categories = ['Sports','Domestic']
#    model = MLP()
#    model.train(sents,categories)
#    text = '皆皆生きているんだ友達なんだ' 
#    print(model.predict(text))
#    print(model.get_score(text))
    
