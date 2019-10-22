'''
Naive Bayes classifier
'''
from . import classifier
import json
import math
import MeCab
import pickle


class NaiveBayes(classifier.Classifier):
    '''
    Naive Bayes classifier
    '''
    def __init__(self):
        self.__vocabularies = set()
        self.__word_count = {}
        self.__category_count = {}

    def __tokenize(self, sentence):
        '''
        Tokenize sentence
        :param sentence: input sentence
        :type sentence: String
        :return: word list of sentence
        :rtype: list
        '''
        tagger = MeCab.Tagger('-Owakati')
        mecab_result = tagger.parse(sentence).strip()
        return tuple(mecab_result.split(' '))

    def __count_category(self, category):
        '''
        Count up number of category documents
        :param category: category of documents
        :type category: String
        '''
        self.__category_count.setdefault(category, 0)
        self.__category_count[category] += 1

    def __count_word(self, word, category):
        '''
        Count up number of words
        :param word: target word
        :type word: String
        :param category: category of documents
        :type category: String
        '''
        self.__word_count.setdefault(category, {})
        self.__word_count[category].setdefault(word, 0)
        self.__word_count[category][word] += 1
        self.__vocabularies.add(word)

    def __prior_prob(self, category):
        '''
        Calculate prior distribution P(category)
        :param category: category of documents
        '''
        num_categories = sum(self.__category_count.values())
        num_docs_of_category = self.__category_count[category]
        return num_docs_of_category / num_categories

    def __word_prob(self, word, category):
        '''
        Calculate conditional probability P(word|category)
        :param word: target word
        :param category: category of documents
        '''
        if word in self.__word_count[category]:
            num_word_in_category = self.__word_count[category][word] + 1
        else:
            num_word_in_category = 1
        num_words_in_category = sum(self.__word_count[category].values())
        num_vocab = len(self.__vocabularies)
        return num_word_in_category / (num_words_in_category + num_vocab)

    def __score(self, words, category):
        '''
        Calculate conditional probability P(doc|category)
        '''
        score = math.log(self.__prior_prob(category))
        for word in words:
            score += math.log(self.__word_prob(word, category))
        return score

    def train(self, document, category):
        '''
        Train the model
        '''
        words = self.__tokenize(document)
        for word in words:
            self.__count_word(word, category)
        self.__count_category(category)

    def predict(self, document):
        '''
        Predict a most likely category
        '''
        word_list = {} 
        words = self.__tokenize(document)
        for category in self.__category_count.keys():
            word_list[category] = self.__score(words, category)
        return max(word_list.items(), key=lambda x:x[1])[0] 

    def get_score(self, document):
        '''
        Get scores (e.g. Probability) for each category
        '''
        word_list = {}
        words = self.__tokenize(document)
        for category in self.__category_count.keys():
            word_list[category] = self.__score(words, category)
        return word_list
