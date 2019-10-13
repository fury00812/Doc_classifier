'''
Base class for classifier model
'''


class Classifier:
    '''
    Base class for classifier models
    '''
    def __init__(self):
        pass

    def train(self, document, category):
        '''
        Train the model
        '''
        pass

    def predict(self, document):
        '''
        Predict a most likely category
        '''
        pass

    def get_score(self, document):
        '''
        Get scores (e.g. Probability) for each category
        '''
        pass
