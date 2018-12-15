import numpy as np
import pandas as pd
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from time import time
from collections import Counter


class Lemmatization:
    
    lemmatizer = WordNetLemmatizer()
    trainData = pd.DataFrame()
    testData = pd.DataFrame()
    
    def __init__(self):
        self.trainData = pd.read_csv('../InputFiles/train.csv')
        self.testData = pd.read_csv('../InputFiles/test.csv')
        
        self.trainData['lem_question1'] = self.trainData.question1.apply(self.lemmatize)
        self.trainData['lem_question2'] = self.trainData.question2.apply(self.lemmatize)
        self.testData['lem_question1'] = self.testData.question1.apply(self.lemmatize)
        self.testData['lem_question2'] = self.testData.question2.apply(self.lemmatize)
        
        self.deal_with_null_values()
        
        self.trainData.to_csv('../LematizedFiles/trainlem.csv')
        self.testData.to_csv('../LematizedFiles/testlem.csv')
        
    
    #Different tags in Word Net to determine type of character
    def isNoun(self, tag):
        return tag in ['NN', 'NNS', 'NNP', 'NNPS']

    def isVerb(self, tag):
        return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

    def isAdverb(self, tag):
        return tag in ['RB', 'RBR', 'RBS']

    def isAdjective(self, tag):
        return tag in ['JJ', 'JJR', 'JJS']
    
    def lemmatize(self, string):
        if string == '':
            return string
        tokens = word_tokenize(string)
        token_pos = pos_tag(tokens)
        tokens_lemma = []
        for word, tag in token_pos:
            wn_tag = self.toWordNet(tag)
            if wn_tag is None:
                wn_tag = wn.NOUN
            lemma = self.lemmatizer.lemmatize(word, wn_tag)
            tokens_lemma.append(lemma)
        return ' '.join(tokens_lemma)
    
    #Penn Tree Bank ie, text corpus
    def toWordNet(self, tag):
        if self.isAdjective(tag):
            return wn.ADJ
        elif self.isNoun(tag):
            return wn.NOUN
        elif self.isAdverb(tag):
            return wn.ADV
        elif self.isVerb(tag):
            return wn.VERB
        return None
    
    #Getting null values for some unsure reason, removing those null values here.
    def deal_with_null_values(self):

        self.trainData = self.trainData[pd.notnull(trainData['lem_question1'])]
        self.trainData = self.trainData[pd.notnull(trainData['lem_question2'])]
    

#Starting Point
if __name__ == '__main__':
    #Read csv files
    obj = Lemmatization()