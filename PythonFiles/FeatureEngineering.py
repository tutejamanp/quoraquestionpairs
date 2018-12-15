#Importing necessary libraries
import pandas as pd
import numpy as np
import string
import csv
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from numpy import loadtxt
from sklearn import preprocessing 
from fuzzywuzzy import fuzz
import re, math
from collections import Counter


#Various Function Definitions for Feature Engineering
class FeatureEngineeringFunctions:
    
    #Given question2 column is not entirely string
    #Converting question2 to string and in the test set 
    #Even question1 column might have such problem hence converting that too    
    def convert_to_string(self, train):
        train['question1'] = train['question1'].astype(str)
        train['question2'] = train['question2'].astype(str)
        train['lem_question1'] = train['lem_question1'].astype(str)
        train['lem_question2'] = train['lem_question2'].astype(str)


    #Porter Stemming
    def portertok(self, text):
        import nltk
        from nltk.stem.porter import PorterStemmer
        porter = PorterStemmer()
        tokens = nltk.word_tokenize(text)
        return " ".join(porter.stem(word) for word in tokens) 


    #Counts of common words
    #First convert to string then lower case then split on the basis of space then create a set
    #then check the common elements of both set and take count of it
    def count_common(self, question):
        q1, q2 = question
        return len((set(str(q1).lower().split(' ')) & set(str(q2).lower().split(' '))))/len((set(str(q1).lower().split(' ')).union(set(str(q2).lower().split(' ')))))


    #Comparing each question pair
    def check_each_pair(self, question1, question2):
        print(question1, question2)

    #Total unique words in both question pair
    def total_unique_words(self, question):
        q1, q2 = question
        return len(set(str(q1)).union(set(str(q2))))


    #Counts the total number of words present
    def words_count(self, question):
        return len(str(question).split(' '))


    #Gives the length of the string
    def length(self, question):
        return len(str(question))

    #Word count ratio
    def word_count_ratio(self, question):
        q1, q2 = question
        l1 = len(set(str(q1)))*1.0 
        l2 = len(set(str(q2)))
        if l2 == 0:
            return np.nan
        if l1 / l2:
            return l2 / l1
        else:
            return l1 / l2

    #Remove Punctuations
    def remove_punc(self, question):
        return question.translate(str.maketrans('', '', string.punctuation))

    #Same First Word
    def same_first_word(self, question):
        q1, q2 = question
        return floatset(set(str(q1).lower().split(' '))[0] == set(str(q2).lower().split(' '))[0])
    
    #Get the cosine values for each question pair
    def get_cosine(myDataFrame):
        WORD = re.compile(r'\w+')

        vec1 = text_to_vector(str(myDataFrame['lem_question1']))
        vec2 = text_to_vector(str(myDataFrame['lem_question2']))
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum([vec1[x] * vec2[x] for x in intersection])
        sum1 = sum([vec1[x]**2 for x in vec1.keys()])
        sum2 = sum([vec2[x]**2 for x in vec2.keys()])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)        
        if not denominator:
            return 0.0
        else:
            return float(numerator) / denominator
        
    #Converts text to vector    
    def text_to_vector(text):
        words = WORD.findall(text)
        return Counter(words)


#Reading the csv file
#Applying Feature Engineering
class FeatureEngineering:
    
    def get_feature_engineered_data(self, flag):

        #Creating an object of Data Exploration 
        obj = FeatureEngineeringFunctions()
        
        if flag==1:
            train = pd.read_csv('../LematizedFiles/trainlem.csv', engine='python')
        else:
            train = pd.read_csv('../LematizedFiles/testlem.csv',engine='python')

        obj.convert_to_string(train)

        #Data Parameter used for some feature engineering attributes
        EPSILON = 0.0000001

        #Map works element wise on a series
        #Apply works row/column wise on a dataframe
        train['q1_word_num'] = train['question1'].map(obj.words_count)
        train['q2_word_num'] = train['question2'].map(obj.words_count)

        train['q1_length'] = train['question1'].map(obj.length)
        train['q2_length'] = train['question2'].map(obj.length)


        train['word_num_difference'] = abs(train.q1_word_num - train.q2_word_num)
        train['length_difference'] = abs(train.q1_length - train.q2_length)

        train['q1_has_fullstop'] = train.question1.apply(lambda x: int('.' in x))
        train['q2_has_fullstop'] = train.question2.apply(lambda x: int('.' in x))

        train['q1_digit_count'] = train.question1.apply(lambda question: sum([word.isdigit() for word in question]))
        train['q2_digit_count'] = train.question2.apply(lambda question: sum([word.isdigit() for word in question]))
        train['digit_count_difference'] = abs(train.q1_digit_count - train.q2_digit_count)

        train['q1_capital_char_count'] = train.question1.apply(lambda question: sum([word.isupper() for word in question]))
        train['q2_capital_char_count'] = train.question2.apply(lambda question: sum([word.isupper() for word in question]))
        train['capital_char_count_difference'] = abs(train.q1_capital_char_count - train.q2_capital_char_count)

        train['q1_has_math_expression'] = train.question1.apply(lambda x: int('[math]' in x))
        train['q2_has_math_expression'] = train.question2.apply(lambda x: int('[math]' in x))      

        train['common_words'] = train[['question1', 'question2']].apply(obj.count_common, axis=1)
        train['lem_common_words'] = train[['lem_question1', 'lem_question2']].apply(obj.count_common, axis=1)

        train['log_word_share'] = np.log(train[['question1', 'question2']].apply(obj.count_common, axis=1) + EPSILON)
        train['lem_log_word_share'] = np.log(train[['lem_question1', 'lem_question2']].apply(obj.count_common, axis=1) + EPSILON)

        train['word_share_squared'] = (train[['question1', 'question2']].apply(obj.count_common, axis=1) ** 2)
        train['lem_word_share_squared'] = (train[['lem_question1', 'lem_question2']].apply(obj.count_common, axis=1) ** 2)


        train['word_share_sqrt'] = np.sqrt(train[['question1', 'question2']].apply(obj.count_common, axis=1))
        train['lem_word_share_sqrt'] = np.sqrt(train[['lem_question1', 'lem_question2']].apply(obj.count_common, axis=1))

        train['log_length_difference'] = np.log(train.length_difference + EPSILON)
        train['length_difference_squared'] = train.length_difference ** 2
        train['length_difference_sqrt'] = np.sqrt(train.length_difference)

        train['log_lem_tfidf'] = np.log(train.lem_tfidf_word_match + EPSILON)
        train['lem_tfidf_squared'] = train.lem_tfidf_word_match ** 2
        train['lem_tfidf_sqrt'] = np.sqrt(train.lem_tfidf_word_match)

        train['total_unique_words'] = train[['question1', 'question2']].apply(obj.total_unique_words, axis=1)
        train['word_count_ratio'] = train[['question1', 'question2']].apply(obj.word_count_ratio, axis=1)

       
        train['fuzz_qratio'] = train.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1)
        train['fuzz_WRatio'] = train.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1)
        train['fuzz_partial_ratio'] = train.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1)
        train['fuzz_partial_token_set_ratio'] = train.apply(lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
        train['fuzz_partial_token_sort_ratio'] = train.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
        train['fuzz_token_set_ratio'] = train.apply(lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
        train['fuzz_token_sort_ratio'] = train.apply(lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)


        train['cosine_score'] = train.apply(get_cosine, axis=1, raw=True)

        features = ['q1_word_num', 'q2_word_num', 'word_num_difference', 'q1_length', 'q2_length',
                'length_difference', 'q1_has_fullstop', 'q2_has_fullstop', 'q1_digit_count', 'q2_digit_count',
                'digit_count_difference', 'q1_capital_char_count', 'q2_capital_char_count',
                'capital_char_count_difference', 'q1_has_math_expression', 'q2_has_math_expression', 
                'log_length_difference', 'log_word_share', 'word_share_squared', 'word_share_sqrt', 'length_difference_squared', 'length_difference_sqrt', 'common_words',
                'lem_common_words','lem_log_word_share','lem_word_share_squared','lem_word_share_sqrt','tfidf_word_match',
                'lem_tfidf_word_match', 'intersection_count', 'log_lem_tfidf','lem_tfidf_squared','lem_tfidf_sqrt',
                   'total_unique_words', 'word_count_ratio','fuzz_qratio','fuzz_WRatio','fuzz_partial_ratio','fuzz_partial_token_set_ratio',
                   'fuzz_partial_token_sort_ratio','fuzz_token_set_ratio','fuzz_token_sort_ratio', 'cosine_score',
                   'q1_freq', 'q2_freq']


        if flag==1:
            target = 'is_duplicate'
            X = train[features]
            Y = train[target]
            return X,Y
        else:
            X = train[features]
            return X




class StartingClass:
    
    def __init__(self):
        
        #Feature Engineering
        obj = FeatureEngineering()
        X_total,Y_total = obj.get_feature_engineered_data(1)
        test_X = obj.get_feature_engineered_data(0)
        
        X_total['is_duplicate'] = Y_total
        
        X_total.to_csv('../FeatureEngineeringFiles/featured_train.csv', index=False)
        test_X.to_csv('../FeatureEngineeringFiles/featured_test.csv', index=False)


#Starting Point
if __name__ == '__main__':
    #Read csv files
    obj = StartingClass()
