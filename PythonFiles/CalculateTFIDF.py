import numpy as np
import pandas as pd
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from time import time
from collections import Counter

class CalculateTFIDF:
    
    if __name__ == '__main__':
        
        trainData = pd.read_csv('../LematizedFiles/trainlem.csv', engine='python')
        testData = pd.read_csv('../LematizedFiles/testlem.csv', engine='python')

        def get_weight(count, eps=10000, min_count=2):
            if count < min_count:
                return 0
            else:
                return 1 / (count + eps)


        eps = 5000 
        words = (" ".join(trainData['lem_question1'])).lower().split()
        counts = Counter(words)
        words2 = (" ".join(trainData['lem_question2'])).lower().split()
        counts2 = Counter(words2)
        totalcount = counts + counts2
        weights = {word: get_weight(count) for word, count in totalcount.items()}

        print('Most common words and weights: \n')
        check_list = sorted(weights.items(), key=lambda x: x[1] if x[1] > 0 else 9999)[:45]
        stops = [i[0] for i in check_list]
        print(stops)



        print('\nLeast common words and weights: ')
        (sorted(weights.items(), key=lambda x: x[1], reverse=True)[:10])
        
        
        def tfidf_word_match_share(row):
            q1words = {}
            q2words = {}
            for word in str(row['lem_question1']).lower().split():
                if word not in stops:
                    q1words[word] = 1
            for word in str(row['lem_question2']).lower().split():
                if word not in stops:
                    q2words[word] = 1
            if len(q1words) == 0 or len(q2words) == 0:
                # The computer-generated chaff includes a few questions that are nothing but stopwords
                return 0

            shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]
            total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]

            R = np.sum(shared_weights) / np.sum(total_weights)
            return R
        
        
        tfidf_train_word_match = trainData.apply(tfidf_word_match_share, axis=1, raw=True)
        tfidf_test_word_match = testData.apply(tfidf_word_match_share, axis=1, raw=True)
        
        
        trainData['lem_tfidf_word_match'] = tfidf_train_word_match
        testData['lem_tfidf_word_match'] = tfidf_test_word_match

        print(tfidf_test_word_match)

        print(trainData.info())


        trainData.to_csv('../LematizedFiles/trainlem.csv', index = False)
        testData.to_csv('../LematizedFiles/testlem.csv', index = False)
        
        
