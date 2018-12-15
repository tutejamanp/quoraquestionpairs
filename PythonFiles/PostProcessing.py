import numpy as np
import pandas as pd
from collections import defaultdict

class PostPreprocessing:
    

    def read_files(self):
        df_train = pd.read_csv("../InputFiles/train.csv")
        df_test = pd.read_csv("../InputFiles/test.csv")
        test_label = np.array(pd.read_csv('../OutputFiles/pickled_output.csv')["is_duplicate"])
        return df_train, df_test, test_label
        
    def post_process(self, df_train, df_test, test_label):

        #Variable initializations
        REPEAT = 2 
        DUP_THRESHOLD = 0.5 
        NOT_DUP_THRESHOLD = 0.1 
        MAX_UPDATE = 0.2 
        DUP_UPPER_BOUND = 0.98 
        NOT_DUP_LOWER_BOUND = 0.01
        
        for i in range(REPEAT):
            dup_neighbors = defaultdict(set)

            for dup, q1, q2 in zip(df_train["is_duplicate"], df_train["question1"], df_train["question2"]): 
                if dup:
                    dup_neighbors[q1].add(q2)
                    dup_neighbors[q2].add(q1)

            for dup, q1, q2 in zip(test_label, df_test["question1"], df_test["question2"]): 
                if dup > DUP_THRESHOLD:
                    dup_neighbors[q1].add(q2)
                    dup_neighbors[q2].add(q1)

            count = 0
            for index, (q1, q2) in enumerate(zip(df_test["question1"], df_test["question2"])): 
                dup_neighbor_count = len(dup_neighbors[q1].intersection(dup_neighbors[q2]))
                if dup_neighbor_count > 0 and test_label[index] < DUP_UPPER_BOUND:
                    update = min(MAX_UPDATE, (DUP_UPPER_BOUND - test_label[index])/2)
                    test_label[index] += update
                    count += 1

        
        for i in range(REPEAT):
            not_dup_neighbors = defaultdict(set)

            for dup, q1, q2 in zip(df_train["is_duplicate"], df_train["question1"], df_train["question2"]): 
                if not dup:
                    not_dup_neighbors[q1].add(q2)
                    not_dup_neighbors[q2].add(q1)

            for dup, q1, q2 in zip(test_label, df_test["question1"], df_test["question2"]): 
                if dup < NOT_DUP_THRESHOLD:
                    not_dup_neighbors[q1].add(q2)
                    not_dup_neighbors[q2].add(q1)

            count = 0
            for index, (q1, q2) in enumerate(zip(df_test["question1"], df_test["question2"])): 
                dup_neighbor_count = len(not_dup_neighbors[q1].intersection(not_dup_neighbors[q2]))
                if dup_neighbor_count > 0 and test_label[index] > NOT_DUP_LOWER_BOUND:
                    update = min(MAX_UPDATE, (test_label[index] - NOT_DUP_LOWER_BOUND)/2)
                    test_label[index] -= update
                    count += 1
        
        submission = pd.DataFrame({'id':df_test["test_id"], 'is_duplicate':test_label})
        submission.to_csv('../OutputFiles/submission.csv', index=False)


#Starting Point
if __name__ == '__main__':
    #Read csv files
    obj = PostPreprocessing()
    df_train, df_test, test_label = obj.read_files()
    obj.post_process(df_train, df_test, test_label)