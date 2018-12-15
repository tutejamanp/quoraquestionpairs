import numpy as np
import pandas as pd
import timeit

class QuestionFrequency:
    
    train_org = pd.DataFrame()
    test_org = pd.DataFrame()
    
    def read_files(self):
        self.train_orig = pd.read_csv('../LematizedFiles/trainlem.csv', engine='python')
        self.test_orig = pd.read_csv('../LematizedFiles/testlem.csv',engine='python')
        return self.train_orig, self.test_orig
    
    def add_freq_features(self):
        
        train_org = self.train_org
        test_org = self.test_org
        
        df1 = self.train_orig[['question1']].copy()
        df2 = self.train_orig[['question2']].copy()
        df1_test = self.test_orig[['question1']].copy()
        df2_test = self.test_orig[['question2']].copy()

        df2.rename(columns = {'question2':'question1'},inplace=True)
        df2_test.rename(columns = {'question2':'question1'},inplace=True)

        train_questions = df1.append(df2)
        train_questions = train_questions.append(df1_test)
        train_questions = train_questions.append(df2_test)
        #train_questions.drop_duplicates(subset = ['qid1'],inplace=True)
        train_questions.drop_duplicates(subset = ['question1'],inplace=True)

        train_questions.reset_index(inplace=True,drop=True)
        questions_dict = pd.Series(train_questions.index.values,index=train_questions.question1.values).to_dict()
        train_cp = self.train_orig.copy()
        test_cp = self.test_orig.copy()
        train_cp.drop(['qid1','qid2'],axis=1,inplace=True)

        test_cp['is_duplicate'] = -1
        test_cp.rename(columns={'test_id':'id'},inplace=True)
        comb = pd.concat([train_cp,test_cp])

        comb['q1_hash'] = comb['question1'].map(questions_dict)
        comb['q2_hash'] = comb['question2'].map(questions_dict)

        q1_vc = comb.q1_hash.value_counts().to_dict()
        q2_vc = comb.q2_hash.value_counts().to_dict()

        def try_apply_dict(x,dict_to_apply):
            try:
                return dict_to_apply[x]
            except KeyError:
                return 0
        
        #map to frequency space
        comb['q1_freq'] = comb['q1_hash'].map(lambda x: try_apply_dict(x,q1_vc) + try_apply_dict(x,q2_vc))
        comb['q2_freq'] = comb['q2_hash'].map(lambda x: try_apply_dict(x,q1_vc) + try_apply_dict(x,q2_vc))
        
        df = pd.DataFrame()
        df['q1_freq'] = comb['q1_freq']
        df['q2_freq'] = comb['q2_freq']
        
        df.tail(1000).to_csv('../LematizedFiles/test_ques_freq.csv')
        df.iloc[0:403287].to_csv('../LematizedFiles/train_ques_freq.csv')
        
    def copy_to_orig_lem(self):
        
        train = pd.read_csv('../LematizedFiles/trainlem.csv', engine='python')
        test = pd.read_csv('../LematizedFiles/testlem.csv',engine='python')
        
        train_freq = pd.read_csv('../LematizedFiles/train_ques_freq.csv', engine='python')
        test_freq = pd.read_csv('../LematizedFiles/test_ques_freq.csv',engine='python')
        
        train['q1_freq'] = train_freq['q1_freq']
        train['q2_freq'] = train_freq['q2_freq']
         
        test['q1_freq'] = test_freq['q1_freq']
        test['q2_freq'] = test_freq['q2_freq']
        
        train.to_csv('../LematizedFiles/trainlem.csv')
        test.to_csv('../LematizedFiles/testlem.csv')
        
        

#Starting Point
if __name__ == '__main__':
    #Read csv files
    obj = QuestionFrequency()
    obj.read_files()
    obj.add_freq_features()
    obj.copy_to_orig_lem()
