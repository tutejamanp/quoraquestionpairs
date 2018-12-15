import numpy as np 
import pandas as pd 
import pandas as pd
import networkx as nx

class IntersectionCount:
    
    def calculate_intersection_count(self):
        train_df = pd.read_csv('../LematizedFiles/trainlem.csv', engine='python')
        test_df = pd.read_csv('../LematizedFiles/testlem.csv', engine='python')

        df = pd.concat([train_df, test_df])


        g = nx.Graph()
        g.add_nodes_from(df.question1)
        g.add_nodes_from(df.question2)
        edges = list(df[['question1', 'question2']].to_records(index=False))
        g.add_edges_from(edges)


        def get_intersection_count(row):
            return(len(set(g.neighbors(row.question1)).intersection(set(g.neighbors(row.question2)))))

        train_ic = pd.DataFrame()
        test_ic = pd.DataFrame()


        train_df['intersection_count'] = train_df.apply(lambda row: get_intersection_count(row), axis=1)
        test_df['intersection_count'] = test_df.apply(lambda row: get_intersection_count(row), axis=1)
        train_ic['intersection_count'] = train_df['intersection_count']
        test_ic['intersection_count'] = test_df['intersection_count']

        train_df.to_csv("../LematizedFiles/trainlem.csv", index=False)
        test_df.to_csv("../LematizedFiles/testlem.csv", index=False)


#Starting Point
if __name__ == '__main__':
    #Read csv files
    obj = IntersectionCount()
    obj.calculate_intersection_count()
    