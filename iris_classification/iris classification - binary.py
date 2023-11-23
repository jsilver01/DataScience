from db_conn import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.model_selection import cross_val_score, KFold, cross_validate
import statistics

class class_iris_classification():
    def __init__(self, import_data_flag=True):
        self.conn, self.cur = open_db()
        if import_data_flag:
            self.import_iris_data()
    
    def import_iris_data(self):
        drop_sql ="""drop table if exists iris;"""
        self.cur.execute(drop_sql)
        self.conn.commit()
    
        create_sql = """
            create table iris (
                id int auto_increment primary key,
                sepal_length float,
                sepal_width float,
                petal_length float,
                petal_width float,
                species varchar(10), 
                enter_date datetime default now() 
                ); 
        """
    
        self.cur.execute(create_sql)
        self.conn.commit()
    
        file_name = 'iris.csv'
        iris_data = pd.read_csv(file_name)
    
        rows = []
    
        insert_sql = """insert into iris(sepal_length, sepal_width, petal_length, petal_width, species)
                        values(%s,%s,%s,%s,%s);"""
    
        for t in iris_data.values:
            rows.append(tuple(t))
    
        self.cur.executemany(insert_sql, rows)
        self.conn.commit()

    def load_data_for_binary_classification(self, species):
        sql = "select * from iris;"
        self.cur.execute(sql)
    
        data = self.cur.fetchall()
        
        #print("data=", data)
    
        #self.X = [ (t['sepal_length'], t['sepal_width'], t['petal_length'], t['petal_width'] ) for t in data ]
        #self.X = [ (t['sepal_length'], t['sepal_width'] ) for t in data ]
        self.X = [ (t['sepal_length'], t['petal_width'] ) for t in data ]
        
        self.X = np.array(self.X)
    
        self.y = [ 1 if (t['species'] == species) else 0 for t in data]
        self.y = np.array(self.y)    
        
        # print(f"X={self.X}")
        # print(f"y={self.y}")
    
    def data_split_train_test(self):
    
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)
        
        print("X_train=", self.X_train)
        print("X_test=", self.X_test)
        print("y_train=", self.y_train)
        print("y_test=", self.y_test)

    def classification_performance_eval_binary(self, y_test, y_predict):
        tp, tn, fp, fn = 0,0,0,0
        
        for y, yp in zip(y_test, y_predict):
            if y == 1 and yp == 1:
                tp += 1
            elif y == 1 and yp == 0:
                fn += 1
            elif y == 0 and yp == 1:
                fp += 1
            else:
                tn += 1
                
        accuracy = (tp+tn)/(tp+tn+fp+fn)
        precision = (tp)/(tp+fp)
        recall = (tp)/(tp+fn)
        f1_score = 2*precision*recall / (precision+recall)
        
        print("accuracy=%f" %accuracy)
        print("precision=%f" %precision)
        print("recall=%f" %recall)
        print("f1 score=%f" %f1_score)

    def train_and_test_dtree_model(self):
        dtree = tree.DecisionTreeClassifier()    
        dtree_model = dtree.fit(self.X_train, self.y_train)
        self.y_predict = dtree_model.predict(self.X_test)

        print(f"self.y_predict[10] = {self.y_predict[:10]}")
        print(f"self.y_test[10] = {self.y_test[:10]}")

    def binary_dtree_KFold_performance(self):
        dtree = tree.DecisionTreeClassifier()
        
        cv_results = cross_validate(dtree, self.X, self.y, cv=5, scoring=['accuracy', 'precision', 'recall', 'f1'])

       #print(cv_results)
        
        for metric, scores in cv_results.items():
            if metric.startswith('test_'):
                print(f'\n{metric[5:]}: {scores.mean():.2f}')    

def binary_dtree_train_test_performance():
        clf = class_iris_classification(import_data_flag=False)

        clf.load_data_for_binary_classification(species='virginica')
        clf.data_split_train_test()
        clf.train_and_test_dtree_model()
        clf.classification_performance_eval_binary(clf.y_test, clf.y_predict)

def binary_dtree_KFold_performance():
        clf = class_iris_classification(import_data_flag=False)
        clf.load_data_for_binary_classification(species='virginica')
        clf.load_data_for_binary_classification(species = '')
        clf.binary_dtree_KFold_performance()

if __name__ == "__main__":
    #binary_dtree_train_test_performance()
    binary_dtree_KFold_performance()