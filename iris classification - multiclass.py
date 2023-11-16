from db_conn import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.model_selection import cross_val_score, KFold, cross_validate
from sklearn.metrics import classification_report, confusion_matrix

class class_iris_classification():
    def __init__(self, import_data_flag=True):
        self.conn, self.cur = open_db()
        if import_data_flag:
            self.import_iris_data()
        

    def import_iris_data(self):
        drop_sql =""" drop table if exists iris;"""
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

  
    
    def load_data_for_multiclass_classification(self):
        sql = "select * from iris;"
        self.cur.execute(sql)
    
        data = self.cur.fetchall()
        
        #print("data=", data)
    
        self.X = [ (t['sepal_length'], t['sepal_width'], t['petal_length'], t['petal_width'] ) for t in data ]
        #self.X = [ (t['sepal_length'], t['sepal_width'] ) for t in data ]
        self.X = [ (t['sepal_length'], t['petal_length'] ) for t in data ]

        self.X = np.array(self.X)
    
        self.y =  [0 if t['species'] == 'setosa' else 1 if t['species'] == 'versicolor' else 2 for t in data]
        self.y = np.array(self.y)    
        
        #print("X=",self.X)
        #print("y=", self.y)
        
        
    #binary랑 동일하다고 한거같음
    def data_split_train_test(self):
    
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)
        
        '''
        print("X_train=", self.X_train)
        print("X_test=", self.X_test)
        print("y_train=", self.y_train)
        print("y_test=", self.y_test)
        '''

    # 여기 부분 수업
    def classification_performance_eval_multiclass(self, y_test, y_predict, output_dict=False):
        target_names=['setosa', 'versicolor', 'virginica' ]
        labels = [0,1,2]
        
        self.classification_report = classification_report(y_test, y_predict, target_names=target_names, labels=labels, output_dict=output_dict)
        self.confusion_matrix = confusion_matrix(y_test, y_predict, labels=labels)
        
        print(f"[classification_report]\n{self.classification_report}")
        print(f"[confusion_matrix]\n{self.confusion_matrix}")


    def train_and_test_dtree_model(self):
        dtree = tree.DecisionTreeClassifier()    
        dtree_model = dtree.fit(self.X_train, self.y_train)
        self.y_predict = dtree_model.predict(self.X_test)
        
        print(f"self.y_predict[:10]={self.y_predict[:10]}")
        print(f"self.y_test[:10]={self.y_test[:10]}")

    def multiclass_dtree_KFold_performance(self):
        accuracy = []
        precision = []
        recall = []
        f1_score = []

        kfold_reports = []
    
        kf = KFold(n_splits=5, random_state=42, shuffle=True)
    
        for train_index, test_index in kf.split(self.X):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            
            dtree = tree.DecisionTreeClassifier()
            dtree_model = dtree.fit(X_train, y_train)
            y_predict = dtree_model.predict(X_test)
            self.classification_performance_eval_multiclass(y_test, y_predict, output_dict=True)

            kfold_reports.append(pd.DataFrame(self.classification_report).transpose())
            
        for s in kfold_reports:
            print('\n', s)
            
        mean_report = pd.concat(kfold_reports).groupby(level=0).mean()
        print('\n\n', mean_report)
        

def multiclass_dtree_train_test_performance():
    clf = class_iris_classification(import_data_flag=False)
    clf.load_data_for_multiclass_classification()
    #다른부분
    clf.data_split_train_test()
    clf.train_and_test_dtree_model()
    
    clf.classification_performance_eval_multiclass(clf.y_test, clf.y_predict)

def multiclass_dtree_KFold_performance():
    clf = class_iris_classification(import_data_flag=False)
    clf.load_data_for_multiclass_classification()
    #다른부분
    clf.multiclass_dtree_KFold_performance()

if __name__ == "__main__":
    #multiclass_dtree_train_test_performance()  #train-test 로 나눠서하는 법
    multiclass_dtree_KFold_performance()
