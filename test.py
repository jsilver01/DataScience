from db_conn import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.model_selection import cross_val_score, KFold, cross_validate
import statistics

class class_iris_classification():
    def __init__(self, import_data_flag = True):
        self.conn, self.cur = open_db()
        if import_data_flag:
            self.import_iris_data()

    def import_iris_data(self):
        drop_sql = """drop talbe if exists iris;"""
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

        rows=[]

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

        self.X = [ (t['sepal_length'], t['petal_width'] ) for t in data ]
        self.X = np.array(self.X)

        self.y = [1 if(t['species'] == species) else 0 for t in data]
        self.y = np.array(self.y)

    def data_split_train_test(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)
        print("X_train=", self.X_train)
        print("X_test=", self.X_test)
        print("y_train=", self.y_train)
        print("y_test=", self.y_test)
        