from db_conn import *
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import statsmodels.api as sm

import time



class class_linear_regression_for_multiple_independent_variable():
    def __init__(self, import_data_flag=True):
        self.conn, self.cur = open_db()
        if import_data_flag:
            self.import_dbscore_data()
        

    def import_dbscore_data(self):
        drop_sql =""" drop table if exists db_score;"""
        self.cur.execute(drop_sql)
        self.conn.commit()
    
        create_sql = """
            CREATE TABLE `db_score` (
              `sno` int NOT NULL,
              `attendance` float DEFAULT NULL,
              `homework` float DEFAULT NULL,
              `discussion` int DEFAULT NULL,
              `midterm` float DEFAULT NULL,
              `final` float DEFAULT NULL,
              `score` float DEFAULT NULL,
              `grade` char(1) DEFAULT NULL,
              enter_date datetime default now(),
              PRIMARY KEY (`sno`)
            ) ;
        """
    
        self.cur.execute(create_sql)
        self.conn.commit()
    
        file_name = 'db_score.xlsx'
        dbscore_data = pd.read_excel(file_name)
    
        rows = []
    
        insert_sql = """insert into db_score(sno, attendance, homework, discussion, midterm, final, score, grade)
                        values(%s,%s,%s,%s,%s,%s,%s,%s);"""
    
        for t in dbscore_data.values:
            rows.append(tuple(t))
    
        self.cur.executemany(insert_sql, rows)
        self.conn.commit()


        print("table created and data loaded")

    def load_data_for_linear_regression(self):
        sql = "select * from db_score;"
        self.cur.execute(sql)
    
        data = self.cur.fetchall()
        
        #print("data=", data)
    
        #3개 넣어서 하라는게 머선말?
        self.X = [ (t['midterm'], t['final']) for t in data ]
        self.X = np.array(self.X)

    
        self.y = [ t['score'] for t in data]
        self.y = np.array(self.y)  
        self.y_label = 'score'
        
        #print("X=",self.X)
        #print("y=", self.y)


    def least_square(self):
        X = sm.add_constant(self.X)
        model = sm.OLS(self.y, X)
        results = model.fit()
        print(results.summary())
        
        
        print(f"params:\n{results.params}")

        self.c, self.m = results.params[0], results.params[1:]
        
        print(f"\nm={self.m}, final c={self.c} from least square")
        

    def gradient_descent(self):
      
        self.start_time = time.time()
        
        epochs = 100000
        min_grad = 0.000001
        learning_rate_m = 0.001
        learning_rate_c = 0.001
        
        num_params = self.X.shape[1]
        
        m = [0.0]*num_params
        c = 0.0
        
        n = len(self.y)
    
        
        for epoch in range(epochs):
    
            c_partial = 0.0        
            m_partial = [0.0]*num_params
    
            for i in range(n):
                y_pred = c
                for j in range(num_params):
                    y_pred += m[j] * self.X[i][j]
                
                c_partial += (y_pred-self.y[i])
                for j in range(num_params):
                    m_partial[j] += (y_pred-self.y[i])*self.X[i][j]
            
            c_partial *= 2/n
            for j in range(num_params):
                m_partial[j] *= 2/n
    
            delta_c = -learning_rate_c * c_partial
            delta_m = [0.0]*num_params
            for j in range(num_params):
                delta_m[j] = -learning_rate_m * m_partial[j]
            
            break_condition = True
            if abs(delta_c) > min_grad:
                break_condition = False
            for j in range(num_params):
                if abs(delta_m[j]) > min_grad:
                    break_condition = False
            
            if break_condition:
                break
      
            c = c + delta_c
            for j in range(num_params):
                m[j] = m[j] + delta_m[j]
            
            if ( epoch % 1000 == 0 ):
                print(f"epoch {epoch}: delta_c={delta_c}, delta_m={delta_m}, c={c}, m={m}")
            
        print(f"c={c}, m={m} from gradient descent")
        self.c, self.m = c, m
        
        self.end_time = time.time()
        print('response time=%f seconds' %(self.end_time - self.start_time) )
 


    def gradient_descent_vectorized(self):
        
        self.start_time = time.time()
        
        epochs = 1000000
        min_grad = 0.000001
        learning_rate_m = 0.001
        learning_rate_c = 0.001
        
        num_params = self.X.shape[1]
        
        m = np.zeros(num_params)
        c = 0.0
        
        n = len(self.y)
        
        for epoch in range(epochs):
    
            y_pred = np.sum( m * self.X, axis=1 ) + c
    
            c_partial = np.sum (2*(y_pred-self.y)) /n
            m_partial = np.dot(2 * (y_pred - self.y), self.X) / n
    
            delta_c = -learning_rate_c * c_partial
            delta_m = -learning_rate_m * m_partial
    
            if abs(delta_c) < min_grad and np.all(np.abs(delta_m) < min_grad):
                break
    
            c += delta_c
            m += delta_m
    
            if ( epoch % 1000 == 0 ):
                print(f"epoch {epoch}: delta_c={delta_c}, delta_m={delta_m}, c={c}, m={m}")    

        print(f"c={c}, m={m} from vectorized gradient descent")
        self.c, self.m = c, m
            
        self.end_time = time.time()
        print('response time=%f seconds' %(self.end_time - self.start_time) )
        

      



if __name__ == "__main__":
    lr = class_linear_regression_for_multiple_independent_variable(import_data_flag=False)
    lr.load_data_for_linear_regression()
    lr.least_square()
    lr.gradient_descent()
    lr.gradient_descent_vectorized() #0.98초 ??
    
    