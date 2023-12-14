from ..db_conn import *
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import statsmodels.api as sm


class class_linear_regression_for_single_independent_variable():
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
        #여기에서 데이터 변경할 수 있음
        self.X = [ t['midterm'] for t in data ]
        self.X = np.array(self.X)
    
    
        self.y = [ t['score'] for t in data]
        self.y = np.array(self.y)    
        
        # print("X=",self.X)
        # print("y=", self.y)
    

    def plot_data(self):
        plt.scatter(self.X, self.y)  # X와 y의 매칭된 값을 2차원 평면 상에 점으로 찍음
        plt.xlabel('homework')  # x축 레이블
        plt.ylabel('score')  # y축 레이블
        plt.title('midterm vs score')  # 그래프 제목
        plt.grid(True)  # 그리드 라인 표시
        plt.show()  # 그래프 표시        
        

    def least_square(self):
        X = sm.add_constant(self.X)
        model = sm.OLS(self.y, X)
        results = model.fit()

        print(results.summary())
        
        
        print(f"params:\n{results.params}")
        
        self.c, self.m = results.params
        
        print(f"\nm={self.m}, final c={self.c} from least square")
        
        
    def plot_graph_after_regression(self, interactive_flag=False):
        if interactive_flag:
            plt.ion()  # 대화식 모드

        plt.scatter(self.X, self.y, label='Data Points')
        
        #line 그리는 부분
        y_pred = self.m * self.X + self.c
        plt.plot(self.X, y_pred, color='red', label='Regression Line')
        
        plt.xlabel('homework')
        plt.ylabel('score')
        plt.title('Regression Line on Scatter Plot')
        plt.legend()  
        plt.grid(True)  
        plt.ylim(-5, max(self.y) + 5)
        
        if interactive_flag:
            plt.draw()  
            plt.pause(0.1)  
            plt.clf() 
        else:
            plt.show() 

    def gradient_descent(self):
        epochs = 100000
        min_grad = 0.00001
        learning_rate = 0.001
        
        self.m = 0.0
        self.c = 0.0
        
        self.plot_graph_after_regression(interactive_flag=True) 
        
        n = len(self.y)
        #여기서부터 gradient descent
        for epoch in range(epochs):
            
            m_partial = 0.0
            c_partial = 0.0
            
            for i in range(n):
                y_pred = self.m * self.X[i] + self.c
                m_partial += (y_pred-self.y[i])*self.X[i]
                c_partial += (y_pred-self.y[i])
            #편미분값
            m_partial *= 2/n
            c_partial *= 2/n
            
            delta_m = -learning_rate * m_partial
            delta_c = -learning_rate * c_partial
            
            if ( abs(delta_m) < min_grad and abs(delta_c) < min_grad ):
                break
            
            self.m += delta_m
            self.c += delta_c
            
            if ( epoch % 1000 == 0 ):
                print("epoch %d: delta_m=%f, delta_c=%f, m=%f, c=%f" %(epoch, delta_m, delta_c, self.m, self.c) )
                self.plot_graph_after_regression(interactive_flag=True)  #plot 그리기
                
        print(f"\nm={self.m}, final c={self.c} from gradient descent")        
        self.plot_graph_after_regression() #최종 Plot 
        


        


if __name__ == "__main__":
    lr = class_linear_regression_for_single_independent_variable(import_data_flag=False)
    lr.load_data_for_linear_regression()
    #lr.plot_data()
    #lr.train_linear_regression()
    lr.least_square()
    #lr.plot_graph_after_regression()
    lr.gradient_descent()