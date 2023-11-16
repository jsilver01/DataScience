import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def mean_median_mode(df):
    mean ={}
    median = {}
    mode = {}
    for c in ['Score','GDP per capita','Social support', 'Healthy life expectancy','Freedom to make life choices', 'Generosity','Perceptions of corruption']:
        mean[c]=df[c].mean() #평균
        median[c]=df[c].median() #중간값
        mode[c] = dict(df[c].mode()) 

    print(f"mean = {mean}\n")
    print(f"median = {median}\n")
    print(f"mode = {mode}\n")

def std_var(df): #표준편차, 분산
    std ={}
    var = {} 
    for c in ['Score','GDP per capita','Social support', 'Healthy life expectancy','Freedom to make life choices', 'Generosity','Perceptions of corruption']:
        std[c]=df[c].std()
        var[c]=df[c].var() 

    print(f"std = {std}\n")
    print(f"var = {var}\n")

def percentile(df):
    p = [x for x in range(0,101,10)]

    for c in ['Score','GDP per capita','Social support', 'Healthy life expectancy','Freedom to make life choices', 'Generosity','Perceptions of corruption']:
        percentile = np.percentile(df[c],p)
        plt.plot(p, percentile,'o-')
        plt.xlabel('percentile')
        plt.ylabel(c)
        plt.xticks(p)
        plt.yticks(np.arange(0,max(percentile)+1,max(percentile)/10).astype('float'))
        plt.show()

def boxplot(df): #박스그래프
    boxplot = df[['Score','GDP per capita','Social support', 'Healthy life expectancy','Freedom to make life choices', 'Generosity','Perceptions of corruption']].boxplot()
    plt.show()

def histogram(df): #히스토그램
    for c in ['Score','GDP per capita','Social support', 'Healthy life expectancy','Freedom to make life choices', 'Generosity','Perceptions of corruption']:
        plt.hist(df[c],facecolor='blue', bins=20)
        plt.xlabel(c)
        plt.show()

def scatter_plot(df):
    for c1 in ['Score','GDP per capita','Social support', 'Healthy life expectancy','Freedom to make life choices', 'Generosity','Perceptions of corruption']:
        for c2 in ['Score','GDP per capita','Social support', 'Healthy life expectancy','Freedom to make life choices', 'Generosity','Perceptions of corruption']:
            if c1 == c2:
                continue
            plt.scatter(df[c1],df[c2])
            plt.xlabel(c1)
            plt.ylabel(c2)
            plt.show()

if __name__ == '__main__':
    csv_file = '2019 happiness index.csv'
    df = pd.read_csv(csv_file)
    mean_median_mode(df)
    std_var(df)
    percentile(df)
    boxplot(df)
    histogram(df)
    scatter_plot(df) 
    print(df)
