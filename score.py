import pandas as pd
import numpy as np

xlfile = 'score.xlsx'

df = pd.read_excel(xlfile)

midterm = df['midterm']
final = df['final']

mean_midterm = midterm.mean()
mean_final = final.mean()

#print("mean_midterm = %d" %mean_midterm)
#print("mean_final = %d" %mean_final)

#print(midterm - mean_midterm) # 벡터값으로 array 형태로 저장
#print((midterm - mean_midterm) * (final - mean_final))

cov = ((midterm - mean_midterm) * (final - mean_final)).mean()
print(cov)  # 33.45627876890359


std_midterm = midterm.std()
std_final = final.std()

corr = cov/(std_midterm * std_final)
print(corr)  # 0.5416101489697124