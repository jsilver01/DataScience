import pandas as pd

xls_file = 'imdb_clean.csv'
df = pd.read_csv(xls_file)
for index, r in df.iterrows():
    print(r['title'],r['release_year'],r['genre'])