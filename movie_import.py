from db_conn import *

import pandas as pd
import numpy as np

file_name = 'top_movies.csv'
df = pd.read_csv(file_name)
df.columns = ['id','movie_name','release_year','watch_time','movie_rating','metascore','gross','votes','description']
df['votes'] = df['votes'].str.replace(',','').astype(int)
df['gross'] = df['gross'].str.replace('#', '').str.replace(',', '').astype(float)
df['gross'] = df['gross'].fillna(0)
df['metascore'] = df['metascore'].replace([np.nan, np.inf, -np.inf],0).astype(int)
df['release_year'] = df['release_year'].str.extract('(\d+)').astype(int)

insert_sql = """insert into topmovie (id, movie_name, release_year,
                    watch_time, movie_rating, metascore, gross, votes,description)
                values(%s, %s, %s, %s,%s, %s,%s, %s, %s);"""

conn, cur = open_db('db_2023')

truncate_sql = """truncate table topmovie;"""
cur.execute(truncate_sql)
conn.commit()

for index, r in df.iterrows():
    t = (r['id'],r['movie_name'],r['release_year'], r['watch_time'],
      r['movie_rating'],r['metascore'],r['gross'],r['votes'],r['description'])
    try:
        cur.execute(insert_sql, t)
    except Exception as e:
        print(t)
        print(e)
        break
        
    
conn.commit()

close_db(conn, cur)