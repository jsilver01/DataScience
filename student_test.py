from db_conn import *

conn, cur = open_db('db_2023')
sql = """select * from student;"""
cur.execute(sql)

row = cur.fetchone() 
while row:
    print(row['sno'],row['sname'])
    row = cur.fetchone()

close_db(conn, cur)