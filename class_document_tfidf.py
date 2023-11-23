import pandas as pd
from db_conn import *

from konlpy.tag import *
### pip install konlpy

#from mecab import MeCab
### pip install python-mecab-ko

import pandas as pd
import os

import matplotlib.pyplot as plt


class class_document_tfidf():
    def __init__(self):
        self.conn, self.cur = open_db()
        self.news_article_excel_file = 'combined_article.xlsx'
        self.pos_tagger = Kkma()
        #self.pos_tagger = Mecab()


    def combine_excel_file(self):
        directory_path = './'
        
        excel_files = [file for file in os.listdir(directory_path) if file.endswith('.xlsx')]
        
        combined_df = pd.DataFrame()
        for file in excel_files:
            try:
                file_path = os.path.join(directory_path, file)
                df = pd.read_excel(file_path)
                df = df[['url', 'title', 'content']]
                combined_df = combined_df.append(df, ignore_index=True)
            except Exception as e:
                print(file_path)
                print(e)
                continue
        
        combined_df.to_excel(self.news_article_excel_file, index=False)        
        
    
    def import_news_article(self):
        drop_sql ="""drop table if exists news_article;"""
        self.cur.execute(drop_sql)
        self.conn.commit()
    
        create_sql = """
            CREATE TABLE IF NOT EXISTS news_article (
              id int auto_increment primary key,
              url varchar(500),
              title varchar(500),
              content TEXT,
              enter_date datetime default now()
            ) ;
        """
    
        self.cur.execute(create_sql)
        self.conn.commit()
    
        file_name = self.news_article_excel_file
        news_article_data = pd.read_excel(file_name)
    
        rows = []
    
        insert_sql = """insert into news_article(url, title, content)
                        values(%s,%s,%s);"""
    
        for _, t in news_article_data.iterrows():
            t = tuple(t)
            try:
                self.cur.execute(insert_sql, t)
            except:
                continue
            #rows.append(tuple(t))
    
        #self.cur.executemany(insert_sql, rows)
        self.conn.commit()


        print("table created and data loaded")     
        
    def extract_nouns(self):
        
        create_sql = """
            drop table if exists extracted_terms;
            drop table if exists term_dict;
            
            create table extracted_terms (
                id int auto_increment primary key,
                doc_id int,
                term varchar(30),
                term_region varchar(10),
                seq_no int,
                enter_date datetime default now()    ,
                index(term)
                );
            
            create table term_dict (
                id int auto_increment primary key,
                term varchar(30),
                enter_date datetime default now(),
                index(term)
                );
        """
        
        self.cur.execute(create_sql)
        self.conn.commit()
        
        
        sql = """select * from news_article;"""
        self.cur.execute(sql)
        
        r = self.cur.fetchone()
        
        noun_terms = set()
        
        rows = []
        
        while r:
            print(f"doc_id={r['id']}")
            i = 0
            title_res = self.pos_tagger.nouns(r['title'])
            content_res = self.pos_tagger.nouns(r['content'])
            
            title_rows = [ (r['id'], t, 'title', i+1) for i, t in enumerate(title_res) ]
            content_rows = [ (r['id'], c, 'content', i+1) for i, c in enumerate(content_res) ]
            
            rows += title_rows
            rows += content_rows

            #print(f"title_res={title_res}")
            #print(f"content_res={content_res}")
            
            noun_terms.update(title_res)
            noun_terms.update(content_res)
            
            r = self.cur.fetchone()
            
        if rows:
            insert_sql = """insert into extracted_terms(doc_id, term, term_region, seq_no)
                            values(%s,%s,%s,%s);"""
            self.cur.executemany(insert_sql, rows)
            self.conn.commit()

        #print(noun_terms)
        print(f"\nnumber of terms = {len(noun_terms)}")
        
        insert_sql = """insert into term_dict(term) values (%s);"""
        self.cur.executemany(insert_sql, list(noun_terms))
        self.conn.commit()
        
    def gen_idf(self):

        create_sql = """
            drop table if exists idf;
            
            create table idf (
                term_id int primary key,
                df int,
                idf float,
                enter_date datetime default now()                
                );
        """
        
        self.cur.execute(create_sql)
        self.conn.commit()
        
        sql = "select count(*) as doc_count from news_article;"
        self.cur.execute(sql)
        self.num_of_docs = self.cur.fetchone()['doc_count']
        
        idf_sql = f""" insert into idf(term_id, df, idf)
                select ntd.id, count(distinct doc_id) as df, log({self.num_of_docs}/count(distinct doc_id)) as idf
                from extracted_terms ent, term_dict ntd
                where ent.term = ntd.term
                group by ntd.id;
            """
        self.cur.execute(idf_sql)        
        self.conn.commit()


    def show_top_df_terms(self):
        sql = """select * from idf idf, term_dict td
                where idf.term_id = td.id
                order by df desc
                ;"""
        self.cur.execute(sql)
        
        res = [ (r['term'], r['df'], r['idf']) for r in self.cur.fetchall() ]

        print("\nTop DF terms:\n")
        
        for r in res[:30]:
            print(f"{r[0]}: df={r[1]}, idf={r[2]}")
            
        # DF 히스토그램 
        df_list = [ r[1] for r in res]
        plt.figure(figsize=(10, 5))
        plt.hist(df_list, bins=100, alpha=0.7, color='blue') 
        plt.title('Histogram of DF')
        plt.xlabel('Document Frequency')
        plt.ylabel('Number of Terms')
        plt.grid(axis='y', alpha=0.75)
        
        plt.show()        


    def show_top_idf_terms(self):
        sql = """select * from idf idf, term_dict td
                where idf.term_id = td.id
                order by idf desc
                ;"""
        self.cur.execute(sql)
        
        res = [ (r['term'], r['df'], r['idf']) for r in self.cur.fetchall() ]

        print("\nTop IDF terms:\n")
        
        for r in res[:30]:
            print(f"{r[0]}: df={r[1]}, idf={r[2]}")


        # IDF 히스토그램 
        idf_list = [ r[2] for r in res]
        plt.figure(figsize=(10, 5))
        plt.hist(idf_list, bins=100, alpha=0.7, color='red') 
        plt.title('Histogram of IDF')
        plt.xlabel('IDF')
        plt.ylabel('Number of Terms')
        plt.grid(axis='y', alpha=0.75)
        
        plt.show()  


    def gen_tfidf(self):

        create_sql = """
            drop table if exists tfidf;
            
            create table tfidf (
                id int auto_increment primary key,
                doc_id int,
                term_id int,
                tf float,
                tfidf float,
                enter_date datetime default now()
                );
        """
        
        self.cur.execute(create_sql)
        self.conn.commit()
        

        tfidf_sql = """ insert into tfidf(doc_id, term_id, tf, tfidf)  
                        select ent.doc_id, ntd.id, count(*) as tf, count(*) * idf.idf as tfidf
                        from extracted_terms ent, term_dict ntd, idf idf
                        where ent.term = ntd.term and ntd.id = idf.term_id
                        group by ent.doc_id, ntd.id;
                    """

        self.cur.execute(tfidf_sql)        
        self.conn.commit()

    def get_keywords_of_document(self, doc):
        sql = f""" 
            select *
            from tfidf tfidf, term_dict td
            where tfidf.doc_id = {doc}
            and tfidf.term_id = td.id
            order by tfidf.tfidf desc
            limit 10;
        """

        self.cur.execute(sql)
        
        r = self.cur.fetchone()
        while r:
            print(f"{r['term']}: {r['tfidf']}")
            r = self.cur.fetchone()
        
    
    def doc_similarity(self, doc1, doc2):
        def cosine_similarity(vec1, vec2):

            dict1 = dict(vec1)
            dict2 = dict(vec2)
            
            common_terms = set(dict1.keys()) & set(dict2.keys())
            dot_product = sum([dict1[term] * dict2[term] for term in common_terms])
            
            vec1_magnitude = sum([val**2 for val in dict1.values()])**0.5
            vec2_magnitude = sum([val**2 for val in dict2.values()])**0.5
            
            if vec1_magnitude == 0 or vec2_magnitude == 0:
                return 0
            else:
                return dot_product / (vec1_magnitude * vec2_magnitude)        
        
        sql1 = f"""select term_id, tfidf from tfidf where doc_id = {doc1};"""
        self.cur.execute(sql1)
        doc1_vector = [(t['term_id'], t['tfidf']) for t in self.cur.fetchall()]
    
        sql1 = f"""select term_id, tfidf from tfidf where doc_id = {doc2};"""
        self.cur.execute(sql1)
        doc2_vector = [(t['term_id'], t['tfidf']) for t in self.cur.fetchall()]
    
        return cosine_similarity(doc1_vector, doc2_vector)
        
        
    def sort_similar_docs(self, doc):
        sim_vector = []

        for i in range(1,339):
            if i == doc:
                continue
            sim = cdb.doc_similarity(doc, i)
            sim_vector.append((i, sim))
        
        sorted_sim_vector = sorted(sim_vector, key=lambda x: x[1], reverse=True)
        print(sorted_sim_vector)
    
        
if __name__ == '__main__':
    cdb = class_document_tfidf()
    #cdb.combine_excel_file()
    #cdb.import_news_article()
    cdb.extract_nouns()
    cdb.gen_idf()
    #cdb.show_top_df_terms()
    #cdb.show_top_idf_terms()
    #cdb.gen_tfidf()
    #cdb.get_keywords_of_document(160)
    #cdb.sort_similar_docs(160)
