from db_conn import *

import os
import pandas as pd
import numpy as np
from konlpy.tag import *

class DocumentVectorModel():
    def __init__(self):
        self.conn, self.cur = open_db()
        self.news_article_excel_file = 'merged_article.xlsx'
        self.pos_tagger = Kkma()

    def combine_excel_file(self):
        directory_path = './'  # 현재 디렉토리
        excel_files = ['combined_article.xlsx', 'combined_article2.xlsx']  # 병합할 파일들 명시

        combined_df = pd.DataFrame()
        for file in excel_files:
            try:
                file_path = os.path.join(directory_path, file)
                df = pd.read_excel(file_path)
                df = df[['url', 'title', 'content']]  # 필요한 컬럼만 선택
                df = df.dropna(subset=['url'])  # 'url' 컬럼이 비어 있는 행 삭제
                df = df.dropna(subset=['title'])
                df = df.dropna(subset=['content'])
                combined_df = pd.concat([combined_df, df], ignore_index=True)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                continue

        # 새로운 파일명으로 엑셀 파일 저장
        combined_df.to_excel(self.news_article_excel_file, index=False)
        print("파일 병합 완료!")

    def import_news_article(self):
        drop_sql ="""drop table if exists news_article;"""
        self.cur.execute(drop_sql)
        self.conn.commit()

        # 엑셀 파일로부터 데이터 불러오기
        file_name = self.news_article_excel_file
        news_article_data = pd.read_excel(file_name)
        news_article_data = news_article_data.fillna('')

        # 테이블 생성
        create_sql = """
            CREATE TABLE IF NOT EXISTS news_article (
              id INT AUTO_INCREMENT PRIMARY KEY,
              url VARCHAR(500),
              title VARCHAR(500),
              content TEXT
            );
        """
        self.cur.execute(create_sql)
        self.conn.commit()

        # 데이터베이스에 데이터 삽입
        insert_sql = """INSERT INTO news_article (url, title, content) VALUES (%s, %s, %s);"""
        for _, row in news_article_data.iterrows():
            try:
                self.cur.execute(insert_sql, (row['url'], row['title'], row['content']))
            except Exception as e:
                print(f"Failed to insert row: {e}")
                continue  # 오류가 발생한 행을 건너뛰고 계속 진행
        self.conn.commit()

        print("테이블 생성 및 데이터 삽입 완료!")

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
                enter_date datetime default now(),
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
            
            noun_terms.update(title_res)
            noun_terms.update(content_res)
            
            r = self.cur.fetchone()
            
        if rows:
            insert_sql = """insert into extracted_terms(doc_id, term, term_region, seq_no)
                            values(%s,%s,%s,%s);"""
            self.cur.executemany(insert_sql, rows)
            self.conn.commit()

        print(f"\nnumber of terms = {len(noun_terms)}")
        
        insert_sql = """insert into term_dict(term) values (%s);"""
        self.cur.executemany(insert_sql, list(noun_terms))
        self.conn.commit()
        print("명사 추출 완료!")


    def gen_idf(self):
        # IDF 값을 계산하여 데이터베이스에 저장하는 메서드
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


    def gen_tfidf(self):
        # TF-IDF 값을 계산하여 데이터베이스에 저장하는 메서드
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

        tfidf_sql = """
                        INSERT INTO tfidf (doc_id, term_id, tf, tfidf)
                        SELECT
                            et.doc_id,
                            td.id AS term_id,
                            COUNT(*) AS tf,
                            COUNT(*) * idf.idf AS tfidf
                        FROM
                            extracted_terms et
                        JOIN
                            term_dict td ON et.term = td.term
                        JOIN
                            idf ON td.id = idf.term_id
                        GROUP BY
                            et.doc_id, td.id;
                    """
        self.cur.execute(tfidf_sql)
        self.conn.commit()


    def get_TOP5_keywords_of_document(self, doc):
        # 특정 문서의 상위 5개 키워드를 조회하는 메서드
        sql = f""" 
            select *
            from tfidf tfidf, term_dict td
            where tfidf.doc_id = {doc}
            and tfidf.term_id = td.id
            order by tfidf.tfidf desc
            limit 5;
        """

        self.cur.execute(sql)
        
        r = self.cur.fetchone()
        print("해당 문서의 Top 5 키워드")
        while r:
            print(f"{r['term']}: {r['tfidf']}")
            r = self.cur.fetchone()


    def cosine_similarity(self, vec1, vec2):
        vec1_array = np.array([val for val in vec1.values()])
        vec2_array = np.array([vec2.get(key, 0) for key in vec1.keys()])
        dot_product = np.dot(vec1_array, vec2_array)
        vec1_magnitude = np.linalg.norm(vec1_array)
        vec2_magnitude = np.linalg.norm(vec2_array)
        
        if vec1_magnitude == 0 or vec2_magnitude == 0:
            return 0
        else:
            return dot_product / (vec1_magnitude * vec2_magnitude) 
        
    def doc_similarity(self, doc1, doc2):
        # 두 문서 간의 코사인 유사도를 계산하는 메서드
        sql1 = f"""select term_id, tfidf from tfidf where doc_id = {doc1};"""
        self.cur.execute(sql1)
        doc1_vector = {t['term_id']: t['tfidf'] for t in self.cur.fetchall()}

        sql2 = f"""select term_id, tfidf from tfidf where doc_id = {doc2};"""
        self.cur.execute(sql2)
        doc2_vector = {t['term_id']: t['tfidf'] for t in self.cur.fetchall()}

        return self.cosine_similarity(doc1_vector, doc2_vector)
    

    def get_TOP3_similar_docs(self, doc):
        # 특정 문서와 가장 유사한 상위 3개 문서를 조회하는 메서드
        sim_vector = []

        for i in range(1,725):
            if i == doc:
                continue
            sim = dv.doc_similarity(doc, i)
            sim_vector.append((i, sim))
        
        sorted_sim_vector = sorted(sim_vector, key=lambda x: x[1], reverse=True)
        print("가까운 문서 3가지 : \n", sorted_sim_vector[:3])
    
    def query_to_vector(self, query):
        # 쿼리에서 명사 추출
        nouns = self.pos_tagger.nouns(query)

        # 쿼리 벡터 생성
        query_vector = {}
        for noun in nouns:
            self.cur.execute("SELECT id FROM term_dict WHERE term = %s", (noun,))
            result = self.cur.fetchone()
            if result:
                term_id = result['id']
                self.cur.execute("SELECT idf FROM idf WHERE term_id = %s", (term_id,))
                result = self.cur.fetchone()
                if result:
                    # TF 값은 쿼리 내의 빈도로 가정
                    tf = query.count(noun)
                    tfidf = tf * result['idf']
                    query_vector[term_id] = tfidf
        return query_vector

    def get_TOP3_documents_for_query(self, query):
        query_vector = self.query_to_vector(query)
        sim_vector = []

        # 모든 문서에 대해 유사도 계산
        self.cur.execute("SELECT id FROM news_article")
        for row in self.cur.fetchall():
            doc_id = row['id']
            self.cur.execute("SELECT term_id, tfidf FROM tfidf WHERE doc_id = %s", (doc_id,))
            doc_vector = dict((res['term_id'], res['tfidf']) for res in self.cur.fetchall())
            sim = self.cosine_similarity(doc_vector, query_vector)
            sim_vector.append((doc_id, sim))

        # 유사도에 따라 상위 3개 문서 선택
        sorted_sim_vector = sorted(sim_vector, key=lambda x: x[1], reverse=True)
        print("사용자 입력 쿼리문에 적합한 문서 3개 : \n",sorted_sim_vector[:3])


if __name__ == '__main__':
    dv = DocumentVectorModel()
    # dv.combine_excel_file()
    # dv.import_news_article()
    # dv.extract_nouns()
    # dv.gen_idf()
    # dv.gen_tfidf()
    dv.get_TOP5_keywords_of_document(5)
    dv.get_TOP3_similar_docs(5)
    dv.get_TOP3_documents_for_query("겨울")