from collections import Counter
import math

# 문서 데이터
docs = [
    "new home sales top forecasts",
    "home sales rise in july",
    "increase in home sales in july",
    "new home sales rise in November"
]

# 각 문서별로 단어 빈도(tf) 계산
tf_docs = [Counter(doc.split()) for doc in docs]

# 각 단어별 문서 빈도(df) 계산
df_terms = Counter()
for doc in tf_docs:
    for term in doc:
        df_terms[term] += 1

# 전체 문서 수
N = len(docs)

# 각 단어별 역문서 빈도(idf) 계산
idf_terms = {term: math.log(N / df) for term, df in df_terms.items()}

# 각 문서별 TF-IDF 벡터 계산
tf_idf_docs = []
for tf_doc in tf_docs:
    tf_idf_doc = {term: (tf / sum(tf_doc.values())) * idf_terms[term] for term, tf in tf_doc.items()}
    tf_idf_docs.append(tf_idf_doc)

# 결과 출력
print("td : ",tf_docs)
print("=========================================")
print("df : ",df_terms)
print("=========================================")
print("idf : ",idf_terms)
print("=========================================")
print("tf-idf : ",tf_idf_docs)
print("=========================================")
