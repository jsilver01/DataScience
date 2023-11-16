
'''
1. minkowski distance for r=1, 2, ...
2. Cosine similarity and dot product
3. most similar k

'''

import numpy as np


def euclidean_distance(vector1, vector2):
    return np.linalg.norm(np.array(vector1) - np.array(vector2))



def minkowski_distance(vector1, vector2, p):
    if p == np.inf:
        return np.max(np.abs(np.array(vector1) - np.array(vector2)))
    else:
        return np.sum(np.abs(np.array(vector1) - np.array(vector2))**p)**(1/p)



def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    vector1_len = np.linalg.norm(vector1)
    vector2_len = np.linalg.norm(vector2)
    
    similarity = dot_product / (vector1_len * vector2_len)
    
    return similarity



def dot_product(vector1, vector2):
    return np.dot(vector1, vector2)  



def topk_vectors(one_vector, vectors, k):

    similarities = [cosine_similarity(one_vector, v) for v in vectors]

    topk_indices = np.argsort(similarities)[-k:][::-1]
    #topk_indices = np.argsort(similarities)[::-1][:k]
    #[start:stop:step]

    topk_vectors = vectors[topk_indices]
    
    print(topk_vectors)
    
#알라뵤



if __name__ == '__main__':

    dim = 10

    vector1 = np.random.randint(0, 100, dim)
    vector2 = np.random.randint(0, 100, dim)

    print(vector1)
    print(vector2)
    
    print(f""" 
          norm1(v1, v2) = {minkowski_distance(vector1, vector2, 1)}
          norm2(v1, v2) b= {minkowski_distance(vector1, vector2, 2)}
          norm_max(v1, v2) = {minkowski_distance(vector1, vector2, np.inf)}
          """)

    '''
    
    num_vectors = 1000
    vectors = np.random.randint(0, 101, (num_vectors, dim))
    
    topk_vectors(vector1, vectors, k=3)


    a = [10, 2, 5, 7]

    sim = np.argsort(a)  
    
    print(sim)
    
    '''