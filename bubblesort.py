import random

def bubble_sort(str):
    N = len(str)
    for j in range(N):
        for k in range(j+1,N):
            if str[j]>str[k]:
                str[j],str[k]= str[k],str[j]
    return str

if __name__ == "__main__":
    str = [ random.randint(1,1001) for _ in range(10)]
    # str = []
    # for x in range(10):
    #   str.appned()
    print(str)
    bubble_sort(str)
    print(str)