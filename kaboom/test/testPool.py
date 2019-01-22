import multiprocessing

def write(i, x):
    print(i, "---", x)

a = ["1","2","3"]
b= 'p'

pool = multiprocessing.Pool(2)
pool.starmap(write, zip(a,itertools.repeat(b))) 
pool.close() 
pool.join()