from multiprocessing import Process, Lock
from timeit import timeit
import numpy as np

def test():
    # l.acquire()
    # for _ in range(2500):
    matrix = np.random.randn(10000)
    # l.release()
    # return matrix

def run():
    p = []
    for _ in range(10000):
        p.append(Process(target=test))
        p[_].start()
    for proc in p:
        proc.join()
    

if __name__ == "__main__":
    lock = Lock()
    print(timeit(run, number=1))
   