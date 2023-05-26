import timeit
from Initial import K
#Booth Function
print("BOOTH FUNCTION:")
N = 1000
M = K
"""
Gradient descent
"""
code_to_test1 = """
import random
import numpy as np

def Derive(x, y):
    F = np.array([10*x+8*y-34,  8*x+10*y-38], dtype=np.float64)
    return F

ans_eff = 0.0001
with open("coordinates.txt", 'r') as f:
    wordlist = f.read().splitlines()
    for i in wordlist:
        x = float(i.split()[0])
        y = float(i.split()[1])
        while np.linalg.norm(Derive(x, y)) > ans_eff:
            derivex, derivey = Derive(x,y)
            x1 = x - 0.1*derivex
            y1 = y - 0.1*derivey
            coord1 = np.array([[x1], [y1]])
            coord = np.array([[x], [y]])
            if np.linalg.norm(coord1-coord) <= ans_eff:
                break
            x = x1
            y = y1
"""
total_time = timeit.timeit(stmt=code_to_test1, number=N)
print("Gradient Descent Time: ",total_time/(M*N))

"""
Newton method
"""
code_to_test2 = """
import random
import numpy as np

def Hessian(x, y):
    H = np.array([[10, 8], [8, 10]], dtype=np.float64)
    return H
def Derive(x, y):
    F = np.array([[10*x+8*y-34], [8*x+10*y-38]], dtype=np.float64)
    return F

ans_eff = 0.0001

with open("coordinates.txt", 'r') as f:
    wordlist = f.read().splitlines()
    for i in wordlist:
        x = float(i.split()[0])
        y = float(i.split()[1])
        coord = np.array([[x], [y]])

        while np.linalg.norm(Derive(coord[0][0], coord[1][0])) > ans_eff:
            H_inv = np.linalg.inv(Hessian(coord[0][0], coord[1][0]))
            F = Derive(coord[0][0], coord[1][0])
            coord1 = coord - np.dot(H_inv, F)
            if np.linalg.norm(coord1-coord) <= ans_eff:
                break
            coord = coord1.copy()

#print(coord)
"""
total_time = timeit.timeit(stmt=code_to_test2, number=N)
print("Newton Time: ", total_time/(M*N))
"""
BFGS
"""
code_to_test3 = """
import random
import numpy as np
from scipy.optimize import minimize


def Objective_function(x):
    return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2

def Derive(x):
    F = np.array([10*x[0]+8*x[1]-34, 8*x[0]+10*x[1]-38], dtype=np.float64)
    return F

ans_eff = 0.0001

with open("coordinates.txt", 'r') as f:
    wordlist = f.read().splitlines()
    for i in wordlist:
        x = float(i.split()[0])
        y = float(i.split()[1])
        coord = np.array([x, y])
        result = minimize(Objective_function, coord , method='BFGS', jac=Derive, tol=ans_eff)
"""
total_time = timeit.timeit(stmt=code_to_test3, number=N)
print("BFGS Time: ", total_time/(M*N))