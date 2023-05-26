import timeit
from Initial import K
#Sphere function
print("SPHERE FUNCTION:")
N = 1000
M = K
"""
Gradient descent
"""
code_to_test1 = """
import numpy as np
def Derive(x, y):
    F = np.array([2*x, 2*y], dtype=np.float64)
    return F
ans_eff = 0.0001
with open("coordinates.txt", 'r') as f:
    wordlist = f.read().splitlines()
    for i in wordlist:
        x = float(i.split()[0])
        y = float(i.split()[1])
        while np.linalg.norm(Derive(x, y)) > ans_eff:
            #print(np.linalg.norm(Derive(x, y)))
            derivex, derivey = Derive(x,y)
            x1 = x - 0.05*derivex
            y1 = y - 0.05*derivey
            coord1 = np.array([[x1], [y1]])
            coord = np.array([[x], [y]])
            if np.linalg.norm(coord1-coord) <= ans_eff:
                break
            x = x1.copy()
            y = y1.copy()
"""
total_time = timeit.timeit(stmt=code_to_test1, number=N)
print("Gradient Descent Time: ",total_time/(M*N))
"""
Newton method
"""
code_to_test2 = """
import numpy as np

def Hessian(x, y):
    H = np.array([[2, 0], [0, 2 ]], dtype=np.float64)
    return H
def Derive(x, y):
    F = np.array([[2*x], [2*y]], dtype=np.float64)
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
import numpy as np
from scipy.optimize import minimize

def Objective_function(x):
    return x[0]**2 + x[1]**2
def Derive(x):
    F = np.array([2*x[0], 2*x[1]], dtype=np.float64)
    return F

ans_eff = 0.0001

with open("coordinates.txt", 'r') as f:
    wordlist = f.read().splitlines()
    for i in wordlist:
        x = float(i.split()[0])
        y = float(i.split()[1])     
        coord = np.array([x, y])
        result = minimize(Objective_function, coord , method='BFGS', jac=Derive, tol=ans_eff)

# solution = result['x']
# evaluation = Objective_function(solution)
#print('Solution: f(%s) = %.5f' % ([round(solution[0], 3), round(solution[1], 3)], evaluation))
"""
total_time = timeit.timeit(stmt=code_to_test3, number=N)
print("BFGS Time: ", total_time/(M*N))