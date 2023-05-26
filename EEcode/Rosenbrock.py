
import timeit
from Initial import K
#ROSENBROCK FUNCTION
print("ROSENBROCK FUNCTION:")

N = 1000
M = K
"""
Gradient descent
"""
code_to_test1 = """
import numpy as np

ans_eff = 0.0001
with open("coordinates.txt", 'r') as f:
    wordlist = f.read().splitlines()
    for i in wordlist:
        x = float(i.split()[0])
        y = float(i.split()[1])
        while abs(1-x) > ans_eff or abs(1-y) > ans_eff:
            derivex, derivey = 400*(x**3) - 400*x*y + 2*x - 2, 200*(y-x*x)
            x = x - 0.000012*derivex
            y = y - 0.000012*derivey
"""
# total_time = timeit.timeit(stmt=code_to_test1, number=N)
# print("Gradient Descent Time: ",total_time/(M*N))
"""
Newton method
"""
code_to_test2 = """
import numpy as np
def Hessian(x, y):
    H = np.array([[1200*x*x - 400*y +2, -400*x], [-400*x, 200]], dtype=np.float64)
    return H
def Derive(x, y):
    F = np.array([[400*(x**3)-400*x*y+2*x-2], [200*(y-x*x)]], dtype=np.float64)
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
"""
total_time = timeit.timeit(stmt=code_to_test2, number=N)
print("Newton Time: ", total_time/(M*N))

code_to_test3 = """
import numpy as np
from scipy.optimize import minimize

def Objective_function(x):
    return (1-x[0])**2 + 100*(x[1]-x[0]*x[0])**2

def Derive(x):
    F = np.array([400*(x[0]**3) - 400*x[0]*x[1] + 2*x[0] - 2, 200*(x[1] - x[0]*x[0])], dtype=np.float64)
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