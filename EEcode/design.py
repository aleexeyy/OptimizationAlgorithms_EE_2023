import matplotlib.pyplot as plt
import autograd.numpy as np

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
from matplotlib import animation
from IPython.display import HTML

from autograd import elementwise_grad, value_and_grad
from scipy.optimize import minimize
from collections import defaultdict
from itertools import zip_longest
from functools import partial
def make_minimize_cb(path=[]):
    
    def minimize_cb(xk):
        path.append(np.copy(xk))

    return minimize_cb
def Newton(x, y, path=[]):
    import numpy as np
    def Hessian(x, y):
        H = np.array([[2, 0], [0, 2 ]], dtype=np.float64)
        return H
    def Derive(x, y):
        F = np.array([[2*x], [2*y]], dtype=np.float64)
        return F

    ans_eff = 0.0001

    coord = np.array([[x], [y]])
    while np.linalg.norm(Derive(coord[0][0], coord[1][0])) > ans_eff:
        path.append([coord[0][0], coord[1][0]])
        H_inv = np.linalg.inv(Hessian(coord[0][0], coord[1][0]))
        F = Derive(coord[0][0], coord[1][0])
        coord1 = coord - np.dot(H_inv, F)
        if np.linalg.norm(coord1-coord) <= ans_eff:
            break
        coord = coord1.copy()
    path.append([coord[0][0], coord[1][0]])
    return path
    
f  = lambda x, y: x**2 + y**2
xmin, xmax, xstep = -20, 20, .3
ymin, ymax, ystep = -40, 40, .3
x, y = np.meshgrid(np.arange(xmin, xmax + xstep, xstep), np.arange(ymin, ymax + ystep, ystep))
z = f(x, y)
minima = np.array([0., 0.])
minima_ = minima.reshape(-1, 1)
dz_dx = elementwise_grad(f, argnum=0)(x, y)
dz_dy = elementwise_grad(f, argnum=1)(x, y)

x0 = np.array([-17., 13.])
func = value_and_grad(lambda args: f(*args))
path_ = [x0]
res = minimize(func, x0=x0, method='BFGS',
               jac=True, tol=1e-20, callback=make_minimize_cb(path_))
#path_ = np.copy(Newton(x0[0], x0[1], path_))
path = np.array(path_).T
fig, ax = plt.subplots(figsize=(10, 6))

ax.contour(x, y, z, levels=np.logspace(0, 5, 35), norm=LogNorm(), cmap=plt.cm.jet)
ax.quiver(path[0,:-1], path[1,:-1], path[0,1:]-path[0,:-1], path[1,1:]-path[1,:-1], scale_units='xy', angles='xy', scale=1, color='k')
ax.plot(*minima_, 'r*', markersize=18)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')

ax.set_xlim((xmin, xmax))
ax.set_ylim((ymin, ymax))
plt.show()