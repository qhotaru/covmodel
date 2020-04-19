# -*- coding: utf-8 -*-
# numerila solution using scipy.integrate
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
 
def lorenz(u, t, p, b, r):
    x = u[0]
    y = u[1]
    z = u[2]
    dxdt = -p * x + p * y
    dydt = -x * z + r * x - y
    dzdt = x * y - b * z
    return([dxdt, dydt, dzdt])
 
# parameters
p =10
b = 8.0 / 3.0
r = 28
# initial values
u0 = [0.1, 0.1, 0.1]
# setting
dt = 0.01
T = 40.0
times = np.arange(0.0, T, dt)
# numerical solution using scipy.integrate
args = (p, b, r)
orbit = odeint(lorenz, u0, times, args)
 
#xorbit = []
#yorbit = []
#zorbit = []
#for x, y, z in orbit:
#    xorbit.append(x)
#    yorbit.append(y)
#    zorbit.append(z)
 
#plot3D using matplotlib
fig = plt.figure()
ax = fig.gca(projection='3d')
#ax.plot(xorbit, yorbit, zorbit)
ax.plot(orbit[:, 0], orbit[:, 1], orbit[:, 2])
ax.set_xlabel('x(t)')
ax.set_ylabel('y(t)')
ax.set_zlabel('z(t)')
plt.show()
