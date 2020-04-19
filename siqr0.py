#-*- coding:utf-8 -*-

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def sir(v, t, r0):
    x = v[0]
    y = v[1]
    r = r0
    if t > 4:
        r=r0*0.41
    dxdt = r * (1 - x - y) * x - x
    dydt = x

    return [dxdt, dydt]

def seiqr(v,t,beta, p, f, yq, yr, yqr):
    s = v[0]
    e = v[1]
    i = v[2]
    q = v[3]
    r = v[4]

    dsdt = - beta * i * s
    dedt = beta * i * s - f * e
    didt = f * e - p * yq * i - (1-p) * yr * i
    dqdt = p * yq * i - yqr * q
    drdt = (1-p) * yr * i + yqr * q
    return [dsdt, dedt, didt, dqdt, drdt]

def main():
    # 初期状態
    de = 5.0  # exposure to infection
    dq = 5.0  # duration of infectious until quarantine
    dr = 14.0 # duration of infectiousness until recovery
    dqr =10.0 # time-lag between diagnosis to recovery

    f   = 1 / de
    yq  = 1 / dq
    yr  = 1 / dr
    yqr = 1 / dqr

    r0 = 2.5
    beta = r0 * yr
    p = 0.5

    x0 = [
        0.999, # s
        0.0,   # e
        0.001,  # i
        0.0,    # q
        0.0     # r
    ]
    
    params = []
    if True:
        p = 0.0
        # for dq in (1,2,3,4,5):
        for dq in (5,):
            yq = 1 / dq
            params.append((beta, p, f, yq, yr, yqr))
    
    t = np.linspace(0.0, 400.0, 100)

    vl = []
    i = 0
    r = 0
    for ppx in params:
        v = odeint(seiqr, x0, t, args = ppx ) 
        vl.append(v)

        print(v)
        s = v[:,0]
        e = v[:,1]
        i = v[:,2]
        q = v[:,3]
        r = v[:,4]
        
        plt.plot(t, i, label='Beta={:.2f} p={} Dq={}'.format(ppx[0], ppx[1], 1/ppx[3]))
        plt.plot(t, r, label='Beta={:.2f} p={} Dq={}'.format(ppx[0], ppx[1], 1/ppx[3]))
    
    plt.legend()
    plt.grid('both')
    plt.title('SEIQR R0={:.2f} beta={:.2f}'.format(r0, beta, p))
    plt.show()
    
if __name__ == '__main__':
    main()
    
