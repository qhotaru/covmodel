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
    dr = 21.0 # duration of infectiousness until recovery
    dqr =10.0 # time-lag between diagnosis to recovery

    f   = 1 / de
    yq  = 1 / dq
    yr  = 1 / dr
    yqr = 1 / dqr

    beta = 1.3
    r0 = 2.5  # = beta / r
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
    for p in (0.0, 0.2, 0.5, 0.8, 1.0):
        params.append( (beta, p, f, yq, yr, yqr) )

    # 計算するインターバル
    # 引数は、「開始時刻」、「終了時刻」、「刻み」の順
    # t = np.arange(0, 10, 0.1)
    t = np.linspace(0.0, 300.0, 50)

    # 積分する
    # x = odeint(f, x0, t)

    vl = []
    for ppx in params:
        v = odeint(seiqr, x0, t, args = ppx ) 
        vl.append(v)

        print(v)
        s = v[:,0]
        e = v[:,1]
        i = v[:,2]
        q = v[:,3]
        r = v[:,4]
        
        plt.plot(t, i, label='Beta={:.2f} p={} Dq={}'.format(ppx[0], ppx[1], 1/ppx[3] ))

    plt.legend()
    plt.title('SIR beta={} p={}'.format(beta, p))
    plt.show()

def main2():
    # 初期状態
    x0 = [
        0.0001,    # 第一式の初期条件
        0     # 第二式の初期条件
    ]

    # 計算するインターバル
    # 引数は、「開始時刻」、「終了時刻」、「刻み」の順
    # t = np.arange(0, 10, 0.1)
    t = np.linspace(1.0, 300.0, 1000)

    # 積分する
    # x = odeint(f, x0, t)
    r0 = 2.5
    v = odeint(sir, x0, t, args = (r0,) )

    # 結果を表示する（とりあえずそのまま print）
    print(v)

    x = v[:,0]
    y = v[:,1]

    o = []
    pre = 0
    for v in x:
        newval = v + pre
        o.append(newval)
        pre = newval
    plt.plot(t, x, label='infected')
    # plt.plot(t, y, label='recovered')
    # plt.plot(t, x+y, label='total')
    # plt.plot(t, o, label='total')

    plt.legend()
    plt.title('SIR R0={}'.format(r0))
    plt.show()
    
if __name__ == '__main__':
    main()
    
