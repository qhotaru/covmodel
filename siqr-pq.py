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
    de =  5.0  # exposure to infection
    dq =  5.0  # duration of infectious until quarantine
    dr =  21.0 # duration of infectiousness until recovery
    dqr = 10.0 # time-lag between diagnosis to recovery

    f   = 1 / de
    yq  = 1 / dq
    yr  = 1 / dr
    yqr = 1 / dqr

    r0 = 2.5
    beta = 1.3
    beta = r0 * yr
    p = 0.5

    params = []

    x0 = [
        0.99999999, # s
        0.0,   # e
        0.00000001,  # i
        0.0,    # q
        0.0     # r
    ]

    pop = 1.2 * 10 ** 8
    xxx = np.linspace(0.0,0.8,10)
    t = np.linspace(0.0, 10000.0, 10000)
    if True:
        for dq in (3, 5,7,9, 11):
            yq = 1 / dq
            curve = []
            for p in xxx:
                # params.append((beta, p, f, yq, yr, yqr))
                param = (beta, p, f, yq, yr, yqr)
                v = odeint(seiqr, x0, t, args=param)
                s = v[:,0]
                e = v[:,1]
                i = v[:,2]
                q = v[:,3]
                r = v[:,4]
                curve.append( np.max( i ) * pop)
                
            plt.plot(xxx, curve, label='Dq={}'.format(dq))
    
    plt.legend()
    plt.title('SEIQR R0={} Beta={:.2f} Dr={}'.format(r0, beta, dr))
    plt.grid('both')
    plt.yscale('log')
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
    t = np.linspace(1.0, 20.0, 1000)

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
    
