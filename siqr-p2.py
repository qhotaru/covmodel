#-*- coding:utf-8 -*-

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import argparse

def sir(v, t, r0):
    x = v[0]
    y = v[1]
    r = r0
    if t > 4:
        r=r0*0.41
    dxdt = r * (1 - x - y) * x - x
    dydt = x

    return [dxdt, dydt]

inlk = 0

def seiqr(v,t,beta, p, f, yq, yr, yqr, dch, curelimit ):
    s = v[0]
    e = v[1]
    i = v[2]
    q = v[3]
    r = v[4]

    if dch > 0 and t > dch:
        p = 0.4

    if q * 0.2 >= curelimit * 0.8:
        beta = beta / 5
    
    dsdt = - beta * i * s
    dedt = beta * i * s - f * e
    didt = f * e - p * yq * i - (1-p) * yr * i
    dqdt = p * yq * i - yqr * q
    drdt = (1-p) * yr * i + yqr * q
    return [dsdt, dedt, didt, dqdt, drdt]

def doparse():
    parser = argparse.ArgumentParser()

    parser.add_argument("-a", action='store_true', help="option")
    parser.add_argument("-p", "--p", nargs='+', type=float, help="p list")
    
    args = parser.parse_args()
    return args

def main(args):
    # 初期状態
    de = 5.0  # exposure to infection
    dq = 5.0  # duration of infectious until quarantine
    dr = 14.0 # duration of infectiousness until recovery
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
        0.999999, # s
        0.0,   # e
        0.000001,  # i
        0.0,    # q
        0.0     # r
    ]

    plist = (0.0,2.0,4.0)
    if args.p:
        print("p={}|".format(p))
        plist = args.p
        
    params = []
    curelimit = 1.0/6.0 * np.power(10.0,-4.0)
    # params.append( (beta, 0.0, f, yq, yr, yqr, 0) )
    # params.append( (beta, 0.0, f, yq, yr, yqr, 50) )
    for p in plist:
        params.append( (beta, p, f, yq, yr, yqr, 0, curelimit) )

    # 計算するインターバル
    # 引数は、「開始時刻」、「終了時刻」、「刻み」の順
    # t = np.arange(0, 10, 0.1)
    yday = 365.0
    years = 90.0
    months = 12.0
    
    t = np.linspace(0.0, yday * years , 1000)
    t2 = t / yday
    pop = 1.2 * np.power(10.0,8)

    xticks = np.linspace(0.0,years ,10)
    xticksm = np.linspace(0.0,years, 10 * 12 )

    xticks = []
    for k in range(0,int(years),5):
        xticks.append(k)

    plt.xticks(xticks)
    plt.ylim(ymin=np.power(10.0,-6) * pop, ymax=np.power(10.0, -3) * pop)

    # plt.xticks(xticksm, minor=True)
    # plt.xticks(xticks, minor=False)
    # plt.xticks(xticksm, minor=True)
    # 積分する
    # x = odeint(f, x0, t)

    
    vl = []
    for ppx in params:
        v = odeint(seiqr, x0, t, args = ppx ) 
        vl.append(v)

        # print(v)
        s = v[:,0]
        e = v[:,1]
        i = v[:,2]
        q = v[:,3]
        r = v[:,4]
        
        plt.plot(t2, pop * i, label='I p={} Dq={} t={}'.format( ppx[1], 1/ppx[3], ppx[6] ))
        plt.plot(t2, pop * q, label='Q p={} Dq={} t={}'.format( ppx[1], 1/ppx[3], ppx[6] ))

    cure = np.full(len(t), curelimit )
    hotel = np.full(len(t), curelimit /0.2 )
    plt.plot(t2,pop*cure, label='current')
    plt.plot(t2,pop*hotel, label='extended')

    
    plt.grid('both')
    plt.yscale('log')
    plt.legend()
    plt.title('SIEQR Lockdown makes 1/5 of Beta'.format( beta, p))
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
    args = doparse()
    main(args)
    
