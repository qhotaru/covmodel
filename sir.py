#-*- coding:utf-8 -*-

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

g = 9.8     # 重力定数
m = 1.0     # 質量
h = 10      # 初期位置

def eee(x,t):
    dxdt = x
    dydt = x
    return [dxdt,dydt]

def sir(v, t, r0):
    x = v[0]
    y = v[1]
    r = r0
    if t > 4:
        r=r0*0.41
    dxdt = r * (1 - x - y) * x - x
    dydt = x

    return [dxdt, dydt]


def main():
    # 初期状態
    x0 = [
        0.0001,    # 第一式の初期条件
        0          # 第二式の初期条件
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

def maineee():
    # 初期状態
    x0 = [
        h,    # 第一式の初期条件
        0     # 第二式の初期条件
    ]

    # 計算するインターバル
    # 引数は、「開始時刻」、「終了時刻」、「刻み」の順
    # t = np.arange(0, 10, 0.1)
    t = np.linspace(0.0, 10.0, 1000)

    # 積分する
    # x = odeint(f, x0, t)
    x = odeint(eee, 1.0, t)

    # 結果を表示する（とりあえずそのまま print）
    print(x)

    plt.plot(t, x)
    plt.show()


if __name__ == '__main__':
    main()
    
