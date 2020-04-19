#-*- coding:utf-8 -*-

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

g = 9.8     # 重力定数
m = 1.0     # 質量
h = 10      # 初期位置

def f(x, t):
    ret = [
        x[1],      # 第一式の右辺
        -g / m     # 第二式の右辺
    ]
    return ret


def main():
    # 初期状態
    x0 = [
        0.999,     # S
        0.0,       # E
        0.001,     # I
        0,         # Q
        0          # R
    ]

    # 計算するインターバル
    # 引数は、「開始時刻」、「終了時刻」、「刻み」の順
    t = np.linspace(1.0, 20.0, 1000)
    # t = np.arange(0, 10, 0.1)

    # 積分する
    x = odeint(f, x0, t)

    # 結果を表示する（とりあえずそのまま print）
    print(x)

    plt.plot(x[:,0], x[:,1])
    plt.show()


if __name__ == '__main__':
    main()
    
