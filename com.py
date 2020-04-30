#-*- coding:utf-8 -*-
#!/usr/bin/python
#
#
#

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.special import comb
import pylab
from scipy import stats
import scipy.stats as st

def domain():
    sns.set()

    n = 60 # 60回の試行
    p = 0.5 # 表が出る確率(帰無仮説)

    # 「反復試行の確率」の定理
    xx = pd.Series([comb(float(n), x)*p**x*(1-p)**(float(n)-x) for x in range(0, n+1)])

    # ヒストグラムの描画
    pylab.figure(figsize=(10,6))
    plt.subplot(2,1,1)
    plt.plot(xx.index, xx)
    plt.xlim(0,n)
    plt.xticks(np.arange(0,n+1,10))

    # 曲線でplot
    #plt.plot(xx.index, xx, color='r', linewidth=4)

    # 表が出る確率0.5、20回中16回表を出すことに成功したときのp値
    # binom_testは両側確率で出てくるので1/2
    results = []
    for i in range(n+1):
        result = stats.binom_test(i, n, p, alternative='greater')
        results.append(1-result)

    border_h = np.full(len(results), 0.95)
    border_l = np.full(len(results), 0.05)
    
    plt.subplot(2,1,2)
    plt.plot(xx.index, results, marker='o')
    plt.plot(xx.index, border_h)
    plt.plot(xx.index, border_l)

    for omote in range(36,39):
        pomote = stats.binom_test(omote, n, p, alternative='greater')
        print('表が{}回回出たときのp値: {:.3f}'.format(omote, pomote))

    plt.show()

def domain2():
    sns.set()

    # 60回の試行
    n = 60
    # 表が出る確率(帰無仮説)

    p = 0.5

    # 「反復試行の確率」の定理
    xx=pd.Series([comb(float(n), x)*p**x*(1-p)**(float(n)-x) for x in range(0, n+1)])

    # ヒストグラムの描画
    pylab.figure(figsize=(10,6))
    plt.subplot(2,2,1)
    plt.plot(xx.index, xx)
    plt.xlim(0,n)
    plt.xticks(np.arange(0,n+1,10))

    # 曲線でplot
    #plt.plot(xx.index, xx, color='r', linewidth=4)

    # 表が出る確率0.5、20回中16回表を出すことに成功したときのp値
    # binom_testは両側確率で出てくるので1/2
    results = []
    for i in range(n+1):
        result = stats.binom_test(i, 60, 0.5, alternative='greater')
        results.append(1-result)
        
    plt.subplot(2,2,2)
    plt.plot(xx.index, results)
    
    plt.show()

    p1 = stats.binom_test(36, 60, 0.5, alternative='greater')
    p2 = stats.binom_test(37, 60, 0.5, alternative='greater')
    p3 = stats.binom_test(38, 60, 0.5, alternative='greater')
    print('表が36回回出たときのp値: ',p1)
    print('表が37回回出たときのp値: ',p2)
    print('表が38回回出たときのp値: ',p3)


if __name__ == '__main__':
    domain()



#
# EOF
#
