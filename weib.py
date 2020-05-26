import scipy.stats as s
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def weib(x,n,a):
    return (a / n) * (x / n)**(a - 1) * np.exp(-(x / n)**a)

def test1():
    data = np.loadtxt("stack_data.csv")
    print( data )
    (loc, scale) = s.exponweib.fit_loc_scale(data, 1, 1)
    print( loc, scale )

    x = np.linspace(data.min(), data.max(), 1000)
    plt.plot(x, weib(x, loc, scale))
    #plt.hist(data, data.max(), density=True)
    plt.hist(data, data.max())
    plt.show()


def test2():
    np.random.seed(2018)

    x = np.random.normal(50, 10, 1000)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(x, bins=16, range=(40, 80))
    ax.set_xlabel('length [cm]')
    plt.show()    


def test3():
    data = np.loadtxt("stack_data.csv")
    total = data.sum()

    print( data )
    (loc, scale) = s.exponweib.fit_loc_scale(data, 1, 1)
    print( loc, scale )

    x = np.linspace(data.min(), data.max(), 1000)
    plt.plot(x, weib(x, loc, scale) * total)
    #plt.hist(data, data.max(), density=True)
    plt.hist(data)
    plt.show()

def test4():
    sns.set()
    sns.set_style('whitegrid')
    sns.set_palette('Set1')

    np.random.seed(2018)

    x1 = np.random.normal(40, 10, 1000)
    x2 = np.random.normal(80, 20, 1000)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(x1, bins=50, alpha=0.6)
    ax.hist(x2, bins=50, alpha=0.6)
    ax.set_xlabel('length [cm]')
    plt.show()

def test5():
    plt.style.use('default')
    sns.set()
    sns.set_style('whitegrid')
    sns.set_palette('Set1')

    np.random.seed(2018)

    x1 = np.random.normal(40, 10, 1000)
    x2 = np.random.normal(80, 20, 1000)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.hist([x1, x2], bins=50)
    ax.set_xlabel('length [cm]')

    plt.show()

def test6_stack():
    plt.style.use('default')
    sns.set()
    sns.set_style('whitegrid')
    sns.set_palette('Set1')
    
    np.random.seed(2018)

    x1 = np.random.normal(40, 10, 1000)
    x2 = np.random.normal(80, 20, 1000)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.hist([x1, x2], bins=50, stacked=True)

    plt.show()
    
if __name__ == '__main__':
    sns.set()
    # sns.set_style('whitegrid')
    # sns.set_palette('gray')
    # test1()
    # test2()
    # test3()
    test4()
    # test5()
    # test6_stack()

#
#
#
