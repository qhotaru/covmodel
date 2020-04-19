#-*- coding:utf-8 -*-

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import argparse

class model:
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

    x0 = [
        0.999999, # s
        0.0,   # e
        0.000001,  # i
        0.0,    # q
        0.0     # r
    ]
    tchange = 0
    plist = (0.0, 0.2, 0.4)
    curelimit = 1.0/6.0 * np.power(10.0,-4.0)
    tchange = 0
    yday = 365.0
    years = 90.0
    months = 12.0

    param = (beta, p, f, yq, yr, yqr, tchange, 0)

    def seiqr(v,t,beta, p, f, yq, yr, yqr, dch, curelimit ):
        s = v[0]
        e = v[1]
        i = v[2]
        q = v[3]
        r = v[4]

        if dch > 0 and t > dch:
            p = 0.4

        if curelimit > 0 and q * 0.2 >= curelimit * 0.8:
            beta = beta / 5
    
        dsdt = - beta * i * s
        dedt = beta * i * s - f * e
        didt = f * e - p * yq * i - (1-p) * yr * i
        dqdt = p * yq * i - yqr * q
        drdt = (1-p) * yr * i + yqr * q
        return [dsdt, dedt, didt, dqdt, drdt]

    def sir(v, t, r0):
        x = v[0]
        y = v[1]
        r = r0
        if t > 4:
            r=r0*0.41
        dxdt = r * (1 - x - y) * x - x
        dydt = x

        return [dxdt, dydt]

def doparse():
    parser = argparse.ArgumentParser()

    parser.add_argument("-a", action='store_true', help="option")
    parser.add_argument("-p", "--p", nargs='+', type=float, help="p list")
    parser.add_argument("-v", "--valiation1", action='store_true', help="variation")
    parser.add_argument("-d", "--dq", action='store_true', help="variation")
    parser.add_argument("--pq", action='store_true', help="show pq graph")
    
    args = parser.parse_args()
    return args

def showpic(title):
    
    plt.grid('both')
    plt.yscale('log')
    plt.legend()
    plt.title(title)
    plt.show()

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

    # beta = 1.3
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
    plist = (0.0, 0.2, 0.4)
    curelimit = 1.0/6.0 * np.power(10.0,-4.0)
    tchange = 0
    yday = 365.0
    years = 90.0
    months = 12.0

    if args.p:
        print("p={}|".format(p))
        plist = args.p
        
    params = []
    for p in plist:
        params.append( (beta, p, f, yq, yr, yqr, tchange, 0) )

    # t = np.arange(0, 10, 0.1)
    
    xmax = yday * 10
    t = np.linspace(0.0, xmax, 1000)
    t2 = t
    pop = 1.2 * np.power(10.0,8)

    xticks = np.linspace(0.0,xmax ,11)
    xticksm = np.linspace(0.0,years, 10 * 12 )

    plt.xticks(xticks)
    # plt.ylim(ymin=np.power(10.0,-6) * pop, ymax=np.power(10.0, -3) * pop)
    # plt.ylim(ymin=1, ymax=np.power(10.0, -3) * pop)
    plt.ylim(ymin=10.0, ymax=10**8)

    vl = []
    for ppx in params:
        v = odeint(model.seiqr, model.x0, t, args = ppx ) 
        vl.append(v)

        # print(v)
        s = v[:,0]
        e = v[:,1]
        i = v[:,2]
        q = v[:,3]
        r = v[:,4]
        
        plt.plot(t2, pop * i, label='I p={} Dq={} t={}'.format( ppx[1], 1/ppx[3], ppx[6] ))
        # plt.plot(t2, pop * q, label='Q p={} Dq={} t={}'.format( ppx[1], 1/ppx[3], ppx[6] ))

    cure = np.full(len(t), curelimit )
    hotel = np.full(len(t), curelimit /0.2 )
    plt.plot(t2,pop*cure, label='current')
    plt.plot(t2,pop*hotel, label='extended')

    showpic('SEIQR')
    
def dodq(args):
    # 初期状態
    de = 5.0  # exposure to infection
    dq = 5.0  # duration of infectious until quarantine
    dr = 14.0 # duration of infectiousness until recovery
    dqr =10.0 # time-lag between diagnosis to recovery

    f   = 1 / de
    yq  = 1 / dq
    yr  = 1 / dr
    yqr = 1 / dqr

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
    plist = (0.0, 0.2, 0.4)
    curelimit = 1.0/6.0 * np.power(10.0,-4.0)
    tchange = 0
    yday = 365.0
    years = 90.0
    months = 12.0

    if args.p:
        print("p={}|".format(p))
        plist = args.p

#
#  configure parameter
#
    params = []
    p = 0.8
    dqlist = (5,5.1,6,8)
    for dq in dqlist:
        yq = 1.0 / dq
        params.append( (beta, p, f, yq, yr, yqr, tchange, 0) )

    # t = np.arange(0, 10, 0.1)
    
    xmax = yday * 10
    t = np.linspace(0.0, xmax, 1000)
    t2 = t
    pop = 1.2 * np.power(10.0,8)

    xticks = np.linspace(0.0,xmax ,11)
    xticksm = np.linspace(0.0,years, 10 * 12 )

    plt.xticks(xticks)
    # plt.ylim(ymin=np.power(10.0,-6) * pop, ymax=np.power(10.0, -3) * pop)
    # plt.ylim(ymin=1, ymax=np.power(10.0, -3) * pop)
    plt.ylim(ymin=10.0, ymax=10**8)

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
        # plt.plot(t2, pop * q, label='Q p={} Dq={} t={}'.format( ppx[1], 1/ppx[3], ppx[6] ))

    cure = np.full(len(t), curelimit )
    hotel = np.full(len(t), curelimit /0.2 )
    plt.plot(t2,pop*cure, label='current')
    plt.plot(t2,pop*hotel, label='extended')

    title = "SEIRQ DQ={}".format(dqlist)
    showpic(title)

def dopq():
    # 初期状態

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



if __name__ == '__main__':
    args = doparse()
    if args.valiation1:
        dovali1(args)
        pass
    elif args.dq:
        dodq(args)
    elif args.pq:
        dopq(args)
    else:
        main(args)
    
