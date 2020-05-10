#-*- coding:utf-8 -*-

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import argparse
import pandas as pd

import datetime
from matplotlib.dates import drange
from matplotlib.dates import DateFormatter
from matplotlib.dates import DayLocator
import seaborn as sns

import json

def doparse():
    parser = argparse.ArgumentParser()

    parser.add_argument("-a", action='store_true', help="option")
    parser.add_argument("-p", "--p", nargs='+', type=float, help="p list")
    parser.add_argument("-v", "--valiation1", action='store_true', help="variation")
    parser.add_argument("-d", "--dq", action='store_true', help="variation")
    parser.add_argument("--pq", action='store_true', help="show pq graph")
    parser.add_argument("--qp", action='store_true', help="show Dq graph")

    parser.add_argument("-j", "--jhu", action='store_true', help="show jhu")
    parser.add_argument("--datalist", action='store_true', help="show data list")
    parser.add_argument("--korea", action='store_true', help="show korea")

    parser.add_argument("-x", "--xrange", type=int, help="x axis range")
    parser.add_argument("-l", "--linear", action='store_true', help="show with linear")

    parser.add_argument("--i", action='store_true', help="show I")
    parser.add_argument("--q", action='store_true', help="show Q")
    parser.add_argument("--n", action='store_true', help="show N")

    args = parser.parse_args()
    return args


class realdata:
    filename = "../COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
    # filename = "..\COVID-19\csse_covid_19_data\csse_covid_19_time_series\time_series_covid19_confirmed_global.csv"
    tokyofilename = 'd:/ICF_AutoCapsule_disabled/covid/covid19/data/data.json'
    def __init__(self, args):
        self.tp = None
        self.args = args
        pass

    def load_tokyo(self,args):
        with open(realdata.tokyofilename, encoding='utf-8') as f:
            tokyodic = json.load(f)
            for k, v in tokyodic.items():
                print(k, v)
        pass
    
    def readit(self, filename):
        with open(realdata.filename, encoding='utf-8') as f:
            line = f.readline()
            print(line)
        pass

    def load_korea(self):
        # self.readit(filename)
        # return
        df = pd.read_csv(realdata.filename)
        tp = df.transpose()
        indexname = 'Country/Region'
        cols = tp.loc[indexname,]
        tp.columns = cols
        # print(cols)
        name = "Korea, South"
        korea = tp[name]
        # print(korea)
        kdata = korea[4:]
        print(kdata)
        # print(tp)
        return
    
        tp.set_index(indexname)
        korea = tp[name]
        print(korea)

    def nation_data(self,nationname):
        if self.tp == None:
            df = pd.read_csv(realdata.filename)
            self.tp = df.transpose()
        
            indexname = 'Country/Region'
            cols = self.tp.loc[indexname,]
            self.tp.columns = cols

        name = nationname
        nationcol = self.tp[name]
        # print(nationcol)
        nation = nationcol[4:]
        # print(tp)
        return nation

    def show_graph(self, title):
        plt.grid('both')
        plt.title(title)
        plt.show()
        

    def find_border(self,data,border):
        ix = 0
        for v in data:
            if v >= border:
                break
            ix+=1
        return ix
        
    def view_nation(self,nationname):
        nationdata = self.nation_data(nationname)

        border = 100
        spos = self.find_border(nationdata, border)
        nationdata = nationdata[spos:]

        ll = len(nationdata)
        xx = np.linspace(0.0,ll-1,ll)
        xlabel = pd.to_datetime(nationdata.index)
        delta = datetime.timedelta(days=1)
        dx = drange(xlabel[0], xlabel[-1], delta)

        
        fig, ax = plt.subplots(figsize=(8, 4))

        xticks = xlabel[::7]
        ax.set_xticks(xticks)
        
        # ax.plot(target_df["datetime"], target_df["C1"])
        ax.plot(dx, nationdata[:-1], label=nationname)

        ax.xaxis.set_major_locator(DayLocator(bymonthday=None, interval=7, tz=None))
        ax.xaxis.set_major_formatter(DateFormatter("%m/%d"))

        if not self.args.linear:
            ax.set_yscale('log')
        title = f"{nationname}"

        pop = 5 * 10 ** 7

        print( xx, model.param )
        x0 = model.x0
        x0[2] = 10 ** -4
        v = odeint( model.seiqr, x0, xx, args = model.param )
        
        i = v[:, 2]
        # ax.plot(dx,i[1:] * pop)
        #ax.plot(dx,v[:-1,0] * pop)   # s
        #ax.plot(dx,v[:-1,1] * pop)   # e
        ax.plot(dx,v[:-1,2] * pop, label='i')    # i
        ax.plot(dx,v[:-1,3] * pop, label='q')    # q
        ax.plot(dx, model.p * model.yq * v[:-1,2] * pop, label='n')    # n = p * yq * i
        #ax.plot(dx,v[:-1,4] * pop)   # r

        ax.legend()
        self.show_graph(title)

    def formatter(self):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(target_df["datetime"], target_df["C1"])

        # 軸目盛の設定
        ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=None, interval=7, tz=None))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

        ## 補助目盛りを使いたい場合や時刻まで表示したい場合は以下を調整して使用
        # ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=range(0, 24, 1), tz=None))
        # ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M:%S"))

        # 軸目盛ラベルの回転
        labels = ax.get_xticklabels()
        plt.setp(labels, rotation=45, fontsize=10);

        ax.grid()
        pass
    
    pass



def showpic(title, args):
    
    plt.grid('both')
    if not args.linear:
        plt.yscale('log')
    plt.legend()
    plt.title(title)
    plt.show()

class model:
    de =  5.0  # exposure to infection
    dq =  10.0  # duration of infectious until quarantine
    dr =  14.0 # duration of infectiousness until recovery
    dqr = 10.0 # time-lag between diagnosis to recovery

    dparam = (de, dq, dr, dqr)

    f   = 1 / de
    yq  = 1 / dq
    yr  = 1 / dr
    yqr = 1 / dqr

    r0 = 2.5
    beta = 1.3
    beta = r0 * yr
    p = 0.3

    kpop = 1.2 * 10 ** 8
    initq = 100 / kpop
    initi = 100 / kpop
    inite = 100 / kpop

    x0 = [
        1 - initi - initq - inite,  # s
        inite,                      # e
        initi,              # i
        initq,              # q
        0.0                 # r
    ]
    tchange = 0
    plist = (0.0, 0.2, 0.4)
    curelimit = 1.0/6.0 * np.power(10.0,-4.0)
    tchange = 0
    yday = 365.0
    years = 90.0
    months = 12.0

    param = (beta, p, f, yq, yr, yqr, tchange, 0)

    def __init__(self):
        pass
    
    def seiqr(v, t, beta, p, f, yq, yr, yqr, dch, curelimit ):
        s = v[0]
        e = v[1]
        i = v[2]
        q = v[3]
        r = v[4]

        if False:
            if dch > 0 and t > dch:
                p = 0.4
            if curelimit > 0 and q * 0.2 >= curelimit * 0.8:
                beta = beta / 5

        if False:
            pass
            #  p += int( t / 7 ) * 0.1
            #  if p > 0.9:
            #      p = 0.9

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

    initi = 10.0 ** -7
    inite = 10.0 ** -7
    
    x0 = [
        1 - inite,  # s
        inite,        # e
        0.0,      # i
        0.0,        # q
        0.0         # r
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
    if type(args.xrange ) == int:
        xmax = args.xrange

    print("xrange = {}".format(args.xrange))
    print(f"xmax={xmax} - {args.xrange}")

    t = np.linspace(0.0, xmax, 1000)
    t2 = t
    pop = 1.2 * np.power(10.0,8)

    xticks = np.linspace(0.0,xmax ,11)
    xticksm = np.linspace(0.0,years, 10 * 12 )

    plt.xticks(xticks)
    # plt.ylim(ymin=np.power(10.0,-6) * pop, ymax=np.power(10.0, -3) * pop)
    # plt.ylim(ymin=1, ymax=np.power(10.0, -3) * pop)
    # plt.ylim(ymin=10.0, ymax=10**8)

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
        
        # plt.plot(t2, pop * i, label='I p={} Dq={} t={}'.format( ppx[1], 1/ppx[3], ppx[6] ))
        p = ppx[1]
        yq = ppx[3]
        yqr = ppx[5]
        dch = ppx[6]
        n = p * yq * i

        if args.q:
            plt.plot(t2, pop * q, label='Q p={} Dq={} t={}'.format( ppx[1], 1/ppx[3], ppx[6] ))
        if args.n:
            plt.plot(t2, pop * n, label='N p={} Dq={} t={}'.format( p, 1/yq, dch ))
        if args.i:
            plt.plot(t2, pop * i, label='I p={} Dq={} t={}'.format( p, 1/yq, dch ))
            

    cure = np.full(len(t), curelimit )
    hotel = np.full(len(t), curelimit /0.2 )
    plt.plot(t2,pop*cure, label='current')
    plt.plot(t2,pop*hotel, label='extended')

    showpic('SEIQR', args)
    
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
    showpic(title, args)

def dopq(args):
    # 初期状態

    params = []

    pop = 1.2 * 10 ** 8
    xxx = np.linspace(0.0,0.8,10)
    t = np.linspace(0.0, 10000.0, 10000)

    beta, p, f, yq, yr, yqr, tchanged, cure = model.param
    de, dq, dr, dqr = model.dparam
    r0              = beta / yr

    if True:
        for dq in (3, 5,7,9, 11):
            yq = 1 / dq
            curve = []
            for p in xxx:
                # params.append((beta, p, f, yq, yr, yqr))
                param = (beta, p, f, yq, yr, yqr, 0, 0)
                v = odeint(model.seiqr, model.x0, t, args=param)
                s = v[:,0]
                e = v[:,1]
                i = v[:,2]
                q = v[:,3]
                r = v[:,4]
                curve.append( np.max( i ) * pop)
                
            plt.plot(xxx, curve, label='Dq={}'.format(dq))
    
    title = 'SEIQR PQ R0={} Beta={:.2f} Dr={}'.format(r0, beta, dr)
    showpic(title,args)

def doqp(args):
    beta, p, f, yq, yr, yqr, tchanged, cure = model.param
    de, dq, dr, dqr = model.dparam
    r0              = beta / yr
    pop = 1.2 * 10 ** 8

    xxx = np.linspace(0.0,0.8,9)
    t = np.linspace(0.0, 10000.0, 10000)

    dql = (3, 5,7,9, 11)
    if True:
        for p in xxx:
            curve = []
            for dq in dql:
                yq = 1 / dq
                param = (beta, p, f, yq, yr, yqr, 0, 0)
                v = odeint(model.seiqr, model.x0, t, args=param)
                s = v[:,0]
                e = v[:,1]
                i = v[:,2]
                q = v[:,3]
                r = v[:,4]
                curve.append( np.max( i ) * pop)
                
            plt.plot(dql, curve, label='p={:.2f}'.format(p))
    
    title = 'SEIQR QP R0={} Beta={:.2f} Dr={}'.format(r0, beta, dr)
    xlabel = 'Dq'
    ylabel = 'Max Infected(num of person)'
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    showpic(title,args)

def dojhu(args):
    jhu = realdata(args)
    # jhu.load_korea()
    name = 'Korea, South'
    jhu.view_nation(name)
    
    pass

if __name__ == '__main__':
    args = doparse()

    sns.set()
    
    if args.valiation1:
        dovali1(args)
        pass
    elif args.dq:
        dodq(args)
    elif args.pq:
        dopq(args)
    elif args.qp:
        doqp(args)
    elif args.jhu:
        dojhu(args)
    elif args.korea:
        poi = realdata(args)
        poi.load_korea()
    else:
        main(args)

#
# EOF
#
