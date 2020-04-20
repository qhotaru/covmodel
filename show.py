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

class realdata:
    def __init__(self, args):
        self.tp = None
        self.args = args
        pass

    def readit(self, filename):
        with open(filename, encoding='utf-8') as f:
            line = f.readline()
            print(line)
        pass

    def load_korea(self):
        # filename = "..\..\COVID-19\csse_covid_19_data\csse_covid_19_time_series\time_series_covid19_confirmed_global.csv"
        filename = "../../COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
        # self.readit(filename)
        # return
        df = pd.read_csv(filename)
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
        filename = "../../COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
        if self.tp == None:
            df = pd.read_csv(filename)
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

    def showplot(self, name):
        nation = self.nation_data(name)
        ll = len(nation)
        dd = np.linspace( 0.0, ll-1, ll)
        plt.plot(dd,nation.diff(), label=nation)
        title = name
        # plt.yscale('log')
        plt.grid('both')
        self.show_graph(title)
        pass

    
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
        v = odeint( model.seiqr, model.x0, xx, args = model.param )
        i = v[:, 2]
        # ax.plot(dx,i[1:] * pop)
        #ax.plot(dx,v[:-1,0] * pop)
        #ax.plot(dx,v[:-1,1] * pop)
        ax.plot(dx,v[:-1,2] * pop)
        ax.plot(dx,v[:-1,3] * pop)
        #ax.plot(dx,v[:-1,4] * pop)
        
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



def doparse():
    parser = argparse.ArgumentParser()

    parser.add_argument("-a", action='store_true', help="option")
    parser.add_argument("-p", "--p", nargs='+', type=float, help="p list")
    parser.add_argument("-v", "--valiation1", action='store_true', help="variation")
    parser.add_argument("-d", "--dq", action='store_true', help="variation")
    parser.add_argument("--pq", action='store_true', help="show pq graph")
    parser.add_argument("-j", "--jhu", action='store_true', help="show jhu")
    parser.add_argument("-l", "--linear", action='store_true', help="show with linear")
    parser.add_argument("--korea", action='store_true', help="show korea")
    
    args = parser.parse_args()
    return args

def showpic(title):
    
    plt.grid('both')
    plt.yscale('log')
    plt.legend()
    plt.title(title)
    plt.show()

class model:
    de =  5.0  # exposure to infection
    dq =  10.0  # duration of infectious until quarantine
    dr =  14.0 # duration of infectiousness until recovery
    dqr = 10.0 # time-lag between diagnosis to recovery

    f   = 1 / de
    yq  = 1 / dq
    yr  = 1 / dr
    yqr = 1 / dqr

    r0 = 2.5
    beta = 1.3
    beta = r0 * yr
    p = 0.3

    kpop = 5 * 10 ** 7
    initq = 100 / kpop
    initi = 7000 / kpop

    x0 = [
        1 - initi - initq,  # s
        0.0,                # e
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

        if dch > 0 and t > dch:
            p = 0.4

        if curelimit > 0 and q * 0.2 >= curelimit * 0.8:
            beta = beta / 5

        p += int( t / 7 ) * 0.1
        if p > 0.9:
            p = 0.9

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


def dojhu(args):
    jhu = realdata(args)
    # jhu.load_korea()
    name = 'Korea, South'
    jhu.view_nation(name)
    
    pass

def doshow(args):
    jhu = realdata(args)
    name = 'Iran'
    jhu.showplot( name )

if __name__ == '__main__':
    args = doparse()
    doshow( args )

exit( 0 )
#
#
#
if __name__ == '__main__':
    args = doparse()
    if args.valiation1:
        dovali1(args)
        pass
    elif args.dq:
        dodq(args)
    elif args.pq:
        dopq(args)
    elif args.jhu:
        dojhu(args)
    elif args.korea:
        poi = realdata(args)
        poi.load_korea()
    else:
        main(args)
