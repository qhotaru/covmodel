#-*- coding:utf-8 -*-

from pykakasi import kakasi

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

from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['MS Gothic', 'Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

def doparse():
    parser = argparse.ArgumentParser()

    parser.add_argument("-a", action='store_true', help="option")
    parser.add_argument("-p", "--p", nargs='+', type=float, help="p list")
    parser.add_argument("-v", "--valiation1", action='store_true', help="variation")
    parser.add_argument("-d", "--dq", action='store_true', help="variation")
    parser.add_argument("--pq", action='store_true', help="show pq graph")
    parser.add_argument("--qp", action='store_true', help="show Dq graph")

    # JAG JAPAN
    parser.add_argument("--jag", action='store_true', help="show jag")

    
    # JHU
    parser.add_argument("-j", "--jhu", action='store_true', help="show jhu")
    parser.add_argument("--datalist", action='store_true', help="show data list")
    parser.add_argument("--korea", action='store_true', help="show korea")
    parser.add_argument("--plot", help="plot type, line, bar")
    parser.add_argument("--nation", nargs='+', help="nation option")

    # SWS
    parser.add_argument("--sws", action='store_true', help="show sws bar")
    parser.add_argument("--pref", action='store_true', help="show sws bar")
    parser.add_argument("--option", help="show sws option")
    parser.add_argument("--domestic", action='store_true', help="show domestic data")

    # TOKYO
    parser.add_argument("--tokyo", action='store_true', help="show tokyo")
    parser.add_argument("--death", action='store_true', help="show tokyo death")

    
    # graph
    parser.add_argument("-x", "--xrange", type=int, help="x axis range")
    parser.add_argument("-l", "--linear", action='store_true', help="show with linear")
    parser.add_argument("--diff", action='store_true', help="show diff option")

    parser.add_argument("--i", action='store_true', help="show I")
    parser.add_argument("--q", action='store_true', help="show Q")
    parser.add_argument("--n", action='store_true', help="show N")

    
    args = parser.parse_args()
    return args


class realdata:
    filename = "../COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
    # filename = "..\COVID-19\csse_covid_19_data\csse_covid_19_time_series\time_series_covid19_confirmed_global.csv"
    tokyofilename = 'd:/ICF_AutoCapsule_disabled/covid/covid19/data/data.json'

    header_colnum = 4

    swsfile = '../2019-ncov-japan/50_Data/resultDailyReport.csv'
    swsdomesticfile = '../2019-ncov-japan/50_Data/domesticDailyReport.csv'
    preffile = '../2019-ncov-japan/50_Data/bydate.csv'

    jagfile = 'data/COVID-19.csv'
    
    def __init__(self, args):
        self.tp = None
        self.args = args
        pass

    def load_jag(self, args):
        df1 = pd.read_csv(realdata.jagfile)
        colnames = df1.columns
        # print( df )
        print( colnames )
        # print( df1['受診都道府県'].value_counts() )

        df = df1[df1['受診都道府県'] == '東京都']
        # print(df)
        # return
    
        kakutei = df['確定日']
        onset = df['発症日']
        pcrperson = df['PCR検査実施人数']
        xx = np.linspace(0.0, len(pcrperson)-1, len(pcrperson))
        # vc = pcrperson.value_counts()
        vc = pcrperson.unique()
        print(vc)
        pcrperson = pcrperson.dropna()
        print( pcrperson )
        
        plt.bar(kakutei, pcrperson, label='pcr')
        plt.show()
        return 
        
        # print( kakutei.value_counts() )
        # print( onset.value_counts() )
        vck = kakutei.value_counts()
        sk  = sorted( vck.keys() )
        dkl = []
        for dk in sk:
            # print(dk, vck[dk])
            dkl.append(vck[dk])

        # plt.plot(sk, dkl, label='reported')

        # print( onset.unique() )
        
        vc = onset.value_counts()
        od = []
        odic = {}
        osum = 0
        for o in vc.keys():
            dt = datetime.datetime.strptime(o, '%m/%d/%Y')
            od.append( dt )
            odic[dt] = o

        ss = sorted( od )
        dl = []
        for d in ss:
            dd = odic[d]
            print(d, vc[dd])
            dl.append(vc[dd])
            osum += vc[dd]

        print( "sum = {}".format( osum ) )
        xticks = ss[::7]
        # plt.xticks(xticks)

        xl = len(ss)
        xx = np.linspace(0.0,xl-1, xl)
        # plt.bar(ss,dl,label='onset', color='g')
        plt.bar(xx,dl,label='onset', color='g')
        plt.legend()
        plt.show()
    
    def load_tokyo(self,args):
        with open(realdata.tokyofilename, encoding='utf-8') as f:
            tokyodic = json.load(f)

        for k, v in tokyodic.items():
            if type(v) != dict:
                print(k,v)
            elif args.option and args.option == k:
                # print(k,v)
                pass
            else:
                print(k, v.keys())

        pcrkey = 'inspection_persons'
        if args.option == pcrkey:
            self.show_pcr(tokyodic)

        patkey = 'patients_summary'
        if args.option == patkey:
            self.show_patients_summary(tokyodic)
        patstatkey = 'patients'
        if args.option == patstatkey:
            self.show_patients_stats(tokyodic)
            if args.death:
                self.show_death(tokyodic)

        
    def compare_age(self, dic):
        pass

    def show_death(self, dic):
        patstatkey = 'patients'
        pats = dic[patstatkey]
        data = pats['data']
        df = pd.DataFrame(data)

        ndischarged = len(df['退院'])

        sum = 0
        for k, v in df['退院'].value_counts().iteritems():
            print( k, v )
            sum += v
        pass
        print( "Total {}, threst {}".format( ndischarged, ndischared - sum ))
        
    def show_patients_stats(self,dic):
        patstatkey = 'patients'
        pats = dic[patstatkey]
        data = pats['data']
        df = pd.DataFrame(data)

        ds = []

        # print(df['年代'])
        for d in df['リリース日']:
            dt = datetime.datetime.strptime(d, '%Y-%m-%dT%H:%M:%S.000Z')
            month = dt.strftime( '%m' )
            ds.append(int(month))

        df['month'] = ds
        mlist = sorted(df['month'].unique())

        # print(df['month'])
        v = df['年代']
        self.kakasi_setup()
        l = []
        for x in v:
            wamei = self.kakasi_doconv(x)
            wamei2 = wamei.replace('dai','-')
            wamei3 = wamei2.replace('10toshimiman','0-9')
            wamei4 = wamei3.replace('100toshiijou','>100')
            l.append( wamei4 )
        df['age'] = l
        alist =  sorted( df['age'].unique() )
        # alist =  sorted( df['age'].unique(), key=int )

        df2 = pd.DataFrame(alist,index=alist, columns=['age'])

        for m in mlist:
            poi = []
            dk = df[df['month'] == m]
            # print(dk)
            vc = dk['age'].value_counts()
            for ix in df2.index:
                val = 0
                if ix in vc:
                    val = vc[ix]
                poi.append(val)
            df2[m] = poi

        print(df2)

        bt = None
        for m in mlist:
            if type(bt) == type(None):
                bt = df2[m] - df2[m].values
                
            print(bt)
            plt.bar(df2.index, df2[m], label=str(m), bottom=bt)
            bt = bt + df2[m].values

        plt.title('Tokyo positives per age group')
        plt.xlabel('Age Group')
        plt.ylabel('Numer of positives')
        plt.legend()
        plt.show()

        # plt.legend(["二乗値"], prop={"family":"MS Gothic"})

        pass
    def show_patients_summary(self,dic):
        patkey = 'patients_summary'
        pata   = dic[patkey]
        data   = pata['data']
        positive = []
        datemark = []

        ix = 0
        for elm in data:
            if ix % 7 == 0:
                print("")
            ix += 1
            dd = elm['日付']
            dt = datetime.datetime.strptime(dd, '%Y-%m-%dT%H:%M:%S.000Z')
            md = dt.strftime( '%m/%d' )

            nn = elm['小計']
            print("{:4}".format(nn), end=" ")
            positive.append(nn)
            datemark.append(md)

        ofs = 50
        df = pd.DataFrame(positive, columns=['positive'])
        df['date'] = datemark
        xx = np.linspace(0,len(data), len(data))
        # plt.bar(xx[ofs:], df['positive'][ofs:].rolling(7).mean(), label='positive')
        plt.bar(datemark[ofs:], df['positive'][ofs:].rolling(7).mean(), label='positive')
        # plt.bar(xx[ofs:], df['new'][ofs:].rolling(3).mean(), label='positive')
        title7 = 'Tokyo Newly confirmed rolling 7 days average'
        title3 = 'Tokyo Newly confirmed rolling 3 days average'

        xt = datemark[ofs::7]
        plt.xticks(xt)
        
        plt.title(title7)
        plt.ylabel('Person')
        plt.xlabel('Days')
        plt.legend()
        plt.show()
        
    def show_pcr(self, dic):
        pcrkey = 'inspection_persons'
        inspectp = dic[pcrkey]
        ds = inspectp['datasets']
        desc = ds[0]
        pcr = desc['data']
        ix = 0
        for x in pcr:
            if ix % 7 == 0:
                print("")
            ix += 1
            print( "{:4}".format(x), end=" " )
        pass

        xx = np.linspace(0.0,len(pcr)-1,len(pcr))
        ofs = 20
        plt.bar(xx[ofs:],pcr[ofs:])
        plt.show()

    def kakasi_setup(self):
        kk = kakasi()
        kk.setMode('H', 'a')
        kk.setMode('K', 'a')
        kk.setMode('J', 'a')
        self.conv = kk.getConverter()

    def kakasi_doconv(self, kanji):
        alnum = self.conv.do(kanji)
        return alnum
    
    def load_pref(self, args):
        kakasi_setup()

        # date,北海道,青森,岩手,宮城,秋田,山形,福島,茨城,栃木,群馬,埼玉,千葉,東京,神奈川,新潟,富山,石川,福井,山梨,長野,岐阜,静岡,愛知,三重,滋賀,京都,大阪,兵庫,奈良,和歌山,鳥取,島根,岡山,広島,山口,徳島,香川,愛媛,高知,福岡,佐賀,長崎,熊本,大分,宮崎,鹿児島,沖縄,チャーター便,検疫職員,クルーズ船,伊客船

        df = pd.read_csv(realdata.preffile)
        indexname = 'date'
        prefs = df.columns
        newcols = []
        for colname in prefs:
            alname = self.kakasi_doconv(colname)
            print (alname, end=" ")
            newcols.append( alname )
        # print ( prefs )
        df.columns = newcols

        objs = ['toukyou']
        if args.nation:
            objs = args.nation
        tokyo = 'toukyou'
        data = df[tokyo]
        ndata = len(data)
        xx = np.linspace(0.0, ndata, ndata)
        for obj in objs:
            if args.diff:
                xx1 = xx[7:]
                yy = df[obj]
                yy1 = yy[7:]
                yy2 = yy[0 : ndata - 7]
                yy3 = yy1 - yy2
                print(yy3)
                print("nxx1={} nyy1={} nyy2={} nyy3={}".format(len(xx1), len(yy1), len(yy2), len(yy3)))
                # plt.bar(xx1, yy3[0:ndata-7], label=obj)
                plt.bar(xx, df[obj].diff(7), label=obj)
                plt.ylabel('Confirmed case: Diff from 7 days ago')
            else:
                plt.bar(xx, df[obj], label=obj)
                plt.ylabel('Confirmed')

        plt.xlabel('Days')
        plt.legend()
        plt.show()
        
        return df

    
    def load_domestic(self, args):
# date,pcr,positive,symptom,symptomless,symptomConfirming,hospitalize,mild,severe,confirming,waiting,discharge,death
        df1 = pd.read_csv(realdata.swsdomesticfile)
        df1['ts'] = pd.to_datetime(df1['date'], format='%Y%m%d')
        print( df1 )
        xticks = df1['ts'][::7]
        ax = plt.subplot(1,1,1)

        ax.xaxis.set_major_locator(DayLocator(bymonthday=None, interval=7, tz=None))
        ax.xaxis.set_major_formatter(DateFormatter("%m/%d"))
        ofs = 30


        if True:
            diff = df1['pcr'][ofs:].diff()
            ax.bar(df1['ts'][ofs:], diff.rolling(7).mean(), label='PCR Rolling 7 days', color='y')
            # ax.plot(df1['ts'][ofs:], diff, label='Daily PCR data', color='r')
            ax.set_ylim(ymin=0, ymax=6000)

            ax2 = ax.twinx()
            ax2.xaxis.set_major_locator(DayLocator(bymonthday=None, interval=7, tz=None))
            ax2.xaxis.set_major_formatter(DateFormatter("%m/%d"))
            ax2.plot(df1['ts'][ofs:], df1['positive'][ofs:].diff().rolling(7).mean(), label='positive rolling 7days', color='r')
            ax2.set_ylim(ymin=0)
        
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
            ax2.set_ylabel('Positive Person')
        else:
            pcraday  = df1['pcr'][ofs:].diff().rolling(7).mean()
            posiaday = df1['positive'][ofs:].diff().rolling(7).mean()

            ax.bar(df1['ts'][ofs:], posiaday, label='Positive Rolling 7 days')

            # ax.bar(df1['ts'][ofs:], pcraday - posiaday.values, label='PCR Rolling 7 days', bottom=posiaday)
            ax.plot(df1['ts'][ofs:], pcraday - posiaday.values, label='PCR Rolling 7 days')
            ax.legend(loc='upper left')

        ax.set_title('Japan Positive and PCR test rolling 7 days average from MLHW Japan')
        ax.set_xlabel('Day')
        ax.set_ylabel('PCR Test Person')
        plt.show()
        return df1


    def load_sws(self):
# date,pcr.d,positive.d,symptom.d,symptomless.d,symptomConfirming.d,hospitalize.d,mild.d,severe.d,confirming.d,waiting.d,discharge.d,death.d,pcr.f,positive.f,symptom.f,symptomless.f,symptomConfirming.f,hospitalize.f,mild.f,severe.f,confirming.f,waiting.f,discharge.f,death.f,pcr.x,positive.x,symptom,symptomless,symptomConfirming,hospitalized,mild,severe.x,confirming,waiting,discharge.x,death.x,pcr.y,positive.y,discharge.y,symptomlessDischarge,symptomDischarge,severe.y,death.y,pcr,discharge,pcrDiff,dischargeDiff
        df = pd.read_csv(realdata.swsfile)
        pcr = df['pcr.d']
        positive = df['positive.d']
        xd = df['date']
        return [xd, positive, pcr]

    def find_first_value(self, df, border):
        pos = 0
        for x in df:
            if x >= border:
                return pos
            pos+=1
        return -1
    
    def swsplot(self, args):
        xdate, pos, pcr = self.load_sws()
        xlen = len(pos)
        xx = np.linspace(0.0, xlen, xlen)

        fig = plt.figure()
        ax1  = fig.add_subplot(111)

        at = self.find_first_value(pos, 50)
        xticks = list(range(0, xlen, 7))
        at = 0
        if at == -1:
            at == 0
        else:
            xx = xx[at:]
            pos = pos[at:]
            pcr = pcr[at:]

        print(len(xx), len(pos), len(pcr))
        rate = pos/pcr
        title = 'Japan'
        xlabel = 'Days'
        ylabel = 'Count'
        ylabel2 = 'Rate'

        if args.diff:
            ax1.bar(xx, pos.diff().diff(7), label='positive')
        elif args.option == 'pcr':
            ax1.bar(xx, pcr.diff(), label='positive')
        elif args.option == 'rolling_pcr':
            ax1.plot(xdate, pcr.diff().rolling(7).mean(), label='pcr')

            ax2 = ax1.twinx()
            ax2.xaxis.set_major_locator(DayLocator(bymonthday=None, interval=7, tz=None))
            ax2.xaxis.set_major_formatter(DateFormatter("%m/%d"))
            # ax2.set_xticks(xdate[xticks])

            ax2.plot(xdate, pos.diff().rolling(7).mean(), label='positive', color='g')
            ax2.set_ylabel(ylabel2)
        else:
            ax1.bar(xx, pos.diff(), label='positive')
        # plt.bar(xx, (pcr-pos).diff(), label='pcr')
        # ax2.plot(xx, rate[at:], label='Positive Rate', color='g')
        
        # plt.legend()

        ax1.set_title(title)
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        plt.show()
        pass

    def load_data(self):
        df = pd.read_csv(realdata.filename)
        tp = df.transpose()
        indexname = 'Country/Region'
        cols = tp.loc[indexname,]
        tp.columns = cols
        self.tp = tp
        return tp
        pass
        
    
    def jhuplot(self, args):
        tp = self.load_data()
        nations = args.nation
        if nations == None or len(nations) <= 0:
            nations = ['Japan']
        for nation in nations:
            dataall = tp[nation]
            data = dataall[realdata.header_colnum:]
            ndata = len(data)
            xx = np.linspace(0.0, ndata, ndata)
            if args.plot == 'bar':
                plt.bar(xx, data.diff(), label=nation)
            else:
                plt.plot(xx, data, label=nation)
        plt.ylim(0,)
        plt.show()
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

    # print("Hello")
    
    if args.valiation1:
        dovali1(args)
        pass
    elif args.tokyo:
        # print(" do tokyo" )
        rd = realdata(args)
        rd.load_tokyo(args)
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
    elif args.plot:
        rd = realdata(args)
        rd.jhuplot(args)
    elif args.sws:
        rd = realdata(args)
        rd.swsplot(args)
    elif args.pref:
        rd = realdata(args)
        rd.load_pref(args)
    elif args.jag:
        rd = realdata(args)
        rd.load_jag(args)
    elif args.domestic:
        rd = realdata(args)
        rd.load_domestic(args)
    else:
        main(args)

#
# EOF
#
