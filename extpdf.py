#!/usr/bin/python3
#-*- coding:utf-8 -*-
#
# extpdf.py
#
#
#

import sys
import codecs
import argparse
import re, os

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def doparse():
    parser = argparse.ArgumentParser('extpdf')

    # extpdf
    parser.add_argument("-o", help="output file")
    parser.add_argument("-a", action='store_true', help="show agerate")

    args = parser.parse_args()
    return args

class ext:
    dirname = 'D:\ICF_AutoCapsule_disabled\covid\covmodel\data'
    filename = 'Actualizacion_93_COVID-19.pdf'
    dirname = dirname.replace('\\', '/')
    fullname = dirname + '/' + filename

    def codings(self):
        pass
        # coding: utf-8
        # 標準入力，標準出力，標準エラー出力の文字コードを変更する．
        sys.stdin  = codecs.getreader('utf-8')(sys.stdin)
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr)

    def show_codings(self):
        # 設定された値を確認する．
        print ('sys.stdin.encoding:', sys.stdin.encoding)
        print ('sys.stdout.encoding:', sys.stdout.encoding)
        print ('sys.stderr.encoding:', sys.stderr.encoding)
    
    def __init__(self):
        # self.show_codings()
        # self.codings()
        # self.show_codings()
        pass

    def readit(self, file):
        txt = self.convert_pdf_to_txt(file)
        # decoded = txt.decode('utf-8')
        return txt

    def doit(self):
        txt = self.readit(ext.fullname)
        print(txt)

    def saveit(self, savefile):
        txt = self.readit(ext.fullname)
        with open(savefile, 'w', encoding='utf-8') as f:
            f.write(txt)
        return

    # Table 3 => Pos 1
    # Crupo de => Pos 2
    #* in pos 2 : get digit started lines as rank
    # in pos 2 : Total wil enter pos 3
    # in pos 3 : Confirmados and n enter pos 4
    #* in pos 4 : get confirmados data
    # in pos 4 : Confirmados enter pos 5
    # in pos 5 : Hospitalizados totales enter 6
    # in pos 6 : n enter pos 7
    #* in pos 7 : get data as hositalized
    # in pos 7 : Hospitalizados totales enter 8
    # in pos 8 : UCI enter pos 9
    # in pos 9 : n enter pos 10
    #* in pos 10 : get UCI data
    # in pos 10 : blank enter 11
    # in pos 11..13. : Third n enter 12,13,14
    #* in pos 14 : get fatal data
    # in pos 14 : blank enter FINISH
    # 
    def agerate(self):
        data = self.agerate_read()
        
        pass

    def agerate_read(self):
        dlist = [2,4,7,10,14]
        dlabel = ['tick', 'confirmed', 'hospitalizd', 'icu', 'fatal']
        data = []
        for x in range(20):
            data.append([])
        txt = self.readit(ext.fullname)
        pos = 0
        for line in txt.split('\n'):
            # print("pos={}".format(pos))
            if pos == 0 and re.match(r'Tabla 3.', line):
                pos = 1
            elif pos == 1 and re.match(r'Grupo de', line):
                pos = 100
            elif pos == 100 and re.match(r'Grupo de', line):
                pos = 2
            elif pos == 2:
                if re.match(r'Total', line):
                    pos = 3
                elif re.match(r'[0-9]', line):
                    data[pos].append( line )
                pass
            elif pos == 3 and re.match(r'n', line): # check later
                pos = 4
            elif pos == 4:
                if re.match(r'Confirmados', line):
                    pos = 5
                elif re.match(r'[0-9]',line):
                    data[pos].append(line)
                pass
            elif pos == 5 and re.match(r'Hospitalizados', line):
                pos = 6
            elif pos == 6 and re.match(r'n', line):
                pos = 7
            elif pos == 7:
                if re.match(r'Hospitalizados', line):
                    pos = 8
                elif re.match(r'[0-9]', line):
                    data[pos].append( line )
                pass
            elif pos == 8 and re.match(r'UCI', line ):
                pos = 9
            elif pos == 9 and re.match(r'n', line):
                pos = 10
            elif pos == 10:
                if len(line) <= 0: # blank line
                    pos = 11
                elif re.match(r'[0-9]', line ):
                    data[pos].append( line )
                pass
            elif pos == 11 and re.match(r'n', line):
                pos = 12
            elif pos == 12 and re.match(r'n', line):
                pos = 13
            elif pos == 13 and re.match(r'n', line):
                pos = 14
            elif pos == 14:
                if len(line) <= 0: # Blank
                    pos = 15
                elif re.match(r'[0-9]', line):
                    data[pos].append( line )
                pass
            elif pos == 15:
                break
        pass
        # print(data)
        d2 = []
        for data_index in dlist:
            nd = []
            for s in data[data_index]:
                if re.search(r'[-+]', s):
                    nd.append( s.replace('.','') )
                else:
                    nd.append( int( s.replace('.','') ) )
            d2.append( nd )
            print( nd )
        pass
        # poi
        df = pd.DataFrame(d2)
        dft = df.transpose()
        dft.columns = dlabel
        

        for x in dlabel[2:]:
            dft[x+'_ratio'] = dft[x] / dft[dlabel[1]].values

        dft['tick'][10] = 'Total'

        print(dft)

        dic = {}
        for ix in range(5):
            dic[dlabel[ix]] = d2[ix]

        self.show_spain_df(dft, dlabel)
        # self.show_spain(dic, dlabel )
        return dft

    def show_spain_df(self, df, dlabel):
        sns.set()

        bt = np.zeros(len(df['tick']))
        for yname in dlabel[2:]:
            plt.bar(df[dlabel[0]], df[yname + '_ratio'], label=yname, bottom=bt)
            bt = bt + df[yname + '_ratio'].values
        
        plt.legend()
        plt.show()
        return df
    
    def show_spain(self, dic, dlabel):
        sns.set()
        
        for yname in dlabel[1:]:
            print(dic[yname])
            plt.plot(dic[dlabel[0]], dic[yname][:-1], label=yname)

        plt.legend()
        plt.show()
        return dic
    
    def convert_pdf_to_txt(self,path): # 引数にはPDFファイルパスを指定
        rsrcmgr = PDFResourceManager()
        retstr = StringIO()
        codec = 'utf-8'
        laparams = LAParams()
        laparams.detect_vertical = True # Trueにすることで綺麗にテキストを抽出できる
        device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
        fp = open(path, 'rb')
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        maxpages = 0
        caching = True
        pagenos=set()
        fstr = ''
        for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages,caching=caching, check_extractable=True):
            interpreter.process_page(page)

            str = retstr.getvalue()
            fstr += str

        fp.close()
        device.close()
        retstr.close()
        return fstr

if __name__ == '__main__':
    args = doparse()
    e = ext()
    # e.doit()
    if args.o:
        e.saveit( args.o )
    elif args.a:
        e.agerate()
    else:
        e.doit()

#
# EOF
#
