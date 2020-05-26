
#
#-*- coding:utf-8 -*-
# ocr_card_filter.py
import os
from PIL import Image
import pyocr
import pyocr.builders
import argparse
import numpy as np

def doparse():
    parser = argparse.ArgumentParser()

    parser.add_argument("-t", '--threshold', type=int, help="threshold")
    parser.add_argument("--prop", action='store_true', help="prop")
    parser.add_argument("--prop2", action='store_true', help="prop2 ")
    parser.add_argument("--eval", action='store_true', help="eval")
    
    parser.add_argument("-r","--ratio",  type=float, help="eval")
    args = parser.parse_args()
    return args

class ocr:
    tool = None
    # path_tesseract = "C:\\Program Files (x86)\\Tesseract-OCR"
    path_tesseract = "C:\\Program Files\\Tesseract-OCR"
    imgsrc = "./pics/Screenshot_20200508-122913.png"
    c_max = 200
    ratio = 0.28

    def __init__(self, args, imgsrc = None):
        self.args = args
        if imgsrc == None:
            self.imgsrc = ocr.imgsrc
        else:
            self.imgsrc = imgsrc
        # self.c_max = 169
        self.c_max = ocr.c_max
        if args.threshold:
            self.c_max = args.threshold
        if ocr.tool != None:
            return
        # インストール済みのTesseractのパスを通す
        if ocr.path_tesseract not in os.environ["PATH"].split(os.pathsep):
            os.environ["PATH"] += os.pathsep + ocr.path_tesseract
        # OCRエンジンの取得
        tools = pyocr.get_available_tools()
        # print(path_tesseract)
        # print(tools)
        ocr.tool = tools[0]

    def evaluate(self):
        ratio = ocr.ratio
        if self.args.ratio:
            ratio = args.ratio
        res = self.prop2()
        center = res[0]
        best = 0
        mtext = 0
        dic = {}
        for th in (0.8, 0.9, 1.0, 1.1, 1.2):
            ith = int( th * center )
            text = self.scan( ith )
            text2 = text.replace(' ','')
            ntext = len(text2)
            print( ith, ntext )
            dic[ith] = text
            if ntext > mtext :
                best = ith
                mtext = ntext

        print( '- BEGIN --' )
        print( dic[best] )
        print( '- END --' )
        
    def edit_image1(self, img_rgb, cmax):
        # Use only fullblack or fullwhite basd on threshold of the sum of RGB.
        pixels  = img_rgb.load()
        for j in range(img_rgb.size[1]):
            for i in range(img_rgb.size[0]):
                v3 = sum( pixels[i,j] )
                color = (0,0,0)
                if v3 > cmax * 3:
                    color = (255, 255, 255)
                pixels[i, j] = color
        
        return img_rgb


    def edit_image_orig(self, img_rgb):
        pixels  = img_rgb.load()
        for j in range(img_rgb.size[1]):
            for i in range(img_rgb.size[0]):
                if (pixels[i, j][0] > self.c_max or pixels[i, j][1] > self.c_max or
                        pixels[i, j][2] > self.c_max):
                    pixels[i, j] = (255, 255, 255)
        pass
        return img_rgb
    
    def doit(self):
        result = self.scan( self.c_max )
        print(result)

    def scan(self, cmax):
        img_org = Image.open( self.imgsrc )
        img_rgb = img_org.convert("RGB")

        img = self.edit_image1(img_rgb, cmax)

        # ＯＣＲ実行
        builder = pyocr.builders.TextBuilder()
        result = ocr.tool.image_to_string(img, lang="jpn", builder=builder)

        return result

    def prop(self):
        # 原稿画像の読み込み
        # img_org = Image.open("./card_image/zairyucard_omote.jpg")
        img_org = Image.open("./pics/Screenshot_20200508-122913.png")
        img_rgb = img_org.convert("RGB")
        pixels = img_rgb.load()

        # 原稿画像加工（黒っぽい色以外は白=255,255,255にする）
        hist = [0,0,0]
        for c in range(3):
            hist[c] = np.zeros(30)
        
        for j in range(img_rgb.size[1]):
            for i in range(img_rgb.size[0]):
                for c in range(3):
                    v = pixels[i,j][c]/10
                    hist[c][int(v)] += 1.0

        for n in range(30):
            mark = 0
            for c in range(3):
                v = hist[c][n]
                if v > 100:
                    mark = 1
            if mark == 1:
                print( n, hist[0][n], hist[1][n], hist[2][n] )
        
    def prop2(self, pos=0.28, div=16):
        # 原稿画像の読み込み
        # img_org = Image.open("./card_image/zairyucard_omote.jpg")
        img_org = Image.open("./pics/Screenshot_20200508-122913.png")
        img_rgb = img_org.convert("RGB")
        pixels = img_rgb.load()

        # 原稿画像加工（黒っぽい色以外は白=255,255,255にする）
        hist = [0,0,0,0]
        ndiv = int( 256 / div + 2 )
        for c in range(4):
            hist[c] = np.zeros( ndiv )

        for j in range(img_rgb.size[1]):
            for i in range(img_rgb.size[0]):
                for c in range(3):
                    v = pixels[i,j][c]/ div 
                    hist[c][int(v)] += 1.0
                s = sum(pixels[i,j]) / div / 3
                hist[3][ int(s) ] += 1

        acc = 0
        total = sum(hist[3])
        th = 0
        for n in range( ndiv ):
            mark = 0
            acc += hist[3][n]
            for c in range(4):
                v = hist[c][n]
                if v > 100:
                    mark = 1
            ratio = 1.0 * acc / total
            if mark == 1:
                print( "{} {} {} {} {} {:.1f}".format( n, hist[0][n], hist[1][n], hist[2][n], hist[3][n],  ratio * 100 ))
            pass
            if ratio < pos:
                th = n * div
        return (th, hist[3])
        
    def doit_orig(self):
        # Use only fullblack or fullwhite basd on threshold of the sum of RGB.
        # 原稿画像の読み込み
        # img_org = Image.open("./card_image/zairyucard_omote.jpg")
        img_org = Image.open("./pics/Screenshot_20200508-122913.png")
        img_rgb = img_org.convert("RGB")
        pixels = img_rgb.load()

        # 原稿画像加工（黒っぽい色以外は白=255,255,255にする）
        # c_max = 169
        for j in range(img_rgb.size[1]):
            for i in range(img_rgb.size[0]):
                if (pixels[i, j][0] > self.c_max or pixels[i, j][1] > self.c_max or
                        pixels[i, j][2] > self.c_max):
                    pixels[i, j] = (255, 255, 255)

        # ＯＣＲ実行
        builder = pyocr.builders.TextBuilder()
        result = ocr.tool.image_to_string(img_rgb, lang="jpn", builder=builder)

        print(result)

    def doit5(self):
        # decrease the pixel of less than threshold to 1/10.
        # 原稿画像の読み込み
        # img_org = Image.open("./card_image/zairyucard_omote.jpg")
        img_org = Image.open("./pics/Screenshot_20200508-122913.png")
        img_rgb = img_org.convert("RGB")
        pixels = img_rgb.load()

        # 原稿画像加工（黒っぽい色以外は白=255,255,255にする）
        for j in range(img_rgb.size[1]):
            for i in range(img_rgb.size[0]):
                v3 = 0
                for c in range(3):
                    v3 += pixels[i, j][c]
                if v3 > self.c_max:
                    pixels[i, j] = (255, 255, 255)
                else:
                    v = [0,0,0]
                    for c in range(3):
                        v[c] = int( pixels[i, j][c]/10 )
                    pixels[i, j] = tuple(v)

        # ＯＣＲ実行
        builder = pyocr.builders.TextBuilder()
        result = ocr.tool.image_to_string(img_rgb, lang="jpn", builder=builder)

        print(result)

if __name__ == '__main__':
    args = doparse()

    oi = ocr(args)
    if args.prop:
        oi.prop()
    elif args.eval:
        oi.evaluate()
    elif args.prop2:
        res = oi.prop2()
        th = res[0]
        print( th )
    else:
        oi.doit()

#
# EOF
#
