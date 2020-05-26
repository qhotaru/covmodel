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
    
    args = parser.parse_args()
    return args



class ocr:
    tool = None
    def __init__(self):
        if ocr.tool != None:
            return
        # インストール済みのTesseractのパスを通す
        # path_tesseract = "C:\\Program Files (x86)\\Tesseract-OCR"
        path_tesseract = "C:\\Program Files\\Tesseract-OCR"
        if path_tesseract not in os.environ["PATH"].split(os.pathsep):
            os.environ["PATH"] += os.pathsep + path_tesseract
        # OCRエンジンの取得
        tools = pyocr.get_available_tools()

        # print(path_tesseract)
        # print(tools)
        ocr.tool = tools[0]

    def doit(self, args):
        # 原稿画像の読み込み
        # img_org = Image.open("./card_image/zairyucard_omote.jpg")
        img_org = Image.open("./pics/Screenshot_20200508-122913.png")
        img_rgb = img_org.convert("RGB")
        pixels = img_rgb.load()

        # 原稿画像加工（黒っぽい色以外は白=255,255,255にする）
        # c_max = 169
        c_max = 200 * 3
        if args.threshold:
            c_max = args.threshold * 3
        for j in range(img_rgb.size[1]):
            for i in range(img_rgb.size[0]):
                # if (pixels[i, j][0] > c_max or pixels[i, j][1] > c_max or
                #         pixels[i, j][0] > c_max):
                v3 = 0
                for c in range(3):
                    v3 += pixels[i, j][c]
                if v3 > c_max:
                    pixels[i, j] = (255, 255, 255)
                else:
                    pixels[i, j] = (0, 0, 0)

        # ＯＣＲ実行
        builder = pyocr.builders.TextBuilder()
        result = ocr.tool.image_to_string(img_rgb, lang="jpn", builder=builder)

        print(result)

    def prop(self, args):
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
        
if __name__ == '__main__':
    args = doparse()

    oi = ocr()
    if args.prop:
        oi.prop(args)
    else:
        oi = ocr()
        oi.doit(args)

#
# EOF
#

