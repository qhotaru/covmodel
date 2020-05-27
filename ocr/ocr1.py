#
#-*- coding:utf-8 -*-
# ocr_card.py
import os
from PIL import Image
import pyocr
import pyocr.builders

# 1.インストール済みのTesseractのパスを通す
# path_tesseract = "C:\\Program Files (x86)\\Tesseract-OCR"
path_tesseract = "C:\\Program Files\\Tesseract-OCR"
if path_tesseract not in os.environ["PATH"].split(os.pathsep):
    os.environ["PATH"] += os.pathsep + path_tesseract

# 2.OCRエンジンの取得
tools = pyocr.get_available_tools()
print(tools, path_tesseract)
# exit()

tool = tools[0]

# 3.原稿画像の読み込み
# img_org = Image.open("./card_image/zairyucard_omote.jpg")
img_org = Image.open("./pics/Screenshot_20200508-122913.png")

# 4.ＯＣＲ実行
builder = pyocr.builders.TextBuilder()
result = tool.image_to_string(img_org, lang="jpn", builder=builder)

print(result)