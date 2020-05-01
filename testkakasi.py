#!/usr/bin/env python3
# coding: utf-8
from pykakasi import kakasi

kakasi = kakasi()

kakasi.setMode('H', 'a')
kakasi.setMode('K', 'a')
kakasi.setMode('J', 'a')

conv = kakasi.getConverter()

filename = '本日は晴天なり.jpg'


print("Base", filename)
print("Base type", type(filename))
print("Conv", conv.do(filename))

# print(type(filename.decode('utf-8')))
# print(conv.do(filename.decode('utf-8')))
# print(type(filename.decode('utf-8')))
# print(type(filename))
# print(conv.do(filename))
