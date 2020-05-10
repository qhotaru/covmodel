#!/usr/bin/python3
#-*- coding:utf-8 -*-

import codecs
import io
import argparse

sjis_codec = codecs.lookup("shift_jis") # SJIS
utf_codec  = codecs.lookup("utf_8")     # UTF

def doparse():
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--tos", action='store_true', help="UTF8 to SJIS default")
    parser.add_argument("-u", "--tou", action='store_true', help="SJIS to UTF8")

    parser.add_argument("-o", "--out", default='jag-covid-sjis.csv', help="output filename option")
    parser.add_argument("src", default='jag-covid.csv', help="src file")

    args = parser.parse_args()
    return args


def main(args):

    src_file_path  = args.src
    dest_file_path = args.out

    src_codec = utf_codec
    dest_codec = sjis_codec
    if args.tou:
        print("converting to UTF8")
        src_codec = sjis_codec
        dest_codec = utf_codec
    else:
        print("converting to SJIS")
        
    # ファイルオブジェクトを開く
    with open(src_file_path, "rb") as src, open(dest_file_path, "wb") as dest:
        
        # 変換ストリームを作成
        stream = codecs.StreamRecoder(
            src,                         # src
            dest_codec.encode,           # dest codec
            src_codec.decode,            # src codec
            src_codec.streamreader,      # src streamer
            dest_codec.streamwriter,     # dest streamer
        )
        reader = io.BufferedReader(stream)
        # writer = io.BufferedWriter(stream)

        # 書き込み
        while True:
            data = reader.read1()
            if not data:
                break
            u = data.decode('utf-8')
            # s = u.encode('cp932', errors='ignore')
            # dest.write(s)
            # dest.flush()
            dest_codec.streamwriter.write(data)
            # writer.flush()
#
# main
#
if __name__ == '__main__':
    args = doparse()
    main(args)

#
# EOF
#
