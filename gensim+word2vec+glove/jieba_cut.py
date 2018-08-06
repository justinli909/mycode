#!/usr/bin/env python
#-*- coding: UTF-8 -*-

"""
@version: Python2.7.10
@author:  Justinli

"""
import jieba

def jieba_cut(filename):
    with open(filename, 'rb') as f:
        mydoc = f.read()
        mydoc_cut = jieba.cut(mydoc, cut_all=False)
        mydoc_cut_content = ' '.join(mydoc_cut)
        mydoc_cut_content= mydoc_cut_content.encode('utf-8')
    with open(filename + '_cut', 'wb+') as f1:
        f1.write(mydoc_cut_content)

def write_file(filename, oldname):
    with open(filename, 'rb') as f:
        doc = f.read()
    with open(oldname, 'ab+') as f1:
        f1.write(doc)

myname = ['zh_wiki_00_g_cut','zh_wiki_01_g_cut','zh_wiki_02_g_cut']
new_name = 'zh_wiki_global'

for file in myname:
    write_file('./data/' + file, './data/' + new_name)







