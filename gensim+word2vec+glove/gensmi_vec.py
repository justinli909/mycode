#!/usr/bin/env python
#-*- coding: UTF-8 -*-

"""
@version: Python2.7.10
@author:  Justinli

"""
import jieba
from gensim.models import word2vec
#对你的语料进行分词， 目前用的是jieba分词，为什么要进行分词？ 因为中文语意强大，因此进行分词能够更好的将有意义的词放在一起
#如果你对你的分词结果不满意，也可以认为干预， 比如用jieba.suggest_freq()函数来制定某些分词

def jieba_cut(filename, cut_filename):

    with open(filename, 'rb') as f:
        mycontent = f.read()
        jieba_content = jieba.cut(mycontent, cut_all=False)
        final_file = ' '.join(jieba_content)
        final_file = final_file.encode('utf-8')

    with open(cut_filename, 'wb+') as cut_f:
        cut_f.write(final_file)


def my_word2vec(cut_filename):
    mysetence = word2vec.Text8Corpus(cut_filename)
    #model = word2vec.Word2Vec(mysetence, size=100, min_count=1, window=5, hs=5)
    model = word2vec.Word2Vec(mysetence, size=100, min_count=1, window=5, hs=5)
    model.save('./model/zh_wiki_global.model')



if __name__ == '__main__':
    #filename = r'/Users/apple/Documents/语料/renmingdemingyi.txt'
    #cut_filename = r'/Users/apple/Documents/语料/renmingdemingyi_cut.txt'
    #cut_filename = r'./data/zh_wiki_global'

    #jieba_cut(filename, cut_filename)
    #my_word2vec(cut_filename)
    model = word2vec.Word2Vec.load('./model/zh_wiki_global.model')
    for key in model.similar_by_word(u'爸爸', topn=10):
        print(key)
    print('*****************')
    for key in model.similar_by_word(u'对不起', topn=10):
        print(key)



