#!/usr/bin/env python
#-*- coding: UTF-8 -*-

"""
@version: Python2.7.10
@author:  Justinli

"""
import gensim
from glove_vec import GloveVec

def load(filename):

    myglovevec = GloveVec(filename, 50)
    model = gensim.models.KeyedVectors.load_word2vec_format(myglovevec.get_file()) #GloVe Model
    #model_name = r'/Users/apple/Documents/语料/my_model.npy'
    #model.save(model_name)
    return model

model = load(r'/Users/apple/Documents/语料/glove.6B.50d.txt')

for key in model.similar_by_word(r'brother', topn=10):
    print(key)





