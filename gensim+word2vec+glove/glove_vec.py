#!/usr/bin/env python
#-*- coding: UTF-8 -*-

"""
@version: Python2.7.10
@author:  Justinli

"""


class GloveVec:

    def __init__(self, filename, dim):
        self.filename = filename
        self.count = 0
        self.dim = dim
        self.new_filename = 'glove_gensim_vec.txt'

    def prepare_doc(self):
        f = open(self.filename, 'r')
        for line in f:
            self.count += 1
        new_line = "{} {}".format(self.count, self.dim)

        with open(self.filename, 'r') as oldfile:
            with open(self.new_filename, 'w') as newfile:
                newfile.write(new_line + '\n')
                for line in oldfile:
                    newfile.write(line)

    def get_file(self):
        self.prepare_doc()
        return self.new_filename


