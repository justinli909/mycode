#!/usr/bin/env python
#-*- coding: UTF-8 -*-

"""
@version: Python3.6.1
@author:  Justinli

"""
import nltk
import itertools

FILEPATH_S = '/Users/apple/Desktop/twiiter_sample_s'
FILEPATH_T = '/Users/apple/Desktop/twiiter_sample_t'
EN_WHITELIST = '0123456789abcdefghijklmnopqrstuvwxyz '  # space is included in whitelist
CH_BLACKLIST = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\''
MAX_LENGTH = 30
most_vocab_size = 10000

def read_data(filepath):
    data = open(filepath).readlines()
    data = [line[:-1] for line in data]
    return data


def process_all_data(data_source, data_target,en_ch=True):

    data = data_source + data_target

    if en_ch:
        data_lines = [line.lower() for line in data]
        lines = [filter_line(nline, EN_WHITELIST, en_ch=True) for nline in data_lines]
    else:
        lines = [filter_line(nline, CH_BLACKLIST, en_ch=False) for nline in data]

    data_lines_list = [line.split(" ") for line in lines]
    print(data_lines_list)

    freq_dist = nltk.FreqDist(itertools.chain(*data_lines_list))
    VOCAB = freq_dist.most_common(most_vocab_size)
    int2word = ['<PAD>'] + ['<UNK>'] + ['<GO>'] + ['<EOS>'] + [x[0] for x in VOCAB]
    word2int = dict([(w, i) for i, w in enumerate(int2word)])
    print(word2int)

    for line in data_lines_list:
        for i in range(len(line)):
            line[i] = word2int.get(line[i], '<UNK>')
    print(data_lines_list)

    for line in data_lines_list:
        if len(line) < MAX_LENGTH:
            for _ in range(MAX_LENGTH - len(line)):
                line.append(word2int.get('<PAD>'))

    return data_lines_list

def process_data():
    data_source = read_data(FILEPATH_S)
    data_target = read_data(FILEPATH_T)
    data_lines_list = process_all_data(data_source, data_target,en_ch=True)
    input_source_int = data_lines_list[:len(data_source)]
    output_target_int = data_lines_list[len(data_source):]

    print(input_source_int)
    print('************************')
    print(output_target_int)

    return input_source_int, output_target_int


def filter_line(line, charlist, en_ch=True):
    if en_ch:
        return "".join([ch for ch in line if ch in charlist])
    else:
        return "".join([ch for ch in line if ch not in charlist])
'''
def get_batch(input_source_int, output_target_int,batch_size):
    vocab_size = len(input_source_int + output_target_int)
    for i in range(vocab_size // batch_size):
        input_source_int = input_source_int[i * batch_size:(i+1) * batch_size]
        output_target_int = output_target_int[i * batch_size:(i+1) * batch_size]
        yield input_source_int, output_target_int

input_source_int, output_target_int = process_data()

input_source_int, output_target_int = get_batch(input_source_int, output_target_int, 50)

'''
process_data()
