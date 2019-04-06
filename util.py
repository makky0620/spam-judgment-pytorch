# coding: utf-8

import re
from pprint import pprint

from lang import Lang


def normalize_string(s):
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def load_data():
    lines = open('../assets/SMSSpamCollection.txt').readlines()
    
    pairs = [[normalize_string(s) for s in line.split('\t')] for line in lines]
    
    input_lang = Lang("txt")
    output_lang = Lang("label")

    for pair in pairs:
        input_lang.add_sentence(pair[1])
        output_lang.add_sentence(pair[0])
    
    return input_lang, output_lang, pairs


if __name__ == "__main__":
    load_data()