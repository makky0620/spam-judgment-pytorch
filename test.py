# coding: utf-8

import torch
import pickle
from pprint import pprint

from model import RNN
from util import normalize_string

def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensor_from_sentence(lang, sentence):
    indexes = indexes_from_sentence(lang, sentence)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensors_from_pair(input_lang, output_lang, pair):
    input_tensor = tensor_from_sentence(input_lang, pair[1])
    target_tensor = tensor_from_sentence(output_lang, pair[0])
    return (input_tensor, target_tensor)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open('input.pickle', 'rb') as f:
      input_lang = pickle.load(f)
    with open('target.pickle', 'rb') as f:
      target_lang = pickle.load(f)
    with open('../assets/SMSSpamCollection.txt') as f:
      lines = f.readlines()
      pairs = [[normalize_string(s) for s in line.split('\t')] for line in lines]
    
    # modelのロード
    hidden_size = 256
    model = RNN(input_lang.n_words, target_lang.n_words, hidden_size).to(device)
    param = torch.load("model_data/model4.pth")
    for p in model.parameters():
      print(p)
    model.load_state_dict(param)
    print("-"*50)
    for p in model.parameters():
      print(p)

    input_tensor = tensor_from_sentence(input_lang, pairs[1][1]).to(device)    
    hidden = model.init_hidden().to(device)
    output = torch.zeros(target_lang.n_words).to(device)
    for i in range(input_tensor.size(0)):
      output = model(input_tensor[i], hidden)
    
    print(output)
    print(tensor_from_sentence(target_lang, pairs[1][0]))



    
    
    