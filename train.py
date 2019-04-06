# coding; utf-8

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import random
from copy import deepcopy
import pickle

from util import load_data
from model import RNN

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

    txt_lang, label_lang, pairs = load_data()
    with open('input.pickle', 'wb') as f:
        pickle.dump(txt_lang, f)
    with open('target.pickle', 'wb') as f:
        pickle.dump(label_lang, f)
    hidden_size =256
    hidden_size = 256

    # モデルと損失関数と最適化手法の定義
    model = RNN(txt_lang.n_words, label_lang.n_words, hidden_size).to(device)
    print(model)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # training_pairs = [tensors_from_pair(txt_lang, label_lang, random.choice(pairs)) for i in range(500)]
    training_pairs = [tensors_from_pair(txt_lang, label_lang, pair) for pair in pairs]

    for epoch in range(5):
        random.shuffle(training_pairs)
        for i, pair in enumerate(training_pairs):
            print('\r{0}/{1}'.format(i+1, len(training_pairs)), end=', ')
            hidden = model.init_hidden().to(device)

            input_tensor = pair[0]
            target_tensor = pair[1]

            optimizer.zero_grad()

            output = torch.zeros(label_lang.n_words).to(device)
            for j in range(input_tensor.size(0)):
                output = model(input_tensor[j], hidden) 
            
            loss = criterion(output.view(1, -1), target_tensor[0])
            
            loss.backward()
            optimizer.step()

            if i % 2500 == 2499:
                print("epoch: {}, iter: {}, loss: {}".format(epoch+1, i+1, loss))

    torch.save(deepcopy(model).cpu().state_dict(), 'model_data/model'+str(epoch)+'.pth')







