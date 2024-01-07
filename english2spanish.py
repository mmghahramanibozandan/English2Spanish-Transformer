# -*- coding: utf-8 -*-
"""English2Spanish.ipynb

Mohammadmahdi Ghahramanibozandan
"""
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchinfo import summary
from my_transformer import Transformer
from my_train import train
import warnings
warnings.filterwarnings("ignore")

def aux(line):
  line = line.replace('.', '')
  line = line.replace(',', '')
  line = line.replace('!', '')
  line = line.replace('?', '')
  return line

English_sens = []
Spanish_sens = []
with open('/content/spa.txt', 'r') as f:
  for line in f.readlines():
    en, sp = line.split('CC')[0].split('\t')[:-1]
    en = aux(en.lower())
    sp = '<st> ' + aux(sp) + ' <end>'
    English_sens.append(en)
    Spanish_sens.append(sp)

sen_length = []
for idx in range(len(English_sens)):
  en_len = len(English_sens[idx].split())
  sp_len = len(Spanish_sens[idx].split())
  sen_length.append(max(en_len, sp_len))

MAX_LEN = max(sen_length)
batch_size = 256
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_tokens(source_dataset): # <pad>:0, <unk>:the last token
  dic = {}
  for sen in source_dataset:
    for word in sen.split():
      if word not in dic:
        dic.setdefault(word, 1)
      else:
        dic[word] += 1

  token_dic = {}
  token_dic.setdefault('<pad>', 0)

  sort_dic = dict(sorted(dic.items(), key=lambda x:x[1], reverse=True))

  for idx, word in enumerate(sort_dic):
    token_dic[word] = idx + 1

  token_dic.setdefault('<unk>', len(token_dic))

  return token_dic
SRC_Tokens = create_tokens(English_sens)
TRG_Tokens = create_tokens(Spanish_sens)
idx_to_tok_src = {v:k for (k,v) in SRC_Tokens.items()}
idx_to_tok_trg = {v:k for (k,v) in TRG_Tokens.items()}

def tokenizer(sentence, tokens, max_len):
  words = sentence.split()
  for i in range(len(words)):
    if words[i] in tokens:
      words[i] = tokens[words[i]]
    else:
      words[i] = tokens['<unk>']
  for _ in range(max_len-len(words)):
    words.append(tokens['<pad>'])
  return torch.tensor(words)


class MyDataset(Dataset):
  def __init__(self, SRC, TRG, device):
    self.SRC = SRC
    self.TRG = TRG
    self.device = device
  def __len__(self):
    return len(self.SRC)
  def __getitem__(self, idx):
    src, trg = self.SRC[idx], self.TRG[idx]
    src = tokenizer(src, SRC_Tokens, MAX_LEN)
    trg = tokenizer(trg, TRG_Tokens, MAX_LEN)
    return src.to(self.device), trg.to(self.device)
dataloader = DataLoader(MyDataset(English_sens, Spanish_sens, device),
                        batch_size=batch_size)

src_vocab = len(SRC_Tokens)
trg_vocab = len(TRG_Tokens)

d_model = 40
N = 1
heads = 2

model = Transformer(src_vocab, trg_vocab, d_model, N, heads, MAX_LEN).to(device)
summary(model, [(batch_size, MAX_LEN), (batch_size, MAX_LEN)], dtypes=[torch.long, torch.long])

@torch.no_grad()
def translate(sentence, device):

  sen_SRC = tokenizer(sentence, SRC_Tokens, MAX_LEN).unsqueeze(0).to(device)
  sen_TRG = '<st>'

  while '<end>' not in sen_TRG:

    length = len(sen_TRG.split())
    trg_input = tokenizer(sen_TRG, TRG_Tokens, MAX_LEN).unsqueeze(0)[:, :-1].to(device)
    # see my_train code => remember: trg_input = trg[:,:-1]
    preds = model(sen_SRC, trg_input, src_mask=None, trg_mask=None).squeeze()
    next_word_idx = torch.argmax(preds, dim=-1)[length-1] #IMPORTANT
    sen_TRG += (' ' + idx_to_tok_trg[next_word_idx.item()])

  return sen_TRG

epochs = 200
print_step = 10
lr = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
test_sen = "we spent the night in a cheap hotel"

loss = train(model, optimizer, dataloader, epochs, translate, test_sen, device, print_step)

epochs = 100
print_step = 10
lr = 5e-4
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
test_sen = "we spent the night in a cheap hotel"

loss = train(model, optimizer, dataloader, epochs, translate, test_sen, device, print_step)

epochs = 100
print_step = 10
lr = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
test_sen = "we spent the night in a cheap hotel"

loss = train(model, optimizer, dataloader, epochs, translate, test_sen, device, print_step)

epochs = 50
print_step = 10
lr = 5e-5
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
test_sen = "we spent the night in a cheap hotel"

loss = train(model, optimizer, dataloader, epochs, translate, test_sen, device, print_step)

new_en_sens = ['hello!', 'what is my name?',
               'how are you?', 'she is pretty.',
               'we spent the night in a cheap hotel']

for new_en_sen in new_en_sens:
  spanish = translate(aux(new_en_sen).lower(), device)
  spanish = spanish.replace('<st> ', '').replace(' <end>', '')
  print(new_en_sen, '->', spanish)

import pickle

with open('params.pkl', 'wb') as f:
  pickle.dump([SRC_Tokens, TRG_Tokens, idx_to_tok_trg], f)
