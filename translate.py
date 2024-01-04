from my_transformer import Transformer
import torch
import pickle
import sys

en_sen, path_to_model, path_to_pkl = sys.argv[1], sys.argv[2], sys.argv[3]

with open(path_to_pkl, 'rb') as f:
  SRC_Tokens, TRG_Tokens, idx_to_tok_trg = pickle.load(f)

src_vocab, trg_vocab, d_model, N, heads, MAX_LEN = (10904, 24189, 40, 1, 2, 21)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Transformer(src_vocab, trg_vocab, d_model, N, heads, MAX_LEN).to(device)
model.load_state_dict(torch.load(path_to_model))

def aux(line):
  line = line.replace('.', '')
  line = line.replace(',', '')
  line = line.replace('!', '')
  line = line.replace('?', '')
  return line

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

spanish = translate(aux(en_sen).lower(), device)
spanish = spanish.replace('<st> ', '').replace(' <end>', '')
print(en_sen, '->', spanish)
