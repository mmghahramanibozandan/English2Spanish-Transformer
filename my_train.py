import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math


def train(model, optim, dataloader, epochs, translator, test_sen, device, print_step=1):
    model.train()  
    total_loss = []

    for epoch in range(epochs): 

        losses, step = 0, 0

        for idx, (src, trg) in enumerate(dataloader):
            # src: [batch_size, seq_len_SRC]
            # trg: [batch_size, seq_len_TRG_PRIME]
            trg_input = trg[:,:-1]
            # trg_input: [batch_size, seq_len_TRG=seq_len_TRG_PRIME-1]
            # we dont have to learn what will come after <end> or <pad> tokens.
            
            preds = model(src, trg_input, src_mask=None, trg_mask=None)
            # preds: [batch_size, seq_len_TRG, trg_vocab]

            ys = trg[:, 1:].contiguous() #Right shifted
            # ys: [batch_size, seq_len_TRG]

            optim.zero_grad()

            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys.view(-1), ignore_index=0)
            # ignore_index must be the <pad> token index: 0
            loss.backward()
            optim.step()
            
            losses += loss
            step += 1
        
        total_loss.append(losses.item() / step)

        if epoch % print_step == 0:

          print(f"Epoch: {epoch+1} -> Loss: {total_loss[-1]: .8f}")
          print(f"{test_sen} -> {translator(test_sen, device)}")
          torch.save(model.state_dict(), 'En2Sp.pt')


    return total_loss
