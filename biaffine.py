import logging
import math

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import BertPreTrainedModel
from transformers import BertModel, RobertaModel
from transformers import BertConfig, RobertaConfig
from transformers import BertConfig
from transformers import XLMRobertaModel
from transformers import XLMRobertaConfig
import tower_config as c

# works for both BERT and RoBERTa
def merge_subword_tokens(subword_outputs, word_starts):
    instances = []    

    # handling instance by instance
    for i in range(len(subword_outputs)):
        subword_vecs = subword_outputs[i]
        word_vecs = []
        starts = word_starts[i]
        for j in range(len(starts) - 1):
            k = j + 1
            if starts[k] <= 0:
                break
            elif starts[k] == starts[j]:
                while starts[k] == starts[j]:
                    k += 1

            start = starts[j]
            end = starts[k]
            vecs_range = subword_vecs[start : end]
            word_vecs.append(torch.mean(vecs_range, 0).unsqueeze(0))
        
        instances.append(word_vecs)
    
    t_insts = []

    hidden_size = 768
    zero_tens = torch.zeros(hidden_size).unsqueeze(0)
    zero_tens = zero_tens.to(c.device)

    for inst in instances:
        if len(inst) < c.max_word_len:
            for i in range(c.max_word_len - len(inst)):
                inst.append(zero_tens)
        t_insts.append(torch.cat(inst, dim = 0).unsqueeze(0))

    w_tens = torch.cat(t_insts, dim = 0)
    return w_tens

def get_loss(arc_preds, rel_preds, labels_arc, labels_rel, loss_fn):
    if len(arc_preds.shape) == 2:
        arc_preds = arc_preds.unsqueeze(0)

    pad_value = -2    
    mask = labels_arc.ne(pad_value)

    arc_scores, arcs = arc_preds[mask], labels_arc[mask]
    loss = loss_fn(arc_scores, arcs)

    rel_scores, rels = rel_preds[mask], labels_rel[mask]
    rel_scores = rel_scores[torch.arange(len(arcs)), arcs]
    rel_loss = loss_fn(rel_scores, rels)
    loss += rel_loss

    return loss   

# Credit:
# Class taken from https://github.com/yzhangcs/biaffine-parser
class Biaffine(nn.Module):
    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super(Biaffine, self).__init__()
        
        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(torch.Tensor(n_out,
                                                n_in + bias_x,
                                                n_in + bias_y))
        self.init_weights()

    def extra_repr(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return s

    def init_weights(self):
        nn.init.zeros_(self.weight)

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        
        # [batch_size, n_out, seq_len, seq_len]
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        return s

class RobertaForBiaffineParsing(BertPreTrainedModel):
    config_class = RobertaConfig

    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.hidden_size = config.hidden_size

        self.biaffine_arcs = Biaffine(n_in=config.hidden_size, bias_x=True, bias_y=False)
        self.biaffine_rels = Biaffine(n_in=config.hidden_size, n_out=config.num_labels, bias_x=True, bias_y=True)

        self.dropout = nn.Dropout(config.last_layer_dropout)
        self.loss_fn = CrossEntropyLoss()

    def forward(self, batch):    
        tokids = batch[0]
        att_masks = batch[1]
        word_starts = batch[2]
        word_lengths = batch[3]

        # run through RoBERTa encoder and get vector representations
        out_trans = self.roberta(input_ids = tokids, attention_mask = att_masks)
        outs = self.dropout(out_trans[0])
        word_outputs_deps = merge_subword_tokens(outs, word_starts)

        # adding the CLS representation as the representation for the "root" parse token
        word_outputs_heads = torch.cat([out_trans[1].unsqueeze(1), word_outputs_deps] , dim = 1)

        arc_preds = self.biaffine_arcs(word_outputs_deps, word_outputs_heads)
        arc_preds = arc_preds.squeeze()
        outputs = (arc_preds, )

        rel_preds = self.biaffine_rels(word_outputs_deps, word_outputs_heads)
        rel_preds = rel_preds.permute(0, 2, 3, 1)
        outputs = (rel_preds, ) + outputs

        return outputs

class XLMRobertaForBiaffineParsing(RobertaForBiaffineParsing):
    config_class = XLMRobertaConfig
    