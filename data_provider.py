import codecs
from os import truncate
from typing import Counter
from numpy import random
import transformers
from transformers.data.processors import InputFeatures
import torch
from torch.utils.data import TensorDataset
import numpy as np
import copy
import tower_config as c
import re
from sys import stdin

class InputFeaturesWordsMap(InputFeatures):
    def __init__(self, input_ids, attention_mask=None, token_type_ids=None, word_start_positions = None, real_len = None, label=None):
        super().__init__(input_ids, attention_mask, token_type_ids, label)
        self.word_start_positions = word_start_positions
        self.real_len = real_len 

def language_specific_preprocessing(lang, sent):
    replace_dict = {"º" : "o", "ª" : "a", "²" : "2", "³" : "3", "¹" : "1", "\u200b" : "", "\u200d" : "", "…" : "...", "µ" : "μ", "r̝" : "r", "ˢ" : "s", 
                    "½" : "1/2", "´" : "'", "Ã¯" : "Ã", "11 3" : "113", "˝" : "\"", "ﬂ" : "fl", "？" : "?", "！" : "!", "。" : ".", "，" : ",", 
                    "）" : ")", "（" : "(", "：" : ":", "Ｂ" : "B", "ＯＫ" : "OK", '＋' : '+', 'Ｄ' : "D", "№" : "No", "™" : "TM", "\ufeff" : "", "¾" : "3/4", 
                    "Ǩusṫa" : "Kusṫa", "₂" : "2", '；' : ";", "\u200e" : "", "อำ" : "อํา", "สำ" : "สํา", "คำ" : "คํา", "จำ" : "จํา", "กำ" : "กํา", "ร่ำ" : "ร่ํา", "ทำ" : "ทํา", 
                    "น้ำ" : "น้ํา", "ตำ" : "ตํา", "ดำ" : "ดํา", "งำ" : "งํา", "นำ" :  "นํา", "ต่ำ" : "ต่ํา", "ซ้ำ" : "ซ้ํา", "ย้ำ" : "ย้ํา", "ว่ำ" : "ว่ํา", "ม่ำ" : "ม่ํา", "ลำ" : "ลํา", 
                    "ยำ" : "ยํา", "ย่ำ" : "ย่ํา", "รำ" : "รํา", "ชำ" : "ชํา", "ล่ำ" : "ล่ํา", "ค่ำ" : "ค่ํา", "ค้ำ" : "ค้ํา", ": )" : ":)", "ㄷ" : "ᄃ", "⸢" : "Γ", "⸣" : "Γ", "ḫ" : "h", 
                    "₄" : "4", "₅" : "5", "₁" : "1", "Ḫ" : "H", "₆" : "6", "ᾧ" : "ω", "ὧ" : "ω", "ᾷ" : "α", "ἣ" : "η", "ἳ" : "ι", "ὦ" : "ω", "Ἴ" : "I", "ἲ" : "ι", 
                    "ᾖ" : "η", "Ὑ" : "Y", "ὣ" : "ω", "Ἵ" : "I", "ῄ" : "η", "ῴ" : "ω", "ὤ" : "ω", "ᾐ" : "η", "ὓ" : "ν", "ᾔ" : "η", "ἃ" : "α", "ᾗ" : "η", "Ἤ" : "H",
                    "ᾅ" : "α", "Ὡ" : "Ω", "ὢ" : "ω", "Ῥ" : "P", "ἆ" : "α", "ᾄ" : "α", "ᾠ" : "ω", "Ἥ" : "H", "Ὄ" : "O", "ὒ" : "ν", "Ὕ" : "Y", "Ἲ" : "I", "Ἶ" : "I", 
                    "ῒ" : "ι", "Ἦ" : "H", "Ὠ" : "Ω", "ῂ" : "η", "Ἦ" : "H", "ᾑ" : "η", "Ἢ" : "H", "ῢ" : "ν", "Ὥ" : "Ω", "ὂ" : "ο", "ᾴ" : "α", "Ὦ" : "Ω", "％" : "%", 
                    "Ⅲ" : "III", "℃" : "°C", "և" : "եւ", "\u200c" : "", "ǹ" : "n", "Ǹ" : "N", "\xa0" : "", "㎞" : "km"}
    
    nospacelangs = ["zh"]

    for i in range(len(sent)):
        for k in replace_dict:
            if k in sent[i]:
                sent[i] = sent[i].replace(k, replace_dict[k])
                
            m = re.search('([0-9]\s+[0-9])', sent[i])
            if m:
                orig = m.group(0)
                rep = orig.replace(" ", "")
                sent[i] = sent[i].replace(orig, rep)

        if lang in ["vi", "sv", "kk", "lt", "kmr", "br"] and " " in sent[i]:
            sent[i] = sent[i].replace(" ", "")

    if lang in nospacelangs:
        return ("").join(sent)
    else:
        return (" ").join(sent)
    

def featurize_sents(sentences, tokenizer, lang):
    counter = 0
    featurized_examples = []
    for sent in sentences:
        sent_text = language_specific_preprocessing(lang, sent)
        instance_text = featurize_text_parsing(sent_text, sent, tokenizer, c.max_length, True)
        if not instance_text:
            print("Mismatch between XLM-R's subword tokens and word-level tokenization of the sentence:\n" + sent_text + "\nSentence is discarded.")
            counter += 1
            continue

        featurized_examples.append(instance_text)

    all_input_ids = torch.tensor([fe.input_ids for fe in featurized_examples], dtype=torch.long)
    all_attention_masks = torch.tensor([fe.attention_mask for fe in featurized_examples], dtype=torch.long)
    all_word_start_positions = torch.tensor([fe.word_start_positions for fe in featurized_examples], dtype=torch.long)
    all_lengths = torch.tensor([fe.real_len for fe in featurized_examples], dtype=torch.long)
    
    dataset = TensorDataset(all_input_ids, all_attention_masks, all_word_start_positions, all_lengths) 
    
    print("Instances added: " + str(len(all_input_ids)))
    print("Instances skipped: " + str(counter))

    return dataset
    
def get_pure_subword_string(subword_string):
    return (subword_string[1:] if subword_string.startswith("▁") else subword_string)

def get_word_start_positions(word_tokens, subword_tokens):
    positions = []
    
    index_subwords = 1
    index_words = 0
    
    string_subword = ""
    string_word = ""

    starter_subwords = 1

    prolong_words = False
    prolong_subwords = False
    
    while True:
        if (index_subwords >= len(subword_tokens) - 1) or (index_words >= len(word_tokens)):
            break
        
        if subword_tokens[index_subwords] == '▁':
            if starter_subwords == index_subwords:
                starter_subwords += 1
            index_subwords += 1
            continue

        string_subword += get_pure_subword_string(subword_tokens[index_subwords])
        string_word += word_tokens[index_words]

        if string_word == string_subword:
            prolong_subwords = False
            prolong_words = False

            positions.append(starter_subwords)
            
            index_subwords += 1
            starter_subwords = index_subwords

            index_words += 1
            string_subword = ""
            string_word = ""

        # main approach, for most languages
        elif string_word.startswith(string_subword):
            # handling horrible cases like: ['餘下', '的'] vs. ['餘', '下的'] 
            if prolong_words:
                w = word_tokens[index_words]
                sw = subword_tokens[index_subwords]
                for i in range(len(sw)):
                    if sw[i] == w[0]:
                        string_subword = sw[i:]
                        break
                starter_subwords = index_subwords + 1

            index_subwords += 1
            string_word = ""
    
            prolong_subwords = True
            prolong_words = False

        # for Chinese-like problem, if subword token contains more than one word token
        elif string_subword.startswith(string_word):      
            positions.append(starter_subwords)

            # handling horrible cases like: ['餘', '下的'] vs. ['餘下', '的']
            if prolong_subwords:
                w = word_tokens[index_words]
                sw = subword_tokens[index_subwords]
                for i in range(len(w)):
                    if w[i] == sw[0]:
                        string_word = w[i:]
                        break
                starter_subwords = index_subwords
            
            string_subword = ""
            index_words += 1

            prolong_subwords = False
            prolong_words = True
            
        else:
            print("Non-matching strings between accumulations of subword-level and word-level tokens")
            return None, -1
            
    real_len = len(positions) 

    if real_len != len(word_tokens):
        return None, -1
        
    positions.append(len(subword_tokens) - 1)
    extension = [-1] * (c.max_word_len + 1 - len(positions))
    positions.extend(extension)
    return positions, real_len


def featurize_text_parsing(text, word_tokens, tokenizer, max_length = 510, add_special_tokens = True, label = None):
    inputs = tokenizer.encode_plus(text, add_special_tokens=True, max_length=max_length, truncation = True)
    subword_strings = tokenizer.convert_ids_to_tokens(inputs["input_ids"])
    word_start_positions, real_len = get_word_start_positions(word_tokens, subword_strings) 
    if not word_start_positions:
        return None

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # Zero-pad up to the sequence length.
    pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
    padding_length = max_length - len(input_ids)
    input_ids = input_ids + ([pad_token] * padding_length)
    attention_mask = attention_mask + ([0] * padding_length)
    
    assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
    assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)

    return InputFeaturesWordsMap(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=None, word_start_positions=word_start_positions, real_len = real_len, label = label)

