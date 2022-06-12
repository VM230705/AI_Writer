# -*- coding: utf-8 -*-
########################################################################################################
# reference - https://github.com/BlinkDL/AI-Writer
# author zhihu: https://zhuanlan.zhihu.com/p/423646620
########################################################################################################

from tkinter import ROUND
import numpy as np
import math, json
import torch
import torch.nn as nn
from torch.nn import functional as F

import src.utils
from src.model import GPT, GPTConfig

src.utils.set_seed(42) # 是否固定随机数（固定后每次运行的生成结果都一样）

"""
Parameters
"""
RUN_DEVICE = 'cpu' # gpu 或 dml 或 cpu

MODEL_NAME = 'model/trained-shian-shia-100'
WORD_NAME = 'model/vocab'

ROUND_CONTINUE = True
NUM_OF_RUNS = 1
LENGTH_OF_EACH = 256

top_p = 0.75 
top_p_newline = 0.9

formor = ""

# Input
while ROUND_CONTINUE:
    context = formor + input("Input Context: ")
    current_record = ""

    ctx_len = 512
    n_layer = 6
    n_head = 8

    n_embd = n_head * 64
    n_attn = n_embd
    n_ffn = n_embd

    context = context.strip().split('\n')
    for c in range(len(context)):
        context[c] = context[c].strip().strip('\u3000') # 去掉首尾的空格
    context = '\n' + ('\n'.join(context)).strip()

    with open(WORD_NAME + '.json', "r", encoding="utf-16") as result_file:
        word_table = json.load(result_file)   

    vocab_size = len(word_table)

    train_dataset = lambda: None
    train_dataset.stoi = {v: int(k) for k, v in word_table.items()}
    train_dataset.itos = {int(k): v for k, v in word_table.items()}

    # replace unused charactor
    UNKNOWN_CHAR = train_dataset.stoi['窩']

    print(f'\nLoading model for {RUN_DEVICE}...', end=' ')
    if RUN_DEVICE == 'dml':
        import onnxruntime as rt
        sess_options = rt.SessionOptions()
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
        sess_options.enable_mem_pattern = False
        rt_session = rt.InferenceSession(MODEL_NAME + '.onnx', sess_options=sess_options, providers=['DmlExecutionProvider'])
        rt_session.set_providers(['DmlExecutionProvider'])
    else:
        model = GPT(GPTConfig(vocab_size, ctx_len, n_layer=n_layer, n_head=n_head, n_embd=n_embd, n_attn=n_attn, n_ffn=n_ffn))
        m2 = torch.load(MODEL_NAME + '.pth', map_location='cpu').state_dict()
        for i in range(n_layer):
            prefix = f'blocks.{i}.attn.'
            time_w = m2[prefix + 'time_w']
            time_alpha = m2[prefix + 'time_alpha']
            time_beta = m2[prefix + 'time_beta']
            
            TT = ctx_len
            T = ctx_len

            # 對time_w做padding 最低維度的前面加0個單位，後面加TT個單位
            w = F.pad(time_w, (0, TT))
            # tile 後面的tuple裡的數值是w每個維度要重複的次數
            # 後面的應該要是tuple，不知道為啥
            w = torch.tile(w, [TT])
            w = w[:, :-TT].reshape(-1, TT, 2 * TT - 1)
            w = w[:, :, TT-1:]
            w = w[:, :T, :T] * time_alpha[:, :, :T] * time_beta[:, :T, :]
            
            m2[prefix + 'time_ww'] = w
            del m2[prefix + 'time_w']
            del m2[prefix + 'time_alpha']
            del m2[prefix + 'time_beta']
        if RUN_DEVICE == 'gpu':
            model = model.cuda()
        model.load_state_dict(m2)

    print('done:', MODEL_NAME, '&', WORD_NAME)

    ##############################################################################

    for run in range(NUM_OF_RUNS):

        x = np.array([train_dataset.stoi.get(s, UNKNOWN_CHAR) for s in context], dtype=np.int64)

        real_len = len(x)
        print_begin = 0
            
        for i in range(LENGTH_OF_EACH):

            if i == 0:
                print(('-' * 60) + '\n' + context.replace('\n', '\n  ').strip('\n'), end = '')
                print_begin = real_len

            with torch.no_grad():
                if RUN_DEVICE == 'dml':
                    if real_len < ctx_len:
                        xxx = np.pad(x, (0, ctx_len - real_len))
                    else:
                        xxx = x
                    out = rt_session.run(None, {rt_session.get_inputs()[0].name: [xxx[-ctx_len:]]})
                    out = torch.tensor(out[0])
                else:
                    xxx = torch.tensor(x[-ctx_len:], dtype=torch.long)[None,...]
                    if RUN_DEVICE == 'gpu':
                        xxx = xxx.cuda()
                    out, _ = model(xxx)
                out[:, :, UNKNOWN_CHAR] = -float('Inf')
            pos = -1 if real_len >= ctx_len else real_len - 1

            if train_dataset.itos[int(x[real_len-1])] == '\n':
                char = src.utils.sample_logits(out, pos, temperature=1.0, top_p=top_p_newline)
            else:
                char = src.utils.sample_logits(out, pos, temperature=1.0, top_p=top_p)
        
            x = np.append(x, char)
            real_len += 1

            if i % 2 == 1 or i == LENGTH_OF_EACH-1 or i < 10 or RUN_DEVICE != 'gpu':
                completion = ''.join([train_dataset.itos[int(i)] for i in x[print_begin:real_len]])
                
                # get context and record it
                a_content = completion.replace('\n', '\n  ')
                current_record += a_content

                print(a_content, end = '', flush=True)
                print_begin = real_len
                
        print()
        

        # Update Formor
        formor += context + current_record