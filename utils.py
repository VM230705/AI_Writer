########################################################################################################
# reference - https://github.com/BlinkDL/AI-Writer
# author zhihu: https://zhuanlan.zhihu.com/p/423646620
########################################################################################################

import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

def to_float(x):
    return x.cpu().detach().numpy().flatten()[0].astype(float)

def sample_logits(logits, pos, temperature=1.0, top_p=None):
    logits = logits[0][pos, :]

    # softmax 把output縮放到(0,1)之間，且和為1，dim=-1 代表對某一維度的列進行運算
    probs = F.softmax(logits, dim=-1)

    if top_p is not None:
        out = probs.clone()
        sorted_probs, _ = torch.sort(out, descending=True)
        # 計算累加的機率並轉成ndarray
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
        cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)].cpu())
        probs[probs < cutoff] = 0

    if temperature != 1.0:
        probs = probs.pow(1.0 / temperature)
        
    # multinomial 對probs根據機率做取樣一個元素 返回值是一個tensor size=1
    ix = torch.multinomial(probs, num_samples=1)
    return ix[0].cpu()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda:
        torch.cuda.manual_seed_all(seed)
