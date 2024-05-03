from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class MyCaption(nn.Module):
    def __init__(self):
        super(MyCaption, self).__init__()
        self.mode = None

    def forward(self, *args, **kwargs):
        mode = kwargs.get('mode', 'forward')
        if 'mode' in kwargs:
            del kwargs['mode']
        return getattr(self, '_' + mode)(*args, **kwargs)

    def sample_next_word(self, logprobs):
        sampleLogprobs, it = torch.max(logprobs.data, 1)
        it = it.view(-1).long()
        return it, sampleLogprobs
