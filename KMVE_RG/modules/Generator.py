from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
from modules.Caption import MyCaption


def sort_pack_padded_sequence(input, lengths):
    sorted_lengths, indices = torch.sort(lengths, descending=True)
    tmp = pack_padded_sequence(input[indices], sorted_lengths, batch_first=True)
    inv_ix = indices.clone()
    inv_ix[indices] = torch.arange(0, len(indices)).type_as(inv_ix)
    return tmp, inv_ix


def pad_unsort_packed_sequence(input, inv_ix):
    tmp, _ = pad_packed_sequence(input, batch_first=True)
    tmp = tmp[inv_ix]
    return tmp


def pack_wrapper(module, att_feats, att_masks):
    if att_masks is not None:
        packed, inv_ix = sort_pack_padded_sequence(att_feats, att_masks.data.long().sum(1))
        return pad_unsort_packed_sequence(PackedSequence(module(packed[0]), packed[1]), inv_ix)
    else:
        return module(att_feats)


class GenModel(MyCaption):
    def __init__(self, args, tokenizer):
        super(GenModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer.idx2token)
        self.input_encoding_size = args.d_model
        self.rnn_size = args.d_ff
        self.num_layers = args.num_layers
        self.drop_prob_lm = args.drop_prob_lm
        self.max_seq_length = args.max_seq_length
        self.att_feat_size = args.d_vf
        self.att_hid_size = args.d_model

        self.bos_idx = args.bos_idx
        self.eos_idx = args.eos_idx
        self.pad_idx = args.pad_idx

        self.use_bn = args.use_bn

        self.embed = lambda x: x
        self.fc_embed = lambda x: x
        self.att_embed = nn.Sequential(*(
                ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ()) +
                (nn.Linear(self.att_feat_size, self.input_encoding_size),
                 nn.ReLU(),
                 nn.Dropout(self.drop_prob_lm)) +
                ((nn.BatchNorm1d(self.input_encoding_size),) if self.use_bn == 2 else ())))

    def clip_att(self, att_feats, att_masks):
        if att_masks is not None:
            max_len = att_masks.data.long().sum(1).max()
            att_feats = att_feats[:, :max_len].contiguous()
            att_masks = att_masks[:, :max_len].contiguous()
        return att_feats, att_masks

    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        # embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)
        p_att_feats = self.ctx2att(att_feats)

        return fc_feats, att_feats, p_att_feats, att_masks

    def get_logprobs_state(self, it, fc_feats, att_feats, p_att_feats, att_masks, state, output_logsoftmax=1):
        xt = self.embed(it)
        output, state = self.core(xt, p_att_feats, state, att_masks)
        if output_logsoftmax:
            logprobs = F.log_softmax(self.logit(output), dim=1)
        else:
            logprobs = self.logit(output)

        output_weight = output.unsqueeze(-1)
        attn_map = torch.matmul(p_att_feats, output_weight)
        return logprobs, state, attn_map

    def _sample(self, fc_feats, att_feats, att_masks=None):
        opt = self.args.__dict__
        sample_n = int(opt.get('sample_n', 1))
        output_logsoftmax = opt.get('output_logsoftmax', 1)
        decoding_constraint = opt.get('decoding_constraint', 0)
        # import pdb
        # pdb.set_trace()

        batch_size = fc_feats.size(0)

        state = []
        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        seq = fc_feats.new_full((batch_size * sample_n, self.max_seq_length), self.pad_idx, dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size * sample_n, self.max_seq_length, self.vocab_size + 1)

        for t in range(self.max_seq_length + 1):
            if t == 0:  # input <bos>
                it = fc_feats.new_full([batch_size * sample_n], self.bos_idx, dtype=torch.long)

            logprobs, state, attn_map = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state,
                                                      output_logsoftmax=output_logsoftmax)
            if decoding_constraint and t > 0:
                tmp = logprobs.new_zeros(logprobs.size())
                tmp.scatter_(1, seq[:, t - 1].data.unsqueeze(1), float('-inf'))
                logprobs = logprobs + tmp

            if t == self.max_seq_length:
                break
            it, sampleLogprobs = self.sample_next_word(logprobs)

            if t == 0:
                unfinished = it != self.eos_idx
            else:
                it[~unfinished] = self.pad_idx
                logprobs = logprobs * unfinished.unsqueeze(1).float()
                unfinished = unfinished * (it != self.eos_idx)
            seq[:, t] = it
            seqLogprobs[:, t] = logprobs

        return seq, seqLogprobs

    def _evaluate(self, fc_feats, att_feats, att_masks=None):
        opt = self.args.__dict__
        sample_n = int(opt.get('sample_n', 1))
        output_logsoftmax = opt.get('output_logsoftmax', 1)
        decoding_constraint = opt.get('decoding_constraint', 0)

        batch_size = fc_feats.size(0)
        state = []
        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        seq = fc_feats.new_full((batch_size * sample_n, self.max_seq_length), self.pad_idx, dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size * sample_n, self.max_seq_length, self.vocab_size + 1)

        first_sentence = []
        first_attmap = []
        first_sentence_probs = []
        for t in range(self.max_seq_length + 1):
            if t == 0:  # input <bos>
                it = fc_feats.new_full([batch_size * sample_n], self.bos_idx, dtype=torch.long)

            logprobs, state, attn_map = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks,
                                                                state,
                                                                output_logsoftmax=output_logsoftmax)
            if decoding_constraint and t > 0:
                tmp = logprobs.new_zeros(logprobs.size())
                tmp.scatter_(1, seq[:, t - 1].data.unsqueeze(1), float('-inf'))
                logprobs = logprobs + tmp

            if t == self.max_seq_length:
                break
            it, sampleLogprobs = self.sample_next_word(logprobs)

            if t == 0:
                unfinished = it != self.eos_idx
            else:
                it[~unfinished] = self.pad_idx
                logprobs = logprobs * unfinished.unsqueeze(1).float()
                unfinished = unfinished * (it != self.eos_idx)
            seq[:, t] = it
            seqLogprobs[:, t] = logprobs

            log_probs = logprobs[0].cpu()
            probabilities = np.exp(log_probs)
            index = int(it[0].cpu())

            prob = probabilities[index]
            first_attmap.append(attn_map[0])
            first_sentence.append(index)
            first_sentence_probs.append(prob)

            if unfinished.sum() == 0:
                break
        return seq, first_sentence, first_attmap, first_sentence_probs
