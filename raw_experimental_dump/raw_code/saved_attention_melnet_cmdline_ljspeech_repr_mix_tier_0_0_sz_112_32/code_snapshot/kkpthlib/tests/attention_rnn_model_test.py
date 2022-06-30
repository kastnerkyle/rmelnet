from __future__ import print_function
import os
import argparse
import numpy as np
import torch
from collections import namedtuple

import logging
import shutil
from kkpthlib import Embedding
from kkpthlib import Linear
from kkpthlib import SequenceConv1dStack
from kkpthlib import LSTMCell
from kkpthlib import BiLSTMLayer
from kkpthlib import GaussianAttentionCell
from kkpthlib import scan
from kkpthlib import clipping_grad_norm_

#from kkpthlib import DiscreteMixtureOfLogistics
#from kkpthlib import DiscreteMixtureOfLogisticsCost
#from kkpthlib import AdditiveGaussianNoise
#from tfbldr import scan

seq_len = 48
batch_size = 10
window_mixtures = 10
cell_dropout = .925
cell_dropout = 1.
#noise_scale = 8.
prenet_units = 128
n_filts = 128
n_stacks = 3
enc_units = 128
dec_units = 512
emb_dim = 15
truncation_len = seq_len
cell_dropout_scale = cell_dropout
epsilon = 1E-8
forward_init = "truncated_normal"
rnn_init = "truncated_normal"
bn_flag = 0.

#basedir = "/Tmp/kastner/lj_speech/LJSpeech-1.0/"
#ljspeech = rsync_fetch(fetch_ljspeech, "leto01")

# THESE ARE CANNOT BE PAIRED (SOME MISSING), ITERATOR PAIRS THEM UP BY NAME
#wavfiles = ljspeech["wavfiles"]
#jsonfiles = ljspeech["jsonfiles"]

# THESE HAVE TO BE THE SAME TO ENSURE SPLIT IS CORRECT
train_random_state = np.random.RandomState(3122)
valid_random_state = np.random.RandomState(3122)

fake_random_state = np.random.RandomState(1234)

class FakeItr(object):
    def __init__(self, batch_size, seq_len):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.vocabulary_sizes=[44, 44]
        self.n_mel_filters = 80

    def next_masked_batch(self):
        # need to make int "strings" of batch_size, random_len (10-50?)
        # need to make batches of 256, 
        # dummy batch sizes from validation iterator in training code
        mels = fake_random_state.randn(self.seq_len, self.batch_size, 80)
        mel_mask = 0. * mels[..., 0] + 1.
        text = fake_random_state.randint(0, 44, size=(145, self.batch_size, 1)).astype("float32")
        text_mask = 0. * text[..., 0] + 1.
        mask = 0. * text_mask + 1.
        mask_mask = 0. * text_mask + 1.
        reset = 0. * mask_mask[0] + 1.
        reset = reset[:, None]
        # mels = (256, 64, 80)
        # mel_mask = (256, 64)
        # text = (145, 64, 1)
        # text_mask = (145, 64)
        # mask = (145, 64)
        # mask_mask = (145, 64)
        # reset = (64, 1)    
        return mels, mel_mask, text, text_mask, mask, mask_mask, reset


train_itr = FakeItr(batch_size, seq_len)
valid_itr = FakeItr(batch_size, seq_len)
#train_itr = wavfile_caching_mel_tbptt_iterator(wavfiles, jsonfiles, batch_size, seq_len, stop_index=.95, shuffle=True, symbol_processing="chars_only", random_state=train_random_state)
#valid_itr = wavfile_caching_mel_tbptt_iterator(wavfiles, jsonfiles, batch_size, seq_len, start_index=.95, shuffle=True, symbol_processing="chars_only", random_state=valid_random_state)

"""
for i in range(10000):
    print(i)
    mels, mel_mask, text, text_mask, mask, mask_mask, reset = train_itr.next_masked_batch()
print("done")
"""

"""
# STRONG CHECK TO ENSURE NO OVERLAP IN TRAIN/VALID
for tai in train_itr.all_indices_:
    assert tai not in valid_itr.all_indices_
for vai in valid_itr.all_indices_:
    assert vai not in train_itr.all_indices_
"""

random_state = np.random.RandomState(1442)
# use the max of the two blended types...
vocabulary_size = max(train_itr.vocabulary_sizes)
output_size = train_itr.n_mel_filters
prenet_dropout = 0.5
prenet_dropout= 1.

att_w_init = torch.FloatTensor(np.zeros((batch_size, 2 * enc_units)))
att_k_init = torch.FloatTensor(np.zeros((batch_size, window_mixtures)))
att_h_init = torch.FloatTensor(np.zeros((batch_size, dec_units)))
att_c_init = torch.FloatTensor(np.zeros((batch_size, dec_units)))
h1_init = torch.FloatTensor(np.zeros((batch_size, dec_units)))
c1_init = torch.FloatTensor(np.zeros((batch_size, dec_units)))
h2_init = torch.FloatTensor(np.zeros((batch_size, dec_units)))
c2_init = torch.FloatTensor(np.zeros((batch_size, dec_units)))

class GlobalModel(torch.nn.Module):
    def __init__(self):
        super(GlobalModel, self).__init__()
        random_state = np.random.RandomState(1442)
        self.projmel1_obj = Linear([output_size], prenet_units,
                              dropout_flag_prob_keep=prenet_dropout, name="prenet1",
                              random_state=random_state)

        random_state = np.random.RandomState(1442)
        self.projmel2_obj = Linear([prenet_units], prenet_units,
                              dropout_flag_prob_keep=prenet_dropout, name="prenet2",
                              random_state=random_state)


        random_state = np.random.RandomState(1442)
        self.text_emb_obj = Embedding(vocabulary_size, emb_dim, random_state=random_state,
                                         name="text_char_emb")

        random_state = np.random.RandomState(1442)
        self.conv_text_obj = SequenceConv1dStack([emb_dim], n_filts,
                                            batch_norm_flag=bn_flag,
                                            n_stacks=n_stacks,
                                            kernel_sizes=[(1, 1), (3, 3), (5, 5)],
                                            name="enc_conv1",
                                            random_state=random_state)

        # text_mask and mask_mask should be the same, doesn't matter which one we use
        random_state = np.random.RandomState(1442)
        self.bitext_layer_obj = BiLSTMLayer([n_filts],
                                       enc_units,
                                       name="encode_bidir",
                                       init=rnn_init,
                                       random_state=random_state)
        random_state = np.random.RandomState(1442)
        self.attn_obj = GaussianAttentionCell([prenet_units],
                                         2 * enc_units,
                                         dec_units,
                                         attention_scale=1.,
                                         step_op="softplus",
                                         name="att",
                                         random_state=random_state,
                                         init=rnn_init)

        random_state = np.random.RandomState(1442)
        self.rnn1_obj = LSTMCell([prenet_units, 2 * enc_units, dec_units],
                            dec_units,
                            random_state=random_state,
                            name="rnn1", init=rnn_init)

        random_state = np.random.RandomState(1442)
        self.rnn2_obj = LSTMCell([prenet_units, 2 * enc_units, dec_units],
                            dec_units,
                            random_state=random_state,
                            name="rnn2", init=rnn_init)

        random_state = np.random.RandomState(1442)
        self.pred_obj = Linear([dec_units], output_size,
                          name="out_proj",
                          random_state=random_state)

    def forward(self, in_mels, in_mel_mask, out_mels, out_mel_mask, text, text_mask, mask, mask_mask, reset):
        projmel1 = self.projmel1_obj([in_mels],
                                 dropout_flag_prob_keep=prenet_dropout)
        projmel2 = self.projmel2_obj([projmel1],
                                dropout_flag_prob_keep=prenet_dropout)

        text_char_e, t_c_emb = self.text_emb_obj(text)
        text_e = text_char_e

        conv_text = self.conv_text_obj([text_e])

        bitext = self.bitext_layer_obj([conv_text],
                                  input_mask=text_mask)


        def step(inp_t, inp_mask_t,
                 corr_inp_t,
                 att_w_tm1, att_k_tm1, att_h_tm1, att_c_tm1,
                 h1_tm1, c1_tm1, h2_tm1, c2_tm1):

            o = self.attn_obj([corr_inp_t],
                         (att_h_tm1, att_c_tm1),
                         att_k_tm1,
                         bitext,
                         att_w_tm1,
                         input_mask=inp_mask_t,
                         conditioning_mask=text_mask,
                         #attention_scale=1. / 10.,
                         cell_dropout=1.)
            att_w_t, att_k_t, att_phi_t, s = o
            att_h_t = s[0]
            att_c_t = s[1]

            output, s = self.rnn1_obj([corr_inp_t, att_w_t, att_h_t],
                                 h1_tm1, c1_tm1,
                                 input_mask=inp_mask_t,
                                 cell_dropout=cell_dropout)
            h1_t = s[0]
            c1_t = s[1]
            output, s = self.rnn2_obj([corr_inp_t, att_w_t, h1_t],
                                 h2_tm1, c2_tm1,
                                 input_mask=inp_mask_t,
                                 cell_dropout=cell_dropout)
            h2_t = s[0]
            c2_t = s[1]
            return output, att_w_t, att_k_t, att_phi_t, att_h_t, att_c_t, h1_t, c1_t, h2_t, c2_t

        r = scan(step,
                 [in_mels, in_mel_mask, projmel2],
                 [None, att_w_init, att_k_init, None, att_h_init, att_c_init,
                 h1_init, c1_init, h2_init, c2_init])

        # values are close-ish to tf but floating point variances accumulate
        output = r[0]
        att_w = r[1]
        att_k = r[2]
        att_phi = r[3]
        att_h = r[4]
        att_c = r[5]
        h1 = r[6]
        c1 = r[7]
        h2 = r[8]
        c2 = r[9]

        pred = self.pred_obj([output])
        return pred

mod = GlobalModel()
learning_rate = 0.0001
optimizer = torch.optim.Adam(mod.parameters(), learning_rate)

mels, mel_mask, text, text_mask, mask, mask_mask, reset = train_itr.next_masked_batch()
text = torch.FloatTensor(text)
text_mask = torch.FloatTensor(text_mask)
mels = torch.FloatTensor(mels)
mel_mask = torch.FloatTensor(mel_mask)
in_mels = mels[:-1, :, :]
in_mel_mask = mel_mask[:-1]
out_mels = mels[1:, :, :]
out_mel_mask = mel_mask[1:]

pred = mod(in_mels, in_mel_mask, out_mels, out_mel_mask, text, text_mask, mask, mask_mask, reset)
cc = torch.pow(pred - out_mels, 2)
loss = cc.sum(dim=-1).mean()
optimizer.zero_grad()
loss.backward()
clipping_grad_norm_(mod.parameters(), 10.)
optimizer.step()

pred0 = pred
cc0 = cc
loss0 = loss

mels, mel_mask, text, text_mask, mask, mask_mask, reset = train_itr.next_masked_batch()
text = torch.FloatTensor(text)
text_mask = torch.FloatTensor(text_mask)
mels = torch.FloatTensor(mels)
mel_mask = torch.FloatTensor(mel_mask)
in_mels = mels[:-1, :, :]
in_mel_mask = mel_mask[:-1]
out_mels = mels[1:, :, :]
out_mel_mask = mel_mask[1:]


pred = mod(in_mels, in_mel_mask, out_mels, out_mel_mask, text, text_mask, mask, mask_mask, reset)
cc = torch.pow(pred - out_mels, 2)
loss = cc.sum(dim=-1).mean()
optimizer.zero_grad()
loss.backward()
clipping_grad_norm_(mod.parameters(), 10.)
optimizer.step()

loss1 = loss
cc1 = cc
pred1 = pred

d0 = {}
d0["pred"] = pred0
d0["cc"] = cc0
d0["loss"] = loss0

d1 = {}
d1["pred"] = pred1
d1["cc"] = cc1
d1["loss"] = loss1
