from __future__ import print_function
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from functools import reduce
from operator import mul

import gc
import math

import os
import sys
import argparse
import numpy as np
import torch
from torch import nn
import torch.functional as F

from kkpthlib.datasets import fetch_mnist
from kkpthlib.datasets import EnglishSpeechCorpus

from kkpthlib import get_logger

from kkpthlib import Linear
from kkpthlib import Dropout
from kkpthlib import BernoulliCrossEntropyFromLogits
from kkpthlib import DiscretizedMixtureOfLogisticsCrossEntropyFromLogits
from kkpthlib import MixtureOfGaussiansNegativeLogLikelihood
from kkpthlib import relu
from kkpthlib import softmax
from kkpthlib import log_softmax
#from kkpthlib import clipping_grad_norm_
from kkpthlib import clipping_grad_value_
from kkpthlib import ListIterator
from kkpthlib import run_loop
from kkpthlib import HParams
from kkpthlib import Conv2d
from kkpthlib import Conv2dTranspose
from kkpthlib import SequenceConv1dStack
from kkpthlib import BiLSTMLayer

from kkpthlib import AttentionMelNetTier
from kkpthlib import MelNetTier
from kkpthlib import MelNetFullContextLayer
from kkpthlib import CategoricalCrossEntropyFromLogits
from kkpthlib import relu
from kkpthlib import Embedding

from kkpthlib import space2batch
from kkpthlib import batch2space
from kkpthlib import split
from kkpthlib import split_np
from kkpthlib import batch_mean_variance_update
from kkpthlib import interleave
from kkpthlib import scan
from kkpthlib import LSTMCell
from kkpthlib import BiLSTMLayer
from kkpthlib import GaussianAttentionCell

import os
import pwd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="script {}".format(__file__))
    parser.add_argument('--axis_splits', type=str, required=True,
                        help='string denoting the axis splits for the model, eg 2121 starting from first split to last\n')
    parser.add_argument('--tier_input_tag', type=str, required=True,
                        help='the tier and data split this particular model is training, with 0,0 being the first tier, first subsplit\n')
    parser.add_argument('--tier_condition_tag', type=str, default=None,
                        help='the tier and data split this particular model is conditioned by, with 0,0 being the first tier, first subsplit\n')
    parser.add_argument('--size_at_depth', '-s', type=str, required=True,
                        help='size of input data in H,W str format, at the specified depth\n')
    parser.add_argument('--n_layers', '-n', type=int, required=True,
                        help='number of layers the tier will have\n')
    parser.add_argument('--hidden_size', type=int, required=True,
                        help='hidden dimension size for every layer\n')
    parser.add_argument('--cell_type', type=str, required=True,
                        help='melnet cell type\n')
    parser.add_argument('--optimizer', type=str, required=True,
                        help='optimizer type\n')
    parser.add_argument('--learning_rate', type=str, required=True,
                        help='learning rate\n')
    parser.add_argument('--real_batch_size', type=int, required=True,
                        help='real batch size\n')
    parser.add_argument('--virtual_batch_size', type=int, required=True,
                        help='virtual_batch size\n')
    parser.add_argument('--experiment_name', type=str, required=True,
                        help='name of overall experiment, will be combined with some of the arg input info for model save')
    parser.add_argument('--previous_saved_model_path', type=str, default=None,
                        help='path to previously saved checkpoint model')
    parser.add_argument('--previous_saved_optimizer_path', type=str, default=None,
                        help='path to previously saved optimizer')
    parser.add_argument('--n_previous_save_steps', type=str, default=None,
                        help='number of save steps taken for previously run model, used to "replay" the data generator back to the same point')
    args = parser.parse_args()
else:
    # filthy global hack passed from sampling code
    import builtins
    args = builtins.my_args
    # note here that the exact parser arguments will need to be *repeated* in the sampling code

if args.previous_saved_model_path is not None:
    assert args.previous_saved_optimizer_path is not None
    assert args.n_previous_save_steps is not None

input_axis_split_list = [int(args.axis_splits[i]) for i in range(len(args.axis_splits))]
input_size_at_depth = [int(el) for el in args.size_at_depth.split(",")]
input_hidden_size = int(args.hidden_size)
input_n_layers = int(args.n_layers)
input_real_batch_size = int(args.real_batch_size)
input_virtual_batch_size = int(args.virtual_batch_size)
input_cell_type = args.cell_type
if float(input_virtual_batch_size) / float(input_real_batch_size) != input_virtual_batch_size // input_real_batch_size:
    raise ValueError("Got non-divisible virtual batch size, virtual batch size should be a multiple of the real batch size")
input_virtual_to_real_batch_multiple = input_virtual_batch_size // input_real_batch_size
assert input_virtual_to_real_batch_multiple > 0
input_tier_input_tag = [int(el) for el in args.tier_input_tag.split(",")]
input_learning_rate = float(args.learning_rate)
input_optimizer = str(args.optimizer)

assert len(input_size_at_depth) == 2
assert len(input_tier_input_tag) == 2
if args.tier_condition_tag is not None:
    input_tier_condition_tag = [int(el) for el in args.tier_condition_tag.split(",")]
    assert len(input_tier_condition_tag) == 2
else:
    input_tier_condition_tag = None

logger = get_logger()

logger.info("sys.argv call {}".format(__file__))
logger.info("{}".format(" ".join(sys.argv)))

logger.info("\ndirect argparse args to script {}".format(__file__))
for arg in vars(args):
    logger.info("{}={}".format(arg, getattr(args, arg)))

hp = HParams(input_dim=1,
             hidden_dim=input_hidden_size,
             use_device='cuda' if torch.cuda.is_available() else 'cpu',
             optimizer=input_optimizer,
             learning_rate=input_learning_rate,
             #optimizer="SM3",
             #learning_rate=.01,
             #learning_rate=.005,
             melnet_cell_type=input_cell_type,
             clip=3.5,
             n_layers_per_tier=[input_n_layers],
             melnet_init="truncated_normal",
             #attention_type="logistic",
             #attention_type="sigmoid_logistic",
             #attention_type="relative_logistic",
             attention_type="sigmoid_logistic_alt",
             #attention_type="gaussian",
             #attention_type="dca",
             #melnet_init=None,
             # 256 mel channels
             input_symbols=256,
             n_mix=10,
             # mixture of logistics n_mix == 10
             #output_size=2 * 10 + 10,
             output_size=1,
             #phone_input_symbols=52, #len(speech.phone_lookup),
             #ascii_input_symbols=65, #len(speech.ascii_lookup)
             phone_input_symbols=65, #max(len(speech.phone_lookup), len(speech.ascii_lookup))
             ascii_input_symbols=65, #max(len(speech.phone_lookup), len(speech.ascii_lookup))
             input_image_size=input_size_at_depth,
             real_batch_size=input_real_batch_size,
             virtual_batch_size=input_virtual_batch_size,
             random_seed=2122)

data_random_state = np.random.RandomState(hp.random_seed)
folder_base = "/usr/local/data/kkastner/ljspeech_cleaned"
fixed_minibatch_time_secs = 4
fraction_train_split = .9
speech = EnglishSpeechCorpus(metadata_csv=folder_base + "/metadata.csv",
                             wav_folder=folder_base + "/wavs/",
                             alignment_folder=folder_base + "/alignment_json/",
                             symbol_type="representation_mixed",
                             fixed_minibatch_time_secs=fixed_minibatch_time_secs,
                             train_split=fraction_train_split,
                             random_state=data_random_state)

"""
s = 0
utt_fails = 0
successes = 0
while True:
    try:
        els = speech.get_utterances(1, [speech.train_keys[s]], skip_mel=False)
        #els = speech.get_utterances(1, [speech.valid_keys[s]], skip_mel=False)
    except:
        print("utt failure at {}".format(s))
        utt_fails += 1
    # check that all ascii pass testing / formatting
    #cond_seq_data_batch, cond_seq_mask, data_batch, data_mask = speech.format_minibatch(els,
    #                                                                                    quantize_to_n_bins=None,
    #                                                                                    symbol_type="ascii")
    # check that all repr mix pass testing / formatting
    r = speech.format_minibatch(els,
                                quantize_to_n_bins=None,
                                symbol_type="representation_mixed")
    cond_seq_data_repr_mix_batch = r[0]
    cond_seq_repr_mix_mask = r[1]
    cond_seq_repr_mix_mask_mask = r[2]
    cond_seq_data_ascii_batch = r[3]
    cond_seq_ascii_mask = r[4]
    cond_seq_data_phoneme_batch = r[5]
    cond_seq_phoneme_mask = r[6]
    data_batch = r[7]
    data_mask = r[8]

    successes += 1
    print("format test {}".format(s))
    s += 1
    if s >= len(speech.train_keys) - 1:
    #if s >= len(speech.valid_keys) - 1:
        print("finished test here")
        from IPython import embed; embed(); raise ValueError()
"""


"""
els = speech.get_utterances(1, speech.train_keys, skip_mel=False)
cond_seq_data_batch, cond_seq_mask, data_batch, data_mask = speech.format_minibatch(els,
                                                                                    quantize_to_n_bins=None)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.imshow(data_batch[0], cmap="viridis")
plt.savefig("tmp_mel.png")
from IPython import embed; embed(); raise ValueError()
"""

# want to see these every 5000 train samples, every 1250 valid samples have been processed from dataset
n_train_steps_per = 5000 // input_virtual_batch_size
n_valid_steps_per = 1250 // input_virtual_batch_size
# used in the loop
#n_train_steps_per = 1000
#n_valid_steps_per = 250

"""
train_el = speech.get_train_utterances(10)
valid_el = speech.get_train_utterances(10)
speech.format_minibatch(train_el)
print("hrr")
from IPython import embed; embed(); raise ValueError()
"""

def get_hparams():
    return hp

def build_model(hp):
    random_state = np.random.RandomState(hp.random_seed)
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            if input_tier_condition_tag is None:
                # handle text attention separately
                self.embed_ascii = Embedding(hp.ascii_input_symbols, hp.hidden_dim, random_state=random_state,
                                            name="tier_{}_{}_sz_{}_{}_embed_ascii".format(input_tier_input_tag[0], input_tier_input_tag[1], hp.input_image_size[0], hp.input_image_size[1]), device=hp.use_device)

                self.embed_phone = Embedding(hp.phone_input_symbols, hp.hidden_dim, random_state=random_state,
                                            name="tier_{}_{}_sz_{}_{}_embed_phone".format(input_tier_input_tag[0], input_tier_input_tag[1], hp.input_image_size[0], hp.input_image_size[1]), device=hp.use_device)

                self.embed_mask = Embedding(2, hp.hidden_dim, random_state=random_state,
                                            name="tier_{}_{}_sz_{}_{}_embed_mask".format(input_tier_input_tag[0], input_tier_input_tag[1], hp.input_image_size[0], hp.input_image_size[1]), device=hp.use_device)

                #self.conv_text = SequenceConv1dStack([hp.hidden_dim], hp.hidden_dim, n_stacks=3, random_state=random_state,
                #                                     name="tier_{}_{}_sz_{}_{}_conv_text".format(input_tier_input_tag[0], input_tier_input_tag[1], hp.input_image_size[0], hp.input_image_size[1]), device=hp.use_device)
                # divided by 2 so the output is hp.hidden_dim
                self.bilstm_text = BiLSTMLayer([hp.hidden_dim], hp.hidden_dim // 2, random_state=random_state,
                                               init=hp.melnet_init,
                                               name="tier_{}_{}_sz_{}_{}_bilstm_text".format(input_tier_input_tag[0], input_tier_input_tag[1], hp.input_image_size[0], hp.input_image_size[1]),
                                               device=hp.use_device)

                self.mn_t = AttentionMelNetTier([hp.input_symbols], hp.input_image_size[0], hp.input_image_size[1],
                                                hp.hidden_dim, hp.output_size, hp.n_layers_per_tier[0],
                                                cell_type=hp.melnet_cell_type,
                                                has_centralized_stack=True,
                                                has_attention=True,
                                                attention_type=hp.attention_type,
                                                random_state=random_state,
                                                init=hp.melnet_init,
                                                device=hp.use_device,
                                                name="tier_{}_{}_sz_{}_{}_mn".format(input_tier_input_tag[0], input_tier_input_tag[1],
                                                                                     hp.input_image_size[0], hp.input_image_size[1]))
            else:
                self.mn_t = AttentionMelNetTier([hp.input_symbols], hp.input_image_size[0], hp.input_image_size[1],
                                                hp.hidden_dim, hp.output_size, hp.n_layers_per_tier[0],
                                                has_spatial_condition=True,
                                                cell_type=hp.melnet_cell_type,
                                                random_state=random_state,
                                                init=hp.melnet_init,
                                                device=hp.use_device,
                                        name="tier_{}_{}_cond_{}_{}_sz_{}_{}_mn".format(input_tier_input_tag[0], input_tier_input_tag[1],
                                                                                        input_tier_condition_tag[0], input_tier_condition_tag[1],
                                                                                        hp.input_image_size[0], hp.input_image_size[1]))


        def forward(self, x, x_mask=None,
                    spatial_condition=None,
                    memory_condition=None, memory_condition_mask=None,
                    memory_condition_mask_mask=None,
                    batch_norm_flag=0.):
            # for now we don't use the x_mask in the model itself, only in the loss calculations
            if spatial_condition is None:
                assert memory_condition is not None
                mem_a, mem_a_e = self.embed_ascii(memory_condition)
                mem_p, mem_p_e = self.embed_phone(memory_condition)
                # condition mask is 0 where it is ascii, 1 where it is phone
                mem_j = memory_condition_mask[..., None] * mem_p + (1. - memory_condition_mask[..., None]) * mem_a
                mem_m, mem_m_e = self.embed_mask(memory_condition_mask[..., None])

                mem_f = mem_j + mem_m

                # doing bn in 16 bit is sketch to say the least
                #mem_conv = self.conv_text([mem_f], batch_norm_flag)
                # mask based on the actual conditioning mask
                #mem_conv = mem_conv * memory_condition_mask_mask[..., None]
                mem_f = mem_f * memory_condition_mask_mask[..., None]

                # use mask in BiLSTM
                mem_lstm = self.bilstm_text([mem_f], input_mask=memory_condition_mask_mask)
                # x currently batch, time, freq, 1
                # mem time, batch, feat
                # feed mask for attention calculations as well
                mn_out, alignment, attn_extras = self.mn_t([x], memory=mem_lstm, memory_mask=memory_condition_mask_mask)
                self.attention_alignment = alignment
                self.attention_extras = attn_extras
            else:
                mn_out = self.mn_t([x], list_of_spatial_conditions=[spatial_condition])
            return mn_out
    return Model().to(hp.use_device)

if __name__ == "__main__":
    model = build_model(hp)
    use_half = True
    use_mixed_precision = False
    if use_half:
        if hp.optimizer == "adam":
            from kkpthlib.optimizers.fp16_adam import Adam16
            optimizer = Adam16(model.parameters(), hp.learning_rate, eps=1E-6)
        elif hp.optimizer == "SM3":
            from kkpthlib.optimizers.fp16_SM3 import SM3
            optimizer = SM3(model.parameters(), hp.learning_rate, momentum=0.9, beta=0.0, eps=1E-6)
        else:
            raise ValueError("Unknown optimizer given to hyperparameter settings {}".format(hp.optimizer))
    else:
        if hp.optimizer == "adam":
            optimizer = torch.optim.Adam(model.parameters(), hp.learning_rate, eps=1E-6)
        else:
            from SM3 import SM3
            optimizer = SM3(model.parameters(), hp.learning_rate, momentum=0.9, beta=0.0, eps=1E-6)

    # code from https://towardsdatascience.com/adaptive-and-cyclical-learning-rates-using-pytorch-2bf904d18dee
    def cyclical_lr(stepsize, min_lr, max_lr):
        # Scaler: we can adapt this if we do not want the triangular CLR
        scaler = lambda x: .95 * x

        # Lambda function to calculate the LR
        lr_lambda = lambda it: min_lr + (max_lr - min_lr) * relative(it, stepsize)

        # Additional function to see where on the cycle we are
        def relative(it, stepsize):
            cycle = math.floor(1 + it / (2 * stepsize))
            x = abs(it / stepsize - 2 * cycle + 1)
            return max(0, (1 - x)) * scaler(cycle)
        return lr_lambda

    #lr_lambda = lambda step: min(1., ((step + 1.) / warmup_steps))
    step_size = 500
    end_lr = hp.learning_rate
    factor = 20
    # no cyclic lr here
    lr_lambda = lambda step: min(1., step / 1000.)
    #lr_lambda = cyclical_lr(step_size, min_lr=end_lr / factor, max_lr=end_lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


    dataset_name = folder_base.split("/")[-1]
    dataset_max_limit = fixed_minibatch_time_secs
    axis_splits_str = "".join([str(aa) for aa in input_axis_split_list])
    axis_size_str = "{}x{}".format(input_size_at_depth[0], input_size_at_depth[1])
    tier_depth_str = str(input_tier_input_tag[0])


    # hardcoded per-dimension mean and std for mel data from the training iterator, read from a file
    full_cached_mean_std_name_for_experiment = "{}_max{}secs_{}splits_{}sz_{}tierdepth_mean_std.npz".format(dataset_name,
                                                                                                        dataset_max_limit,
                                                                                                        axis_splits_str,
                                                                                                        axis_size_str,
                                                                                                        tier_depth_str)
    mean_std_cache = os.getcwd() + "/mean_std_cache/"
    if not os.path.exists(mean_std_cache):
        os.makedirs(mean_std_cache)
    mean_std_path = mean_std_cache + full_cached_mean_std_name_for_experiment
    if not os.path.exists(mean_std_path):
        print("Estimated mean and std for dataset/tier combo not found, checked {}".format(mean_std_path))
        # calculate estimate over 10000 samples randomly drawn from the speech loader!
        mean_var_random_state = np.random.RandomState(8179)
        # manually set the speech iterator random state to the new one - this preserves the data splits but wont just give the first
        # N minibatches for our mean var estimates
        speech.random_state = mean_var_random_state

        # not an exact thing since we sample with replacement, but should roughly form the mean/var for the train set
        #n_sample_estimate = len(speech.train_keys)
        # 5000 sequences should be plenty
        n_sample_estimate = 5000
        subbatch = 100

        running_mean_vec = None
        running_var_vec = None
        running_sample_count = None
        count = 0

        while count < n_sample_estimate:
            print("mean/var estimate count {} of {}".format(count, n_sample_estimate))
            els = speech.get_utterances(subbatch, speech.train_keys, skip_mel=False)
            r = speech.format_minibatch(els,
                                        quantize_to_n_bins=None)
            cond_seq_data_repr_mix_batch = r[0]
            cond_seq_repr_mix_mask = r[1]
            cond_seq_repr_mix_mask_mask = r[2]
            cond_seq_data_ascii_batch = r[3]
            cond_seq_ascii_mask = r[4]
            cond_seq_data_phoneme_batch = r[5]
            cond_seq_phoneme_mask = r[6]
            data_batch = r[7]
            data_mask = r[8]

            x_mask_t = data_mask[..., None, None] + 0. * data_batch[..., None]
            x_t = data_batch[..., None]
            divisors = [2, 4, 8]
            max_frame_count = x_t.shape[1]
            for di in divisors:
                # nearest divisible number above, works because largest divisor divides by smaller
                # we need something that has a length in time (frames) divisible by 2 4 and 8 due to the nature of melnet
                # same for frequency but frequency is a power of 2 so no need to check it
                q = int(max_frame_count / di)
                if float(max_frame_count / di) == int(max_frame_count / di):
                    max_frame_count = di * q
                else:
                    max_frame_count = di * (q + 1)
            assert max_frame_count == int(max_frame_count)

            axis_splits = input_axis_split_list
            splits_offset = 0
            axis1_m = [2 for a in str(axis_splits)[splits_offset:] if a == "1"]
            axis2_m = [2 for a in str(axis_splits)[splits_offset:] if a == "2"]
            axis1_m = reduce(mul, axis1_m)
            axis2_m = reduce(mul, axis2_m)

            x_in = x_t[:, ::axis1_m, ::axis2_m]
            x_mask_in = x_mask_t[:, ::axis1_m, ::axis2_m]

            """
            all_x_splits = []
            x_t = data_batch[..., None]
            for aa in input_axis_split_list:
                all_x_splits.append(split_np(x_t, axis=aa))
                x_t = all_x_splits[-1][0]
            x_in = all_x_splits[::-1][input_tier_input_tag[0]][input_tier_input_tag[1]]

            all_x_mask_splits = []
            # broadcast mask over frequency so we can downsample
            x_mask_t = data_mask[..., None, None] + 0. * data_batch[..., None]
            for aa in input_axis_split_list:
                all_x_mask_splits.append(split_np(x_mask_t, axis=aa))
                x_mask_t = all_x_mask_splits[-1][0]
            x_mask_in = all_x_mask_splits[::-1][input_tier_input_tag[0]][input_tier_input_tag[1]]
            """

            megabatch_frames = []
            for _i in range(x_in.shape[0]):
                masklen = np.sum(x_mask_in[_i, :, :, 0], axis=0)
                assert np.all(masklen == int(masklen[0]))
                this_x_in = x_in[_i, :int(masklen[0]), :, 0]
                megabatch_frames.append(this_x_in)
            this_batch = np.concatenate(megabatch_frames)

            if running_mean_vec is None:
                # only build megabatch out of non masked frames
                running_mean_vec = np.mean(this_batch, axis=0)
                running_var_vec = np.std(this_batch, axis=0)
                running_sample_count = float(this_batch.shape[0])
            else:
                new_mean, new_var, new_count = batch_mean_variance_update(this_batch, running_mean_vec, running_var_vec, running_sample_count)
                running_mean_vec = new_mean
                running_var_vec = new_var
                running_sample_count = new_count
            count += subbatch
        running_std_vec = np.sqrt(running_var_vec)
        mean = running_mean_vec
        std = running_std_vec
        frame_count = running_sample_count
        np.savez(mean_std_path, mean=mean, std=std, frame_count=frame_count)
        # terminate here because we needed to manipulate the speech loader...
        raise ValueError("Terminating after calculation and caching of mean/std data to {}, rerun to train".format(mean_std_cache))


    if args.previous_saved_model_path is not None:
        assert args.previous_saved_optimizer_path is not None
        assert args.n_previous_save_steps is not None
        n_previous_save_steps = int(args.n_previous_save_steps)

        model_dict = torch.load(args.previous_saved_model_path, map_location=hp.use_device)
        model.load_state_dict(model_dict)

        opt_dict = torch.load(args.previous_saved_optimizer_path, map_location=hp.use_device)
        optimizer.load_state_dict(opt_dict)

        del model_dict
        del opt_dict

        logger.info("Reloaded weights from previous training files based on commandline arguments")
        logger.info("Previous model file: {}".format(args.previous_saved_model_path))
        logger.info("Previous optimizer file: {}".format(args.previous_saved_optimizer_path))
        logger.info("Previous number of save steps: {}".format(n_previous_save_steps))

        # re-run data iterator forward for reload
        logger.info("Replaying data iterator")
        for _i in range(n_previous_save_steps):
            logger.info("Data iterator train step: {} of {}".format(_i * n_train_steps_per, n_previous_save_steps * n_train_steps_per))
            # since mel calculations take the bulk of the time, and we are just replaying the iterator, skip it
            for _j in range(n_train_steps_per):
                train_el = speech.get_train_utterances(input_real_batch_size, skip_mel=True, fastforward_state=True)
                #cond_seq_data_batch, cond_seq_mask, data_batch, data_mask = speech.format_minibatch(train_el)
                batch_norm_flag = 0.
            for _j in range(n_valid_steps_per):
                valid_el = speech.get_valid_utterances(input_real_batch_size, skip_mel=True, fastforward_state=True)
                #cond_seq_data_batch, cond_seq_mask, data_batch, data_mask = speech.format_minibatch(valid_el)
                batch_norm_flag = 1.

    #loss_function = DiscretizedMixtureOfLogisticsCrossEntropyFromLogits()
    # needs 30 for n_mix of 10
    loss_function = MixtureOfGaussiansNegativeLogLikelihood()

    # load in cached mean std
    speech.load_mean_std_from_filepath(mean_std_path)
    saved_mean = speech.cached_mean_vec_[None, None, :, None]
    saved_std = speech.cached_std_vec_[None, None, :, None]
    #saved_torch_mean = torch.tensor(speech.cached_mean_vec_[None, None, :, None]).contiguous().to(hp.use_device)
    #saved_torch_std = torch.tensor(speech.cached_std_vec_[None, None, :, None]).contiguous().to(hp.use_device)

    if use_half:
        assert use_mixed_precision is False
        model.half()  # convert to half precision
        for layer in model.modules():
            layer.half()
        [a.half() for a in model.parameters()]

    scaler = torch.cuda.amp.GradScaler()

    noise_random_state = np.random.RandomState(111212)
    def loop(itr, extras, stateful_args):
        if extras["train"]:
            model.train()
        else:
            model.eval()
        model.zero_grad()
        optimizer.zero_grad()

        total_l = None
        step_count = input_virtual_to_real_batch_multiple
        assert step_count >= 1
        for _step in range(step_count):
            if extras["train"]:
                train_el = speech.get_train_utterances(input_real_batch_size)
                r = speech.format_minibatch(train_el,
                                            quantize_to_n_bins=None)
                cond_seq_data_repr_mix_batch = r[0]
                cond_seq_repr_mix_mask = r[1]
                cond_seq_repr_mix_mask_mask = r[2]
                cond_seq_data_ascii_batch = r[3]
                cond_seq_ascii_mask = r[4]
                cond_seq_data_phoneme_batch = r[5]
                cond_seq_phoneme_mask = r[6]
                data_batch = r[7]
                data_mask = r[8]

                batch_norm_flag = 0.
            else:
                valid_el = speech.get_valid_utterances(input_real_batch_size)
                r = speech.format_minibatch(valid_el,
                                            quantize_to_n_bins=None)
                cond_seq_data_repr_mix_batch = r[0]
                cond_seq_repr_mix_mask = r[1]
                cond_seq_repr_mix_mask_mask = r[2]
                cond_seq_data_ascii_batch = r[3]
                cond_seq_ascii_mask = r[4]
                cond_seq_data_phoneme_batch = r[5]
                cond_seq_phoneme_mask = r[6]
                data_batch = r[7]
                data_mask = r[8]

                batch_norm_flag = 1.

            torch_cond_seq_data_batch = torch.tensor(cond_seq_data_repr_mix_batch[..., None]).contiguous().to(hp.use_device)
            torch_cond_seq_data_mask = torch.tensor(cond_seq_repr_mix_mask).contiguous().to(hp.use_device)
            torch_cond_seq_data_mask_mask = torch.tensor(cond_seq_repr_mix_mask_mask).contiguous().to(hp.use_device)
            """
            data_batch, = next(itr)
            # N H W C
            data_batch = data_batch.reshape(data_batch.shape[0], 28, 28, 1)
            # N C H W
            data_batch = data_batch.transpose(0, 3, 1, 2).astype("int32").astype("float32")
            torch_data_batch = torch.tensor(data_batch).contiguous().to(hp.use_device)
            # N H W 1
            torch_data_batch = torch_data_batch[:, 0][..., None]
            """

            x_mask_t = data_mask[..., None, None] + 0. * data_batch[..., None]
            x_t = data_batch[..., None]
            divisors = [2, 4, 8]
            max_frame_count = x_t.shape[1]
            for di in divisors:
                # nearest divisible number above, works because largest divisor divides by smaller
                # we need something that has a length in time (frames) divisible by 2 4 and 8 due to the nature of melnet
                # same for frequency but frequency is a power of 2 so no need to check it
                q = int(max_frame_count / di)
                if float(max_frame_count / di) == int(max_frame_count / di):
                    max_frame_count = di * q
                else:
                    max_frame_count = di * (q + 1)
            assert max_frame_count == int(max_frame_count)

            axis_splits = input_axis_split_list
            splits_offset = 0
            axis1_m = [2 for a in str(axis_splits)[splits_offset:] if a == "1"]
            axis2_m = [2 for a in str(axis_splits)[splits_offset:] if a == "2"]
            axis1_m = reduce(mul, axis1_m)
            axis2_m = reduce(mul, axis2_m)

            x_in_np = x_t[:, ::axis1_m, ::axis2_m]
            x_mask_in_np = x_mask_t[:, ::axis1_m, ::axis2_m]
            x_in_np = (x_in_np - saved_mean) / saved_std

            """
            all_x_splits = []
            x_t = data_batch[..., None]
            for aa in input_axis_split_list:
                all_x_splits.append(split_np(x_t, axis=aa))
                x_t = all_x_splits[-1][0]
            x_in_np = all_x_splits[::-1][input_tier_input_tag[0]][input_tier_input_tag[1]]
            x_in_np = (x_in_np - saved_mean) / saved_std

            # additive noise? next thing to try

            all_x_mask_splits = []
            # broadcast mask over frequency so we can downsample
            x_mask_t = data_mask[..., None, None] + 0. * data_batch[..., None]
            for aa in input_axis_split_list:
                all_x_mask_splits.append(split_np(x_mask_t, axis=aa))
                x_mask_t = all_x_mask_splits[-1][0]
            x_mask_in_np = all_x_mask_splits[::-1][input_tier_input_tag[0]][input_tier_input_tag[1]]
            """

            x_in = torch.tensor(x_in_np).contiguous().to(hp.use_device)
            x_mask_in = torch.tensor(x_mask_in_np).contiguous().to(hp.use_device)

            if input_tier_condition_tag is not None:
                spatial_cond = all_x_splits[::-1][input_tier_condition_tag[0]][input_tier_condition_tag[1]]
                x_cond_in = torch.tensor(spatial_cond).contiguous().to(hp.use_device)

            if use_half:
                torch_cond_seq_data_batch = torch.tensor(cond_seq_data_repr_mix_batch[..., None]).contiguous().to(hp.use_device).half()
                torch_cond_seq_data_mask = torch.tensor(cond_seq_repr_mix_mask).contiguous().to(hp.use_device).half()
                torch_cond_seq_data_mask_mask = torch.tensor(cond_seq_repr_mix_mask_mask).contiguous().to(hp.use_device).half()
                x_in = torch.tensor(x_in_np).contiguous().to(hp.use_device).half()
                x_mask_in = torch.tensor(x_mask_in_np).contiguous().to(hp.use_device).half()
                if input_tier_condition_tag is not None:
                    x_cond_in = torch.tensor(spatial_cond).contiguous().to(hp.use_device).half()

            with torch.set_grad_enabled(True):
                with torch.cuda.amp.autocast(enabled=use_mixed_precision):
                    if input_tier_condition_tag is None:
                        pred_out = model(x_in, x_mask=x_mask_in,
                                         memory_condition=torch_cond_seq_data_batch,
                                         memory_condition_mask=torch_cond_seq_data_mask,
                                         memory_condition_mask_mask=torch_cond_seq_data_mask_mask,
                                         batch_norm_flag=batch_norm_flag)
                    else:
                        pred_out = model(x_in, x_mask=x_mask_in,
                                         spatial_condition=x_cond_in,
                                         batch_norm_flag=batch_norm_flag)

                    # x_in comes in discretized between 0 and 256, now scale -1 to 1
                    #loss1 = loss_function(pred_out, 2 * (x_in / 256.) - 1., n_mix=hp.n_mix)
                    # for now just do mse
                    #loss1 = (pred_out - x_in) ** 2

                    # calculate loss in 32 but then recast to 16 bit?
                    loss1 = torch.abs(pred_out - x_in)
                    if use_half:
                        loss1 = loss1.float()
                    # need to be very careful here...
                    loss = ((loss1 * x_mask_in / step_count) / x_mask_in.sum()).sum()
                    #loss = ((loss1 * x_mask_in) / x_mask_in.sum()).sum()
                    l = loss.cpu().data.numpy()
                    if np.isnan(l):
                        print("NAN in loss! Dropping into debug")
                        from IPython import embed; embed(); raise ValueError()

                    if use_half:
                        loss = loss.half()

                    if extras["train"]:
                        if use_mixed_precision:
                            #scaler.scale(loss / step_count).backward()
                            scaler.scale(loss).backward()
                        else:
                            #(loss / step_count).backward()
                            loss.backward()
                    else:
                        loss = 0. * loss
                        # need to do backward to avoid memory issues in valid!
                        loss.backward()
                        # wipe the grad immediately
                        model.zero_grad()

                    for n, v in model.named_parameters():
                        if torch.any(torch.isnan(v.grad)):
                            print("Found nan in grad!")
                            from IPython import embed; embed(); raise ValueError()

                    if total_l is None:
                        #total_l = l / step_count
                        total_l = l
                    else:
                        #total_l += l / step_count
                        total_l += l

            # delete intermediates to save memory
            del torch_cond_seq_data_batch
            del torch_cond_seq_data_mask
            del torch_cond_seq_data_mask_mask
            del x_in
            del x_mask_in
            del pred_out
            del loss
            del loss1
            if input_tier_condition_tag is None:
                del model.attention_alignment
                del model.attention_extras
            else:
                del x_cond_in
            gc.collect()
            torch.cuda.empty_cache()

        if extras["train"]:
            #clipping_grad_value_(model.named_parameters(), hp.clip, named_check=True)
            if use_mixed_precision:
                scaler.unscale_(optimizer)
                clipping_grad_value_(model.parameters(), hp.clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                clipping_grad_value_(model.parameters(), hp.clip)
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()
        """
        del torch_cond_seq_data_batch
        del torch_cond_seq_data_mask
        del x_in
        del x_mask_in
        del pred_out
        del loss
        del loss1
        del model.attention_alignment
        del model.attention_extras
        gc.collect()
        torch.cuda.empty_cache()
        """
        return total_l, None, None

    s = {"model": model,
         "optimizer": optimizer,
         "hparams": hp}

    """
    r = loop(speech, {"train": True}, None)
    r2 = loop(speech, {"train": True}, None)
    print(r)
    print(r2)

    rs = []
    for i in range(10):
        print(i)
        rx = loop(speech, {"train": True}, None)
        rs.append(rx)
        print(rs)
    from IPython import embed; embed(); raise ValueError()
    """

    if input_tier_condition_tag is None:
        tag = str(args.experiment_name) + "_tier_{}_{}_sz_{}_{}".format(input_tier_input_tag[0], input_tier_input_tag[1],
                                                                        input_size_at_depth[0], input_size_at_depth[1])
    else:
        tag = str(args.experiment_name) + "_tier_{}_{}_cond_{}_{}_sz_{}_{}".format(input_tier_input_tag[0], input_tier_input_tag[1],
                                                                                   input_tier_condition_tag[0], input_tier_condition_tag[1],
                                                                                   input_size_at_depth[0], input_size_at_depth[1])

    run_loop(loop, speech,
             loop, speech,
             s,
             force_tag=tag,
             n_train_steps_per=n_train_steps_per,
             n_valid_steps_per=n_valid_steps_per)
