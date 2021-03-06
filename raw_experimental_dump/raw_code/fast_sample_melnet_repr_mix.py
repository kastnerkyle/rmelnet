# -*- coding: utf-8 -*-
"""fast_sample_melnet_repr_mix.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1j81QKEhgHZ-iQT43OrcEMvlt8qtS0qYF
"""

!nvidia-smi

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# can pre-build gentle, takes quite some time
# for now, it is prebuilt in my mounted folder and will be copied in later
#!git clone -q https://github.com/lowerquality/gentle.git > /dev/null
#%cd gentle
#!./install.sh

# Author: Kyle Kastner
# License: BSD 3-Clause

import os
import sys
from os.path import exists, join, expanduser
import IPython
from IPython.display import Audio
import matplotlib.pyplot as plt
plt.style.use('classic')

os.chdir(os.path.expanduser("~"))
lib_dir = "kkpthlib"
if not os.path.exists(lib_dir):
  ! git clone https://github.com/kastnerkyle/$lib_dir
  ! cd $lib_dir && git checkout rmelnet_basics 
  ! pip install pretty_midi

os.chdir(os.path.expanduser("~"))
lib_dir = "hmm_tts_build"
if not os.path.exists(lib_dir):
  ! git clone https://github.com/kastnerkyle/$lib_dir
  ! cd $lib_dir && bash install_voices.sh
  ! sudo apt-get install file

os.chdir(os.path.expanduser("~"))
lib_dir = "ez-phones"
if not os.path.exists(lib_dir):
  ! git clone https://github.com/kastnerkyle/$lib_dir
  ! sudo apt-get install bison libtool autoconf swig sox
  ! cd $lib_dir && bash setup.sh

os.chdir(os.path.expanduser("~"))
lib_dir = "stexp"
sub_dir = "stexp/wavernn"
if not os.path.exists(lib_dir):
  ! git clone https://github.com/kastnerkyle/$lib_dir
  ! cd $lib_dir && git checkout working
  ! cd $sub_dir && pip install -r "requirements.txt"
  has_installed = False

lib_dir = "kkpthlib"
os.chdir(join(expanduser("~"), lib_dir))
# generally want 1.9.1 but use nightlies for now
if not has_installed:
    ! pip3 install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/gpu/torch_nightly.html
    ! pip3 install noisereduce
    ! python setup.py build
    ! ln -s ~/kkpthlib/kkpthlib /usr/local/lib/python$(python --version | cut -d " " -f 2 | cut -d "." -f 1).$(python --version | cut -d " " -f 2 | cut -d "." -f 2)/dist-packages/
    has_installed = True

os.chdir(os.path.expanduser("~"))
# copy in prebuilt kaldi/gentle
if not os.path.exists("gentle"):
    ! cp -pr /content/drive/MyDrive/gentle_kaldi_prebuilt/gentle.tar.gz .
    ! tar xzf gentle.tar.gz
    ! chmod -R 755 gentle

"""#Global

"""

#sentence = "Crimson rain pummeled the hero across his helm"                 # 0
sentence = "Who wants to be the king anyway"                                # 1
#sentence = "Concerned citizens called the authority board"                  # 2
#sentence = "Their boat sank into the icy river"                             # 3
#sentence = "Sometimes the old ways are best"                                # 4
#sentence = "The sky above the port was the color of television"             # 5
#sentence = "Surface to air missiles rocked the conflict"                    # 6
#sentence = "Jamie died but she brought him back to life"                    # 7
#sentence = "Be the girl who talks today"                                    # 8

#sentence = "The birch canoe slid on the smooth planks"                      # 9
#sentence = "Glue the sheet to the dark blue background"                     # 10
#sentence = "Its easy to tell the depth of a well"                           # 11
#sentence = "It is easy to tell the depth of a well"                         # 12
#sentence = "These days a chicken leg is a rare dish"                        # 13
#sentence = "Rice is often served in round bowls"                            # 14
#sentence = "The juice of lemons makes fine punch"                           # 15
#sentence = "The box was thrown beside the parked truck"                     # 16
#sentence = "The hogs were fed chopped corn and garbage"                     # 17
#sentence = "Four hours of steady work faced us"                             # 18
#sentence = "Large size in stockings is hard to sell"                        # 19
#sentence = "The Harvard sentences are sample phrases"                       # 20

#sentence = "The boy was there when the sun rose"                            # 21
#sentence = "A rod is used to catch pink salmon"                             # 22
#sentence = "The source of the huge river is the clear spring"               # 23
#sentence = "Kick the ball straight and follow through"                      # 24

frame_offset = 0
split_gap = 0.0
sample_index = "[0,1,2,3,4,5,6,7,8,9]"
#additive_noise_level = .635
additive_noise_level = .33
conditioning_type = "phoneme"
use_half = True
use_double = None
force_ascii_words = None
#force_ascii_words = "anyway"
force_phoneme_words = None
# empirical tuning, higher tau (e.g. from -inf toward 0 and then above) is a softer cutoff
# -1.5 is a "sharp" cutoff, can sometimes cut early
attention_termination_tau = -.35
n_noise_samples = 100
#conditioning_type="ascii"
#force_ascii_words = "helm" 
#conditioning_type="ascii"
#force_phoneme_words = "pummeled,hero"

#sentence = "Their boat sank into the icy river"
#frame_offset = 0
#split_gap = 0.0
#sample_index = "[0,1,2,3,4,5,6,7,8,9]"
#additive_noise_level = .21
#conditioning_type="phoneme"
#force_ascii_words = None #"helm"

os.chdir(expanduser("~"))

ls hmm_tts_build/

!echo $sentence > text.txt

! bash hmm_tts_build/say_it.sh "$sentence" output.wav

! file output.wav

! sox output.wav -b 16 output_16k.wav rate 16k

"""
# TTS"""

from scipy.io import wavfile
from IPython.display import Audio
from kkpthlib.datasets.speech.audio_processing.audio_tools import soundsc

fs, d = wavfile.read("output_16k.wav")
synth_ts = d.astype('float32') / (2 ** 15)
synth_ts -= synth_ts.mean()
Audio(data=soundsc(synth_ts), rate=16000)

cd ez-phones

! ln -snf ../output_16k.wav .

! bash ps_shortcut.sh output_16k.wav > recognition_info.txt

! cat recognition_info.txt

os.chdir(os.path.expanduser("~"))

! ln -nsf ez-phones/recognition_info.txt .

cat recognition_info.txt

ls

# Now that we have the output wav file, and its recognized phonetic content
# use the phoneme timing to build the datastructure needed for the synthesis model
# then run the full sampling pipeline...

os.chdir(os.path.expanduser("~"))

! python gentle/align.py output_16k.wav text.txt > gentle_recognition_info.json

cat gentle_recognition_info.json

! mkdir -p /home/kkastner/_kkpthlib_models/

! mkdir -p /usr/local/data/kkastner/

os.chdir(os.path.expanduser("~"))
#if not os.path.exists("melnet_base_saved_models_reduced.tar.gz"): 
#    ! cp -pr /content/drive/MyDrive/melnet_sampling_resources/melnet_base_saved_models_reduced.tar.gz .
#    ! tar xzf melnet_base_saved_models_reduced.tar.gz
#    ! mv melnet_base_saved_models_reduced/melnet/* ~/_kkpthlib_models
#    ! mv melnet_base_saved_models_reduced/wavernn/*256mel stexp/wavernn
FORCE_RELOAD = False
if not os.path.exists("ljspeech_mean_std_cache.tar.gz") or FORCE_RELOAD:
    ! cp -pr /content/drive/MyDrive/melnet_sampling_resources/ljspeech_mean_std_cache.tar.gz .
    ! tar xzf ljspeech_mean_std_cache.tar.gz
    ! mv mean_std_cache ~/kkpthlib/examples/attention_melnet_cmdline_ljspeech
if not os.path.exists("ljspeech_skiplist_cache.tar.gz") or FORCE_RELOAD:
    ! cp -pr /content/drive/MyDrive/melnet_sampling_resources/ljspeech_skiplist_cache.tar.gz .
    ! tar xzf ljspeech_skiplist_cache.tar.gz
    ! mv skiplist_cache ~/kkpthlib/examples/attention_melnet_cmdline_ljspeech
if not os.path.exists("ljspeech_repr_mix_mean_std_cache.tar.gz") or FORCE_RELOAD:
    ! cp -pr /content/drive/MyDrive/melnet_sampling_resources/ljspeech_repr_mix_mean_std_cache.tar.gz .
    ! tar xzf ljspeech_repr_mix_mean_std_cache.tar.gz
    ! mv mean_std_cache ~/kkpthlib/examples/attention_melnet_cmdline_ljspeech_repr_mix
if not os.path.exists("ljspeech_repr_mix_skiplist_cache.tar.gz") or FORCE_RELOAD:
    ! cp -pr /content/drive/MyDrive/melnet_sampling_resources/ljspeech_repr_mix_skiplist_cache.tar.gz .
    ! tar xzf ljspeech_repr_mix_skiplist_cache.tar.gz
    ! mv skiplist_cache ~/kkpthlib/examples/attention_melnet_cmdline_ljspeech_repr_mix
if not os.path.exists("mini_robovoice_c_25k.tar.gz") or FORCE_RELOAD:
    ! cp -pr /content/drive/MyDrive/melnet_sampling_resources/mini_robovoice_c_25k.tar.gz .
    ! tar xzf mini_robovoice_c_25k.tar.gz
    ! mv mini_robovoice_c_25k /usr/local/data/kkastner/mini_robovoice_c_25k
if not os.path.exists("mini_ljspeech_cleaned.tar.gz") or FORCE_RELOAD:
    ! cp -pr /content/drive/MyDrive/melnet_sampling_resources/mini_ljspeech_cleaned.tar.gz .
    ! tar xzf mini_ljspeech_cleaned.tar.gz
    ! cp -pr mini_ljspeech_cleaned /usr/local/data/kkastner/mini_ljspeech_cleaned
    ! mv mini_ljspeech_cleaned /usr/local/data/kkastner/ljspeech_cleaned
if not os.path.exists("saved_attention_melnet_cmdline_ljspeech_repr_mix_tier_0_0_sz_112_32.tar.gz") or FORCE_RELOAD:
    ! cp -pr /content/drive/MyDrive/melnet_sampling_resources/saved_attention_melnet_cmdline_ljspeech_repr_mix_tier_0_0_sz_112_32.tar.gz .
    ! tar xzf saved_attention_melnet_cmdline_ljspeech_repr_mix_tier_0_0_sz_112_32.tar.gz
    ! mv saved_attention_melnet_cmdline_ljspeech_repr_mix_tier_0_0_sz_112_32 /home/kkastner/_kkpthlib_models/saved_attention_melnet_cmdline_ljspeech_repr_mix_tier_0_0_sz_112_32
if not os.path.exists("wavernn_ljspeech_downsample_4_0_0_alt.tar.gz") or FORCE_RELOAD:
    ! cp -pr /content/drive/MyDrive/melnet_sampling_resources/wavernn_ljspeech_downsample_4_0_0_alt.tar.gz .
    ! tar xzf wavernn_ljspeech_downsample_4_0_0_alt.tar.gz
    ! mv checkpoint_step000450000.pth /root/stexp/wavernn_ljspeech_downsample_4_0_0_alt/checkpoint_step000450000.pth
if not os.path.exists("wavernn_ljspeech_fullres_cleanup.tar.gz") or FORCE_RELOAD:
    ! cp -pr /content/drive/MyDrive/melnet_sampling_resources/wavernn_ljspeech_fullres_cleanup.tar.gz .
    ! tar xzf wavernn_ljspeech_fullres_cleanup.tar.gz
    ! mv checkpoint_step000520000.pth /root/stexp/wavernn_ljspeech_fullres_cleanup/checkpoint_step000520000.pth

ls

cmd_dir = os.path.expanduser("~") + "/kkpthlib/examples/attention_melnet_cmdline_ljspeech_repr_mix"

#cd ~/kkpthlib/examples/attention_melnet_cmdline_ljspeech/

#! git pull

os.chdir(cmd_dir)

pwd

ls

! ln -nsf ../../../gentle_recognition_info.json .

! git pull

os.chdir(cmd_dir)

full_sample_cmd = "CUDA_VISIBLE_DEVICES=0 python sample_attention_melnet_cmdline_ljspeech_repr_mix.py"
full_sample_cmd += " --axis_split=21212"
full_sample_cmd += " --tier_input_tag=0,0"
full_sample_cmd += " --size_at_depth=112,32"
full_sample_cmd += " --n_layers=5"
full_sample_cmd += " --hidden_size=256"
full_sample_cmd += " --cell_type=gru"
full_sample_cmd += " --optimizer=adam"
full_sample_cmd += " --learning_rate=1E-4"
full_sample_cmd += " --real_batch_size=1"
full_sample_cmd += " --virtual_batch_size=1"
full_sample_cmd += " --bias_data_frame_offset={}".format(str(frame_offset))
full_sample_cmd += " --bias_split_gap={}".format(str(split_gap))
full_sample_cmd += " --output_dir=generated_samples"
full_sample_cmd += " --experiment_name=attn_tts_ljspeech_XXX"
if conditioning_type is not None:
    full_sample_cmd += " --force_conditioning_type='{}'".format(conditioning_type)
if sample_index is not None:
    full_sample_cmd += " --use_sample_index={}".format(str(sample_index))
if force_ascii_words is not None:
    full_sample_cmd += " --force_ascii_words={}".format(str(force_ascii_words))
if force_phoneme_words is not None:
    full_sample_cmd += " --force_phoneme_words={}".format(str(force_phoneme_words))
if attention_termination_tau is not None:
    full_sample_cmd += " --attention_termination_tau={}".format(str(attention_termination_tau))
full_sample_cmd += " --custom_conditioning_json=gentle_recognition_info.json"
full_sample_cmd += " --override_dataset_path='/usr/local/data/kkastner/mini_ljspeech_cleaned'"
full_sample_cmd += " --force_end_punctuation=~"
if n_noise_samples is not None:
    full_sample_cmd += " --n_noise_samples={}".format(n_noise_samples)
if use_half is not None:
    full_sample_cmd += " --use_half"
if use_double is not None:
    full_sample_cmd += " --use_double"
full_sample_cmd += " --additive_noise_level={}".format(additive_noise_level)
full_sample_cmd += " /home/kkastner/_kkpthlib_models/saved_attention_melnet_cmdline_ljspeech_repr_mix_tier_0_0_sz_112_32/saved_models/permanent_model-147264.pth"

print(full_sample_cmd)

#python sample_attention_melnet_cmdline_ljspeech.py --axis_split=21212 --tier_input_tag=0,0 --size_at_depth=112,32 --n_layers=5 --hidden_size=256 --cell_type=gru --optimizer=adam --learning_rate=1E-4 --real_batch_size=1 --virtual_batch_size=1 --bias_data_frame_offset=0 --bias_split_gap=0.05 --output_dir=plot_200_samples --experiment_name=attn_tts_ljspeech_119 --custom_conditioning_json=gentle_recognition_info.json --override_dataset_path=/usr/local/data/kkastner/mini_ljspeech_cleaned /home/kkastner/_kkpthlib_models/attention_melnet_cmdline_ljspeech_01-32-52_2022-21-02_af1c26_attn_tts_ljspeech_127_finetune_tier_0_0_sz_112_32/saved_models/checkpoint_model-109200.pth

#! git pull

ls /home/kkastner/_kkpthlib_models/

! $full_sample_cmd

"""# Attention

"""

ls generated_samples_*/sampled_forced_images

from IPython.display import display, Image
used_seeds = sample_index.replace("[", "").replace("]", "").split(",")
for u in used_seeds:
    display(Image(filename='generated_samples_bias{}/sampled_forced_images/attn_0.png'.format(u)))

import numpy as np
import json

# random state used for random variance on edge cases
lcl_var_random_state = np.random.RandomState(11789)
seed_speeds = []
#used_seeds = sample_index.replace("[", "").replace("]", "").split(",")
for u in used_seeds:
    bias_path = "/root/kkpthlib/examples/attention_melnet_cmdline_ljspeech_repr_mix/generated_samples_bias{}/sampled_forced_images/bias_information.txt".format(u)
    attention_path = "/root/kkpthlib/examples/attention_melnet_cmdline_ljspeech_repr_mix/generated_samples_bias{}/sampled_forced_images/attention_termination_x0.txt".format(u)
    text_path = "/root/kkpthlib/examples/attention_melnet_cmdline_ljspeech_repr_mix/generated_samples_bias{}/sampled_forced_images/text_info.json".format(u)
    attention_npy_path = "/root/kkpthlib/examples/attention_melnet_cmdline_ljspeech_repr_mix/generated_samples_bias{}/sampled_forced_images/attn_activation.npy".format(u)

    with open(bias_path, "r") as f:
        l = f.readlines()
        start_frame = int(l[1].strip().split(":")[1])
    with open(attention_path, "r") as f:
       l = f.readlines()
       end_frame = int(l[1].strip().split(":")[1])
    with open(text_path, "r") as f:
        text_json = json.load(f) 

    attn_npy_data = np.load(attention_npy_path)[:, 0, 0, :]
   
    diff = end_frame - start_frame
    # if for some reason the detected end frame is before the start, it is bad
    if diff < 0:
        diff = np.inf
    # hard coded end for now (112 is the total time len) - means the attention didn't finish
    # possibly that it finished AT the last step, still no good
    # dont trust this , use percentile instead
    #if end_frame == 111:
    #    diff = np.inf
    horiz_breaks = np.where(np.sum(attn_npy_data > np.percentile(attn_npy_data, 95), axis=0) == 0)[0]
    if len(horiz_breaks) > 0:
        diff = np.inf

    # there should be at least 1 vertical step with nothing
    vert_breaks = np.where(np.sum(attn_npy_data > np.percentile(attn_npy_data, 95), axis=1) == 0)[0]
    if len(vert_breaks) < 1:
        diff = np.inf
    
    # check the "energy" of the attention (95th+ percentile) goes through the entire sequence
    # no gaps in time (attention usually breaks apart in failure cases, gives values < 95 percentile)
    #plt.imshow(attn_npy_data > np.percentile(attn_npy_data, 95), cmap="viridis")
    breaks = np.where(np.sum(attn_npy_data > np.percentile(attn_npy_data, 90), axis=1) == 0)[0] 
    if len(breaks) > 0:
        # see if any breaks occur within our phrase
        if np.any((breaks >= start_frame) & (breaks <= end_frame)):
            diff = np.inf

    ljkey = [k for k in text_json.keys() if "LJ" in k][0]
    for w in text_json[ljkey]["full_alignment"]["words"]:
        # check that all words are within the 4 second window
        # otherwise the attention might seem to work well but cut off
        # since the conditioning was cut off
        if w["end"] >= 4:
            diff = np.inf

    if diff != np.inf:
        thresh_attn = attn_npy_data > np.percentile(attn_npy_data, 90)
        vv = np.where(thresh_attn)
        s_i = np.where(vv[0] == start_frame)[0]
        e_i = np.where(vv[0] == end_frame)[0]
        if len(s_i) < 1 or len(e_i) < 1:
            rank = np.inf
        else:
            avg_slope = (np.mean(e_i) - np.mean(s_i)) / (end_frame - start_frame)
            attn_avg = []
            for _ii in range(start_frame, end_frame):
                el_i = np.where(vv[0] == _ii)[0]
                if len(el_i) > 0:
                    lcl_mean = np.mean(vv[1][el_i])
                    attn_avg.append(lcl_mean)
                else:
                    rank = np.inf
                    break
            attn_avg = np.array(attn_avg)
            grad_avg_attn_diff = (np.abs(attn_avg[1:] - attn_avg[:-1]) - avg_slope) ** 2
            # var or lowest high-percentile value...
            rank = np.percentile(grad_avg_attn_diff, 80)
            #rank = np.max(grad_avg_attn_diff)
    else:
        rank = np.inf
 
    seed_speeds.append((u, attn_npy_data, start_frame, end_frame, rank, diff))
print("All seed info")
# sort by rank
print([(el[0], el[2], el[3], el[4], el[5]) for el in seed_speeds])

# be sure to remove filtered generations
filtered = [s for s in seed_speeds if (s[-1] < np.inf) and (s[-2] < np.inf)]
def metric_fn(x):
    # the longer the output relative to the prime, the lower the overall score
    #v1 = float(x[2]) / (np.abs(x[3] - x[2]))
    # the farther this output length is from the median over all seeds
    med_len = np.median([np.abs(t[3] - t[2]) for t in filtered])
    start_frame = x[2]
    end_frame = x[3]
    # do the rate INCLUDING the prime?
    e_i = x[1][end_frame]
    s_i = x[1][start_frame]
    avg_rate = (np.argmax(e_i) - np.argmax(s_i)) / float((end_frame - start_frame))
    # instead of avg rate compare to grad ahead and grad behind, todo
    per_step_rate_gap = [(np.argmax(x[1][_ii]) - np.argmax(x[1][_ii - 1])) - avg_rate
                         for _ii in range(start_frame, end_frame)]
    v1 = np.sum(np.abs(per_step_rate_gap))# - np.min(per_step_rate_gap))
    # nearness to the median length for all examples
    #v2 = (1. + np.abs(np.abs(x[3] - x[2]) - med_len))
    # gap between max and median, lower gap reduces overall score
    #v3 = (1. + np.max(np.max(x[1], axis=1)[x[2]:x[3]]) - np.median(np.max(x[1], axis=1)[x[2]:x[3]]))
    # gap between max and min, lower gap reduces overall score
    #v4 = (1. + np.abs(np.max(np.max(x[1], axis=1)[x[2]:x[3]]) - np.min(np.max(x[1], axis=1)[x[2]:x[3]])))
    # min floor scale, higher min will reduce overall score
    #v5 = 1. / (1 + np.min(np.max(x[1], axis=1)[x[2]:x[3]]))
    #v4 = 1. / (1 + np.min(x[1][x[2]:x[3]]))
    # final score, lower is better
    return v1 #* v5 #* v2 #* v3 * v4 * v5
  
# sort by metric function
sorted_final = sorted(filtered, key=metric_fn) 
best_seed_info = sorted_final[0]

print("Top seeds sorted by attention metric")
print([(el[0], el[2], el[3], el[4], el[5]) for el in sorted_final])

"""
rank_sorted = sorted(seed_speeds, key=lambda x: x[4])[:7]

# sort by rate to get top k (up to 4 as long as they arent invalid (np.inf in rank or diff))
topk = [r for r in rank_sorted if (r[-1] < np.inf) and (r[-2] < np.inf)]
topk_sorted = sorted(topk, key=lambda x: x[5])[:6]

# sort by attention variance
topk2 = topk_sorted
topk2_sorted = sorted(topk2, key=lambda x: np.var(np.max(x[1], axis=1)[x[2]:x[3]]))
topk2_sorted = topk2_sorted[:5]

# then final sort by attention "concentration"
# lower values will be more concentrated
topk3_sorted = sorted(topk2_sorted, key=lambda x: (x[1] > np.percentile(x[1], 90))[x[2]:x[3]].sum())
topk3_sorted = topk3_sorted[:3]

# sort by average max value of attention
sorted_final = sorted(topk3_sorted, key=lambda x: np.abs(np.max(np.max(x[1], axis=1)[x[2]:x[3]]) - np.min(np.max(x[1], axis=1)[x[2]:x[3]])))
best_seed_info = sorted_final[0]

print("Top seeds sorted by attention mean")
print([(el[0], el[2], el[3], el[4], el[5]) for el in sorted_final])
"""

"""
# finally, sort again to get final result
topk_sorted_by_rank = sorted(topk3_sorted, key=lambda x: x[4])
topk_sorted_by_rate = sorted(topk3_sorted, key=lambda x: x[5])
topk_sorted_by_prime = sorted(topk3_sorted, key=lambda x: x[2])[::-1]
# prefer primes close to avg...
#if len(topk_sorted_by_prime) > 2:
#    topk_sorted_by_prime = [topk_sorted_by_prime[1], topk_sorted_by_prime[0], topk_sorted_by_prime[2]]

print("Top seeds sorted by rank")
print([(el[0], el[2], el[3], el[4], el[5]) for el in topk_sorted_by_rank])
print("Top seeds sorted by rate")
print([(el[0], el[2], el[3], el[4], el[5]) for el in topk_sorted_by_rate])
print("Top seeds sorted by closeness to avg prime")
print([(el[0], el[2], el[3], el[4], el[5]) for el in topk_sorted_by_prime])

if len(topk_sorted_by_rank) > 1:
    # we want the method that does the best on all metrics...
    # tie break on rank
    combined_scores = []
    for n1, k1 in enumerate(topk_sorted_by_rank):
        for n2, k2 in enumerate(topk_sorted_by_rate):
            for n3, k3 in enumerate(topk_sorted_by_prime):
                if k1 == k2 and k1 == k3:
                    combined_scores.append((n1 + 1) ** 2 + (n2 + 1) ** 2 + (n3 + 1) ** 2 + .001 * k1[5])
                    break
    print("Final scores")
    print([(tup[0], [e for _nn, e in enumerate(tup[1]) if _nn != 1]) for tup in zip(combined_scores, topk_sorted_by_rank)])
    final = [(_n, tup[0], tup[1]) for _n, tup in enumerate(zip(combined_scores, topk_sorted_by_rank))]
    sorted_final = sorted(final, key=lambda x: x[1])
    best_seed_info = sorted_final[0][2]
else:
    best_seed_info = topk_sorted_by_rank[0]
""";

print("Autoselected best seed {}, start frame {}, end frame {}".format(best_seed_info[0], best_seed_info[2], best_seed_info[3]))
best_seed = int(best_seed_info[0])

# can override best seed autoselected for manual listening
#best_seed = 2

"""# Post-Net

"""

# example plots
'''
thresh_attn = attn_npy_data > np.percentile(attn_npy_data, 95)
vv = np.where(thresh_attn)
attn_avg = []
for _ii in range(thresh_attn.shape[0]):
    el_i = np.where(vv[0] == _ii)[0]
    if len(el_i) > 0:
        lcl_mean = np.mean(vv[1][el_i])
    else:
        lcl_mean = 0
    attn_avg.append(lcl_mean)
attn_avg = np.array(attn_avg)
grad_avg_attn = (attn_avg[1:] - attn_avg[:-1]) ** 2
rank = np.var(grad_avg_attn)
plt.imshow(thresh_attn, cmap="viridis")
plt.figure()
plt.plot(attn_avg)
plt.figure()
plt.plot(grad_avg_attn)
print(rank)
''';

cd ~/stexp/wavernn_ljspeech_downsample_4_0_0_alt/

#! git pull

ls

if not os.path.exists("eval"):
    os.mkdir("eval")

full_wavernn_sample_cmd = "CUDA_VISIBLE_DEVICES='' python reconstruct_npy.py"
full_wavernn_sample_cmd += " /root/kkpthlib/examples/attention_melnet_cmdline_ljspeech_repr_mix/generated_samples_bias{}/sampled_forced_images/unnormalized_samples.npy".format(best_seed)
full_wavernn_sample_cmd += " --checkpoint=/root/stexp/wavernn_ljspeech_downsample_4_0_0_alt/checkpoint_step000450000.pth"
full_wavernn_sample_cmd += " --bias_information=/root/kkpthlib/examples/attention_melnet_cmdline_ljspeech_repr_mix/generated_samples_bias{}/sampled_forced_images/bias_information.txt".format(best_seed)
full_wavernn_sample_cmd += " --bias_data_frame_offset=0"
full_wavernn_sample_cmd += " --bias_data_frame_offset_right=0"
full_wavernn_sample_cmd += " --fixed_pad_left=0"
full_wavernn_sample_cmd += " --fixed_pad_right=0"
full_wavernn_sample_cmd += " --attention_information=/root/kkpthlib/examples/attention_melnet_cmdline_ljspeech_repr_mix/generated_samples_bias{}/sampled_forced_images/attention_termination_x0.txt".format(best_seed)

! $full_wavernn_sample_cmd

"""# Raw, Full & Cleaned"""

fs, d = wavfile.read("eval/eval_checkpoint_step000450000_wav_0.wav")
synth_ts = d.astype('float32') / (2 ** 15)
synth_ts -= synth_ts.mean()
Audio(data=soundsc(synth_ts), rate=22050)

fs, d = wavfile.read("eval/eval_checkpoint_step000450000_wav_0_full.wav")
synth_ts = d.astype('float32') / (2 ** 15)
synth_ts -= synth_ts.mean()
Audio(data=soundsc(synth_ts), rate=22050)

import noisereduce as nr 
import shutil
#load data 
if not os.path.exists("noisy_example.wav"):
    shutil.copy2("/content/drive/MyDrive/melnet_sampling_resources/noisy_example.wav", "noisy_example.wav")
noise_rate, noise_data = wavfile.read("noisy_example.wav") 
# select section of data that is noise 
noise_part = noise_data[-18000:-5000] 

rate, data = wavfile.read("eval/eval_checkpoint_step000450000_wav_0.wav") 
# perform noise reduction 
cleaned_data = nr.reduce_noise(y=data, y_noise=noise_part, sr=rate) 
wavfile.write("cleaned_data.wav", rate, cleaned_data)
fs, d = wavfile.read("cleaned_data.wav")
Audio(data=d, rate=22050)

ls

# take top 3 audio files, run through recognizer and selected best?

# plug in "quality improvement" wavernn

cd ~/stexp/wavernn_ljspeech_fullres_cleanup

upres_wavernn_sample_cmd = "CUDA_VISIBLE_DEVICES='' python reconstruct_wav.py"
upres_wavernn_sample_cmd += " /root/stexp/wavernn_ljspeech_downsample_4_0_0_alt/cleaned_data.wav"
upres_wavernn_sample_cmd += " --checkpoint=/root/stexp/wavernn_ljspeech_fullres_cleanup/checkpoint_step000520000.pth"

#upres_wavernn_sample_cmd = "CUDA_VISIBLE_DEVICES='' python reconstruct_wav.py"
#upres_wavernn_sample_cmd += " /root/stexp/wavernn_ljspeech_downsample_4_0_0_alt/eval/eval_checkpoint_step000450000_wav_0.wav"
#upres_wavernn_sample_cmd += " --checkpoint=/root/stexp/wavernn_ljspeech_fullres_cleanup/checkpoint_step000520000.pth"

! $upres_wavernn_sample_cmd

"""# Upres"""

fs, d = wavfile.read("eval/eval_checkpoint_step000520000_wav_0_full.wav")
Audio(data=d, rate=22050)

#cd /home/kkastner/_kkpthlib_models/saved_attention_melnet_cmdline_ljspeech_repr_mix_tier_0_0_sz_112_32/saved_models/

#! git pull

import time
while True:
    time.sleep(2)

# plot the groundtruth
text_path = "/root/kkpthlib/examples/attention_melnet_cmdline_ljspeech_repr_mix/generated_samples_bias4/teacher_forced_images/text_info.json"
small_npy_path = "/root/kkpthlib/examples/attention_melnet_cmdline_ljspeech_repr_mix/generated_samples_bias4/teacher_forced_images/small_x0.npy"
data_npy_path = "/root/kkpthlib/examples/attention_melnet_cmdline_ljspeech_repr_mix/generated_samples_bias4/teacher_forced_images/data_x0.npy"

small_data = np.load(small_npy_path)
full_data = np.load(data_npy_path)
with open(text_path, "r") as f:
    text_json = json.load(f)

import pprint
pprint.pprint(text_json)

plt.imshow(full_data.T, interpolation="nearest", cmap="viridis", origin="lower")
plt.axis("off");
plt.savefig("/root/full_mel.png", transparent=True, bbpx_inches="tight", pad_inches=0)

plt.imshow(small_data.T, interpolation="nearest", cmap="viridis", origin="lower")
plt.axis("off");
plt.savefig("/root/small_mel.png", transparent=True, bbox_inches="tight", pad_inches=0)

# plot the sample and attention
plot_sample = 9
sample_text_path = "/root/kkpthlib/examples/attention_melnet_cmdline_ljspeech_repr_mix/generated_samples_bias{}/sampled_forced_images/text_info.json".format(plot_sample)
bias_path = "/root/kkpthlib/examples/attention_melnet_cmdline_ljspeech_repr_mix/generated_samples_bias{}/sampled_forced_images/bias_information.txt".format(plot_sample)
termination_path = "/root/kkpthlib/examples/attention_melnet_cmdline_ljspeech_repr_mix/generated_samples_bias{}/sampled_forced_images/attention_termination_x0.txt".format(plot_sample)
attn_npy_path = "/root/kkpthlib/examples/attention_melnet_cmdline_ljspeech_repr_mix/generated_samples_bias{}/sampled_forced_images/attn_activation.npy".format(plot_sample)
sample_npy_data_path = "/root/kkpthlib/examples/attention_melnet_cmdline_ljspeech_repr_mix/generated_samples_bias{}/sampled_forced_images/unnormalized_samples.npy".format(plot_sample)

attn_npy_data = np.load(attn_npy_path)
sample_npy_data = np.load(sample_npy_data_path)
with open(sample_text_path, "r") as f:
    sample_text_json = json.load(f) 

with open(bias_path, "r") as f:
    l = f.readlines()
    start_frame = int(l[1].strip().split(":")[1])

with open(termination_path, "r") as f:
    l = f.readlines()
    end_frame = int(l[1].strip().split(":")[1])

text_tup = [('$', 0.0), ('ih', 1.0), ('n', 1.0), (' ', 0.0), ('ah', 1.0), ('d', 1.0), ('ih', 1.0), ('sh', 1.0), ('ah', 1.0), ('n', 1.0), (' ', 0.0), ('dh', 1.0), ('ah', 1.0), (' ', 0.0), ('dh', 1.0), ('eh', 1.0), ('r', 1.0), (' ', 0.0), ('b', 1.0), ('ow', 1.0), ('t', 1.0), (' ', 0.0), ('s', 1.0), ('ae', 1.0), ('ng', 1.0), ('k', 1.0), (' ', 0.0), ('ih', 1.0), ('n', 1.0), ('t', 1.0), ('uw', 1.0), (' ', 0.0), ('dh', 1.0), ('iy', 1.0), (' ', 0.0), ('ay', 1.0), ('s', 1.0), ('iy', 1.0), (' ', 0.0), ('r', 1.0), ('ih', 1.0), ('v', 1.0), ('er', 1.0), ('~', 0.0)]
#text_tup = [('$', 0.0), ('hh', 1.0), ('uw', 1.0), ('z', 1.0), (' ', 0.0), ('l', 1.0), ('ay', 1.0), ('f', 1.0), (' ', 0.0), ('dh', 1.0), ('eh', 1.0), ('r', 1.0), (' ', 0.0), ('b', 1.0), ('ow', 1.0), ('t', 1.0), (' ', 0.0), ('s', 1.0), ('ae', 1.0), ('ng', 1.0), ('k', 1.0), (' ', 0.0), ('ih', 1.0), ('n', 1.0), ('t', 1.0), ('uw', 1.0), (' ', 0.0), ('dh', 1.0), ('iy', 1.0), (' ', 0.0), ('ay', 1.0), ('s', 1.0), ('iy', 1.0), (' ', 0.0), ('r', 1.0), ('ih', 1.0), ('v', 1.0), ('er', 1.0), ('~', 0.0)]
#text_tup = [('$', 0.0), ('ae', 1.0), ('n', 1.0), ('d', 1.0), (' ', 0.0), ('dh', 1.0), ('iy', 1.0), (' ', 0.0), ('ae', 1.0), ('g', 1.0), ('r', 1.0), ('ah', 1.0), ('g', 1.0), ('ih', 1.0), ('t', 1.0), (' ', 0.0), ('ah', 1.0), ('m', 1.0), ('aw', 1.0), ('n', 1.0), ('t', 1.0), (' ', 0.0), ('dh', 1.0), ('eh', 1.0), ('r', 1.0), (' ', 0.0), ('b', 1.0), ('ow', 1.0), ('t', 1.0), (' ', 0.0), ('s', 1.0), ('ae', 1.0), ('ng', 1.0), ('k', 1.0), (' ', 0.0), ('ih', 1.0), ('n', 1.0), ('t', 1.0), ('uw', 1.0), (' ', 0.0), ('dh', 1.0), ('iy', 1.0), (' ', 0.0), ('ay', 1.0), ('s', 1.0), ('iy', 1.0), (' ', 0.0), ('r', 1.0), ('ih', 1.0), ('v', 1.0), ('er', 1.0), ('~', 0.0)]
#text_tup = [('$', 0.0), ('w', 1.0), ('ay', 1.0), ('l', 1.0), ('s', 1.0), ('t', 1.0), (' ', 0.0), ('l', 1.0), ('ah', 1.0), ('d', 1.0), ('g', 1.0), ('ey', 1.0), ('t', 1.0), ('eh', 1.0), (' ', 0.0), ('dh', 1.0), ('eh', 1.0), ('r', 1.0), (' ', 0.0), ('b', 1.0), ('ow', 1.0), ('t', 1.0), (' ', 0.0), ('s', 1.0), ('ae', 1.0), ('ng', 1.0), ('k', 1.0), (' ', 0.0), ('ih', 1.0), ('n', 1.0), ('t', 1.0), ('uw', 1.0), (' ', 0.0), ('dh', 1.0), ('iy', 1.0), (' ', 0.0), ('ay', 1.0), ('s', 1.0), ('iy', 1.0), (' ', 0.0), ('r', 1.0), ('ih', 1.0), ('v', 1.0), ('er', 1.0), ('~', 0.0)]

print(len(text_tup))
print(attn_npy_data[:, 0, 0, :].shape)
print(text_tup)
print(text_tup[15:])

print(start_frame)
print(end_frame)

start_text_frame = 15
attn_crop = attn_npy_data[start_frame:end_frame, 0, 0, start_text_frame:].astype("float32")

plt.imshow(attn_crop.T, interpolation="nearest", cmap="viridis", origin="upper")
tok = [t[0] for t in text_tup][start_text_frame:]
print(tok)
ax = plt.gca()
plt.yticks(np.arange(len(tok)))
ax.set_yticklabels(tok)
ax.xaxis.set_ticks_position('none') 
ax.yaxis.set_ticks_position('none')
ax.xaxis.set_ticks([])
plt.savefig("/root/attn_{}.png".format(plot_sample), transparent=True, bbox_inches="tight", pad_inches=0) 
#plt.axis("off");

plt.imshow(sample_npy_data[0, start_frame:end_frame, :, 0].T,
           origin="lower", cmap="viridis", interpolation="nearest")
ax = plt.gca()
ax.xaxis.set_ticks_position('none') 
ax.yaxis.set_ticks_position('none')
ax.xaxis.set_ticks([])
ax.yaxis.set_ticks([])
plt.savefig("/root/sample_{}.png".format(plot_sample), transparent=True, bbox_inches="tight", pad_inches=0)

sampled_full_mel_path = "/root/stexp/wavernn_ljspeech_fullres_cleanup/eval/mel.npy"
sampled_full_mel_data = np.load(sampled_full_mel_path)

sampled_full_mel_data.shape

plt.imshow(sampled_full_mel_data,
           origin="lower", cmap="viridis", interpolation="nearest")
ax = plt.gca()
ax.xaxis.set_ticks_position('none') 
ax.yaxis.set_ticks_position('none')
ax.xaxis.set_ticks([])
ax.yaxis.set_ticks([])
plt.savefig("/root/sample_full_{}.png".format(plot_sample), transparent=True, bbox_inches="tight", pad_inches=0)

#! cp -pr ~/kkpthlib/examples/attention_melnet_cmdline/log.log /content/drive/MyDrive/melnet_sampling_outputs/
#! cp -pr ~/kkpthlib/examples/attention_melnet_cmdline/tier* /content/drive/MyDrive/melnet_sampling_outputs/
#! cp -pr ~/kkpthlib/examples/attention_melnet_cmdline/combined_unnormalized_samples.npy /content/drive/MyDrive/melnet_sampling_outputs/
#! cp -pr eval/ /content/drive/MyDrive/melnet_sampling_outputs/