import os
import numpy as np
import json
import time
import copy
from scipy import signal
from scipy.io import wavfile
from collections import OrderedDict
from .frontends import EnglishPhonemeLookup
from .frontends import EnglishASCIILookup
from .audio_processing.audio_tools import herz_to_mel, mel_to_herz
from .audio_processing.audio_tools import stft, istft

import os
import pwd

def get_username():
    return pwd.getpwuid(os.getuid())[0]

class EnglishSpeechCorpus(object):
    """
    Frontend processing inspired by r9y9 (Ryuichi Yamamoto) DeepVoice3 implementation
    https://github.com/r9y9/deepvoice3_pytorch

    which in turn variously is inspired by librosa filter implementations, Tacotron implementation by Keith Ito
    some modifications from more recent TTS work
    """
    def __init__(self, metadata_csv, wav_folder, alignment_folder=None, remove_misaligned=True, cut_on_alignment=True,
                       train_split=0.9,
                       min_length_words=3, max_length_words=100,
                       min_length_symbols=7, max_length_symbols=200,
                       min_length_time_secs=2, max_length_time_secs=None,
                       extract_subsequences=True,
                       symbol_type="phoneme",
                       fixed_minibatch_time_secs=6,
                       build_skiplist=True,
                       bypass_checks=False,
                       combine_all_into_valid=False,
                       sample_rate=22050,
                       random_state=None):
        self.metadata_csv = metadata_csv
        self.wav_folder = wav_folder
        self.alignment_folder = alignment_folder
        self.random_state = random_state
        self.train_split = train_split

        self.symbol_type = symbol_type

        self.min_length_words = min_length_words
        self.max_length_words = max_length_words
        self.min_length_symbols = min_length_symbols
        self.max_length_symbols = max_length_symbols
        self.min_length_time_secs = min_length_time_secs
        if max_length_time_secs is None:
            max_length_time_secs = fixed_minibatch_time_secs
        self.max_length_time_secs = max_length_time_secs
        self.extract_subsequences = extract_subsequences

        self.bypass_checks = bypass_checks
        if bypass_checks == True:
            raise ValueError("Unable to bypass safety checks at this time - do cleaning at the dataset level")
        self.build_skiplist = build_skiplist
        self.combine_all_into_valid = combine_all_into_valid

        if get_username() == "root":
            print("WARNING: detected colab environment (due to username 'root'), using mini_robovoice settings to sample!\nThese settings will use all data in {} for both train and valid sets!".format(metadata_csv))
            self.bypass_checks = True
            self.combine_all_info_valid = True
            self.build_skiplist = False

        self.cached_mean_vec_ = None
        self.cached_std_vec_ = None
        self.cached_count_ = None

        self.sample_rate = sample_rate

        # initial number of mels to design with, will be further reduced later
        self.n_mels = 256

        self.cut_on_alignment = cut_on_alignment
        self.remove_misaligned = remove_misaligned

        self.mel_freq_min = 125
        self.mel_freq_max = 7600

        # increased overlap to compensate for downsampling in freq
        #self.stft_size = 6 * 256 #2 * 960 #6 * 256
        #self.stft_step = 256

        self.stft_size = 6 * 200 #2 * 960 #6 * 256
        self.stft_step = 200

        self._split_gap_time_s = 0.0

        # preemphasis filter
        self.preemphasis_coef = 0.97
        self.ref_level_db = 20
        self.min_level_db = -90

        info = {}
        with open(metadata_csv, encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                info[parts[0]] = {}
                info[parts[0]]["metadata_transcript"] = parts[2]

        self.misaligned_keys = []
        self.aligned_keys = []
        start_time = time.time()
        if alignment_folder is not None:
            for k in sorted(info.keys()):
                alignment_info_json = alignment_folder + k + ".json"
                if self.bypass_checks:
                    try:
                        with open(alignment_info_json, "r") as read_file:
                            alignment = json.load(read_file)
                        skip_example = False
                        for el in alignment["words"]:
                            if el["case"] != "success":
                                #print("skipping {} due to unaligned word".format(k))
                                skip_example = True
                        if skip_example == True:
                            self.misaligned_keys.append(k)
                        else:
                            self.aligned_keys.append(k)
                    except:
                        continue
                else:
                    with open(alignment_info_json, "r") as read_file:
                        alignment = json.load(read_file)
                    skip_example = False
                    for el in alignment["words"]:
                        if el["case"] != "success":
                            #print("skipping {} due to unaligned word".format(k))
                            skip_example = True
                    if skip_example == True:
                        self.misaligned_keys.append(k)
                    else:
                        self.aligned_keys.append(k)

        if self.combine_all_into_valid:
            shuf_keys = self.aligned_keys
            self.train_keys = self.aligned_keys
            self.valid_keys = self.aligned_keys
        else:
            shuf_keys = copy.deepcopy(self.aligned_keys)
            random_state.shuffle(shuf_keys)
            splt = int(self.train_split * len(shuf_keys))
            self.train_keys = shuf_keys[:splt]
            self.valid_keys = shuf_keys[splt:]

        self._batch_utts_queue = []
        self._batch_used_keys_queue = []

        """
        # code used to pre-calculate information about pauses and gaps
        start_time = time.time()
        all_gaps = []
        for _n, k in enumerate(sorted(info.keys())):
            print("{} of {}".format(_n, len(info.keys())))
            if k in self.misaligned_keys:
                continue
            fs, d, this_melspec, this_info = self._fetch_utterance(k, skip_mel=True)
            # start stop boundaries
            start_stop = [(el["start"], el["end"]) for el in this_info[k]["full_alignment"]["words"]]
            gaps = []
            last_end = 0
            for _s in start_stop:
                gaps.append(_s[0] - last_end)
                last_end = _s[1]
            final_gap = (len(d) / float(fs)) - last_end
            gaps.append(final_gap)
            all_gaps.extend(gaps)
        stop_time = time.time()
        print("total iteration time")
        print(stop_time - start_time)
        gap_arr = np.array(all_gaps)
        # assume any negatives are failed alignment - just give min pause duration aka basically 0
        gap_arr[gap_arr < 0] = 0.
        # look at histograms to pick values...
        # A Large Scale Study of Multi-lingual Pause Duration
        # longest category only occuring in spontaneous speech
        # we split the smaller chunk into 0.01, 0.05, 0.125, .5 as approximate "halving" of remaining values. With the .5 value adding a separate
        # category for "long tail" silence that may be difficult to model with attention
        # https://web.archive.org/web/20131114035947/http://aune.lpl.univ-aix.fr/projects/aix02/sp2002/pdf/campione-veronis.pdf
        import matplotlib.pyplot as plt
        n, bins, patches = plt.hist(gap_arr, 100, density=True)
        plt.savefig("tmp1.png")
        plt.close()

        n, bins, patches = plt.hist(gap_arr[gap_arr > 0.01], 100, density=True)
        plt.savefig("tmp2.png")
        plt.close()

        n, bins, patches = plt.hist(gap_arr[gap_arr > 0.0625], 100, density=True)
        plt.savefig("tmp3.png")
        plt.close()

        n, bins, patches = plt.hist(gap_arr[gap_arr > 0.1325], 100, density=True)
        plt.savefig("tmp4.png")
        plt.close()

        n, bins, patches = plt.hist(gap_arr[gap_arr > 0.25], 100, density=True)
        plt.savefig("tmp5.png")
        plt.close()
        """
        self.pause_duration_breakpoints = [0.01, 0.0625, 0.1325, 0.25]

        self.phone_lookup = EnglishPhonemeLookup()
        # each sil starts with ! , so !0, !1, !2, !3, !4

        sil_val = len(self.phone_lookup.keys())
        # +1 because 0:0.01, 0.01:0.0625, 0.0625:0.1325, 0.1325:0.25, 0.25:inf
        for _i in range(len(self.pause_duration_breakpoints) + 1):
            self.phone_lookup["!{}".format(_i)] = sil_val
            sil_val += 1

        # add start symbol
        self.phone_lookup["$"] = sil_val
        sil_val += 1
        # add continuation symbol
        self.phone_lookup["&"] = sil_val
        sil_val += 1
        # add eos symbol
        self.phone_lookup["~"] = sil_val
        sil_val += 1
        # add pad symbol
        self.phone_lookup["_"] = sil_val
        assert len(self.phone_lookup.keys()) == len(np.unique(list(self.phone_lookup.keys())))

        self.ascii_lookup = EnglishASCIILookup()

        sil_val = len(self.ascii_lookup.keys())
        self.ascii_lookup[" "] = sil_val
        sil_val += 1

        # we overload long pauses into 2 groups? potentially

        special = [s for s in "!\',-.:;?"]
        for s in special:
            self.ascii_lookup[s] = sil_val
            sil_val += 1

        # add start symbol
        self.ascii_lookup["$"] = sil_val
        sil_val += 1
        # add continuation symbol
        self.ascii_lookup["&"] = sil_val
        sil_val += 1
        # add eos symbol
        self.ascii_lookup["~"] = sil_val
        sil_val += 1
        # add pad symbol
        self.ascii_lookup["_"] = sil_val
        assert len(self.ascii_lookup.keys()) == len(np.unique(list(self.ascii_lookup.keys())))

        if self.build_skiplist:
            # should be deterministic if we run the same script 2x
            random_state_val = random_state.randint(10000)
            skiplist_base_dir = os.getcwd() + "/skiplist_cache/"
            skiplist_base_path = skiplist_base_dir + "{}".format(metadata_csv.split(".csv")[0].replace("/", "-"))
            skiplist_train_path = skiplist_base_path + "_keyval{}_train_skip.txt".format(random_state_val)
            keeplist_train_path = skiplist_base_path + "_keyval{}_train_keep.txt".format(random_state_val)

            skiplist_valid_path = skiplist_base_path + "_keyval{}_valid_skip.txt".format(random_state_val)
            keeplist_valid_path = skiplist_base_path + "_keyval{}_valid_keep.txt".format(random_state_val)

            if not os.path.exists(skiplist_base_dir):
                os.mkdir(skiplist_base_dir)

            if not all([os.path.exists(p) for p in [skiplist_train_path, keeplist_train_path, skiplist_valid_path, keeplist_valid_path]]):
                # info for skip / keep lists is missing, must create it
                checks = ["noise", "sil", "oov", "#", "laughter", "<eps>"]

                train_failed_checks = OrderedDict()
                train_passed_checks = OrderedDict()
                for n, k in enumerate(self.train_keys):
                    try:
                        utt = self.get_utterances(1, [self.train_keys[n]], do_not_filter=True)
                    except:
                        train_failed_checks[k] = [(n, k, 0)]
                        continue
                    utt_key = list(utt[0][3].keys())[0]
                    words = utt[0][3][utt_key]["full_alignment"]["words"]
                    for i in range(len(words)):
                        for j in range(len(words[i]["phones"])):
                            p = words[i]["phones"][j]["phone"]
                            for c in checks:
                                if c in p:
                                    if k not in train_failed_checks:
                                        train_failed_checks[k] = [(n, k, c)]
                                    else:
                                        train_failed_checks[k].append((n, k, c))
                    if k not in train_failed_checks:
                        train_passed_checks[k] = True
                    print("Building skiplist for train" + "," + str(n) + "," + k + " :::  " + utt[0][3][utt_key]["transcript"])
                # be sure there havent somehow been elements put into both lists
                for tk in train_passed_checks.keys():
                    assert tk not in train_failed_checks

                valid_failed_checks = OrderedDict()
                valid_passed_checks = OrderedDict()
                for n, k in enumerate(self.valid_keys):
                    try:
                        utt = self.get_utterances(1, [self.valid_keys[n]], do_not_filter=True)
                    except:
                        valid_failed_checks[k] = [(n, k, 0)]
                        continue
                    utt_key = list(utt[0][3].keys())[0]
                    words = utt[0][3][utt_key]["full_alignment"]["words"]
                    for i in range(len(words)):
                        for j in range(len(words[i]["phones"])):
                            p = words[i]["phones"][j]["phone"]
                            for c in checks:
                                if c in p:
                                    if k not in valid_failed_checks:
                                        valid_failed_checks[k] = [(n, k, c)]
                                    else:
                                        valid_failed_checks[k].append((n, k, c))
                    if k not in valid_failed_checks:
                        valid_passed_checks[k] = True
                    print("Building skiplist for valid" + "," + str(n) + "," + k + " :::  " + utt[0][3][utt_key]["transcript"])

                for vk in valid_passed_checks.keys():
                    assert vk not in valid_failed_checks
                    assert vk not in train_passed_checks
                    assert vk not in train_failed_checks
                for vk in valid_failed_checks.keys():
                    assert vk not in train_passed_checks
                    assert vk not in train_failed_checks
                # be sure there havent somehow been elements put into both lists, and no train/valid cross-pollination

                # we are to the critical point, writing the info out!
                with open(skiplist_train_path, "w") as f:
                    f.write("\n".join(list(train_failed_checks)))
                    print("Wrote skiplist for {}".format(skiplist_train_path))

                with open(keeplist_train_path, "w") as f:
                    f.write("\n".join(list(train_passed_checks)))
                    print("Wrote keeplist for {}".format(keeplist_train_path))

                with open(skiplist_valid_path, "w") as f:
                    f.write("\n".join(list(valid_failed_checks)))
                    print("Wrote skiplist for {}".format(skiplist_valid_path))

                with open(keeplist_valid_path, "w") as f:
                    f.write("\n".join(list(valid_passed_checks)))
                    print("Wrote keeplist for {}".format(keeplist_valid_path))

            # now we can assume the skip/keep lists exist, lets double check the current train/val keys against the one written out
            # then prune down
            with open(keeplist_train_path, "r") as f:
                loaded_train_keep_keys = [el.strip() for el in f.readlines()]
            with open(keeplist_valid_path, "r") as f:
                loaded_valid_keep_keys = [el.strip() for el in f.readlines()]
            self.train_keep_keys = [k for k in self.train_keys if k in loaded_train_keep_keys]
            self.valid_keep_keys = [k for k in self.valid_keys if k in loaded_valid_keep_keys]
        else:
            self.train_keep_keys = [k for k in self.train_keys]
            self.valid_keep_keys = [k for k in self.valid_keys]

        # sanity check we didnt delete the whole dataset
        assert len(self.train_keep_keys) > (len(self.train_keys) // 10)
        assert len(self.valid_keep_keys) > (len(self.valid_keys) // 10)
        # sanity check train keys wasnt so short we got 0
        assert len(self.train_keep_keys) > 0
        assert len(self.valid_keep_keys) > 0

        # metadata key value
        self.metadata_exact_lookup = {}
        with open(self.metadata_csv, "r", encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                text_tup = (parts[1], parts[2])
                assert parts[0] not in self.metadata_exact_lookup
                self.metadata_exact_lookup[parts[0]] = text_tup

        for t in self.train_keys:
            if t not in self.metadata_exact_lookup:
                raise ValueError("Mismatched key {} in train between metadata file and transcript files".format(t))
        for v in self.valid_keys:
            if v not in self.metadata_exact_lookup:
                raise ValueError("Mismatched key {} in valid between metadata file and transcript files".format(v))


    def get_utterances(self, size, all_keys, skip_mel=False,
                       fastforward_state=False,
                       min_length_words=None, max_length_words=None,
                       min_length_symbols=None, max_length_symbols=None,
                       min_length_time_secs=None, max_length_time_secs=None,
                       extract_subsequences=None,
                       debug_print_filtered=False,
                       allow_zero_length_utt=False,
                       do_not_filter=False):
        # allow_zero_length_utt only used for debug
        if min_length_words is None:
            min_length_words = self.min_length_words
        if max_length_words is None:
            max_length_words = self.max_length_words
        if min_length_symbols is None:
            min_length_symbols = self.min_length_symbols
        if max_length_symbols is None:
            max_length_symbols = self.max_length_symbols
        if min_length_time_secs is None:
            min_length_time_secs = self.min_length_time_secs
        if max_length_time_secs is None:
            max_length_time_secs = self.max_length_time_secs
        if extract_subsequences is None:
            extract_subsequences = self.extract_subsequences

        utts = []
        used_keys = []
        # get a bigger extent, so if some don't match out filters we can keep going
        idx = self.random_state.choice(len(all_keys), 100 * size)
        if fastforward_state:
            # the internal logic of fetch utterance can call the shuffle up to 4 or 5 times
            # call it several times so that once we are done fastforwarding, the state is *definitely* in a new place
            aa = [1, 2, 3]
            for i in idx:
                self.random_state.shuffle(aa)
                self.random_state.shuffle(aa)
                self.random_state.shuffle(aa)
                self.random_state.shuffle(aa)
                self.random_state.shuffle(aa)
            return []
        # don't re-iterate duplicates, and don't sort them (so it stays in random order defined by random choice)
        u_idx = idx[np.sort(np.unique(idx, return_index=True)[1])]
        for this_idx in u_idx:
            utt = self._fetch_utterance(all_keys[this_idx], skip_mel=skip_mel)
            utt[3]["flagged_for_subsequence"] = False

            # fs, d, melspec, info
            core_key = list(utt[-1].keys())[0]
            word_length = len(utt[-1][core_key]["transcript"].split(" "))
            # add on 4 window buffer to time length calc
            time_length = (len(utt[1]) + (4 * self.stft_size)) / float(utt[0])
            #print("{}".format(utt[-1][core_key]["transcript"]))
            #print("time_length {}".format(time_length))
            phoneme_parts = [len(el["phones"]) for el in utt[-1][core_key]["full_alignment"]["words"]]
            # add on len(phoneme_parts) - 1 to account for added spaces
            phoneme_length = sum(phoneme_parts) + len(phoneme_parts) - 1
            char_length = len(utt[-1][core_key]["transcript"])
            # just use the min for filtering, should be close in length for most cases
            symbol_length = min(char_length, phoneme_length)
            if do_not_filter:
                pass
            else:
                # categorically reject sentences which are too long
                if word_length > max_length_words or word_length < min_length_words:
                    if debug_print_filtered:
                        print("wl reject {}".format(this_idx))
                    continue
                if symbol_length > max_length_symbols or symbol_length < min_length_symbols:
                    if debug_print_filtered:
                        print("sl reject {}".format(this_idx))
                    continue
                if time_length < min_length_time_secs:
                    if debug_print_filtered:
                        print("tl min reject {}".format(this_idx))
                    continue
                if self.extract_subsequences == False:
                    if time_length > max_length_time_secs:
                        continue
                else:
                    if time_length >= min_length_time_secs:
                        # since self.extract_subsequences was set at init, we check that there is at least 1 valid subsequence
                        # when splitting on word boundaries
                        # looking for gaps between words of a moderate length for the ideal split (>.01s)
                        this_k = list(utt[3].keys())[0]
                        aligned_words = utt[3][this_k]["full_alignment"]["words"]
                        w_0 = np.array([w["end"] for w in aligned_words[:-1]])
                        w_1 = np.array([w["start"] for w in aligned_words[1:]])
                        gaps = w_1 - w_0
                        if np.any(gaps > self._split_gap_time_s):
                            # there is at least one valid split point
                            # now we check that the split subsequence has at least one part
                            # which is a valid length subsequence aka
                            proposed_splits = list(np.where(gaps > self._split_gap_time_s)[0])
                            # sort in order of largest to smallest gap, so we prefer the largest gap if it meets conditions below
                            proposed_splits = np.argsort(gaps)[::-1][:len(proposed_splits)]
                            all_proposed_subwords = []
                            for p in proposed_splits:
                                if p == (len(aligned_words) - 1):
                                    # skip checking if it is at the very end
                                    continue
                                # +1 because we use the *end* of the entries!
                                all_proposed_subwords.append(aligned_words[:p+1])
                                all_proposed_subwords.append(aligned_words[p+1:])

                            for subwords in all_proposed_subwords:
                                subword_length = len(subwords)

                                subphoneme_length = [len(sw["phones"]) for sw in subwords]
                                subchar_length = [len(sw["alignedWord"]) for sw in subwords]

                                # add on len(phoneme_parts) - 1 to account for added spaces
                                subphoneme_length = sum(subphoneme_length) + len(subphoneme_length) + 1
                                subchar_length = sum(subchar_length) + len(subchar_length) + 1
                                # just use the min for filtering, should be close in length for most cases
                                subsymbol_length = min(subchar_length, subphoneme_length)
                                subtime_length = subwords[-1]["end"] - subwords[0]["start"]

                                if subword_length > max_length_words or subword_length < min_length_words:
                                    if debug_print_filtered:
                                        print("sw wl reject {}".format(this_idx))
                                    continue
                                if subsymbol_length > max_length_symbols or subsymbol_length < min_length_symbols:
                                    if debug_print_filtered:
                                        print("sw sl reject {}".format(this_idx))
                                    continue
                                if subtime_length < min_length_time_secs:
                                    if debug_print_filtered:
                                        print("sw tl min reject {}".format(this_idx))
                                    continue
                                utt[3]["flagged_for_subsequence"] = True
                                break
                        else:
                            if debug_print_filtered:
                                print("no splits {}".format(this_idx))
                            # no splits ignore this utt for processing
                            continue

            utts.append(utt)
            used_keys.append(all_keys[this_idx])
            if len(utts) >= size:
                break

        if len(utts) < size:
            if not allow_zero_length_utt:
                raise ValueError("Unable to build correct length in get_utterances! Something has gone very wrong, debug this!")

        self._batch_used_keys_queue.append(used_keys)
        self._batch_used_keys_queue = self._batch_used_keys_queue[-5:]

        self._batch_utts_queue.append(utts)
        self._batch_utts_queue = self._batch_utts_queue[-5:]
        return utts

    def get_train_utterances(self, size, skip_mel=False, fastforward_state=False):
        if self.build_skiplist:
            # we skip elements which had poor recognition
            return self.get_utterances(size, self.train_keep_keys, skip_mel=skip_mel, fastforward_state=fastforward_state)
        else:
            return self.get_utterances(size, self.train_keys, skip_mel=skip_mel, fastforward_state=fastforward_state)

    def get_valid_utterances(self, size, skip_mel=False, fastforward_state=False):
        if self.build_skiplist:
            # we skip elements which had poor recognition
            return self.get_utterances(size, self.valid_keep_keys, skip_mel=skip_mel, fastforward_state=fastforward_state)
        else:
            return self.get_utterances(size, self.valid_keys, skip_mel=skip_mel, fastforward_state=fastforward_state)

    def load_mean_std_from_filepath(self, filepath):
        if not os.path.exists(filepath):
            raise ValueError("Unable to find mean/std file at {}".format(filepath))
        d = np.load(filepath)
        self.cached_mean_vec_ = d["mean"].copy()
        self.cached_std_vec_ = d["std"].copy()
        self.cached_count_ = d["frame_count"].copy()

    def format_minibatch(self, utterances,
                         symbol_type=None,
                         is_sampling=False,
                         pause_duration_breakpoints=None,
                         write_out_debug_info=False,
                         quantize_to_n_bins=None):
        if symbol_type == None:
            symbol_type = self.symbol_type
        if pause_duration_breakpoints is None:
            pause_duration_breakpoints = self.pause_duration_breakpoints

        phoneme_sequences = []
        phoneme_nonspacing_sequences = []
        ascii_spacing_sequences = []
        ascii_nonspacing_sequences = []
        melspec_sequences = []

        fs, _, _, _ = utterances[0]
        overlap_len = ((self.max_length_time_secs * fs) + self.stft_size) % self.stft_size
        max_frame_count = (((self.max_length_time_secs * fs) + self.stft_size) - overlap_len) / self.stft_step
        divisors = [2, 4, 8]
        for di in divisors:
            # nearest divisble number above, works because largest divisor divides by smaller
            # we need something that has a length in time (frames) divisible by 2 4 and 8 due to the nature of melnet
            # same for frequency but frequency is a power of 2 so no need to check it
            q = int(max_frame_count / di)
            if float(max_frame_count / di) == int(max_frame_count / di):
                max_frame_count = di * q
            else:
                max_frame_count = di * (q + 1)
        assert max_frame_count == int(max_frame_count)

        if symbol_type not in ["phoneme", "representation_mixed", "ascii"]:
            raise ValueError("Unsupported symbol_type {}".format(symbol_type))

        if symbol_type in ["phoneme", "representation_mixed", "ascii"]:
            if self.alignment_folder is None:
                raise ValueError("symbol_type phoneme minibatch formatting not supported without 'aligment_folder' argument to speech corpus init!")
            for u_i, utt in enumerate(utterances):
                fs, d, melspec, al = utt
                k = list(al.keys())[0]
                words = al[k]["full_alignment"]["words"]
                # start_to_end is no crop
                crop_type = "start_to_end"

                if al["flagged_for_subsequence"] == True or melspec.shape[0] > max_frame_count:
                    # assume gap check was done already
                    w_0 = np.array([w["end"] for w in words[:-1]])
                    w_1 = np.array([w["start"] for w in words[1:]])
                    gaps = w_1 - w_0
                    # there is at least one valid split point
                    # now we check that the split subsequence has at least one part
                    # which is a valid length subsequence aka
                    proposed_splits = list(np.where(gaps > self._split_gap_time_s)[0])
                    # sort in order of largest to smallest gap, so we prefer the largest gap if it meets conditions below
                    # add 1 because the split is at the end of the word
                    proposed_splits = np.argsort(gaps)[::-1][:len(proposed_splits)] + 1

                    start_to_mid = []
                    mid_to_mid = []
                    mid_to_end = []
                    # assume we can always cut on the end of a word to make a chunk fit within "fixed_minibatch_time_secs"
                    for ii, p in enumerate(proposed_splits):
                        # here we build "fake" subsets, putting in 3 categories
                        # start:somewhere
                        # somewhere:somewhere
                        # somewhere:end
                        # want to (ideally) do a 33% chance for each type
                        # need to check a lot per each split
                        # if nothing else, we *can* split on a random word to make it fit the minibatch size / max_time_length_secs
                        all_non_self_splits = [p2 for jj, p2 in enumerate(proposed_splits) if jj != ii]
                        all_non_self_splits = [0] + all_non_self_splits + [len(words)]
                        # inclusive split since gap check was at the end
                        for jj, p2 in enumerate(all_non_self_splits):
                            le = min(p, p2)
                            re = max(p, p2)
                            tsplit = words[le:re]
                            # 4 window buffer on check (2 each side)
                            sub_dur = tsplit[-1]["end"] - tsplit[0]["start"] + (1. / fs * 4 * self.stft_size)
                            if sub_dur > self.min_length_time_secs and sub_dur < self.max_length_time_secs:
                                if le == 0:
                                    # last entry in tuple for whether it was cropped or whole
                                    start_to_mid.append((tsplit, le, re, False))
                                elif le != 0 and re != len(words):
                                    mid_to_mid.append((tsplit, le, re, False))
                                elif re == len(words):
                                    mid_to_end.append((tsplit, le, re, False))
                                else:
                                    print("fall through case in crop check should never happen, 1")
                                    from IPython import embed; embed(); raise ValueError()
                            elif sub_dur > self.max_length_time_secs:
                                # if still too long, crop it short to the nearest word to self.max_length_time_secs 
                                # if has start or mid le crop the end
                                # if end crop from start to be sure and include end
                                if le == 0 or re != len(words):
                                    # was beginning or mid chunk, crop the end of segment
                                    re_tmp = le + 1
                                    while True:
                                        tsplit_tmp = words[le:re_tmp]
                                        sub_dur_tmp = tsplit_tmp[-1]["end"] - tsplit_tmp[0]["start"] + (1. / fs * 4 * self.stft_size)
                                        # include a 4 window buffer on the max length check
                                        if (sub_dur_tmp + (1./fs * 4 * self.stft_size)) > self.max_length_time_secs:
                                            # we assume here that one word cannot put us above the max length
                                            tsplit = words[le:re_tmp - 1]
                                            if le == 0:
                                                start_to_mid.append((tsplit, le, re_tmp -1, True))
                                            else:
                                                mid_to_mid.append((tsplit, le, re_tmp -1, True))
                                            break
                                        if re_tmp >= re:
                                            print("fall through case in crop check should never happen, 2")
                                            from IPython import embed; embed(); raise ValueError()
                                            break
                                        re_tmp = re_tmp + 1
                                else:
                                    # crop the front, was an end segment
                                    le_tmp = re - 1
                                    while True:
                                        tsplit_tmp = words[le_tmp:re]
                                        sub_dur_tmp = tsplit_tmp[-1]["end"] - tsplit_tmp[0]["start"] + (1. / fs * 4 * self.stft_size)
                                        # include a 4 window buffer on the max length check
                                        if sub_dur_tmp > self.max_length_time_secs: # we assume here that one word cannot put us above the max length
                                            tsplit = words[le_tmp + 1:re]
                                            mid_to_end.append((tsplit, le_tmp, re, True))
                                            break
                                        if le_tmp <= le:
                                            print("fall through case in crop check should never happen, 3")
                                            from IPython import embed; embed(); raise ValueError()
                                            break
                                        le_tmp = le_tmp - 1
                            else:
                                # chunk was too short to use due to configured min length
                                pass
                    # choose a single crop to use from availables
                    # prefer non-cropped versions, but if a cropped one is necessary use that instead
                    selected_seq = None
                    for is_cropped in [False, True]:
                        start_to_mid_tmp = [t for t in start_to_mid if t[-1] == is_cropped]
                        mid_to_mid_tmp = [t for t in mid_to_mid if t[-1] == is_cropped]
                        mid_to_end_tmp = [t for t in mid_to_end if t[-1] == is_cropped]
                        choices = [0, 1, 2]
                        comb = [start_to_mid_tmp, mid_to_mid_tmp, mid_to_end_tmp]
                        # shuffle choices of start to mid mid to mid mid to end
                        self.random_state.shuffle(choices)
                        for c in choices:
                            if len(comb[c]) == 0:
                                pass
                            else:
                                selected = self.random_state.choice(len(comb[c]))
                                selected_seq = comb[c][selected]
                                if c == 0:
                                    crop_type = "start_to_mid"
                                elif c == 1:
                                    crop_type = "mid_to_mid"
                                elif c == 2:
                                    crop_type = "mid_to_end"
                                break
                        if selected_seq is not None:
                            break

                        if is_cropped and selected_seq is None:
                            print("ERROR IN CROP SELECTION! DEBUG THIS")
                            from IPython import embed; embed(); raise ValueError()
                    # now that we (finally) selected the subcrop, cut d to match (accounting for the fact that we already cut off some samples)!
                    # recalculate melspec
                    # set words to the part we cut
                    # replace d with sliced version, recalculate melspec, split && truncate the alignment to a subsequence which is valid
                    # according to the initial rules of the data loader
                    words = selected_seq[0]

                    # some amount may have already been trimmed, need to account for this when calculating the offset slice bounds
                    # since the original alignment times are based on a wav file with no cuts!
                    pre_offset = utt[3]["start_offset_precut_in_samples"]

                    end_in_samples = fs * words[-1]["end"]
                    end_in_samples -= pre_offset
                    end_in_samples = int(max(end_in_samples, 0))

                    start_in_samples = fs * words[0]["start"]
                    start_in_samples -= pre_offset
                    start_in_samples = int(max(start_in_samples, 0))

                    gap_samp = end_in_samples - start_in_samples + (4 * self.stft_size)
                    gap_s = gap_samp / float(fs)
                    if gap_s > self.max_length_time_secs:
                        print("ERROR FORMATTING CROP SAMPLE, DEBUG!")
                        from IPython import embed; embed(); raise ValueError()

                    # get new wav and melspec cuts
                    # should we window the wav cut?
                    # already has pre-emph filtering...
                    d2 = copy.deepcopy(d[start_in_samples:end_in_samples])
                    melspec2 = copy.deepcopy(self._melspectrogram_preprocess(d2, fs))
                    if melspec2.shape[0] > max_frame_count:
                        print("ERROR IN FRAME COUNT FOR CROP SAMPLE, DEBUG!")
                        from IPython import embed; embed(); raise ValueError()

                    del melspec
                    del d
                    melspec = melspec2
                    d = d2

                melspec_sequences.append(melspec)

                # this part makes all the phonetic sequences
                phone_groups = [el["phones"] for el in words]#al[k]["full_alignment"]["words"]]
                start_stop = [(el["start"], el["end"]) for el in words]#al[k]["full_alignment"]["words"]]
                gaps = []
                last_end = 0
                for _s in start_stop:
                    gaps.append(_s[0] - last_end)
                    last_end = _s[1]
                final_gap = (len(d) / float(fs)) - last_end
                gaps.append(final_gap)
                gaps_arr = np.array(gaps)
                v = len(pause_duration_breakpoints) - 1
                gap_idx_groups = []
                prev_pd = -np.inf
                for pd in pause_duration_breakpoints:
                    gap_idx_groups.append((gaps_arr >= prev_pd) & (gaps_arr < pd))
                    prev_pd = pd
                gap_idx_groups.append((gaps_arr >= prev_pd))
                for _n, gig in enumerate(gap_idx_groups):
                    gaps_arr[gig] = _n
                # reverse iterate pause duration breakpoints to quantized gap values
                gaps_arr = gaps_arr.astype("int32")
                phone_group_syms = [[pgi["phone"] for pgi in pg] for pg in phone_groups]

                # change flat phones and gaps based on if it was start_to_mid mid_to_mid mid_to_end or un-cropped (start to end)
                if "start_to" in crop_type:
                    # start of sentence symbol
                    flat_phones_and_gaps = ["$"]
                else:
                    # continuation symbol
                    flat_phones_and_gaps = ["&"]
                for _n in range(len(phone_group_syms)):
                    flat_phones_and_gaps.append("!{}".format(gaps_arr[_n]))
                    flat_phones_and_gaps.extend(phone_group_syms[_n])

                if "to_end" in crop_type:
                    # eos symbol
                    flat_phones_and_gaps.append("~")
                else:
                    # continuation symbol
                    flat_phones_and_gaps.append("&")
                flat_phones_and_gaps.append("!{}".format(gaps_arr[-1]))
                seq_as_ints = [self.phone_lookup[s.split("_")[0]] for s in flat_phones_and_gaps]
                phoneme_sequences.append(seq_as_ints)

                phoneme_nonspacing_sequences.append(phone_group_syms)

                # this part will make the word sequences
                # lower? or keep upper case?
                #TODO
                ascii_groups = [el["alignedWord"] for el in words]#al[k]["full_alignment"]["words"]]
                ascii_exact_text_tup = self.metadata_exact_lookup[k]
                ascii_group_syms = [ag for ag in ascii_groups]

                true_text = ascii_exact_text_tup[1]
                # assume that split on spaces is possible
                true_text_group_syms = true_text.split(" ")

                def hamming(a, b):
                    return len([i for i in filter(lambda x: x[0] != x[1], zip(a, b))])

                whitelist = set('abcdefghijklmnopqrstuvwxyz')
                def begend(a, b):
                    j_a = "".join(a)
                    j_b = "".join(b)
                    i = min(10, min(len(j_a), len(j_b)))
                    c_a = ''.join(filter(whitelist.__contains__, j_a.strip().lower()))
                    c_b = ''.join(filter(whitelist.__contains__, j_b.strip().lower()))
                    e = c_a[-i:] == c_b[-i:]
                    #b = c_a[:i] == c_b[:i]
                    return e #and b

                def approx_contains(sub, pri, hamming_error=1):
                    # returns all possible substring matches
                    # allow 1 mismatch per match group (some issues with apostophe)
                    # pri[matches[0][0]:matches[0][1]]
                    # create all possible slicings of pri by length of sub
                    slice_starts = [el for el in range(0, len(pri) - len(sub) + 1)]
                    slice_ends = [ss + len(sub) for ss in slice_starts]
                    pri_slice = [pri[ss:se] for ss, se in zip(slice_starts, slice_ends)]
                    exact_matches = [(ss, se) for (ss, se), prs in zip(zip(slice_starts, slice_ends), pri_slice) if prs == sub]
                    # we can have 1 mismatched word and still get a match
                    hamming_matches = [(ss, se) for (ss, se), prs in zip(zip(slice_starts, slice_ends), pri_slice) if hamming(prs, sub) <= hamming_error]
                    begend_matches = [(ss, se) for (ss, se), prs in zip(zip(slice_starts, slice_ends), pri_slice)
                                      if begend(prs, sub)]
                    return exact_matches, hamming_matches, begend_matches
                cleaned_true_text_group_syms = [''.join(filter(whitelist.__contains__, true_text_group_syms[_n].strip().lower()))
                                                for _n in range(len(true_text_group_syms))]
                # do the match on cleaned ascii syms rather than the base, catches some edge cases like apostrophes
                cleaned_ascii_group_syms = [''.join(filter(whitelist.__contains__, ascii_group_syms[_n].strip().lower()))
                                            for _n in range(len(ascii_group_syms))]
                r, r_hamming, r_begend = approx_contains(cleaned_ascii_group_syms, cleaned_true_text_group_syms)
                if len(r) == 0:
                   if is_sampling:
                       # if we are sampling and don't get a match, just fall back to the ascii and call it a day
                       true_text_sub_group_syms = ascii_group_syms
                   else:
                       if len(r_hamming) > 0:
                           true_text_sub_group_syms = true_text_group_syms[r_hamming[0][0]:r_hamming[0][1]]
                       else:
                           if len(r_begend) > 0:
                               true_text_sub_group_syms = true_text_group_syms[r_begend[0][0]:r_begend[0][1]]
                           else:
                               # fall back to just using the ascii group syms rather than the "true text"
                               true_text_sub_group_syms = ascii_group_syms
                else:
                    # if we have multiple identical matches, take the first one
                    # may be edge cases with ends of sentences, or commas but chalk it up as a loss
                    true_text_sub_group_syms = true_text_group_syms[r[0][0]:r[0][1]]

                # pause related symbols ',;'
                # now go through and check if commas, other pause related specials line up with gaps
                # if there is a mismatch, we fall back to the default (what was in the transcript file)
                final_true_text_group_syms = copy.deepcopy(ascii_group_syms)
                for _n in range(len(true_text_sub_group_syms)):
                    t = final_true_text_group_syms[_n]
                    if "," not in t and ";" not in t:
                        final_true_text_group_syms[_n] = true_text_sub_group_syms[_n]
                    elif t[-1] != "," or t[-1] != ";":
                        # weird potential edge case here, let it just be whatever the original (cleaned up) ascii_group_syms was
                        continue
                    else:
                        # had a trailing , or ;, handle it below
                        final_true_text_group_syms[_n] = true_text_sub_group_syms[_n]
                # remove any extraneous symbols
                symlist = "".join(sorted(self.ascii_lookup.keys()))
                final_true_text_group_syms = [''.join(filter(symlist.__contains__, final_true_text_group_syms[_n]))
                                              for _n in range(len(final_true_text_group_syms))]

                # change flat ascii and gaps based on if it was start_to_mid mid_to_mid mid_to_end or un-cropped (start to end)
                if "start_to" in crop_type:
                    # start of sentence symbol
                    ascii_spacing_terms = [["$"]]
                    ascii_nonspacing_terms  = []
                else:
                    # continuation symbol
                    ascii_spacing_terms = [["&"]]
                    ascii_nonspacing_terms = []

                # we always use the "text" version of the spaces in this
                # so we build a list of chunks that can be selected for representation mixing
                # 1 per word, 1 per gap (~2x len(final_true_text_group_syms)
                # gap part must contain , or ; if it is present. once we detect it is in the group_sym remove it from the word
                # and add it to the spacing term
                # looks the same when joined, but matters for repr mix selection

                for _n in range(len(final_true_text_group_syms)):
                    # find and replace group syms with the ascii exact equivalent???
                    # use 1 because it has the number expansions in ljspeech
                    t = final_true_text_group_syms[_n]
                    ascii_nonspacing_terms.append([t])
                    # don't add spacing for last element
                    if _n == (len(final_true_text_group_syms) - 1):
                        break

                    if "," not in t and ";" not in t:
                        ascii_spacing_terms.append([" "])
                    else:
                        # had a trailing , or ;, handle it below
                        # ones with no trailing , or ; should have been handled before
                        # g is the gap size
                        # 0 or 1 should represent a value between 0 and 0.0625 according to the duration breakpoints
                        # 2 and up should be anything above that
                        #self.pause_duration_breakpoints = [0.01, 0.0625, 0.1325, 0.25]
                        # this means that the buckets are 0 .... 1 ....  2 ..... 3 .... 4
                        g = gaps_arr[_n]
                        if t[-1] == ",":
                            if g >= 2:
                                ascii_spacing_terms.append([", "])
                            else:
                                ascii_spacing_terms.append([" "])
                        elif t[-1] == ";":
                            if g >= 3:
                                ascii_spacing_terms.append(["; "])
                            else:
                                if g >= 2:
                                    ascii_spacing_terms.append([", "])
                                else:
                                    ascii_spacing_terms.append([" "])
                if "to_end" in crop_type:
                    # eos symbol
                    if final_true_text_group_syms[-1][-1] == "!":
                        ascii_spacing_terms.append(["!~"])
                    elif final_true_text_group_syms[-1][-1] == "?":
                        ascii_spacing_terms.append(["?~"])
                    else:
                        ascii_spacing_terms.append([".~"])
                else:
                    # continuation symbol
                    ascii_spacing_terms.append(["&"])

                # gaps_arr[-1] unused in ascii (we don't have silence labeled)
                # we use this to construct both the repr mix and the ascii later
                # slightly different processing than phoneme
                ascii_spacing_sequences.append(ascii_spacing_terms)
                ascii_nonspacing_sequences.append(ascii_nonspacing_terms)

                if write_out_debug_info:
                    # test write to listen to some samples
                    from kkpthlib.datasets.speech.audio_processing.audio_tools import soundsc
                    fldr = "tmp_test_debug_minibatch_format/"
                    if not os.path.exists(fldr):
                        os.mkdir(fldr)
                    wavfile.write(fldr + "out_{}.wav".format(u_i), fs, soundsc(d))
                    full_w = al[list(al.keys())[0]]["transcript"]
                    part_w = " ".join([w["word"] for w in words])
                    with open(fldr + "out_{}.txt".format(u_i), "w") as f:
                        f.write("Part: {}\n".format(part_w))
                        f.write("Full: {}\n".format(full_w))
        else:
            raise ValueError("Unknown symbol_type {} specified!".format(symbol_type))

        # phoneme sequences were pre-constructed so this is easier
        # pad it out so all are same length
        max_seq_len = max([len(ps) for ps in phoneme_sequences])
        # mask and padded sequence
        input_seq_mask = [[1.] * len(ps) + [0.] * (max_seq_len - len(ps)) for ps in phoneme_sequences]
        input_seq_mask = np.array(input_seq_mask).T
        phoneme_sequences = [ps + (max_seq_len - len(ps)) * [self.phone_lookup["_"]] for ps in phoneme_sequences]
        phoneme_sequences = np.array(phoneme_sequences).astype("float32").T

        melspec_seq_mask = [[1.] * ms.shape[0] + [0.] * int(max_frame_count - ms.shape[0]) for ms in melspec_sequences]
        melspec_seq_mask = np.array(melspec_seq_mask)
        padded_melspec_sequences = []
        for el_i, ms in enumerate(melspec_sequences):
            if ms.shape[0] > max_frame_count:
                print("got melspec shape larger than theoretical max frame count?")
                from IPython import embed; embed(); raise ValueError()

            melspec_padded = 0. * melspec[:1, :] + np.zeros((int(max_frame_count), 1)).astype("float32")
            melspec_padded[:len(ms)] = ms
            padded_melspec_sequences.append(melspec_padded)
        padded_melspec_sequences = np.array(padded_melspec_sequences)
        if quantize_to_n_bins is not None:
            assert mean_std_per_bin_normalization is False
            n_bins = quantize_to_n_bins
            bins = np.linspace(0., 1., num=n_bins, endpoint=True)
            quantized_melspec_sequences = np.digitize(padded_melspec_sequences, bins)
        else:
            quantized_melspec_sequences = padded_melspec_sequences

        if symbol_type == "phoneme":
            return phoneme_sequences, input_seq_mask.astype("float32"), quantized_melspec_sequences.astype("float32"), melspec_seq_mask.astype("float32")

        # ascii and repr mix actually constructed from the groups here 
        ascii_sequences = []
        repr_mixed_sequences = []
        repr_mixed_sequences_masks = []
        # be sure we have the same amount of sequences (should always be true)
        assert len(ascii_nonspacing_sequences) == len(ascii_spacing_sequences)
        assert len(ascii_nonspacing_sequences) == len(phoneme_nonspacing_sequences)
        for _n in range(len(ascii_nonspacing_sequences)):
            a_spacing = ascii_spacing_sequences[_n]
            a_nonspacing = ascii_nonspacing_sequences[_n]
            p_nonspacing = phoneme_nonspacing_sequences[_n]

            this_ascii_sequence = []
            this_repr_mixed_sequence = []
            this_repr_mixed_sequence_mask = []

            this_ascii_sequence.extend(a_spacing[0])
            this_repr_mixed_sequence.extend(a_spacing[0])
            # 0 is ascii, 1 is phoneme here
            this_repr_mixed_sequence_mask.append(0)
            assert len(a_nonspacing) == len(p_nonspacing)
            if len(a_spacing) != (len(a_nonspacing) + 1):
                # something has gone weird with the spacing. Bail and set spacing to all " "
                # with the correct beginning and ending symbols (based on the cut type)
                # this happens very rarely
                if "start_to" in crop_type:
                    new_a_spacing = [["$"]]
                elif "mid_to" in crop_type:
                    new_a_spacing = [["&"]]
                else:
                    raise ValueError("Unknown crop_type {}".format(crop_type))

                for _s in range(len(a_nonspacing) - 1):
                    new_a_spacing.append([" "])

                if "to_end" in crop_type:
                    new_a_spacing.append([".~"])
                elif "to_mid" in crop_type:
                    new_a_spacing.append(["&"])
                else:
                    raise ValueError("Unknown crop_type {}".format(crop_type))
                assert len(new_a_spacing) == (len(a_nonspacing) + 1)
                a_spacing = new_a_spacing

            # 0 is ascii, 1 is phoneme for each "word"
            # if we use 0.5, should be 50/50 choice
            choosing = (self.random_state.rand(len(a_nonspacing)) > 0.5).astype("int32")
            for _n, c in enumerate(choosing):
                # build ascii sequence at the same time for convenience
                this_ascii_sequence.extend([el for el in a_nonspacing[_n][0]])
                this_ascii_sequence.extend([el for el in a_spacing[_n + 1][0]])
                if c == 0:
                    this_repr_mixed_sequence.extend([el for el in a_nonspacing[_n][0]])
                    this_repr_mixed_sequence.extend([el for el in a_spacing[_n + 1][0]])
                    this_repr_mixed_sequence_mask.extend([0] * len(a_nonspacing[_n][0]))
                    this_repr_mixed_sequence_mask.extend([0] * len(a_spacing[_n + 1][0]))
                elif c == 1:
                    # p_nonspacing has multi groups, so have to handle slightly differently
                    this_repr_mixed_sequence.extend(p_nonspacing[_n])
                    this_repr_mixed_sequence.extend([el for el in a_spacing[_n + 1][0]])
                    this_repr_mixed_sequence_mask.extend([1] * len(p_nonspacing[_n]))
                    this_repr_mixed_sequence_mask.extend([0] * len(a_spacing[_n + 1][0]))
                    # always use ascii spacing terms even for phones
                else:
                    raise ValueError("Some unknown error when generating masks")
            ascii_sequences.append(this_ascii_sequence)
            repr_mixed_sequences.append(this_repr_mixed_sequence)
            repr_mixed_sequences_masks.append(this_repr_mixed_sequence_mask)

        # now we need to convert pad, convert batches to ints and create masks for the batches
        assert len(ascii_sequences) == len(repr_mixed_sequences)
        assert len(repr_mixed_sequences) == len(repr_mixed_sequences_masks)
        max_ascii_seq_len = max([len(a_s) for a_s in ascii_sequences])
        max_repr_mixed_seq_len = max([len(r_s) for r_s in repr_mixed_sequences])

        input_ascii_seq = [a_s + (max_ascii_seq_len - len(a_s)) * ["_"] for a_s in ascii_sequences]
        input_ascii_seq_mask = [[1.] * len(a_s) + [0.] * (max_ascii_seq_len - len(a_s)) for a_s in ascii_sequences]

        input_ascii_seq_lu = [[self.ascii_lookup[el] for el in a_s] for a_s in input_ascii_seq]


        input_repr_mixed_seq = [r_s + (max_repr_mixed_seq_len - len(r_s)) * ["_"] for r_s in repr_mixed_sequences]
        input_repr_mixed_seq_mask = [r_s_m + [0.] * (max_repr_mixed_seq_len - len(r_s_m)) for r_s_m in repr_mixed_sequences_masks]
        input_repr_mixed_seq_mask_mask = [[1.] * len(r_s) + [0.] * (max_repr_mixed_seq_len - len(r_s)) for r_s in repr_mixed_sequences]

        input_repr_mixed_seq_lu = [[self.ascii_lookup[el] if el_t == 0
                                    else self.phone_lookup[el.split("_")[0].lower()]
                                    for el, el_t in zip(r_s, r_s_m)]
                                    for r_s, r_s_m in zip(input_repr_mixed_seq, input_repr_mixed_seq_mask)]

        input_ascii_seq_lu = np.array(input_ascii_seq_lu).astype("float32").T
        input_ascii_seq_mask = np.array(input_ascii_seq_mask).T
        assert input_ascii_seq_lu.shape == input_ascii_seq_mask.shape

        input_repr_mixed_seq_lu = np.array(input_repr_mixed_seq_lu).astype("float32").T
        input_repr_mixed_seq_mask = np.array(input_repr_mixed_seq_mask).T
        input_repr_mixed_seq_mask_mask = np.array(input_repr_mixed_seq_mask_mask).T
        assert input_repr_mixed_seq_lu.shape == input_repr_mixed_seq_mask.shape
        assert input_repr_mixed_seq_mask.shape == input_repr_mixed_seq_mask_mask.shape

        if symbol_type == "ascii":
            return input_ascii_seq_lu, input_ascii_seq_mask.astype("float32"), quantized_melspec_sequences.astype("float32"), melspec_seq_mask.astype("float32")
        elif symbol_type == "representation_mixed":
            pack = []
            pack.append(input_repr_mixed_seq_lu)
            pack.append(input_repr_mixed_seq_mask.astype("float32"))
            pack.append(input_repr_mixed_seq_mask_mask.astype("float32"))

            pack.append(input_ascii_seq_lu)
            pack.append(input_ascii_seq_mask.astype("float32"))

            pack.append(phoneme_sequences)
            pack.append(input_seq_mask.astype("float32"))

            pack.append(quantized_melspec_sequences.astype("float32"))
            pack.append(melspec_seq_mask.astype("float32"))
            return pack
        else:
            raise ValueError("Unhandled symbol_type {}".format(symbol_type))

    def _fetch_utterance(self, basename, skip_mel=False):
        # fs, d, melspec, info
        this_info = {}
        this_info[basename] = {}
        wav_path = self.wav_folder + "/" + basename + ".wav"
        fs, d = wavfile.read(wav_path)
        d = d.astype('float32') / (2 ** 15)
        if self.alignment_folder is not None:
            # alignment from gentle
            alignment_info_json = self.alignment_folder + "/" + basename + ".json"
            with open(alignment_info_json, "r") as read_file:
                alignment = json.load(read_file)
            end_in_samples = fs * alignment["words"][-1]["end"]
            end_in_samples += 2 * self.stft_size
            end_in_samples = int(end_in_samples)
            # add a little bit of extra, if the cut is a ways before the end
            start_in_samples = fs * alignment["words"][0]["start"]
            start_in_samples -= 2 * self.stft_size
            start_in_samples = int(max(start_in_samples, 0))
            # cut a little bit before the start
            if self.cut_on_alignment:
                overall_len = len(d)
                d = d[start_in_samples:end_in_samples]
                this_info["original_len_samples"] = overall_len
                this_info["start_offset_precut_in_samples"] = start_in_samples
                this_info["end_offset_precut_in_samples"] = overall_len - end_in_samples
            else:
                this_info["original_len_samples"] = overall_len
                this_info["start_offset_precut_in_samples"] = 0
                this_info["end_offset_precut_in_samples"] = 0
            this_info[basename]["full_alignment"] = alignment
            this_info[basename]["transcript"] = alignment["transcript"]
        if skip_mel:
            return fs, d, None, this_info
        # T, F melspec
        melspec = self._melspectrogram_preprocess(d, fs)
        # check full validity outside the core fetch
        return fs, d, melspec, this_info

    def melspectrogram_denormalize(self, ms):
        from IPython import embed; embed(); raise ValueError()

    def _old_melspectrogram_preprocess(self, data, sample_rate):
        # takes in a raw sequence scaled between -1 and 1 (such as loaded from a wav file)

        # 'Center freqs' of mel bands - uniformly spaced between limits
        x = data
        sr = sample_rate

        n_mels = self.n_mels

        fmin = self.mel_freq_min
        fmax = self.mel_freq_max

        n_fft = self.stft_size
        n_step = self.stft_step

        # preemphasis filter
        coef = self.preemphasis_coef
        b = np.array([1.0, -coef], x.dtype)
        a = np.array([1.0], x.dtype)
        preemphasis_filtered = signal.lfilter(b, a, x)

        # mel weights
        weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype="float32")

        fftfreqs = np.linspace(0, float(sr) / 2., int(1 + n_fft // 2), endpoint=True)

        min_mel = herz_to_mel(fmin)
        max_mel = herz_to_mel(fmax)
        mels = np.linspace(min_mel, max_mel, n_mels + 2)
        mel_f = mel_to_herz(mels)[:, 0]

        fdiff = np.diff(mel_f)
        ramps = np.subtract.outer(mel_f, fftfreqs)

        for i in range(n_mels):
            # lower and upper slopes for all bins
            lower = -ramps[i] / float(fdiff[i])
            upper = ramps[i + 2] / float(fdiff[i + 1])

            # .. then intersect them with each other and zero
            weights[i] = np.maximum(0., np.minimum(lower, upper))
        # slaney style norm
        enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
        weights *= enorm[:, np.newaxis]
        mel_weights = weights

        # do stft
        ref_level_db = self.ref_level_db
        min_level_db = self.min_level_db
        def _amp_to_db(a):
            min_level = np.exp(min_level_db / 20. * np.log(10))
            return 20 * np.log10(np.maximum(min_level, a))

        abs_stft = np.abs(stft(preemphasis_filtered, fftsize=n_fft, step=n_step, real=True))
        melspec_ref = _amp_to_db(np.dot(mel_weights, abs_stft.T)) - ref_level_db
        melspec_clip = np.clip((melspec_ref - min_level_db) / -min_level_db, 0, 1)
        return melspec_clip.T

    def _melspectrogram_preprocess(self, data, sample_rate):
        # takes in a raw sequence scaled between -1 and 1 (such as loaded from a wav file)

        # 'Center freqs' of mel bands - uniformly spaced between limits
        x = data
        sr = sample_rate

        n_mels = self.n_mels

        fmin = self.mel_freq_min
        fmax = self.mel_freq_max

        n_fft = self.stft_size
        n_step = self.stft_step

        # preemphasis filter
        coef = self.preemphasis_coef
        b = np.array([1.0, -coef], x.dtype)
        a = np.array([1.0], x.dtype)
        preemphasis_filtered = signal.lfilter(b, a, x)

        # mel weights
        # nfft - 1 because onesided=False cuts off last bin
        weights = np.zeros((n_mels, n_fft - 1), dtype="float32")

        fftfreqs = np.linspace(0, float(sr) / 2., n_fft - 1, endpoint=True)

        min_mel = herz_to_mel(fmin)
        max_mel = herz_to_mel(fmax)
        mels = np.linspace(min_mel, max_mel, n_mels + 2)
        mel_f = mel_to_herz(mels)[:, 0]

        fdiff = np.diff(mel_f)
        ramps = np.subtract.outer(mel_f, fftfreqs)

        for i in range(n_mels):
            # lower and upper slopes for all bins
            lower = -ramps[i] / float(fdiff[i])
            upper = ramps[i + 2] / float(fdiff[i + 1])

            # .. then intersect them with each other and zero
            weights[i] = np.maximum(0., np.minimum(lower, upper))
        # slaney style norm
        enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
        weights *= enorm[:, np.newaxis]
        mel_weights = weights

        # do stft
        ref_level_db = self.ref_level_db
        min_level_db = self.min_level_db
        def _amp_to_db(a):
            min_level = np.exp(min_level_db / 20. * np.log(10))
            return 20 * np.log10(np.maximum(min_level, a))

        # ONE SIDED MUST BE FALSE!!!!!!!!
        abs_stft = np.abs(stft(preemphasis_filtered, fftsize=n_fft, step=n_step, real=True, compute_onesided=False))
        melspec_ref = _amp_to_db(np.dot(mel_weights, abs_stft.T)) - ref_level_db
        melspec_clip = np.clip((melspec_ref - min_level_db) / -min_level_db, 0, 1)
        return melspec_clip.T
