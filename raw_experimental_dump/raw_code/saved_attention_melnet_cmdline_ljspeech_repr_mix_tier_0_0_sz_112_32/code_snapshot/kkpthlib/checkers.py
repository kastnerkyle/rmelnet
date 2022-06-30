from __future__ import print_function
from collections import defaultdict
import numpy as np
import os
from music21 import roman, stream, chord, midi, corpus, interval, pitch
import json
import shutil
from operator import itemgetter
from itertools import groupby, cycle
import json
import base64
try:
    import cPickle as pickle
except ImportError:
    import _pickle as pickle
import gc
import time
from collections import OrderedDict
from .core import get_cache_dir
from .datasets import music_json_to_midi
from .datasets import music21_parse_and_save_json
from .checker_html_reporter import make_index_html_string
from .checker_html_reporter import make_index_html_string2
from .checker_html_reporter import make_website_string
from .checker_html_reporter import make_plot_json
from .checker_html_reporter import midi_to_name_lookup 

class GroupDict(OrderedDict):
    def __init__(self):
        super(GroupDict, self).__init__()

    def groupby_tag(self):
        """
        'pivot' so keys are reduced, and matches

        turns:

        GroupDict([(('76:74:73:78:80:81:81:80->78', 1, 55),
                    [('bwv145-a.A-major-transposed.metajson', 0, 51),
                     ('bwv145-a.A-major-transposed.metajson', 18, 51)])])

        into

        OrderedDict([('bwv145-a.A-major-transposed.metajson',
                      [(0, 51, '76:74:73:78:80:81:81:80->78', 1, 55),
                       (18, 51, '76:74:73:78:80:81:81:80->78', 1, 55)])])

        where the first 2 indices are the step into key (bwv145) and its length
              the match key
              then the index into the example and its length

        """
        all_v = []
        for k, v in self.items():
            all_v.extend([vi[0] for vi in v])

        o = OrderedDict()

        done = []
        for p in all_v:
            if p in done:
                continue
            # key -> list 
            # [(('76:74:73:78:80:81:81:80->78', 1, 55),
            #  [('bwv145-a.A-major-transposed.metajson', 0, 51),
            #   ('bwv145-a.A-major-transposed.metajson', 18, 51)])]
            temp = [(k, [vi for vi in v if p in vi]) for k, v in self.items()]
            # prune empties
            temp = [t for t in temp if len(t[1]) > 0]
            # name of tag
            name = temp[0][1][0][0]
            for el in temp:
               # value of format
               # [(0, 51, '76:74:73:78:80:81:81:80->78', 1, 55),
               # (18, 51, '76:74:73:78:80:81:81:80->78', 1, 55)]
               o[name] = [(e[1], e[2]) + el[0] for e in el[1]]
            done.append(p)
        return o


class Trie(object):
    """
    order_insert
    order_search

    are the primary methods
    trie = Trie()

    trie.order_insert(3, "string")

    or

    trie.order_insert(3, [<hashable_obj_instance1>, <hashable_obj_instance2>, ...]

    can optionally pass a tag in with_attribution_tag arguemnt
    """
    def __init__(self):
        self.root = defaultdict()
        self._end = "_end"
        self.orders = []
        self.attribution_tags = {}

    def insert(self, list_of_items):
        current = self.root
        for item in list_of_items:
            current = current.setdefault(item, {})
        current.setdefault(self._end)
        self.orders = sorted(list(set(self.orders + [len(list_of_items)])))

    def order_insert(self, order, list_of_items, with_attribution_tag=None):
        s = 0
        e = order
        while e < len(list_of_items):
            # + 1 due to numpy slicing
            e = s + order + 1
            el = list_of_items[s:e]
            self.insert(el)
            if with_attribution_tag:
                tk_seq = [str(eel).encode("ascii", "ignore").decode("latin-1") for eel in el]
                tk = ":".join(tk_seq[:-1]) + "->" + tk_seq[-1]
                if tk not in self.attribution_tags:
                    self.attribution_tags[tk] = []
                # tag is name of file, number of steps into sequence associated with that file, total number of items in the file
                self.attribution_tags[tk].append((with_attribution_tag, s, len(list_of_items)))
            s += 1

    def search(self, list_of_items):
        # items of the list should be hashable
        # returns True if item in Trie, else False
        if len(list_of_items) not in self.orders:
            raise ValueError("item {} has invalid length {} for search, only {} supported".format(list_of_items, len(list_of_items), self.orders))
        current = self.root
        for item in list_of_items:
            if item not in current:
                return False
            current = current[item]
        if self._end in current:
            return True
        return False

    def order_search(self, order, list_of_items, return_attributions=False):
        # returns true if subsequence at offset is found
        s = 0
        e = order
        searches = []
        attributions = GroupDict()
        while e < len(list_of_items):
            # + 1 due to numpy slicing
            e = s + order + 1
            el = list_of_items[s:e]
            ss = self.search(el)
            if ss and return_attributions:
                if not ss:
                    attributions.append(None)
                else:
                    tk_seq = [str(eel).encode("ascii", "ignore").decode("latin-1") for eel in el]
                    tk = ":".join(tk_seq[:-1]) + "->" + tk_seq[-1]
                    attributions[(tk, s, len(list_of_items))] = self.attribution_tags[tk]
            searches.append(ss)
            s += 1
        if return_attributions:
            return searches, attributions
        else:
            return searches


class MaxOrder(object):
    def __init__(self, max_order):
        assert max_order >= 2
        self.orders = list(range(1, max_order + 1))
        self.order_tries = [Trie() for n in self.orders]
        self.max_order = max_order

    def insert(self, list_of_items, with_attribution_tag=None):
        """
        a string, or a list of elements
        """
        if len(list_of_items) - 1 < self.max_order:
            raise ValueError("item {} to insert shorter than max_order!".format(list_of_items))

        for n, i in enumerate(self.orders):
            self.order_tries[n].order_insert(i, list_of_items, with_attribution_tag=with_attribution_tag)

    def included_at_index(self, list_of_items, return_attributions=False):
        """
        return a list of list values in [None, True, False]
        where None is pre-padding, True means the subsequence of list_of_items at that point is included
        False means not included

        attributions returned as custom OrderedDict of OrderedDict

        with extra methods for grouping / gathering
        e.g. for attr returned

        attr["order_8"].keys() will show all the keys with matches in the data
        attr["order_8"][key_name] will show the files, places, and total length of the match for all attribute tags
        attr["order_8"].groupby_tag() will "pivot" the matches to show all matches, with a list of the tag names and positions
        """
        longest = len(list_of_items) - 1
        all_res = []
        all_attr = OrderedDict()
        for n, i in enumerate(self.orders):
            if return_attributions:
                res, attr = self.order_tries[n].order_search(i, list_of_items, return_attributions=return_attributions)
            else:
                res = self.order_tries[n].order_search(i, list_of_items, return_attributions=return_attributions)
            # need to even these out with padding
            if len(res) < longest:
                res = [None] * (longest - len(res)) + res
            all_res.append(res)
            if return_attributions:
                all_attr["order_{}".format(i)] = attr
        if return_attributions:
            return all_res, all_attr
        else:
            return all_res

    def satisfies_max_order(self, list_of_items):
        if len(list_of_items) - 1 < self.max_order:
            return True
        matched = self.included_at_index(list_of_items)
        true_false_order = [any([mi for mi in m if mi is not None]) for m in matched]
        if all(true_false_order[:-1]):
            # if all the previous are conained, guarantee the last one IS contained
            return not true_false_order[-1]
        else:
            # if some of the previous ones were false, max order is satisfied
            return True

# Following functions from 
# https://www.kaggle.com/wfaria/midi-music-data-extraction-using-music21
def note_count(measure, count_dict):
    bass_note = None
    for chord in measure.recurse().getElementsByClass('Chord'):
        # All notes have the same length of its chord parent.
        note_length = chord.quarterLength
        for note in chord.pitches:
            # If note is "C5", note.name is "C". We use "C5"
            # style to be able to detect more precise inversions.
            note_name = str(note)
            if (bass_note is None or bass_note.ps > note.ps):
                bass_note = note
            if note_name in count_dict:
                count_dict[note_name] += note_length
            else:
                count_dict[note_name] = note_length
    return bass_note


def simplify_roman_name(roman_numeral):
    # Chords can get nasty names as "bII#86#6#5",
    # in this method we try to simplify names, even if it ends in
    # a different chord to reduce the chord vocabulary and display
    # chord function clearer.
    ret = roman_numeral.romanNumeral
    inversion_name = None
    inversion = roman_numeral.inversion()
    # Checking valid inversions.
    if ((roman_numeral.isTriad() and inversion < 3) or
            (inversion < 4 and
                 (roman_numeral.seventh is not None or roman_numeral.isSeventh()))):
        inversion_name = roman_numeral.inversionName()
    if (inversion_name is not None):
        ret = ret + str(inversion_name)
    elif (roman_numeral.isDominantSeventh()): ret = ret + "M7"
    elif (roman_numeral.isDiminishedSeventh()): ret = ret + "o7"
    return ret

def harmonic_reduction(part):
    ret_roman = []
    ret_chord = []
    temp_midi = stream.Score()
    temp_midi_chords = part.chordify()
    temp_midi.insert(0, temp_midi_chords)
    music_key = temp_midi.analyze('key')
    max_notes_per_chord = 4
    # bug in music21? chordify can return a thing without measures...
    if len(temp_midi_chords.measures(0, None)) == 0:
        print("chordify returned 0 measure stream, attempting to fix...")
        tt = stream.Stream()
        for tmc in temp_midi_chords:
            mm = stream.Measure()
            mm.insert(tmc)
            tt.append(mm)
        temp_midi_chords = tt

    for m in temp_midi_chords.measures(0, None): # None = get all measures.
        if (type(m) != stream.Measure):
            continue
        # Here we count all notes length in each measure,
        # get the most frequent ones and try to create a chord with them.
        count_dict = dict()
        bass_note = note_count(m, count_dict)
        if (len(count_dict) < 1):
            ret_roman.append("-") # Empty measure
            ret_chord.append(chord.Chord(["C0", "C0", "C0", "C0"]))
            continue
        sorted_items = sorted(count_dict.items(), key=lambda x:x[1])
        sorted_notes = [item[0] for item in sorted_items[-max_notes_per_chord:]]
        measure_chord = chord.Chord(sorted_notes)
        # Convert the chord to the functional roman representation
        # to make its information independent of the music key.
        roman_numeral = roman.romanNumeralFromChord(measure_chord, music_key)
        ret_roman.append(simplify_roman_name(roman_numeral))
        ret_chord.append(measure_chord)
    return ret_roman, ret_chord

def music21_from_midi(midi_path):
    mf = midi.MidiFile()
    mf.open(midi_path)
    mf.read()
    mf.close()
    return midi.translate.midiFileToStream(mf)

def get_metadata(p, original_filepath_to_piece):
    piece_container = {}
    piece_container["original_filepath"] = original_filepath_to_piece
    piece_container["parts"] = []
    piece_container["parts_times"] = []
    piece_container["parts_cumulative_times"] = []
    piece_container["parts_names"] = []
    roman_names, chord_objects = harmonic_reduction(p)
    # list of list of chord pitches in SATB order 
    chord_pitch_names = [[str(cpn) for cpn in cn.pitches][::-1] for cn in chord_objects]
    functional_names = [cn.commonName for cn in chord_objects]
    pitched_functional_names = [cn.pitchedCommonName for cn in chord_objects]
    piece_container["chord_pitches"] = chord_pitch_names
    piece_container["functional_names"] = functional_names
    piece_container["pitched_functional_names"] = pitched_functional_names
    piece_container["roman_names"] = roman_names

    for i, pi in enumerate(p.parts):
        piece_container["parts"].append([])
        piece_container["parts_times"].append([])
        piece_container["parts_cumulative_times"].append([])
        piece_container["parts_names"].append(pi.id)
        part = []
        part_time = []
        for n in pi.flat.notesAndRests:
            if n.isChord:
                continue

            if n.isRest:
                part.append(0)
            else:
                part.append(n.pitch.midi)
            part_time.append(n.duration.quarterLength)
        piece_container["parts"][i] += part
        piece_container["parts_times"][i] += part_time
        piece_container["parts_cumulative_times"][i] += list(np.cumsum(part_time))
    return piece_container


def save_metadata_to_json(piece_container, fpath):
    j = json.dumps(piece_container, indent=4)
    with open(fpath, "w") as f:
         print(j, file=f)


def get_music21_metadata(list_of_musicjson_or_midi_files, explicit_metapath_tag, only_pieces_with_n_voices=[4], assume_cached=False,
                         metapath="_kkpthlib_cache", verbose=False):
    """
    build metadata for plagiarism checks
    """
    #metapath = get_cache_dir() + "_{}_metadata".format(abs(hash(tuple(list_of_musicjson_or_midi_files))) % 10000)
    metapath = get_cache_dir() + "_{}_metadata".format(explicit_metapath_tag)
    if assume_cached:
        print("Already cached files found in {}!".format(metapath))
        files_list = [metapath + os.sep + f for f in os.listdir(metapath) if ".metajson" in f]
        if not os.path.exists(metapath):
            raise ValueError("{} does not exist, cannot assume cached!".format(metapath))
        return {"files": files_list}
    else:
        if os.path.exists(metapath):
            raise ValueError("Folder {} already exists even though assume_cached=False! Cowardly refusing to continue - delete manually!".format(metapath))

    midi_cache_path = get_cache_dir() + "_{}_midicache".format(explicit_metapath_tag)
    piece_paths = []
    for f in list_of_musicjson_or_midi_files:
        if f.endswith(".json"):
            if not os.path.exists(midi_cache_path):
                os.mkdir(midi_cache_path)
            basename = f.split(os.sep)[-1].split(".json")[0]
            out_midi = midi_cache_path + os.sep + basename + ".midi"
            if not os.path.exists(out_midi):
                music_json_to_midi(f, out_midi)
            piece_paths.append(out_midi)
        elif f.endswith(".midi") or f.endswith(".mid"):
            if not os.path.exists(midi_cache_path):
                os.mkdir(midi_cache_path)
            basename = f.split(os.sep)[-1].split(".mid")[0]
            out_path = midi_cache_path + os.sep + basename + ".midi"
            if not os.path.exists(out_path):
                shutil.copy2(f, out_path)
            piece_paths.append(out_path)
        else:
            raise ValueError("Unknown file type for file {}, expected .json (MusicJSON) or .midi/.mid".format(f))
    if not os.path.exists(metapath):
        os.mkdir(metapath)

    print("Not yet cached, processing...")
    print("Total number of pieces to process from music21: {}".format(len(piece_paths)))

    for it, piece in enumerate(piece_paths):
        p = music21_from_midi(piece)
        if len(p.parts) not in only_pieces_with_n_voices:
            print("Skipping file {}, {} due to undesired voice count...".format(it, p_bach))
            continue

        if len(p.metronomeMarkBoundaries()) != 1:
            print("Skipping file {}, {} due to unknown or multiple tempo changes...".format(it, p_bach))
            continue

        print("Processing {} of {}, {} ...".format(it, len(piece_paths), piece))
        stripped_extension_name = ".".join(os.path.split(piece)[1].split(".")[:-1])
        base_fpath = metapath + os.sep + stripped_extension_name
        skipped = False
        k = p.analyze('key')
        dp = get_metadata(p, piece)
        if os.path.exists(base_fpath + ".metajson"):
            pass
        else:
            save_metadata_to_json(dp, base_fpath + ".metajson")

    files_list = [metapath + os.sep + f for f in os.listdir(metapath) if ".metajson" in f]
    return {"files": files_list}


def build_music_plagiarism_checkers(metajson_files, roman_reduced_max_order=10, roman_checker_max_order=10,
                                    pitched_functional_max_order=10,
                                    functional_max_order=10,
                                    pitches_max_order=10):
    """
    given list of metajson files, builds all the tries for checking plagiarism
    """
    roman_reduced_checker = MaxOrder(roman_reduced_max_order)
    roman_checker = MaxOrder(roman_checker_max_order)
    pitched_functional_checker = MaxOrder(pitched_functional_max_order)
    functional_checker = MaxOrder(functional_max_order)

    soprano_pitch_checker = MaxOrder(pitches_max_order)
    alto_pitch_checker = MaxOrder(pitches_max_order)
    tenor_pitch_checker = MaxOrder(pitches_max_order)
    bass_pitch_checker = MaxOrder(pitches_max_order)

    soprano_pitch_duration_checker = MaxOrder(pitches_max_order)
    alto_pitch_duration_checker = MaxOrder(pitches_max_order)
    tenor_pitch_duration_checker = MaxOrder(pitches_max_order)
    bass_pitch_duration_checker = MaxOrder(pitches_max_order)

    for n, jf in enumerate(metajson_files):
        print("growing plagiarism checker {}/{}".format(n + 1, len(metajson_files)))
        with open(jf) as f:
            data = json.load(f)
        tag = data["original_filepath"]
        roman_names = data["roman_names"]
        roman_reduced_names = [x[0] for x in groupby(roman_names)]
        roman_checker.insert(roman_names, with_attribution_tag=tag)

        pitched_functional_names = data["pitched_functional_names"]
        functional_names = data["functional_names"]

        # pad with empty stuff if < max order?
        if len(roman_reduced_names) <= roman_reduced_max_order:
            print("{} roman reduced names < max order, padding with '-'".format(jf))
            roman_reduced_names = roman_reduced_names + ["-"] * roman_reduced_max_order
            roman_reduced_names = roman_reduced_names[:roman_reduced_max_order + 1]
        roman_reduced_checker.insert(roman_reduced_names, with_attribution_tag=tag)
        pitched_functional_checker.insert(pitched_functional_names, with_attribution_tag=tag)
        functional_checker.insert(functional_names, with_attribution_tag=tag)

        soprano_pitch_checker.insert(data["parts"][0], with_attribution_tag=tag)
        alto_pitch_checker.insert(data["parts"][1], with_attribution_tag=tag)
        tenor_pitch_checker.insert(data["parts"][2], with_attribution_tag=tag)
        bass_pitch_checker.insert(data["parts"][3], with_attribution_tag=tag)

        assert len(data["parts"][0]) == len(data["parts_times"][0])
        soprano_pitch_duration_checker.insert(list(zip(data["parts"][0], data["parts_times"][0])), with_attribution_tag=tag)

        assert len(data["parts"][1]) == len(data["parts_times"][1])
        alto_pitch_duration_checker.insert(list(zip(data["parts"][1], data["parts_times"][1])), with_attribution_tag=tag)

        assert len(data["parts"][2]) == len(data["parts_times"][2])
        tenor_pitch_duration_checker.insert(list(zip(data["parts"][2], data["parts_times"][2])), with_attribution_tag=tag)

        assert len(data["parts"][3]) == len(data["parts_times"][3])
        bass_pitch_duration_checker.insert(list(zip(data["parts"][3], data["parts_times"][3])), with_attribution_tag=tag)

    return {"roman_names_checker": roman_checker,
            "roman_reduced_names_checker": roman_reduced_checker,
            "pitched_functional_checker": pitched_functional_checker,
            "functional_checker": functional_checker,
            "soprano_pitch_checker": soprano_pitch_checker,
            "alto_pitch_checker": alto_pitch_checker,
            "tenor_pitch_checker": tenor_pitch_checker,
            "bass_pitch_checker": bass_pitch_checker,
            "soprano_pitch_duration_checker": soprano_pitch_duration_checker,
            "alto_pitch_duration_checker": alto_pitch_duration_checker,
            "tenor_pitch_duration_checker": tenor_pitch_duration_checker,
            "bass_pitch_duration_checker": bass_pitch_duration_checker}


def extract_plot_info_from_file(midi_or_musicjson_file_path, match_info=None, match_type=None, match_note_color="red"):
    # match_info[0] is the key itself
    # match_info[1] is the "number" of the match, if the match occurs multiple times we need to know which match to color
    tmp_midi_path = "_rpttmp.midi"
    tmp_json_path = "_rpttmp.json"
    if os.path.exists(tmp_midi_path):
        os.remove(tmp_midi_path)
    if os.path.exists(tmp_json_path):
        os.remove(tmp_json_path)

    f = midi_or_musicjson_file_path
    if f.endswith(".json"):
        music_json_to_midi(f, tmp_midi_path)
    elif f.endswith(".midi") or f.endswith(".mid"):
        shutil.copy2(f, tmp_midi_path)

    p = music21_from_midi(tmp_midi_path)
    core_name = tmp_json_path
    music21_parse_and_save_json(p, core_name, tempo_factor=1)

    with open(tmp_json_path, "r") as f:
        music_json_data = json.load(f)
    l = []
    marked_l = []
    for _p in range(len(music_json_data["parts"])):
        parts = music_json_data["parts"][_p]
        parts_times = music_json_data["parts_times"][_p]
        parts_cumulative_times = music_json_data["parts_cumulative_times"][_p]
        assert len(parts) == len(parts_times)
        assert len(parts_times) == len(parts_cumulative_times)
        if match_info is not None:
            if match_type == "pitch_duration":
                el_vals = list(zip(parts, parts_times))
                part_note_names = [(midi_to_name_lookup[el_i[0]] if el_i[0] != 0 else "0", el_i[1]) for el_i in el_vals]
            elif match_type == "pitch":
                part_note_names = [midi_to_name_lookup[tt] if tt != 0 else "0" for tt in parts]
            def contains(sub, pri):
                # print (contains((1,2,3),(1,2,3)))
                # https://stackoverflow.com/questions/3847386/how-to-test-if-a-list-contains-another-list
                M, N = len(pri), len(sub)
                i, LAST = 0, M-N+1
                while True:
                    try:
                        found = pri.index(sub[0], i, LAST) # find first elem in sub
                    except ValueError:
                        return False
                    if pri[found:found+N] == sub:
                        return [found, found+N-1]
                    else:
                        i = found+1
            # try to do subsequence match here
            # need to use match_info because there can be multiple matches in a file... don't rematch first over and over
            start_step = 0
            matched_at = []
            run_steps = 0
            while start_step <= (len(part_note_names) - len(match_info[0])):
                search_res = contains(match_info[0], part_note_names[start_step:])
                if search_res is not False:
                    matched_at.append((start_step + search_res[0], start_step + search_res[1]))
                    start_step = start_step + search_res[0] + 1
                else:
                    break

        for _s in range(len(parts)):
            # should be ok to skip rests...
            if parts[_s] == 0:
                continue
            d = parts_times[_s]
            l.append((parts[_s], parts_cumulative_times[_s] - d, d))
            if match_info is not None:
                if len(matched_at) > 0:
                    which_match = matched_at[match_info[1]]
                    match_steps = list(range(which_match[0], which_match[1] + 1))
                    if _s in match_steps:
                        marked_l.append(True)
                    else:
                        marked_l.append(False)
                else:
                    marked_l.append(False)
            else:
                marked_l.append(False)

    # want them all to end at the same place
    #last_step = max([t[1] for t in l])
    #last_step_dur = max([t[2] for t in l if t[1] == last_step])
    #end_time = last_step + last_step_dur

    end_time = 120
    r = make_plot_json(l, notes_to_highlight=marked_l, match_note_color=match_note_color)
    return [r, l, marked_l]


def write_html_report_for_musicjson(midi_or_musicjson_file_path, html_report_write_directory, match_info=None, match_type=None, report_index_value=0, info_tag=None):
    # match_info[0] is the key itself
    # match_info[1] is the "number" of the match, if the match occurs multiple times we need to know which match to color
    tmp_midi_path = "_rpttmp.midi"
    tmp_json_path = "_rpttmp.json"
    if os.path.exists(tmp_midi_path):
        os.remove(tmp_midi_path)
    if os.path.exists(tmp_json_path):
        os.remove(tmp_json_path)

    f = midi_or_musicjson_file_path
    if f.endswith(".json"):
        music_json_to_midi(f, tmp_midi_path)
    elif f.endswith(".midi") or f.endswith(".mid"):
        shutil.copy2(f, tmp_midi_path)

    p = music21_from_midi(tmp_midi_path)
    core_name = tmp_json_path
    music21_parse_and_save_json(p, core_name, tempo_factor=1)

    with open(tmp_json_path, "r") as f:
        music_json_data = json.load(f)
    l = []
    marked_l = []
    for _p in range(len(music_json_data["parts"])):
        parts = music_json_data["parts"][_p]
        parts_times = music_json_data["parts_times"][_p]
        parts_cumulative_times = music_json_data["parts_cumulative_times"][_p]
        assert len(parts) == len(parts_times)
        assert len(parts_times) == len(parts_cumulative_times)
        if match_info is not None:
            if match_type == "pitch_duration":
                el_vals = list(zip(parts, parts_times))
                part_note_names = [(midi_to_name_lookup[el_i[0]] if el_i[0] != 0 else "0", el_i[1]) for el_i in el_vals]
            elif match_type == "pitch":
                part_note_names = [midi_to_name_lookup[tt] if tt != 0 else "0" for tt in parts]
            else:
                raise ValueError("Unknown match_type specified in write_html_report_for_musicjson")
            def contains(sub, pri):
                # print (contains((1,2,3),(1,2,3)))
                # https://stackoverflow.com/questions/3847386/how-to-test-if-a-list-contains-another-list
                M, N = len(pri), len(sub)
                i, LAST = 0, M-N+1
                while True:
                    try:
                        found = pri.index(sub[0], i, LAST) # find first elem in sub
                    except ValueError:
                        return False
                    if pri[found:found+N] == sub:
                        return [found, found+N-1]
                    else:
                        i = found+1
            # try to do subsequence match here
            # need to use match_info because there can be multiple matches in a file... don't rematch first over and over
            start_step = 0
            matched_at = []
            run_steps = 0
            while start_step <= (len(part_note_names) - len(match_info[0])):
                search_res = contains(match_info[0], part_note_names[start_step:])
                if search_res is not False:
                    matched_at.append((start_step + search_res[0], start_step + search_res[1]))
                    start_step = start_step + search_res[0] + 1
                else:
                    break

        for _s in range(len(parts)):
            # should be ok to skip rests...
            if parts[_s] == 0:
                continue
            d = parts_times[_s]
            l.append((parts[_s], parts_cumulative_times[_s] - d, d))
            if match_info is not None:
                if len(matched_at) > 0:
                    which_match = matched_at[match_info[1]]
                    match_steps = list(range(which_match[0], which_match[1] + 1))
                    if _s in match_steps:
                        marked_l.append(True)
                    else:
                        marked_l.append(False)
                else:
                    marked_l.append(False)
            else:
                marked_l.append(False)

    # want them all to end at the same place
    #last_step = max([t[1] for t in l])
    #last_step_dur = max([t[2] for t in l if t[1] == last_step])
    #end_time = last_step + last_step_dur

    end_time = 120
    r = make_plot_json(l, notes_to_highlight=marked_l)

    """
    # write out the json + javascript
    with open("notesOnlyJSON.js", "w") as f:
        f.write(r)
    """

    report_dir = html_report_write_directory + os.sep + "test_report"
    if not os.path.exists(report_dir):
        os.mkdir(report_dir)

    # write out the html with minor modifications for lane names
    with open(tmp_midi_path, "rb") as f:
         base64_midi = base64.b64encode(f.read()).decode("utf-8")

    if midi_or_musicjson_file_path.endswith(".json"):
        raise ValueError("Need to use midi file for report generation for now! got {}".format(midi_or_musicjson_file_path))

    midi_name = midi_or_musicjson_file_path.split(os.sep)[-1]
    w = make_website_string(javascript_note_data_string=r, end_time=end_time, info_tag=info_tag, report_index_value=report_index_value,
                            midi_name=midi_name, base64_midi=base64_midi)

    with open(report_dir + os.sep + "report{}.html".format(report_index_value), "w") as f:
        f.write(w)

    if os.path.exists(tmp_midi_path):
        os.remove(tmp_midi_path)
    if os.path.exists(tmp_json_path):
        os.remove(tmp_json_path)


def evaluate_music_against_checkers(midi_or_musicjson_file_path, checkers, write_checker_results_to_directory=None,
                                    report_maxorder_violations=0):
    """ report_maxorder_violations = 0 means to also save and report files that violate the absolute maxorder for a given checker.
        setting to -1, -2 etc will report maxorder - 1, maxorder - 2 violations as well.

        for reading the report, the attribution tag is a tuple of name of file, number of steps into sequence associated with that file, total number of items in the file """
    tmp_midi_path = "_tmp.midi"
    if os.path.exists(tmp_midi_path):
        os.remove(tmp_midi_path)

    f = midi_or_musicjson_file_path
    if f.endswith(".json"):
        music_json_to_midi(f, tmp_midi_path)
    elif f.endswith(".midi") or f.endswith(".mid"):
        shutil.copy2(f, tmp_midi_path)

    p = music21_from_midi(tmp_midi_path)
    dp = get_metadata(p, tmp_midi_path)
    roman_names = dp["roman_names"]
    roman_reduced_names = [x[0] for x in groupby(roman_names)]
    functional_names = dp["functional_names"]
    pitched_functional_names = dp["pitched_functional_names"]

    full_report_status = ""

    roman_names_max_order_ok = checkers["roman_names_checker"].satisfies_max_order(roman_names)
    roman_names_matrix, roman_names_attr = checkers["roman_names_checker"].included_at_index(roman_names, return_attributions=True)
    s = "Roman names checker reports no max-order copying: {}\n".format(roman_names_max_order_ok)
    print(s)
    full_report_status += s

    roman_reduced_names_max_order_ok = checkers["roman_reduced_names_checker"].satisfies_max_order(roman_reduced_names)
    roman_reduced_names_matrix, roman_reduced_names_attr = checkers["roman_reduced_names_checker"].included_at_index(roman_reduced_names, return_attributions=True)
    s = "Roman reduced names checker reports no max-order copying: {}\n".format(roman_reduced_names_max_order_ok)
    print(s)
    full_report_status += s

    functional_names_max_order_ok = checkers["functional_checker"].satisfies_max_order(functional_names)
    functional_names_matrix, functional_names_attr = checkers["functional_checker"].included_at_index(functional_names, return_attributions=True)
    s = "Functional names checker reports no max-order copying: {}\n".format(functional_names_max_order_ok)
    print(s)
    full_report_status += s

    pitched_functional_names_max_order_ok = checkers["pitched_functional_checker"].satisfies_max_order(pitched_functional_names)
    pitched_functional_names_matrix, pitched_functional_names_attr = checkers["pitched_functional_checker"].included_at_index(pitched_functional_names, return_attributions=True)
    s = "Pitched functional names checker reports no max-order copying: {}\n".format(pitched_functional_names_max_order_ok)
    print(s)
    full_report_status += s

    soprano_parts = dp["parts"][0]
    alto_parts = dp["parts"][1]
    tenor_parts = dp["parts"][2]
    bass_parts = dp["parts"][3]

    soprano_pitch_max_order_ok = checkers["soprano_pitch_checker"].satisfies_max_order(soprano_parts)
    soprano_pitch_matrix, soprano_pitch_attr = checkers["soprano_pitch_checker"].included_at_index(soprano_parts, return_attributions=True)
    s = "Soprano pitch checker reports no max-order copying: {}\n".format(soprano_pitch_max_order_ok)
    print(s)
    full_report_status += s


    alto_pitch_max_order_ok = checkers["alto_pitch_checker"].satisfies_max_order(soprano_parts)
    alto_pitch_matrix, alto_pitch_attr = checkers["alto_pitch_checker"].included_at_index(alto_parts, return_attributions=True)
    s = "Alto pitch checker reports no max-order copying: {}\n".format(alto_pitch_max_order_ok)
    print(s)
    full_report_status += s

    tenor_pitch_max_order_ok = checkers["tenor_pitch_checker"].satisfies_max_order(tenor_parts)
    tenor_pitch_matrix, tenor_pitch_attr = checkers["tenor_pitch_checker"].included_at_index(tenor_parts, return_attributions=True)
    s = "Tenor pitch checker reports no max-order copying: {}\n".format(tenor_pitch_max_order_ok)
    print(s)
    full_report_status += s

    bass_pitch_max_order_ok = checkers["bass_pitch_checker"].satisfies_max_order(bass_parts)
    bass_pitch_matrix, bass_pitch_attr = checkers["bass_pitch_checker"].included_at_index(bass_parts, return_attributions=True)
    s = "Bass pitch checker reports no max-order copying: {}\n".format(bass_pitch_max_order_ok)
    print(s)
    full_report_status += s

    assert len(dp["parts"][0]) == len(dp["parts_times"][0])
    soprano_parts_durations = list(zip(dp["parts"][0], dp["parts_times"][0]))

    assert len(dp["parts"][1]) == len(dp["parts_times"][1])
    alto_parts_durations = list(zip(dp["parts"][1], dp["parts_times"][1]))

    assert len(dp["parts"][2]) == len(dp["parts_times"][2])
    tenor_parts_durations = list(zip(dp["parts"][2], dp["parts_times"][2]))

    assert len(dp["parts"][3]) == len(dp["parts_times"][3])
    bass_parts_durations = list(zip(dp["parts"][3], dp["parts_times"][3]))

    soprano_pitch_duration_max_order_ok = checkers["soprano_pitch_duration_checker"].satisfies_max_order(soprano_parts_durations)
    soprano_pitch_duration_matrix, soprano_pitch_duration_attr = checkers["soprano_pitch_duration_checker"].included_at_index(soprano_parts_durations, return_attributions=True)
    s = "Soprano pitch duration checker reports no max-order copying: {}\n".format(soprano_pitch_duration_max_order_ok)
    print(s)
    full_report_status += s

    alto_pitch_duration_max_order_ok = checkers["alto_pitch_duration_checker"].satisfies_max_order(alto_parts_durations)
    alto_pitch_duration_matrix, alto_pitch_duration_attr = checkers["alto_pitch_duration_checker"].included_at_index(alto_parts_durations, return_attributions=True)
    s = "Alto pitch duration checker reports no max-order copying: {}\n".format(alto_pitch_duration_max_order_ok)
    print(s)
    full_report_status += s

    tenor_pitch_duration_max_order_ok = checkers["tenor_pitch_duration_checker"].satisfies_max_order(tenor_parts_durations)
    tenor_pitch_duration_matrix, tenor_pitch_duration_attr = checkers["tenor_pitch_duration_checker"].included_at_index(tenor_parts_durations, return_attributions=True)
    s = "Tenor pitch duration checker reports no max-order copying: {}\n".format(tenor_pitch_duration_max_order_ok)
    print(s)
    full_report_status += s

    bass_pitch_duration_max_order_ok = checkers["bass_pitch_duration_checker"].satisfies_max_order(bass_parts_durations)
    bass_pitch_duration_matrix, bass_pitch_duration_attr = checkers["bass_pitch_duration_checker"].included_at_index(bass_parts_durations, return_attributions=True)
    s = "Bass pitch duration checker reports no max-order copying: {}\n".format(bass_pitch_duration_max_order_ok)
    print(s)
    full_report_status += s

    if os.path.exists(tmp_midi_path):
        os.remove(tmp_midi_path)

    if write_checker_results_to_directory is not None:
        # write out diagnostic reports, copy in relevant songs...
        if not os.path.exists(write_checker_results_to_directory):
            os.mkdir(write_checker_results_to_directory)

        import pprint
        reports_dict = OrderedDict()
        reports_dict["soprano_pitch"] = soprano_pitch_attr
        reports_dict["alto_pitch"] = alto_pitch_attr
        reports_dict["tenor_pitch"] = tenor_pitch_attr
        reports_dict["bass_pitch"] = bass_pitch_attr
        reports_dict["roman_names"] = roman_names_attr
        reports_dict["roman_reduced_names"] = roman_reduced_names_attr
        reports_dict["functional_names"] = functional_names_attr
        reports_dict["pitched_functional_names"] = pitched_functional_names_attr
        reports_dict["soprano_pitch_duration"] = soprano_pitch_duration_attr
        reports_dict["alto_pitch_duration"] = alto_pitch_duration_attr
        reports_dict["tenor_pitch_duration"] = tenor_pitch_duration_attr
        reports_dict["bass_pitch_duration"] = bass_pitch_duration_attr
        for k, v in reports_dict.items():
            output_s = pprint.pformat(v)
            cleaned_infile_path = "_".join("_".join(midi_or_musicjson_file_path.split(os.sep)).split("."))
            subfolder = write_checker_results_to_directory + os.sep + cleaned_infile_path
            if not os.path.exists(subfolder):
                os.mkdir(subfolder)
            subsubfolder = subfolder + os.sep + k
            if not os.path.exists(subsubfolder):
                os.mkdir(subsubfolder)
            output_path = subsubfolder + os.sep + "{}_{}_report.txt".format(cleaned_infile_path, k)
            with open(output_path, "w") as f:
                f.write(output_s)
            # tag is name of file, number of steps into sequence associated with that file, total number of items in the file
            check_file_copy_dir = subsubfolder + os.sep + "file_to_check_against"
            # copy original file in here
            maxorder_match_dir = subsubfolder + os.sep + "maxorder_matches"
            # copy related matches in here
            print("Saving report {}".format(output_path))
            # checkers[k + "_checker"].order_tries[-1].attribution_tags
            # works because this is an ordered dict
            last_key = list(v.keys())[-1]

            if report_maxorder_violations != 0:
                raise ValueError("Currently only supports maxorder_violations = 0 aka report the highest maxorder violations from check")

            if "names" in k:
                # skip fancy plots for 'name' analysis for now...
                if len(v[last_key]) > 0:
                    for ki, vi in v[last_key].items():
                        for el in vi:
                            match_fpath = el[0]
                            match_step = el[1]
                            every_match_file.append(match_fpath)
                            if os.path.exists(match_fpath):
                                if not os.path.exists(check_file_copy_dir):
                                    os.mkdir(check_file_copy_dir)

                                # multiple copies but makes things simpler
                                shutil.copy2(midi_or_musicjson_file_path, check_file_copy_dir)

                                if not os.path.exists(maxorder_match_dir):
                                    os.mkdir(maxorder_match_dir)
                                shutil.copy2(match_fpath, maxorder_match_dir)
                continue

            if "duration" in k:
                match_type = "pitch_duration"
            else:
                match_type = "pitch"

            # if there are violations, iterate them and copy the files to "maxorder_match_dir"
            every_match_file = [midi_or_musicjson_file_path]
            every_match_tree = OrderedDict()
            colorlist = ["yellow", "red", "green", "blue", "darkorange", "purple", "saddlebrown", "coral", "darkcyan"]
            colorlist = cycle(colorlist)
            if len(v[last_key]) > 0:
                report_index_value = 0
                report_index_names = []
                for ki, vi in v[last_key].items():
                    # tuple format is filename, first
                    # sort by file name?
                    for el in vi:
                        match_fpath = el[0]
                        match_step = el[1]
                        every_match_file.append(match_fpath)
                        if os.path.exists(match_fpath):
                            if not os.path.exists(check_file_copy_dir):
                                os.mkdir(check_file_copy_dir)

                            # multiple copies but makes things simpler
                            shutil.copy2(midi_or_musicjson_file_path, check_file_copy_dir)

                            if not os.path.exists(maxorder_match_dir):
                                os.mkdir(maxorder_match_dir)
                            shutil.copy2(match_fpath, maxorder_match_dir)

                            if report_index_value == 0:
                                # 0th report is always for the TRUE data
                                info_tag = ""
                                info_tag += "\n<br>Source report report{}\n<br>Source file: {}\n<br>This is the data we matched against!<br>\n".format(report_index_value, midi_or_musicjson_file_path)
                                # do we need to convert midi to musicjson here

                                # loop through all keys?
                                reduced_match_strings = []
                                for _ki, _vi in v[last_key].items():
                                    if match_type == "pitch_duration":
                                        el_vals = [eval(el) for el in (":".join(_ki[0].split("->"))).split(":")]
                                        step_el = _ki[1]
                                        match_note_names = [(midi_to_name_lookup[el_i[0]] if el_i[0] != 0 else "0", el_i[1]) for el_i in el_vals]
                                        match_str = ":".join([str(e) for e in match_note_names])
                                    elif match_type == "pitch":
                                        el_vals = [int(el) for el in (":".join(_ki[0].split("->"))).split(":")]
                                        step_el = _ki[1]
                                        match_note_names = [midi_to_name_lookup[el_i] if el_i != 0 else "0" for el_i in el_vals]
                                        match_str = ":".join(match_note_names)
                                    if match_str in reduced_match_strings:
                                        # skip strings that we already handled
                                        continue
                                    else:
                                        # which match will always be 0
                                        # write out 1 file for every match type
                                        reduced_match_strings.append(match_str)

                                        write_html_report_for_musicjson(midi_or_musicjson_file_path, subsubfolder,
                                                                        match_info=(match_note_names, 0),
                                                                        match_type=match_type,
                                                                        report_index_value=report_index_value,
                                                                        info_tag=info_tag)
                                        report_index_value += 1
                                        lcl_name = midi_or_musicjson_file_path.split(os.sep)[-1]
                                        report_index_names.append(lcl_name)

                            # now generate html report...
                            match_fname = match_fpath.split(os.sep)[-1]
                            info_tag = ""
                            info_tag += "\n<br>Report report{}\n<br>Checker: {}\n<br>Match sequence: {}\n<br>Matched against file: {}\n<br>Match start step: {}\n<br>Query file: {}\n<br>Query start step: {}\n<br>".format(report_index_value, k, ki[0], match_fname, match_step, midi_or_musicjson_file_path, ki[1])
                            if match_type == "pitch_duration":
                                # ki NOT _ki!
                                el_vals = [eval(el) for el in (":".join(ki[0].split("->"))).split(":")]
                                step_el = ki[1]
                                match_note_names = [(midi_to_name_lookup[el_i[0]] if el_i[0] != 0 else "0", el_i[1]) for el_i in el_vals]
                            elif match_type == "pitch":
                                el_vals = [int(el) for el in (":".join(ki[0].split("->"))).split(":")]
                                step_el = ki[1]
                                match_note_names = [midi_to_name_lookup[el_i] if el_i != 0 else "0" for el_i in el_vals]

                            # find the number of matches for this key, and send "which one" to the html writer... kinda hacky
                            #el_vals = [int(el) for el in (":".join(ki[0].split("->"))).split(":")]
                            #step_el = ki[1]
                            #match_note_names = [midi_to_name_lookup[el_i] if el_i != 0 else "0" for el_i in el_vals]

                            which_match = 0
                            n_matches = 0
                            # first get total number of matches
                            for _el in vi:
                                if el[0] == _el[0]:
                                    n_matches += 1

                            for _el in vi:
                                if el[0] == _el[0]:
                                    if el[1] != _el[1]:
                                        which_match += 1
                                    if el[1] == _el[1]:
                                        break

                            if match_fpath not in every_match_tree:
                                every_match_tree[match_fpath] = []
                            every_match_tree[match_fpath].append([ki[0], ki[1], which_match, n_matches])

                            write_html_report_for_musicjson(match_fpath, subsubfolder,
                                                            match_info=(match_note_names, which_match),
                                                            match_type=match_type,
                                                            report_index_value=report_index_value,
                                                            info_tag=info_tag)
                            report_index_value += 1
                            report_index_names.append(match_fname)
                        else:
                            print("path fail?")
                            from IPython import embed; embed(); raise ValueError()

                # dedupe every_match
                dedupe = []
                for em in every_match_file:
                    if em in dedupe:
                        continue
                    else:
                        dedupe.append(em)
                every_match_file = dedupe

                base64_midis = []
                for em in every_match_file:
                    # write out the html with minor modifications for lane names
                    with open(em, "rb") as f:
                         base64_midi = base64.b64encode(f.read()).decode("utf-8")
                    base64_midis.append(base64_midi)

                all_javascript_note_info = []
                all_match_info = []
                all_marked_l = []
                all_marked_color = []
                for em in every_match_file:
                    if em in every_match_tree:
                        # need to recombine? then make one plot with both colors???

                        for _s in range(len(every_match_tree[em])):
                            match_str = every_match_tree[em][_s][0]
                            if match_type == "pitch_duration":
                                el_vals = [eval(el) for el in (":".join(match_str.split("->"))).split(":")]
                                match_note_names = [(midi_to_name_lookup[el_i[0]] if el_i[0] != 0 else "0", el_i[1]) for el_i in el_vals]
                            elif match_type == "pitch":
                                el_vals = [int(el) for el in (":".join(match_str.split("->"))).split(":")]
                                match_note_names = [midi_to_name_lookup[el_i] if el_i != 0 else "0" for el_i in el_vals]
                            which_match = every_match_tree[em][_s][2]
                            match_info = (match_note_names, which_match)

                            this_color = next(colorlist)
                            r, l, marked_l = extract_plot_info_from_file(em, match_info=match_info, match_type=match_type, match_note_color=this_color)
                            all_javascript_note_info.append(r)

                            all_marked_l.append(marked_l)
                            all_marked_color.append((em, this_color))

                            if match_info is None:
                                match_info = (em, "", 0)
                            else:
                                match_info = (em,) + match_info
                            all_match_info.append(match_info)
                    else:
                        match_info = None
                        this_color = next(colorlist)
                        r, l, marked_l = extract_plot_info_from_file(em, match_info=match_info, match_type=match_type, match_note_color=this_color)
                        all_javascript_note_info.append(r)

                        all_marked_l.append(marked_l)
                        all_marked_color.append((em, this_color))

                        if match_info is None:
                            match_info = (em, "", 0)
                        else:
                            match_info = (em,) + match_info
                        all_match_info.append(match_info)

                if len(every_match_tree) > 0:
                    all_match_keys = OrderedDict()
                    for _k in every_match_tree.keys():
                        for _v in every_match_tree[_k]:
                            all_match_keys[_v[0]] = []
                            for _c in all_marked_color:
                                if _c[0] == _k:
                                    all_match_keys[_v[0]].append(_c)

                    # back match into the source file to make the final plot
                    all_marked_l = []
                    all_marked_l_color = []
                    for _k in all_match_keys.keys():
                        match_str = _k
                        if match_type == "pitch_duration":
                            el_vals = [eval(el) for el in (":".join(match_str.split("->"))).split(":")]
                            match_note_names = [(midi_to_name_lookup[el_i[0]] if el_i[0] != 0 else "0", el_i[1]) for el_i in el_vals]
                        else:
                            el_vals = [int(el) for el in (":".join(match_str.split("->"))).split(":")]
                            match_note_names = [midi_to_name_lookup[el_i] if el_i != 0 else "0" for el_i in el_vals]
                        match_info = (match_note_names, 0)
                        this_color = all_match_keys[_k][0][-1]
                        r, l, marked_l = extract_plot_info_from_file(midi_or_musicjson_file_path, match_info=match_info, match_type=match_type, match_note_color=this_color)
                        all_marked_l.append(marked_l)
                        all_marked_l_color.append(this_color)

                    final_marked_l = []
                    for _n in range(len(all_marked_l[0])):
                        step_colors = []
                        for _j in range(len(all_marked_l)):
                            if all_marked_l[_j][_n] != False:
                                step_colors.append(all_marked_l_color[_j])
                        if len(step_colors) > 0:
                            final_marked_l.append(step_colors[0])
                        else:
                            final_marked_l.append(False)
                    r = make_plot_json(l, notes_to_highlight=final_marked_l, match_note_color="magenta")
                    # replace 0th element with the colored version
                    all_javascript_note_info[0] = r

                w = make_index_html_string2(base64_midis, [em.split(os.sep)[-1] for em in every_match_file], all_javascript_note_info, all_match_info)
                if not os.path.exists(subsubfolder + os.sep + "test_report"):
                    os.makedirs(subsubfolder + os.sep + "test_report")

                with open(subsubfolder + os.sep + "test_report" + os.sep + "0_index.html", "w") as f:
                    f.write(w)

                """
                w = make_index_html_string([("report{}".format(i), report_index_names[i]) for i in range(report_index_value)])
                with open(subsubfolder + os.sep + "test_report" + os.sep + "0_index.html", "w") as f:
                    f.write(w)
                """
        print("report complete")
        from IPython import embed; embed(); raise ValueError()

'''
if __name__ == "__main__":
    # these two samples are identical for a while, not a bad idea to use these to test
    #path1 = "midi_samples_2142/temp20.midi"
    #path2 = "midi_samples_13/temp0.midi"
    #p1 = music21_from_midi(path1)
    #p2 = music21_from_midi(path2)
    #dp1 = get_metadata(p1)
    #dp2 = get_metadata(p2)

    # if it is the first time running this scripts, set assume_cached to false!
    meta = get_music21_bach_metadata(assume_cached=True)
    metafiles = meta["files"]
    skip = False
    if not skip:
        cached_checkers_path = "cached_checkers.pkl"
        if not os.path.exists(cached_checkers_path):
            checkers = build_plagiarism_checkers(metafiles)
            # disabling gc can help speed up pickle
            gc.disable()
            print("Caching checkers to {}".format(cached_checkers_path))
            start = time.time()
            with open(cached_checkers_path, 'wb') as f:
                pickle.dump(checkers, f, protocol=-1)
            end = time.time()
            print("Time to cache {}s".format(end - start))
            gc.enable()
        else:
            print("Loading cached checkers from {}".format(cached_checkers_path))
            start = time.time()
            with open(cached_checkers_path, 'rb') as f:
                checkers = pickle.load(f)
            end = time.time()
            print("Time to load {}s".format(end - start))
    else:
        checkers = build_plagiarism_checkers(metafiles)

    midi_path = "midi_samples/temp0.midi"
    evaluate_midi_against_checkers(midi_path, checkers)
    from IPython import embed; embed(); raise ValueError()

    #corpus = ["random", "randint", "randnight"]
    #max_order = 4
    #checker = MaxOrder(max_order)
    #[checker.insert(c) for c in corpus]

    #checker.insert("purple")

    #a = checker.satisfies_max_order("purp")
    #b = checker.satisfies_max_order("purpt")
    #c = checker.satisfies_max_order("purpl")
    #d = checker.satisfies_max_order("purple")
    #e = checker.satisfies_max_order("purplez")
    #f = checker.satisfies_max_order("purplezz")
    #print(a)
    #print(b)
    #print(c)
    #print(d)
    #print(e)
    #print(f)
    #from IPython import embed; embed(); raise ValueError()
'''
