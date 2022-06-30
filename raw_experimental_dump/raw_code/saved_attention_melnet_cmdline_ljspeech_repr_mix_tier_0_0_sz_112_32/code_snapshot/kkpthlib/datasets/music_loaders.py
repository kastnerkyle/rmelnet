from __future__ import print_function
from music21 import corpus, interval, pitch, converter, note, chord, repeat
from music21.midi import MidiTrack, MidiFile, MidiEvent, DeltaTime
import os
import json
import time
import struct
import fnmatch
import numpy as np
import itertools
import copy
try:
    import cPickle as pickle
except:
    import pickle
import copy
from ..core import get_logger
from ..data import LookupDictionary
from .loaders import get_kkpthlib_dataset_dir
from .midi_instrument_map import midi_instruments_number_to_name
from .midi_instrument_map import midi_instruments_name_to_number
import collections

# https://www.audiolabs-erlangen.de/resources/MIR/FMP/C1/C1S2_MIDI.html
import pretty_midi

logger = get_logger()

def midi_parse_and_save_json(midi_path, fpath, tempo_factor=1, quarter_quantization_factor=0.015625, transpose=0,
                             instrument_names_keeplist=None, instrument_names_droplist=None):
    # quantization is for forming "parts" / "voices"
    piece_container = {}
    piece_container["parts"] = []
    piece_container["parts_times"] = []
    piece_container["parts_cumulative_times"] = []
    piece_container["parts_names"] = []

    midi_data = pretty_midi.PrettyMIDI(midi_path)

    if instrument_names_keeplist is not None:
        assert instrument_names_droplist is None

    if instrument_names_droplist is not None:
        assert instrument_names_keeplist is None

    last = -1
    all_note_list = []
    for instrument in midi_data.instruments:
        if instrument_names_keeplist is not None:
            if instrument.name not in instrument_names_keeplist:
                continue
        elif instrument_names_droplist is not None:
            if instrument.name in instrument_names_droplist:
                continue
        note_list = []
        for note in instrument.notes:
            note_list.append(note)

            start = note.start
            end = note.end
            if end > last:
                last = end
            pitch = note.pitch
            velocity = note.velocity
        all_note_list.append(note_list)

    spq = -1
    qbpm = -1

    # set ppq to 220 to line up with magenta and pretty_midi
    ppq = 220
    #https://stackoverflow.com/questions/2038313/converting-midi-ticks-to-actual-playback-seconds
    piece_container["seconds_per_quarter"] = spq
    piece_container["quarter_beats_per_minute"] = qbpm
    piece_container["pulses_per_quarter"] = ppq

    every_possible_quantization_break = []
    cur = 0.0
    q_v = quarter_quantization_factor
    while True:
        if cur > last + (last * .05):
            break
        every_possible_quantization_break.append(cur)
        cur += q_v

    all_parts = []
    all_parts_times = []
    all_parts_cumulative_times = []

    max_voices = 100
    all_el_lu = collections.OrderedDict()
    active_el_lu = [collections.OrderedDict() for _ in range(max_voices)]
    active_el = [[None] * len(every_possible_quantization_break) for _ in range(max_voices)]
    # quantize and bucket so we can form approximate "voices"
    for _i, ni in enumerate(all_note_list):
        for _j, el_n in enumerate(note_list):

            def quantize_round(raw_v):
                if float(raw_v) % q_v == 0:
                    ind = int(float(raw_v) // q_v)
                else:
                    lower_ind = int(float(raw_v) / q_v)
                    upper_ind = int(float(raw_v) / q_v) + 1
                    nq_ind = float(raw_v) / q_v
                    if abs(nq_ind - lower_ind) < abs(upper_ind - nq_ind):
                        ind = lower_ind
                    else:
                        ind = upper_ind
                return ind

            ind = quantize_round(el_n.start)
            e_ind = quantize_round(el_n.end)

            for _v in range(max_voices):
                empty = True
                overlaps = []
                for _q in range(ind, e_ind + 1):
                    if active_el[_v][_q] != None:
                        empty = False
                    overlaps.append(_q)

                if empty:
                    for _q in range(ind, e_ind + 1):
                        if _q not in all_el_lu:
                            all_el_lu[_q] = []
                        all_el_lu[_q].append((_v, el_n))

                        active_el[_v][_q] = _j
                    active_el_lu[_v][ind] = _j
                    break
                else:
                    if _v == (max_voices - 1):
                        print("all non empty")
                        from IPython import embed; embed(); raise ValueError()

    def _mk_parts(lu, el, all_lu, voice_match):
        this_parts = []
        this_parts_times = []
        this_parts_cumulative_times = []

        last_note_boundary = 0
        for _key in sorted(lu.keys()):
            sounding_at = all_lu[_key]
            sounding_note = [sa for sa in sounding_at if sa[0] == voice_match]
            if len(sounding_note) > 1:
                print("multiple matches????")
                from IPython import embed; embed(); raise ValueError()
            elif len(sounding_note) == 0:
                print("no matches????")
                from IPython import embed; embed(); raise ValueError()
            else:
                sounding_note = sounding_note[0][1]

            if sounding_note.start != last_note_boundary:
                this_parts.append(0)
                this_parts_times.append(sounding_note.start - last_note_boundary)
                this_parts_cumulative_times.append(sounding_note.start)

            time_length = float(sounding_note.end - sounding_note.start) #len(matched) * q_v
            offset = float(sounding_note.start) #matched[0] * q_v
            extent = float(sounding_note.end)
            this_parts.append(sounding_note.pitch + transpose)
            this_parts_times.append(time_length)
            this_parts_cumulative_times.append(extent)
            last_note_boundary = sounding_note.end
        return this_parts, this_parts_times, this_parts_cumulative_times

    for _v in range(max_voices):
        if len(active_el_lu[_v].keys()) > 0:
            this_alt_parts, this_alt_parts_times, this_alt_parts_cumulative_times = _mk_parts(active_el_lu[_v],
                                                                                              active_el[_v],
                                                                                              all_el_lu,
                                                                                              _v)
            all_parts.append(this_alt_parts)
            all_parts_times.append(this_alt_parts_times)
            all_parts_cumulative_times.append(this_alt_parts_cumulative_times)

    piece_container["parts"] = all_parts
    piece_container["parts_times"] = all_parts_times
    piece_container["parts_cumulative_times"] = all_parts_cumulative_times

    j = json.dumps(piece_container, indent=4)
    with open(fpath, "w") as f:
         print(j, file=f)


def music21_parse_and_save_json_squopo(p, fpath, tempo_factor=1):
    piece_container = {}
    piece_container["parts"] = []
    piece_container["parts_times"] = []
    piece_container["parts_cumulative_times"] = []
    piece_container["parts_names"] = []
    # we check for multiple timings when loading files, usually
    spq = p.metronomeMarkBoundaries()[0][-1].secondsPerQuarter()
    qbpm = p.metronomeMarkBoundaries()[0][-1].getQuarterBPM()
    # set ppq to 220 to line up with magenta and pretty_midi
    ppq = 220
    #https://stackoverflow.com/questions/2038313/converting-midi-ticks-to-actual-playback-seconds
    piece_container["seconds_per_quarter"] = spq
    piece_container["quarter_beats_per_minute"] = qbpm
    piece_container["pulses_per_quarter"] = ppq
    for i, pi in enumerate(p.parts):
        piece_container["parts"].append([])
        piece_container["parts_times"].append([])
        piece_container["parts_cumulative_times"].append([])
        piece_container["parts_names"].append(pi.id)
        part = []
        part_time = []
        for n in pi.flat.notesAndRests:
            if n.isRest:
                part.append(0)
            else:
                part.append(n.pitch.midi)
            part_time.append(n.duration.quarterLength * tempo_factor)
        piece_container["parts"][i] += part
        piece_container["parts_times"][i] += part_time
        piece_container["parts_cumulative_times"][i] += list(np.cumsum(part_time))
    j = json.dumps(piece_container, indent=4)
    with open(fpath, "w") as f:
         print(j, file=f)


def music21_parse_and_save_json_qqq(p, fpath, tempo_factor=1, quarter_quantization_factor=0.015625):
    piece_container = {}
    piece_container["parts"] = []
    piece_container["parts_times"] = []
    piece_container["parts_cumulative_times"] = []
    piece_container["parts_names"] = []
    # we check for multiple timings when loading files, usually
    spq = p.metronomeMarkBoundaries()[0][-1].secondsPerQuarter()
    qbpm = p.metronomeMarkBoundaries()[0][-1].getQuarterBPM()
    # set ppq to 220 to line up with magenta and pretty_midi
    ppq = 220
    #https://stackoverflow.com/questions/2038313/converting-midi-ticks-to-actual-playback-seconds
    piece_container["seconds_per_quarter"] = spq
    piece_container["quarter_beats_per_minute"] = qbpm
    piece_container["pulses_per_quarter"] = ppq

    """
    # looks like there should be 5 slots...
    # but get 1, 3, and 6 for verticality?
    for _i, pi in enumerate(p.parts):
        n_sounding_parts = 0
        pt = pi.asTimespans()
        iv = pt.getVerticalityAt(0.0)
        while True:
            max_notes = len(iv.pitchSet)
            n_sounding_parts = max(max_notes, n_sounding_parts)
            iv_new = iv.nextVerticality
            if max_notes > 4:
                pass
                #print("wut")
                #from IPython import embed; embed(); raise ValueError()
            if iv_new is None:
                break
            iv = iv_new
        print(n_sounding_parts)
    """

    last = -1
    for _i, pi in enumerate(p.parts):
        for el_n in pi.flat.notesAndRests:
            last = float(el_n.offset) + float(el_n.duration.quarterLength)

    every_possible_quantization_break = []
    cur = 0.0
    q_v = quarter_quantization_factor
    while True:
        if cur > last:
            break
        every_possible_quantization_break.append(cur)
        cur += q_v

    these_parts = []
    for _i, pi in enumerate(p.parts):
        corrected_parts = []
        for _j, el_n in enumerate(pi.flat.notesAndRests):
            if el_n.duration.quarterLength == 0:
                continue

            # skip chords?
            if el_n.isChord:
                for notes_el in el_n.notes:
                    corrected_parts.append(notes_el)
                    corrected_parts[-1].offset = el_n.offset
            else:
                corrected_parts.append(el_n)
        these_parts.append(corrected_parts)

    all_parts = []
    all_parts_times = []
    all_parts_cumulative_times = []

    max_voices = 12
    all_el_lu = collections.OrderedDict()
    active_el_lu = [collections.OrderedDict() for _ in range(max_voices)]
    active_el = [[None] * len(every_possible_quantization_break) for _ in range(max_voices)]
    for _i, pi in enumerate(these_parts):
        for _j, el_n in enumerate(pi):
            if el_n.duration.quarterLength == 0:
                continue
            if el_n.isRest:
                continue

            # skip chords?
            if el_n.isChord:
                continue

            ind = None
            e_ind = None

            def quantize_round(raw_v):
                if float(raw_v) % q_v == 0:
                    ind = int(float(raw_v) // q_v)
                else:
                    lower_ind = int(float(raw_v) / q_v)
                    upper_ind = int(float(raw_v) / q_v) + 1
                    nq_ind = float(raw_v) / q_v
                    if abs(nq_ind - lower_ind) < abs(upper_ind - nq_ind):
                        ind = lower_ind
                    else:
                        ind = upper_ind
                return ind

            ind = quantize_round(el_n.offset)
            e_ind = quantize_round(float(el_n.offset) + float(el_n.duration.quarterLength))

            for _v in range(max_voices):
                empty = True
                overlaps = []
                for _q in range(ind, e_ind + 1):
                    if active_el[_v][_q] != None:
                        empty = False
                    overlaps.append(_q)

                if empty:
                    for _q in range(ind, e_ind + 1):
                        if _q not in all_el_lu:
                            all_el_lu[_q] = []
                        all_el_lu[_q].append((_v, el_n))

                        active_el[_v][_q] = _j
                    active_el_lu[_v][ind] = _j
                    break
                else:
                    if _v == (max_voices - 1):
                        print("all non empty")
                        from IPython import embed; embed(); raise ValueError()

    # max cardinality
    mk = [len(all_el_lu[k]) for k in all_el_lu.keys()]
    mm = max(mk)
    mok = [k for k in all_el_lu.keys() if len(all_el_lu[k]) == mm]

    def _mk_parts(lu, el, all_lu, voice_match):
        this_parts = []
        this_parts_times = []
        this_parts_cumulative_times = []

        s = 0
        last_boundary = 0
        for _key in sorted(lu.keys()):
            # add rest part 
            if s != _key:
                last_boundary = s
                while True:
                    if s < _key:
                        s += 1
                    else:
                        break
                matched = [m for m in range(last_boundary, s)]
                # what is with all these super short rests
                time_length = len(matched) * q_v
                offset = matched[0] * q_v
                extent = offset + time_length
                #this_parts.append(0)
                #this_parts_times.append(time_length)
                #this_parts_cumulative_times.append(extent)

            if s != _key:
                s = _key

            while True:
                if s >= len(el):
                    break

                if el[s] == lu[_key]:
                    s += 1
                else:
                    break
            matched = [m for m in range(_key, s)]
            if len(matched) == 0:
                print("0 length note dur")
                from IPython import embed; embed(); raise ValueError()
            sounding_at = all_lu[_key]
            sounding_note = [sa for sa in sounding_at if sa[0] == voice_match]
            if len(sounding_note) > 1:
                print("multiple matches????")
                from IPython import embed; embed(); raise ValueError()
            elif len(sounding_note) == 0:
                print("no matches????")
                from IPython import embed; embed(); raise ValueError()
            else:
                sounding_note = sounding_note[0][1]

            time_length = float(sounding_note.duration.quarterLength) #len(matched) * q_v
            offset = float(sounding_note.offset) #matched[0] * q_v
            extent = offset + time_length
            this_parts.append(sounding_note.pitch.midi)
            this_parts_times.append(time_length)
            this_parts_cumulative_times.append(extent)
        return this_parts, this_parts_times, this_parts_cumulative_times

    for _v in range(max_voices):
        if len(active_el_lu[_v].keys()) > 0:
            this_alt_parts, this_alt_parts_times, this_alt_parts_cumulative_times = _mk_parts(active_el_lu[_v],
                                                                                              active_el[_v],
                                                                                              all_el_lu,
                                                                                              _v)
            all_parts.append(this_alt_parts)
            all_parts_times.append(this_alt_parts_times)
            all_parts_cumulative_times.append(this_alt_parts_cumulative_times)

    piece_container["parts"] = all_parts
    piece_container["parts_times"] = all_parts_times
    piece_container["parts_cumulative_times"] = all_parts_cumulative_times

    j = json.dumps(piece_container, indent=4)
    with open(fpath, "w") as f:
         print(j, file=f)


def music21_parse_and_save_json_squog(p, fpath, tempo_factor=1, quarter_quantization_factor=0.015625):
    piece_container = {}
    piece_container["parts"] = []
    piece_container["parts_times"] = []
    piece_container["parts_cumulative_times"] = []
    piece_container["parts_names"] = []
    # we check for multiple timings when loading files, usually
    spq = p.metronomeMarkBoundaries()[0][-1].secondsPerQuarter()
    qbpm = p.metronomeMarkBoundaries()[0][-1].getQuarterBPM()
    # set ppq to 220 to line up with magenta and pretty_midi
    ppq = 220
    #https://stackoverflow.com/questions/2038313/converting-midi-ticks-to-actual-playback-seconds
    piece_container["seconds_per_quarter"] = spq
    piece_container["quarter_beats_per_minute"] = qbpm
    piece_container["pulses_per_quarter"] = ppq

    """
    # looks like there should be 5 slots...
    # but get 1, 3, and 6 for verticality?
    for _i, pi in enumerate(p.parts):
        n_sounding_parts = 0
        pt = pi.asTimespans()
        iv = pt.getVerticalityAt(0.0)
        while True:
            max_notes = len(iv.pitchSet)
            n_sounding_parts = max(max_notes, n_sounding_parts)
            iv_new = iv.nextVerticality
            if max_notes > 4:
                pass
                #print("wut")
                #from IPython import embed; embed(); raise ValueError()
            if iv_new is None:
                break
            iv = iv_new
        print(n_sounding_parts)
    """

    last = -1
    for _i, pi in enumerate(p.parts):
        for el_n in pi.flat.notesAndRests:
            last = float(el_n.offset) + float(el_n.duration.quarterLength)

    every_possible_quantization_break = []
    cur = 0.0
    q_v = quarter_quantization_factor
    while True:
        if cur > last:
            break
        every_possible_quantization_break.append(cur)
        cur += q_v

    these_parts = []
    for _i, pi in enumerate(p.parts):
        corrected_parts = []
        for _j, el_n in enumerate(pi.flat.notesAndRests):
            if el_n.duration.quarterLength == 0:
                continue

            # skip chords?
            if el_n.isChord:
                for notes_el in el_n.notes:
                    corrected_parts.append(notes_el)
                    corrected_parts[-1].offset = el_n.offset
            else:
                corrected_parts.append(el_n)
        these_parts.append(corrected_parts)

    all_parts = []
    all_parts_times = []
    all_parts_cumulative_times = []

    max_voices = 12
    all_el_lu = collections.OrderedDict()
    active_el_lu = [collections.OrderedDict() for _ in range(max_voices)]
    active_el = [[None] * len(every_possible_quantization_break) for _ in range(max_voices)]
    for _i, pi in enumerate(these_parts):
        for _j, el_n in enumerate(pi):
            if el_n.duration.quarterLength == 0:
                continue
            if el_n.isRest:
                continue

            # skip chords?
            if el_n.isChord:
                continue

            ind = None
            e_ind = None

            def quantize_round(raw_v):
                if float(raw_v) % q_v == 0:
                    ind = int(float(raw_v) // q_v)
                else:
                    lower_ind = int(float(raw_v) / q_v)
                    upper_ind = int(float(raw_v) / q_v) + 1
                    nq_ind = float(raw_v) / q_v
                    if abs(nq_ind - lower_ind) < abs(upper_ind - nq_ind):
                        ind = lower_ind
                    else:
                        ind = upper_ind
                return ind

            ind = quantize_round(el_n.offset)
            e_ind = quantize_round(float(el_n.offset) + float(el_n.duration.quarterLength))

            for _v in range(max_voices):
                empty = True
                overlaps = []
                for _q in range(ind, e_ind + 1):
                    if active_el[_v][_q] != None:
                        empty = False
                    overlaps.append(_q)

                if empty:
                    for _q in range(ind, e_ind + 1):
                        if _q not in all_el_lu:
                            all_el_lu[_q] = []
                        all_el_lu[_q].append((_v, el_n))

                        active_el[_v][_q] = _j
                    active_el_lu[_v][ind] = _j
                    break
                else:
                    if _v == (max_voices - 1):
                        print("all non empty")
                        from IPython import embed; embed(); raise ValueError()

    # max cardinality
    mk = [len(all_el_lu[k]) for k in all_el_lu.keys()]
    mm = max(mk)
    mok = [k for k in all_el_lu.keys() if len(all_el_lu[k]) == mm]

    def _mk_parts(lu, el, all_lu, voice_match):
        this_parts = []
        this_parts_times = []
        this_parts_cumulative_times = []

        s = 0
        last_boundary = 0
        for _key in sorted(lu.keys()):
            # add rest part 
            if s != _key:
                last_boundary = s
                while True:
                    if s < _key:
                        s += 1
                    else:
                        break
                matched = [m for m in range(last_boundary, s)]
                # what is with all these super short rests
                time_length = len(matched) * q_v
                offset = matched[0] * q_v
                extent = offset + time_length
                this_parts.append(0)
                this_parts_times.append(time_length)
                this_parts_cumulative_times.append(extent)

            if s != _key:
                s = _key

            while True:
                if s >= len(el):
                    break

                if el[s] == lu[_key]:
                    s += 1
                else:
                    break
            matched = [m for m in range(_key, s)]
            if len(matched) == 0:
                print("0 length note dur")
                from IPython import embed; embed(); raise ValueError()
            time_length = len(matched) * q_v
            offset = matched[0] * q_v
            extent = offset + time_length
            sounding_at = all_lu[_key]
            sounding_note = [sa for sa in sounding_at if sa[0] == voice_match]
            if len(sounding_note) > 1:
                print("multiple matches????")
                from IPython import embed; embed(); raise ValueError()
            elif len(sounding_note) == 0:
                print("no matches????")
                from IPython import embed; embed(); raise ValueError()
            else:
                sounding_note = sounding_note[0][1]
            this_parts.append(sounding_note.pitch.midi)
            this_parts_times.append(time_length)
            this_parts_cumulative_times.append(extent)
        return this_parts, this_parts_times, this_parts_cumulative_times

    for _v in range(max_voices):
        if len(active_el_lu[_v].keys()) > 0:
            this_alt_parts, this_alt_parts_times, this_alt_parts_cumulative_times = _mk_parts(active_el_lu[_v],
                                                                                              active_el[_v],
                                                                                              all_el_lu,
                                                                                              _v)
            all_parts.append(this_alt_parts)
            all_parts_times.append(this_alt_parts_times)
            all_parts_cumulative_times.append(this_alt_parts_cumulative_times)

    piece_container["parts"] = all_parts
    piece_container["parts_times"] = all_parts_times
    piece_container["parts_cumulative_times"] = all_parts_cumulative_times

    j = json.dumps(piece_container, indent=4)
    with open(fpath, "w") as f:
         print(j, file=f)


def music21_parse_and_save_json_squig(p, fpath, tempo_factor=1, max_possible_voices=5, quarter_quantization_factor=.041666666):
    piece_container = {}
    piece_container["parts"] = []
    piece_container["parts_times"] = []
    piece_container["parts_cumulative_times"] = []
    piece_container["parts_names"] = []
    # we check for multiple timings when loading files, usually
    spq = p.metronomeMarkBoundaries()[0][-1].secondsPerQuarter()
    qbpm = p.metronomeMarkBoundaries()[0][-1].getQuarterBPM()
    # set ppq to 220 to line up with magenta and pretty_midi
    ppq = 220
    #https://stackoverflow.com/questions/2038313/converting-midi-ticks-to-actual-playback-seconds
    piece_container["seconds_per_quarter"] = spq
    piece_container["quarter_beats_per_minute"] = qbpm
    piece_container["pulses_per_quarter"] = ppq

    """
    # looks like there should be 5 slots...
    # but get 1, 3, and 6 for verticality?
    for _i, pi in enumerate(p.parts):
        n_sounding_parts = 0
        pt = pi.asTimespans()
        iv = pt.getVerticalityAt(0.0)
        while True:
            max_notes = len(iv.pitchSet)
            n_sounding_parts = max(max_notes, n_sounding_parts)
            iv_new = iv.nextVerticality
            if max_notes > 4:
                pass
                #print("wut")
                #from IPython import embed; embed(); raise ValueError()
            if iv_new is None:
                break
            iv = iv_new
        print(n_sounding_parts)
    """

    last = -1
    for _i, pi in enumerate(p.parts):
        for el_n in pi.flat.notesAndRests:
            last = float(el_n.offset) + float(el_n.duration.quarterLength)

    every_possible_quantization_break = []
    cur = 0.0
    q_v = quarter_quantization_factor
    while True:
        if cur > last:
            break
        every_possible_quantization_break.append(cur)
        cur += q_v

    all_parts = []
    all_parts_times = []
    all_parts_cumulative_times = []

    leftovers = []

    these_parts = []
    for _i, pi in enumerate(p.parts):
        corrected_parts = []
        for _j, el_n in enumerate(pi.flat.notesAndRests):
            if el_n.duration.quarterLength == 0:
                continue

            # skip chords?
            if el_n.isChord:
                for notes_el in el_n.notes:
                    corrected_parts.append(notes_el)
                    corrected_parts[-1].offset = el_n.offset
            else:
                corrected_parts.append(el_n)
        these_parts.append(corrected_parts)

    # should we even use snap gravity
    snap_gravity = [-1] * len(every_possible_quantization_break)

    # for now lets just make a ton of parts and see how many there are...
    carryover_all_el_lu = None
    carryover_active_el_lu = None
    carryover_active_el = None
    for _i, pi in enumerate(these_parts):
        # attempt to "pack" each voice using leftovers from the last
        if carryover_all_el_lu is not None:
            all_el_lu = carryover_all_el_lu
            active_el_lu = carryover_active_el_lu
            active_el = carryover_active_el
        else:
            all_el_lu = collections.OrderedDict()
            active_el_lu = collections.OrderedDict()
            active_el = [None] * len(every_possible_quantization_break)

        max_voices = 5
        alt_active_el_lu = [collections.OrderedDict() for _ in range(max_voices)]
        alt_active_el = [[None] * len(every_possible_quantization_break) for _ in range(max_voices)]

        for _j, el_n in enumerate(pi):
            if el_n.duration.quarterLength == 0:
                continue
            if el_n.isRest:
                continue

            # skip chords?
            if el_n.isChord:
                continue

            all_el_lu[_j] = el_n

            ind = None
            e_ind = None
            if float(el_n.offset) % q_v == 0:
                ind = int(float(el_n.offset) // q_v)

            if (float(el_n.offset) + float(el_n.duration.quarterLength)) % q_v == 0:
                e_ind = int((float(el_n.offset) + float(el_n.duration.quarterLength)) // q_v)

            if ind is not None and e_ind is not None:
                skip = False
                for _q in range(ind, e_ind + 1):
                    if active_el[_q] is not None:
                        skip = True
                if not skip:
                    for _q in range(ind, e_ind + 1):
                        active_el[_q] = _j
                    active_el_lu[ind] = _j

        for _j, el_n in enumerate(pi):
            if el_n.duration.quarterLength == 0:
                continue
            if el_n.isRest:
                continue
            if _j in active_el:
                continue

            # skip chords?
            if el_n.isChord:
                continue

            ind = int(float(el_n.offset) // q_v)
            e_ind = int((float(el_n.offset) + float(el_n.duration.quarterLength)) // q_v)

            # if all the slots are empty just smash it in there
            empty = True
            overlaps = []
            for _q in range(ind, e_ind + 1):
                if active_el[_q] != None:
                    empty = False
                    overlaps.append(_q)

            '''
            if _i > 1:
                if _j > 1:
                    print("got here")
                    from IPython import embed; embed(); raise ValueError()
            '''

            if empty:
                for _q in range(ind, e_ind + 1):
                    active_el[_q] = _j
                active_el_lu[ind] = _j
            else:
                multiple = False
                if len(overlaps) <= 2:
                    if overlaps[0] == ind:
                        # if theres only a 1 or 2 step overlap at the start, and the overlapper is a multistep note, just overwrrite (prefer to preserve onsets to endings) 

                        st = e_ind + 1
                        if active_el[e_ind] != None:
                            st = e_ind

                        if active_el[ind] == active_el[ind - 1]:
                            for _q in range(ind, st):
                                active_el[_q] = _j
                            active_el_lu[ind] = _j
                    elif overlaps[0] == e_ind:
                        # if there's only one overlap at the end, just truncate
                        for _q in range(ind, e_ind):
                            active_el[_q] = _j
                        active_el_lu[ind] = _j
                    else:
                        multiple = True

                if multiple:
                    # need to use multiple active_els now, no way around it
                    # there was an overlap
                    filled = False
                    for _v in range(max_voices):
                        empty = True
                        overlaps = []
                        for _q in range(ind, e_ind + 1):
                            if alt_active_el[_v][_q] != None:
                                empty = False
                                overlaps.append(_q)

                        if empty:
                            for _q in range(ind, e_ind + 1):
                                alt_active_el[_v][_q] = _j
                            alt_active_el_lu[_v][ind] = _j
                            filled = True
                            break
                        else:
                            if len(overlaps) == 1:
                                if overlaps[0] == ind:
                                    # if theres only a 1 step overlap at the start, and the overlapper is a multistep note, just overwrrite (prefer to preserve onsets to endings) 

                                    st = e_ind + 1
                                    if alt_active_el[_v][ind] == alt_active_el[_v][ind - 1]:
                                        for _q in range(ind, st):
                                            alt_active_el[_v][_q] = _j
                                        alt_active_el_lu[_v][ind] = _j
                                    filled = True
                                    break
                                else:
                                    print("secondary non start but singular overlap")
                                    from IPython import embed; embed(); raise ValueError()
                            else:
                                pass
                                # wasn't empty, continue...
                                #print("secondary multi overlap")
                                #from IPython import embed; embed(); raise ValueError()

                    if filled == False:
                        print("none open?")
                        from IPython import embed; embed(); raise ValueError()

        '''
        # now that we have a sequential part,
        # stick into a parts, parts_times, parts_cumulative_times structure with rests to fill the gaps
        this_parts = []
        this_parts_times = []
        this_parts_cumulative_times = []

        s = 0
        last_boundary = 0
        for _key in sorted(active_el_lu.keys()):
            # add rest part
            if s != _key:
                last_boundary = s
                while True:
                    if s < _key:
                        s += 1
                    else:
                        break
                matched = [m for m in range(last_boundary, s)]
                if len(matched) <= 2:
                    # 2 means no rest <= 16th
                    # just stretch? instead of all these minimal time rests
                    this_parts_times[-1] += len(matched) * q_v
                    this_parts_cumulative_times[-1] += len(matched) * q_v
                elif len(matched) == 0:
                    print("0 length match rest")
                    from IPython import embed; embed(); raise ValueError()
                    pass
                else:
                    # what is with all these super short rests
                    time_length = len(matched) * q_v
                    offset = matched[0] * q_v
                    extent = offset + time_length
                    this_parts.append(0)
                    this_parts_times.append(time_length)
                    this_parts_cumulative_times.append(extent)

            if s != _key:
                s = _key

            while True:
                if active_el[s] == active_el_lu[_key]:
                    s += 1
                else:
                    break
            matched = [m for m in range(_key, s)]
            if len(matched) == 0:
                print("0 length note dur")
                from IPython import embed; embed(); raise ValueError()
            time_length = len(matched) * q_v
            offset = matched[0] * q_v
            extent = offset + time_length
            this_parts.append(all_el_lu[active_el_lu[_key]].pitch.midi)
            this_parts_times.append(time_length)
            this_parts_cumulative_times.append(extent)

        '''

        def _mk_parts(lu, el, all_lu):
            this_parts = []
            this_parts_times = []
            this_parts_cumulative_times = []

            s = 0
            last_boundary = 0
            for _key in sorted(lu.keys()):
                # add rest part 
                if s != _key:
                    last_boundary = s
                    while True:
                        if s < _key:
                            s += 1
                        else:
                            break
                    matched = [m for m in range(last_boundary, s)]
                    if len(matched) <= 2:
                        # 2 means no rest <= 16th
                        # just stretch? instead of all these minimal time rests
                        this_parts_times[-1] += len(matched) * q_v
                        this_parts_cumulative_times[-1] += len(matched) * q_v
                    elif len(matched) == 0:
                        print("0 length match rest")
                        from IPython import embed; embed(); raise ValueError()
                        pass
                    else:
                        # what is with all these super short rests
                        time_length = len(matched) * q_v
                        offset = matched[0] * q_v
                        extent = offset + time_length
                        this_parts.append(0)
                        this_parts_times.append(time_length)
                        this_parts_cumulative_times.append(extent)

                if s != _key:
                    s = _key

                while True:
                    if s >= len(el):
                        break

                    if el[s] == lu[_key]:
                        s += 1
                    else:
                        break
                matched = [m for m in range(_key, s)]
                if len(matched) == 0:
                    print("0 length note dur")
                    from IPython import embed; embed(); raise ValueError()
                time_length = len(matched) * q_v
                offset = matched[0] * q_v
                extent = offset + time_length
                this_parts.append(all_lu[lu[_key]].pitch.midi)
                this_parts_times.append(time_length)
                this_parts_cumulative_times.append(extent)
            return this_parts, this_parts_times, this_parts_cumulative_times

        this_parts, this_parts_times, this_parts_cumulative_times = _mk_parts(active_el_lu, active_el, all_el_lu)
        all_parts.append(this_parts)
        all_parts_times.append(this_parts_times)
        all_parts_cumulative_times.append(this_parts_cumulative_times)

        # see if there's any non-zero length alt keys
        n_leftover = 0
        for _v in range(max_voices):
            if len(alt_active_el_lu[_v].keys()) > 0:
                this_alt_parts, this_alt_parts_times, this_alt_parts_cumulative_times = _mk_parts(alt_active_el_lu[_v],
                                                                                                  alt_active_el[_v],
                                                                                                  all_el_lu)
                all_parts.append(this_alt_parts)
                all_parts_times.append(this_alt_parts_times)
                all_parts_cumulative_times.append(this_alt_parts_cumulative_times)
                # for now, don't carryover / collapse
                #n_leftover += 1

        if n_leftover > 1:
            print("more than one leftover alt there")
            from IPython import embed; embed(); raise ValueError()

            this_alt_parts, this_alt_parts_times, this_alt_parts_cumulative_times = _mk_parts(alt_active_el_lu[_v],
                                                                                              alt_active_el[_v],
                                                                                              all_el_lu)
        elif n_leftover == 1:
            carryover_all_el_lu = copy.deepcopy(alt_active_el_lu[0])
            carryover_active_el_lu = copy.deepcopy(alt_active_el_lu[0])
            carryover_active_el = copy.deepcopy(alt_active_el[0])
        elif n_leftover == 0:
            carryover_all_el_lu = None
            carryover_active_el_lu = None
            carryover_active_el = None

    piece_container["parts"] = all_parts
    piece_container["parts_times"] = all_parts_times
    piece_container["parts_cumulative_times"] = all_parts_cumulative_times

    j = json.dumps(piece_container, indent=4)
    with open(fpath, "w") as f:
         print(j, file=f)


def music21_parse_and_save_json(p, fpath, tempo_factor=1):
    print("eeee")
    from IPython import embed; embed(); raise ValueError()
    piece_container = {}
    piece_container["parts"] = []
    piece_container["parts_times"] = []
    piece_container["parts_cumulative_times"] = []
    piece_container["parts_names"] = []
    # we check for multiple timings when loading files, usually
    spq = p.metronomeMarkBoundaries()[0][-1].secondsPerQuarter()
    qbpm = p.metronomeMarkBoundaries()[0][-1].getQuarterBPM()
    # set ppq to 220 to line up with magenta and pretty_midi
    ppq = 220
    #https://stackoverflow.com/questions/2038313/converting-midi-ticks-to-actual-playback-seconds
    piece_container["seconds_per_quarter"] = spq
    piece_container["quarter_beats_per_minute"] = qbpm
    piece_container["pulses_per_quarter"] = ppq
    for i, pi in enumerate(p.parts):
        piece_container["parts"].append([])
        piece_container["parts_times"].append([])
        piece_container["parts_cumulative_times"].append([])
        piece_container["parts_names"].append(pi.id)
        part = []
        part_time = []
        for n in pi.flat.notesAndRests:
            if n.isRest:
                part.append(0)
            else:
                part.append(n.pitch.midi)
            part_time.append(n.duration.quarterLength * tempo_factor)
        piece_container["parts"][i] += part
        piece_container["parts_times"][i] += part_time
        piece_container["parts_cumulative_times"][i] += list(np.cumsum(part_time))
    j = json.dumps(piece_container, indent=4)
    with open(fpath, "w") as f:
         print(j, file=f)


def check_fetch_jsb_chorales(only_pieces_with_n_voices=[4], verbose=True):
    """
    if os.path.exists(get_kkpthlib_dataset_dir() + os.sep + "jsb_chorales_json"):
        dataset_path = get_kkpthlib_dataset_dir("jsb_chorales_json")
        # if the dataset already exists, assume the preprocessing is already complete
        return dataset_path
    """
    dataset_path = get_kkpthlib_dataset_dir("jsb_chorales_json")

    all_bach_paths = corpus.getComposer('bach')

    if verbose:
        logger.info("JSB Chorales not yet cached, processing...")
        logger.info("Total number of Bach pieces to process from music21: {}".format(len(all_bach_paths)))

    all_file_keys = []
    all_file_information = collections.OrderedDict()

    for it, p_bach in enumerate(all_bach_paths):
        if "riemenschneider" in str(p_bach):
            # skip certain files we don't care about
            continue
            # WANT analysis=True but breaks expander/repeat finder
        el = corpus.parse(str(p_bach))
        # do we have to manually expand/repeat the analysis part of the stream to match other parts?
        #lu = str(el.filePath).split("/")[-1]
        #if lu != "bwv347.mxl":
        #    continue
        has_expansion = [False] * len(el.parts)
        for _ii in range(len(el.parts)):
            try:
                # this errors for the last part, which corresponds to the textual annotations
                new_p = el.parts[_ii].expandRepeats()
                has_expansion[_ii] = True
            except:
                # the last part is the "analysis=True", and the repeat Expander doesnt directly handle it
                # but in all testing, analysis was the same length as the other 4 parts so we can manually handle it
                # (? double check this with an assert on the last part length)
                if _ii == (len(el.parts) - 1) and all(has_expansion[:_ii]):
                    has_expansion[_ii] = True
                continue

        global_measure_map = None
        for _ii in range(len(el.parts)):
            try:
                new_p = el.parts[_ii].expandRepeats()
            except:
                continue
            all_meas_hashes = []
            for meas in el.parts[_ii].recurse().getElementsByClass('Measure'):
                #Expander.measureMap gives the wrong indices on bwv347
                # believe to be due to this line https://github.com/cuthbertLab/music21/blob/master/music21/repeat.py#L850
                # plus the fact that bwv347 has measure named 4 and 4a but the measureMap logic seems to auto add suffix "a" to repeats
                # meaning 0 1 2 3 4 0 1 2 3 4 5 becomes 0 1 2 3 4 0a 1a 2a 3a 4a 5 but 5 is already named 4a in the XML! 

                # need to do similar logic by hand, using hashes of notes
                all_note_hashes = []
                for note in meas.recurse().getElementsByClass("Note"):
                    n_hash = hash((note.pitch, note.duration.quarterLength, note.offset))
                    all_note_hashes.append(n_hash)
                all_meas_hashes.append(tuple(all_note_hashes))

            # build lookup table which holds the index of the FIRST occurence of a measure hash
            # here we assume that if 2 measures hash the same, using the first one is fine
            # theorectically could have a hash collision but seems.... unlikely
            h_lu = {}
            for m, m_hash in enumerate(all_meas_hashes):
                if m_hash not in h_lu:
                    h_lu[m_hash] = [m]
                else:
                    h_lu[m_hash].append(m)

            # measure hashes for expansion
            all_meas_hashes_exp = []
            for meas in el.parts[_ii].expandRepeats().recurse().getElementsByClass('Measure'):
                #Expander.measureMap gives the wrong indices on bwv347
                #need to do similar logic by hand, using hashes of notes and then hashes of the measure
                # we go part by part because analysis=True in the iterator breaks expandReapeats() at the score level
                all_note_hashes = []
                for note in meas.recurse().getElementsByClass("Note"):
                    n_hash = hash((note.pitch, note.duration.quarterLength, note.offset))
                    all_note_hashes.append(n_hash)
                all_meas_hashes_exp.append(tuple(all_note_hashes))

            match_indices = []
            for m1, m_hash1 in enumerate(all_meas_hashes_exp):
                if len(h_lu[m_hash1]) == 1:
                    match_indices.append(h_lu[m_hash1][0])
                else:
                    match_indices.append(h_lu[m_hash1])

            if global_measure_map is None:
                global_measure_map = match_indices
            else:
                assert len(match_indices) == len(global_measure_map)
                # merge new estimate with old global measure map
                # theres only 1 choice in the global, be sure the 1 global choice 
                # exists in our new match set, then set it.
                # if both have multiple, use all the keys that exist in *both* sets
                # think of this like a superposition of states, we loop through
                # reducing or eliminating them until only 1 option remains
                for ttt in range(len(match_indices)):
                    if hasattr(global_measure_map[ttt], "__len__"):
                        subset = []
                        if hasattr(match_indices[ttt], "__len__"):
                            for mik in match_indices[ttt]:
                                if mik in global_measure_map[ttt]:
                                    subset.append(mik)
                            global_measure_map[ttt] = subset
                        else:
                            assert match_indices[ttt] in global_measure_map[ttt]
                            global_measure_map[ttt] = match_indices[ttt]
                    else:
                        if hasattr(match_indices[ttt], "__len__"):
                            assert global_measure_map[ttt] in match_indices[ttt]
                        else:
                            assert global_measure_map[ttt] == match_indices[ttt]

        if global_measure_map is not None:
            # if global measure map is None then we don't need to check for repeats etc
            # now that global measure map is formed we have 1 more step
            # there may still be multi-key matches in the list
            # collapse them, preferring keys which "continue" a sequence
            # eg [0 1 2 [3, 12] 4 5] prefers 3 over 12
            # preference for key which is 1 away from both left and right
            # then preference for key which is 1 away from left
            # then preference for key 1 away from right
            # however, if none of the keys are within 1 of *either* of the neighbors, take the lowest value
            final_gmmap = []
            # do 2 passes, fill in all the easy ones then go back and fill the multi-key matches
            for gmmk in global_measure_map:
                if not hasattr(gmmk, "__len__"):
                    final_gmmap.append(gmmk)
                else:
                    final_gmmap.append(None)

            for _n, gmmk in enumerate(global_measure_map):
                if final_gmmap[_n] is None:
                    left_ok = False
                    right_ok = False
                    # use shortcuts to prevent some if statements
                    if _n >= 1:
                        left_ok = True
                    if _n <= (len(global_measure_map) - 2):
                        right_ok = True

                    # default, lowest prio choice is just the min
                    if final_gmmap[_n] is None:
                        final_gmmap[_n] = min(global_measure_map[_n])

                    # if we can make a right hand match, check it and set
                    if right_ok and final_gmmap[_n + 1] is not None:
                        if final_gmmap[_n + 1] - 1 in gmmk:
                            final_gmmap[_n] = final_gmmap[_n + 1] - 1

                     # if we can make a left hand match, check it and set
                    if left_ok and final_gmmap[_n - 1] is not None:
                        if final_gmmap[_n - 1] + 1 in gmmk:
                            final_gmmap[_n] = final_gmmap[_n - 1] + 1

                    # if we can make both a left and right hand match, check and set
                    if left_ok and right_ok and final_gmmap[_n - 1] is not None and final_gmmap[_n + 1] is not None:
                        if final_gmmap[_n - 1] + 1 == final_gmmap[_n + 1] - 1:
                            if final_gmmap[_n - 1] + 1 in gmmk:
                                final_gmmap[_n] = final_gmmap[_n - 1] + 1

            # be sure there are no None values left in final_gmmap
            assert all([a != None for a in final_gmmap])
            # now assign it
            global_measure_map = final_gmmap

        try:
            last = len(el.parts[0].expandRepeats().recurse().getElementsByClass('Measure'))
        except:
            last = len(el.parts[0].recurse().getElementsByClass('Measure'))
        #if lu == "bwv347.mxl":
        #    el.show("text")
        #    print(global_measure_map)
        #    raise ValueError()

        if global_measure_map is None:
            global_measure_map = list(range(last + 1))

        # sequentially roman numeral, measure, starting offset, duration
        romans = []
        # sequentially key change, measure, starting offset
        keys = []
        # sequentially time signature, measure, starting offset, numerator, denominator
        time_signatures = []
        # note value, note name with octave, measure, starting offset, duration
        notes_and_durations = {"Soprano": [],
                               "Alto": [],
                               "Tenor": [],
                               "Bass": []}
        for _n, j in enumerate(global_measure_map):
            romans.append([])
            keys.append([])
            time_signatures.append([])
            for key in ["Soprano", "Alto", "Tenor", "Bass"]:
                notes_and_durations[key].append([])
            # check if measure-by-index and .measures are the same?
            # if they are different we probably need to do something special for time signatures / romans
            for _ii in range(len(el.parts)):
                try:
                    p = list(el.parts[_ii].recurse().getElementsByClass("Measure"))[j]
                    p_class = el.parts[_ii]
                except:
                    if j >= len(list(el.parts[_ii].recurse().getElementsByClass("Measure"))) - 1:
                        continue
                    if _ii >= len(el.parts) - 1:
                        continue
                # start with crucial structural information - time signature, roman annotations, keys
                # might need to do something very weird with measures with stuff like 4 4a since the annotations arent split in this way...
                if str(p_class).split(" ")[-1].split(">")[0] not in ["Soprano", "Alto", "Tenor", "Bass"]:
                    for pi in p.flat:
                        if "RomanNumeral" in pi.classes and "||" not in str(pi):
                            romans[-1].append((str(pi).split(" ")[1], _n, j, pi.offset, pi.duration.quarterLength))
                        if "Key" in pi.classes:
                            keys[-1].append((str(pi), _n, j, pi.offset))
                        if "TimeSignature" in pi.classes:
                            time_signatures[-1].append((str(pi), _n, j, pi.offset, pi.numerator, pi.denominator))
            for _ii in range(len(el.parts)):
                try:
                    p = list(el.parts[_ii].recurse().getElementsByClass("Measure"))[j]
                    p_class = el.parts[_ii]
                except:
                    if j >= len(list(el.parts[_ii].recurse().getElementsByClass("Measure"))) - 1:
                        continue
                    if _ii >= len(el.parts) - 1:
                        continue
                p = list(el.parts[_ii].recurse().getElementsByClass("Measure"))[j]
                p_class = el.parts[_ii]
                if str(p_class).split(" ")[-1].split(">")[0] in ["Soprano", "Alto", "Tenor", "Bass"]:
                    key = str(p_class).split(" ")[-1][:-1]
                    for pi in p.flat:
                        if "Note" in pi.classes or "Rest" in pi.classes:
                            if "Note" in pi.classes:
                                notes_and_durations[key][-1].append((pi.pitch.midi, pi.nameWithOctave, _n, j, pi.offset, pi.duration.quarterLength))
                            else:
                                notes_and_durations[key][-1].append((0, "R", _n, j, pi.offset, pi.duration.quarterLength))
        '''
        for _n, j in enumerate(global_measure_map):
            romans.append([])
            keys.append([])
            time_signatures.append([])
            for key in ["Soprano", "Alto", "Tenor", "Bass"]:
                notes_and_durations[key].append([])
            for p in el.measure(j).parts:
                # start with crucial structural information - time signature, roman annotations, keys
                if str(p).split(" ")[-1].split(">")[0] not in ["Soprano", "Alto", "Tenor", "Bass"]:
                    for pi in p.flat:
                        if "RomanNumeral" in pi.classes and "||" not in str(pi):
                            romans[-1].append((str(pi).split(" ")[1], _n, j, pi.offset, pi.duration.quarterLength))
                        if "Key" in pi.classes:
                            keys[-1].append((str(pi), _n, j, pi.offset))
                        if "TimeSignature" in pi.classes:
                            time_signatures[-1].append((str(pi), _n, j, pi.offset, pi.numerator, pi.denominator))
            for p in el.measure(j).parts:
                if str(p).split(" ")[-1].split(">")[0] in ["Soprano", "Alto", "Tenor", "Bass"]:
                    key = str(p).split(" ")[-1][:-1]
                    for pi in p.flat:
                        if "Note" in pi.classes or "Rest" in pi.classes:
                            if "Note" in pi.classes:
                                notes_and_durations[key][-1].append((pi.pitch.midi, pi.nameWithOctave, _n, j, pi.offset, pi.duration.quarterLength))
                            else:
                                notes_and_durations[key][-1].append((0, "R", _n, j, pi.offset, pi.duration.quarterLength))
        '''
        lu = str(el.filePath).split("/")[-1]
        # will be 5 when we do analysis == True, pieces without analysis will be 4
        if ".krn" in lu or lu == "bwv41.6.mxl" or len(el.parts) not in [4,]:
            print("skipped file {}".format(lu))
            continue
        print("processed file {}".format(lu))

        all_file_keys.append(lu)
        d = collections.OrderedDict()
        k = el.analyze('key')
        # cannot store music21 :(
        #d["stream"] = el
        d["original_fpath"] = str(p_bach)
        d["measure_map"] = global_measure_map
        d["romans"] = romans
        d["keys"] = keys
        d["time_signatures"] = time_signatures
        d["notes"] = notes_and_durations
        all_file_information[lu] = copy.deepcopy(d)
        from IPython import embed; embed(); raise ValueError()

        if len(p.parts) not in only_pieces_with_n_voices:
            if verbose:
                logger.info("Skipping file {}, {} due to undesired voice count...".format(it, p_bach))
            continue

        if len(p.metronomeMarkBoundaries()) != 1:
            if verbose:
                logger.info("Skipping file {}, {} due to unknown or multiple tempo changes...".format(it, p_bach))
            continue

        if verbose:
            logger.info("Processing {}, {} ...".format(it, p_bach))

        if verbose:
            logger.info("Original key: {}".format(k))
        stripped_extension_name = ".".join(os.path.split(p_bach)[1].split(".")[:-1])
        base_fpath = dataset_path + os.sep + stripped_extension_name
        try:
            if os.path.exists(base_fpath + ".json"):
                if verbose:
                    logger.info("File exists {}, skipping...".format(base_fpath))
            else:
                if 'major' in k.name:
                    kt = "major"
                elif 'minor' in k.name:
                    kt = "minor"
                core_name = base_fpath + ".{}-{}-original.json".format(k.name.split(" ")[0], kt)
                music21_parse_and_save_json(p, core_name, tempo_factor=1)
                if verbose:
                    logger.info("Writing {}".format(core_name))
            for t in ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]:
                if 'major' in k.name:
                    kt = "major"
                elif 'minor' in k.name:
                    kt = "minor"
                else:
                    raise AttributeError('Unknown key {}'.format(kn.name))

                transpose_fpath = base_fpath + ".{}-{}-transposed.json".format(t, kt)
                if os.path.exists(transpose_fpath):
                    if verbose:
                        logger.info("File exists {}, skipping...".format(transpose_fpath))
                    continue

                i = interval.Interval(k.tonic, pitch.Pitch(t))
                pn = p.transpose(i)
                #kn = pn.analyze('key')
                music21_parse_and_save_json(pn, transpose_fpath, tempo_factor=1)
                if verbose:
                    logger.info("Writing {}".format(transpose_fpath))
        except Exception as e:
            if verbose:
                logger.info(e)
                logger.info("Skipping {} due to unknown error".format(p_bach))
            continue
    return dataset_path


def fetch_jsb_chorales(verbose=True):
    jsb_dataset_path = check_fetch_jsb_chorales(verbose=verbose)
    json_files = [jsb_dataset_path + os.sep + fname for fname in sorted(os.listdir(jsb_dataset_path)) if ".json" in fname]
    return {"files": json_files}


def check_fetch_pop909(verbose=True):
    if False and os.path.exists(get_kkpthlib_dataset_dir() + os.sep + "pop909_json"):
        dataset_path = get_kkpthlib_dataset_dir("pop909_json")
        # if the dataset already exists, assume the preprocessing is already complete
        return dataset_path

    pop_base_dataset_path = get_kkpthlib_dataset_dir() + os.sep + "POP909-Dataset"
    if not os.path.exists(pop_base_dataset_path):
        raise ValueError("POP909-Dataset directory not found at {}, retrive this from https://github.com/music-x-lab/POP909-Dataset".format(oio_base_dataset_path))
    dataset_path = get_kkpthlib_dataset_dir("pop909_json")

    # POP909-Dataset/POP909
    pop_subpath = pop_base_dataset_path + os.sep + "POP909"
    midifiles = [f
                 for dirpath, dirnames, files in os.walk(pop_subpath)
                 for f in fnmatch.filter(files, '*.mid')]
    all_pop_paths = []
    for m in midifiles:
        if "-v" in m:
            all_pop_paths.append(pop_subpath + os.sep + m[:3] + os.sep + "versions" + os.sep + m)
        else:
            all_pop_paths.append(pop_subpath + os.sep + m[:3] + os.sep + m)

    all_pop_paths = sorted(all_pop_paths)

    if verbose:
        logger.info("POP909 not yet cached, processing...")
        logger.info("Total number of POP909 pieces to process from music21: {}".format(len(all_pop_paths)))
    for it, p_pop in enumerate(all_pop_paths):

        try:
            p = converter.parseFile(p_pop, format="midi")
        except:
            logger.info("Skipping file {}, {} music21 parse failed...".format(it, p_pop))

        """
        if len(p.parts) not in only_pieces_with_n_voices:
            if verbose:
                logger.info("Skipping file {}, {} due to undesired voice count...".format(it, p_pop))
            continue
        """

        if len(p.metronomeMarkBoundaries()) != 1:
            if verbose:
                logger.info("Skipping file {}, {} due to unknown or multiple tempo changes...".format(it, p_pop))
            continue

        if verbose:
            logger.info("Processing {}, {} ...".format(it, p_pop))

        k = p.analyze('key')
        if verbose:
            logger.info("Original key: {}".format(k))
        stripped_extension_name = ".".join(os.path.split(p_pop)[1].split(".")[:-1])
        base_fpath = dataset_path + os.sep + stripped_extension_name

        """
        if os.path.exists(base_fpath + ".json"):
            if verbose:
                logger.info("File exists {}, skipping...".format(base_fpath))
        else:
            if 'major' in k.name:
                kt = "major"
            elif 'minor' in k.name:
                kt = "minor"
            core_name = base_fpath + ".{}-{}-original.json".format(k.name.split(" ")[0], kt)
            midi_parse_and_save_json(p_pop, core_name, tempo_factor=1)
            logger.info("Writing {}".format(core_name))
        """

        instruments_to_keep = ["PIANO"]
        try:
            if os.path.exists(base_fpath + ".json"):
                if verbose:
                    logger.info("File exists {}, skipping...".format(base_fpath))
            else:
                if 'major' in k.name:
                    kt = "major"
                elif 'minor' in k.name:
                    kt = "minor"
                core_name = base_fpath + ".{}-{}-original.json".format(k.name.split(" ")[0], kt)
                midi_parse_and_save_json(p_pop, core_name, tempo_factor=1,
                                         instrument_names_keeplist=instruments_to_keep)
                logger.info("Writing {}".format(core_name))
            orig_key = k
            orig_key_type = kt
            for t in ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]:
                if 'major' in k.name:
                    kt = "major"
                elif 'minor' in k.name:
                    kt = "minor"
                else:
                    raise AttributeError('Unknown key {}'.format(kn.name))

                transpose_fpath = base_fpath + ".{}-{}-transposed.json".format(t, kt)
                if os.path.exists(transpose_fpath):
                    if verbose:
                        logger.info("File exists {}, skipping...".format(transpose_fpath))
                    continue

                i = interval.Interval(k.tonic, pitch.Pitch(t))
                pn = p.transpose(i)
                #kn = pn.analyze('key')
                dist = int(i.chromatic.cents // 100)
                if abs(dist) > 6:
                    if dist > 0:
                        dist = -12 + dist
                    else:
                        dist = 12 + dist
                midi_parse_and_save_json(p_pop, transpose_fpath, tempo_factor=1, transpose=dist)
                #music21_parse_and_save_json(pn, transpose_fpath, tempo_factor=1)
                if verbose:
                    logger.info("Writing {} using transpose {} from base key {} to {}".format(transpose_fpath, dist, orig_key.name, t))
        except Exception as e:
            if verbose:
                logger.info(e)
                logger.info("Skipping {} due to unknown error".format(p_pop))
            continue
    return dataset_path


def fetch_pop909():
    dataset_base_path = check_fetch_pop909()
    print("nhkjdwjakld")
    from IPython import embed; embed(); raise ValueError()
    json_files = [jsb_dataset_path + os.sep + fname for fname in sorted(os.listdir(jsb_dataset_path)) if ".json" in fname]
    return {"files": json_files}


def _populate_track_from_data(data, index, program_changes=None):
    """
    example program change
    https://github.com/cuthbertLab/music21/blob/a78617291ed0aeb6595c71f82c5d398ebe604ef4/music21/midi/__init__.py

    instrument_sequence = [(instrument, ppq_adjusted_time)]
    """
    mt = MidiTrack(index)
    t = 0
    tlast = 0
    pc_counter = 0
    for d, p, v in data:
        # need these "blank" time events
        # between each real event
        # to parse correctly
        dt = DeltaTime(mt)
        dt.time = 0 #t - tLast
        dt.channel = index
        # add to track events
        mt.events.append(dt)

        if program_changes is not None:
            if pc_counter >= len(program_changes):
                pass
            elif t >= program_changes[pc_counter][1] and tlast <= program_changes[pc_counter][1]:
                # crossed a program change event
                pc = MidiEvent(mt)
                if program_changes[pc_counter][0] not in midi_instruments_name_to_number.keys():
                    raise ValueError("Passed program change name {} not found in kkpthlib/datasets/midi_instrument_map.py".format(program_changes[pc_counter][0]))
                inst_num = midi_instruments_name_to_number[program_changes[pc_counter][0]]
                # convert from 1 indexed ala pretty-midi to 0 indexed ala music21...
                inst_num = inst_num - 1
                pc.type = "PROGRAM_CHANGE"
                pc.channel = index
                pc.time = None
                pc.data = inst_num
                mt.events.append(pc)

                # need these "blank" time events
                # between each real event
                # to parse correctly
                dt = DeltaTime(mt)
                dt.time = 0 #t - tLast
                # add to track events
                mt.events.append(dt)

                pc_counter += 1

        me = MidiEvent(mt)
        me.type = "NOTE_ON"
        me.channel = index
        me.time = None #d
        me.pitch = p
        me.velocity = v
        mt.events.append(me)

        # add note off / velocity zero message
        dt = DeltaTime(mt)
        dt.time = d
        dt.channel = index
        # add to track events
        mt.events.append(dt)

        me = MidiEvent(mt)
        me.type = "NOTE_ON"
        me.channel = index 
        me.time = None #d
        me.pitch = p
        me.velocity = 0
        mt.events.append(me)

        tlast = t
        t += d

    # add end of track
    dt = DeltaTime(mt)
    dt.time = 0
    dt.channel = index
    mt.events.append(dt)

    me = MidiEvent(mt)
    me.type = "END_OF_TRACK"
    me.channel = index
    me.data = '' # must set data to empty string
    mt.events.append(me)
    return mt


def write_music_json(json_data, out_name, default_velocity=120):
    """
    assume data is formatted in "music JSON" format
    """
    data = json.loads(json_data)
    ppq = data["pulses_per_quarter"]
    qbpm = data["quarter_beats_per_minute"]
    spq = data["seconds_per_quarter"]

    parts = data["parts"]
    parts_times = data["parts_times"]
    # https://github.com/cuthbertLab/music21/blob/c6fc39204c16c47d1c540b545d0c9869a9cafa8f/music21/midi/__init__.py#L1471
    if "parts_velocities" not in data:
        # handle rests
        parts_velocities = [[default_velocity if pi != 0 else 0 for pi in p] for p in parts]
    else:
        print("handle velocities in write_music_json")
        from IPython import embed; embed(); raise ValueError()

    with open(out_name, "w") as f:
         json.dump(data, f, indent=4)


_program_presets = {
                    "dreamy_r_preset": [("Sitar", 30),
                                        ("Orchestral Harp", 40),
                                        ("Acoustic Guitar (nylon)", 40),
                                        ("Pan Flute", 20)],
                    "dreamy_preset": [("Pan Flute", 20),
                                      ("Acoustic Guitar (nylon)", 40),
                                      ("Orchestral Harp", 40),
                                      ("Sitar", 30)],
                    "zelda_preset": [("Pan Flute", 10),
                                     ("Acoustic Guitar (nylon)", 25),
                                     ("Acoustic Guitar (nylon)", 16),
                                     ("Acoustic Guitar (nylon)", 20)],
                    "nylon_preset": [("Acoustic Guitar (nylon)", 20),
                                     ("Acoustic Guitar (nylon)", 25),
                                     ("Acoustic Guitar (nylon)", 16),
                                     ("Acoustic Guitar (nylon)", 20)],
                    "organ_preset": [("Church Organ", 50),
                                     ("Church Organ", 30),
                                     ("Church Organ", 30),
                                     ("Church Organ", 40)],
                    "grand_piano_preset": [("Acoustic Grand Piano", 50),
                                           ("Acoustic Grand Piano", 30),
                                           ("Acoustic Grand Piano", 30),
                                           ("Acoustic Grand Piano", 40)],
                    "electric_piano_preset": [("Electric Piano 1", 50),
                                              ("Electric Piano 1", 30),
                                              ("Electric Piano 1", 30),
                                              ("Electric Piano 1", 40)],
                    "harpsichord_preset": [("Harpsichord", 50),
                                           ("Harpsichord", 30),
                                           ("Harpsichord", 30),
                                           ("Harpsichord", 40)],
                    "woodwind_preset": [("Oboe", 50),
                                        ("English Horn", 30),
                                        ("Clarinet", 30),
                                        ("Bassoon", 40)],
                   }


def music_json_to_midi(json_file, out_name, tempo_factor=.5,
                       default_velocity=120,
                       voice_program_map=None,
                       verbose=True):
    """
    string (filepath) or json.dumps object

    tempo factor .5, twice as slow
    tempo factor 2, twice as fast

    voice_program_map {0: [(instrument_name, time_in_quarters)],
                       1: [(instrument_name, time_in_quarters)]}
    voices ordered SATB by default

    instrument names for program changes defined in kkpthlib/datasets/midi_instrument_map.py

    An example program change, doing harpsichord the first 8 quarter notes then a special
    mix as used by Music Transformer, Huang et. al.

    and recommended by
    https://musescore.org/en/node/109121

    a = "Harpsichord"
    b = "Harpsichord"
    c = "Harpsichord"
    d = "Harpsichord"

    e = "Oboe"
    f = "English Horn"
    g = "Clarinet"
    h = "Bassoon"

    # key: voice
    # values: list of tuples (instrument, time_in_quarter_notes_to_start_using) - optionally (instrument, time_in_quarters, global_amplitude)
    # amplitude should be in [0 , 127]
    m = {0: [(a, 0), (e, 8)],
         1: [(b, 0), (f, 8)],
         2: [(c, 0), (g, 8)],
         3: [(d, 0), (h, 8)]}

    or

    m = {0: [(a, 0, 60), (e, 8, 40)],
         1: [(b, 0, 30), (f, 8, 30)],
         2: [(c, 0, 30), (g, 8, 30)],
         3: [(d, 0, 40), (h, 8, 50)]}

    Alternatively, support "auto" groups which set custom voices and amplitudes
    a = "harpsichord_preset"
    b = "woodwind_preset"
    m = {0: [(a, 0), (b, 8)],
         1: [(a, 0), (b, 8)],
         2: [(a, 0), (b, 8)],
         3: [(a, 0), (b, 8)]}

    valid preset values:
                "dreamy_r_preset"
                "dreamy_preset"
                "zelda_preset"
                "nylon_preset"
                "organ_preset"
                "grand_piano_preset"
                "electric_piano_preset"
                "harpsichord_preset"
                "woodwind_preset"
    """
    if json_file.endswith(".json"):
        with open(json_file) as f:
            data = json.load(f)
    else:
        data = json.loads(json_file)
    #[u'parts', u'parts_names', u'parts_cumulative_times', u'parts_times']
    #['parts_velocities'] optional

    ppq = data["pulses_per_quarter"]
    qbpm = data["quarter_beats_per_minute"]
    spq = data["seconds_per_quarter"]

    parts = data["parts"]
    parts_times = data["parts_times"]
    parts_cumulative_times = data["parts_cumulative_times"]

    pm = pretty_midi.PrettyMIDI()
    for _i in range(len(parts)):
        inst = pretty_midi.Instrument(0)
        pm.instruments.append(inst)
        for _j in range(len(parts[_i])):
            if parts[_i][_j] == 0:
                continue
            inst.notes.append(pretty_midi.Note(pitch=parts[_i][_j], velocity=100, start=parts_cumulative_times[_i][_j] - parts_times[_i][_j],
                                               end=parts_cumulative_times[_i][_j]))
    pm.write(out_name)


def music_json_to_midi_old(json_file, out_name, tempo_factor=.5,
                       default_velocity=120,
                       voice_program_map=None,
                       verbose=True):
    """
    string (filepath) or json.dumps object

    tempo factor .5, twice as slow
    tempo factor 2, twice as fast

    voice_program_map {0: [(instrument_name, time_in_quarters)],
                       1: [(instrument_name, time_in_quarters)]}
    voices ordered SATB by default

    instrument names for program changes defined in kkpthlib/datasets/midi_instrument_map.py

    An example program change, doing harpsichord the first 8 quarter notes then a special
    mix as used by Music Transformer, Huang et. al.

    and recommended by
    https://musescore.org/en/node/109121

    a = "Harpsichord"
    b = "Harpsichord"
    c = "Harpsichord"
    d = "Harpsichord"

    e = "Oboe"
    f = "English Horn"
    g = "Clarinet"
    h = "Bassoon"

    # key: voice
    # values: list of tuples (instrument, time_in_quarter_notes_to_start_using) - optionally (instrument, time_in_quarters, global_amplitude)
    # amplitude should be in [0 , 127]
    m = {0: [(a, 0), (e, 8)],
         1: [(b, 0), (f, 8)],
         2: [(c, 0), (g, 8)],
         3: [(d, 0), (h, 8)]}

    or

    m = {0: [(a, 0, 60), (e, 8, 40)],
         1: [(b, 0, 30), (f, 8, 30)],
         2: [(c, 0, 30), (g, 8, 30)],
         3: [(d, 0, 40), (h, 8, 50)]}

    Alternatively, support "auto" groups which set custom voices and amplitudes
    a = "harpsichord_preset"
    b = "woodwind_preset"
    m = {0: [(a, 0), (b, 8)],
         1: [(a, 0), (b, 8)],
         2: [(a, 0), (b, 8)],
         3: [(a, 0), (b, 8)]}

    valid preset values:
                "dreamy_r_preset"
                "dreamy_preset"
                "zelda_preset"
                "nylon_preset"
                "organ_preset"
                "grand_piano_preset"
                "electric_piano_preset"
                "harpsichord_preset"
                "woodwind_preset"
    """
    if json_file.endswith(".json"):
        with open(json_file) as f:
            data = json.load(f)
    else:
        data = json.loads(json_file)
    #[u'parts', u'parts_names', u'parts_cumulative_times', u'parts_times']
    #['parts_velocities'] optional

    ppq = data["pulses_per_quarter"]
    qbpm = data["quarter_beats_per_minute"]
    spq = data["seconds_per_quarter"]

    parts = data["parts"]
    parts_times = data["parts_times"]
    # https://github.com/cuthbertLab/music21/blob/c6fc39204c16c47d1c540b545d0c9869a9cafa8f/music21/midi/__init__.py#L1471
    if "parts_velocities" not in data:
        # handle rests
        parts_velocities = [[default_velocity if pi != 0 else 0 for pi in p] for p in parts]
    else:
        print("handle velocities in json_to_midi")

    all_mt = []
    for i in range(len(parts)):
        assert len(parts[i]) == len(parts_velocities[i])
        assert len(parts[i]) == len(parts_times[i])
        program_changes = voice_program_map[i] if voice_program_map is not None else None
        this_part_velocity = [parts_velocities[i][j] for j in range(len(parts[i]))]
        if program_changes is not None:
            # remap presets
            program_changes_new = []
            for pc in program_changes:
                if pc[0] in _program_presets.keys():
                    p = _program_presets[pc[0]]
                    program_changes_new.append((p[i][0], pc[1], p[i][1]))
                else:
                    program_changes_new.append(pc)
            program_changes = program_changes_new

        if program_changes is not None:
            program_changes = [(pg[0], int(pg[1] * ppq)) if len(pg) < 3 else (pg[0], int(pg[1] * ppq), pg[2]) for pg in program_changes]
            this_part_velocity_new = []
            pg_counter = 0
            last_step_tick = 0
            current_step_tick = 0
            current_velocity = default_velocity
            for j in range(len(parts[i])):
                last_step_tick = current_step_tick
                current_step_tick += int(parts_times[i][j] * ppq)
                if len(program_changes[pg_counter]) < 3:
                    this_part_velocity_new.append(this_part_velocity[j])
                else:
                    # if it is the last program change then we just stay on that
                    if pg_counter < len(program_changes) - 1:
                        # check for tick boundary
                        # if we are exactly on a boundary... change it now
                        if last_step_tick <= int(program_changes[pg_counter + 1][1]) and current_step_tick >= int(program_changes[pg_counter + 1][1]):
                            pg_counter += 1
                    current_velocity = default_velocity if len(program_changes[pg_counter]) < 3 else program_changes[pg_counter][2]
                    this_part_velocity_new.append(current_velocity)
            this_part_velocity = this_part_velocity_new

        track_data = [[int(parts_times[i][j] * ppq), parts[i][j], this_part_velocity[j] if parts[i][j] != 0 else 0] for j in range(len(parts[i]))]

        # do global velocity modulations...
        # + 1 to account for MidiTrack starting at 1
        mt = _populate_track_from_data(track_data, i + 1, program_changes=program_changes)
        all_mt.append(mt)

    mf = MidiFile()
    # multiply by half to get proper feel on bach at least, not sure why...
    # see for example bwv110.7
    # https://www.youtube.com/watch?v=1WWR4PQZdjo
    mf.ticksPerQuarterNote = int(ppq * tempo_factor)
    # ticks (pulses) / quarter * quarters / second 
    mf.ticksPerSecond = int(ppq * (1. / float(spq)))

    for mt in all_mt:
        mf.tracks.append(mt)

    mf.open(out_name, 'wb')
    mf.write()
    mf.close()


def piano_roll_from_music_json_file(json_file, default_velocity=120, quantization_rate=.25, n_voices=4,
                                    separate_onsets=True, onsets_boundary=100, as_numpy=True):
    """
    return list of list [[each_voice] n_voices] or numpy array of shape (time_len, n_voices)
    """
    with open(json_file) as f:
        data = json.load(f)
    ppq = data["pulses_per_quarter"]
    qbpm = data["quarter_beats_per_minute"]
    spq = data["seconds_per_quarter"]

    parts = data["parts"]
    parts_times = data["parts_times"]
    parts_cumulative_times = data["parts_cumulative_times"]
    # https://github.com/cuthbertLab/music21/blob/c6fc39204c16c47d1c540b545d0c9869a9cafa8f/music21/midi/__init__.py#L1471
    if "parts_velocities" not in data:
        default_velocity = default_velocity
        parts_velocities = [[default_velocity] * len(p) for p in parts]
    else:
        parts_velocities = data["parts_velocities"]
    end_in_quarters = max([max(p) for p in parts_cumulative_times])
    # clock is set currently by the fact that "sixteenth" is the only option
    # .25 due to "sixteenth"
    clock = np.arange(0, max(max(parts_cumulative_times)), quantization_rate)
    # 4 * for 4 voices
    raster_end_in_steps = n_voices * len(clock)

    roll_voices = [[] for _ in range(n_voices)]
    # use these for tracking if we cross a change event
    p_i = [0] * n_voices
    for c in clock:
        # voice
        for v in range(len(parts)):
            if p_i[v] >= len(parts[v]):
                continue
            else:
                current_note = parts[v][p_i[v]]
            next_change_time = parts_cumulative_times[v][p_i[v]]
            new_onset = False
            if c >= next_change_time:
                # we hit a boundary, swap notes
                p_i[v] += 1
                if p_i[v] >= len(parts[v]):
                    continue
                else:
                    current_note = parts[v][p_i[v]]
                next_change_time = parts_cumulative_times[v][p_i[v]]
                new_onset = True
            if c == 0. or new_onset:
                if current_note != 0:
                    if separate_onsets:
                        roll_voices[v].append(current_note + onsets_boundary)
                    else:
                        roll_voices[v].append(current_note)
                else:
                    # rests have no "onset"
                    roll_voices[v].append(current_note)
            else:
               roll_voices[v].append(current_note)
    if any([len(rv) == 0 for rv in roll_voices]):
        # print a warning here?
        roll_voices = [rv for rv in roll_voices if len(rv) > 0]
    if as_numpy:
        # truncate to shortest if ragged and needs to be numpy
        min_len = min([len(rv) for rv in roll_voices])
        roll_voices = [rv[:min_len] for rv in roll_voices]
        roll_voices = np.array(roll_voices).T
    return roll_voices


def pitch_duration_velocity_lists_from_music_json_file(json_file, default_velocity=120, n_voices=4,
                                                       add_measure_values=True,
                                                       measure_value=99,
                                                       measure_quarters=8,
                                                       force_quarters_match=False,
                                                       trim_uneven_last=False,
                                                       fill_value=-1):
    """
    return list of list [[each_voice] n_voices] or numpy array of shape (time_len, n_voices)
    """
    with open(json_file) as f:
        data = json.load(f)
    ppq = data["pulses_per_quarter"]
    qbpm = data["quarter_beats_per_minute"]
    spq = data["seconds_per_quarter"]

    parts = data["parts"]
    parts_times = data["parts_times"]
    parts_cumulative_times = data["parts_cumulative_times"]
    # https://github.com/cuthbertLab/music21/blob/c6fc39204c16c47d1c540b545d0c9869a9cafa8f/music21/midi/__init__.py#L1471
    if "parts_velocities" not in data:
        default_velocity = default_velocity
        parts_velocities = [[default_velocity] * len(p) for p in parts]
    else:
        parts_velocities = data["parts_velocities"]

    n_steps = max([len(p) for p in parts])

    # shared starts
    tmp = [set(parts_cumulative_times[i]) for i in range(len(parts))]
    # assume there at least 2 parts or more
    candidates = tmp[0]
    for i in range(len(tmp)):
        candidates = candidates.intersection(tmp[i])

    mod_candidates = [c for c in sorted(list(candidates)) if c % measure_quarters == 0]
    # skip 0
    full_enumeration = [c for c in list(range(1, int(mod_candidates[-1]) + measure_quarters)) if c % measure_quarters == 0]
    if force_quarters_match:
        try:
            assert all([f_e in mod_candidates for f_e in full_enumeration])
        except:
            raise AttributeError("force quarters match failed! measure_quarters {}, file {}".format(measure_quarters, json_file))
        candidates = mod_candidates

    measure_stops = [[] for p in parts]
    for v in range(len(parts)):
        last_measure_boundary = -1

        for s in range(n_steps):
            if s >= len(parts[v]):
                continue
            p_s = parts[v][s]
            d_s = parts_times[v][s]
            p_c_s = parts_cumulative_times[v][s]
            p_v_s = parts_velocities[v][s]

            if p_c_s in candidates:
                if p_c_s >= 4.0:
                    if v == 0:
                        if last_measure_boundary == -1 or s - last_measure_boundary > 3:
                            measure_stops[v].append((s + 1, p_c_s))
                            last_measure_boundary = s
                    elif v > 0:
                        if p_c_s in [el[1] for el in measure_stops[0]]:
                            measure_stops[v].append((s + 1, p_c_s))
                            last_measure_boundary = s

    # back to front in order to avoid issues with indexes and shifting
    new_parts = [[] for p in parts]
    new_parts_times = [[] for p in parts]
    new_parts_cumulative_times = [[] for p in parts]
    new_parts_velocities = [[] for p in parts]
    for v in range(len(parts)):
        new_parts[v].extend(parts[v])
        new_parts_times[v].extend(parts_times[v])
        new_parts_cumulative_times[v].extend(parts_cumulative_times[v])
        new_parts_velocities[v].extend(parts_velocities[v])

        for pt in measure_stops[v][::-1]:
            new_parts[v].insert(pt[0], 99)
            new_parts_times[v].insert(pt[0], 0)
            new_parts_velocities[v].insert(pt[0], 0)

        new_parts_cumulative_times[v] = [el for el in np.cumsum([0] + new_parts_times[v])[:-1]]
        # set duration to -1 now that we calculate the new cumulative times
        new_parts_times[v] = [new_parts_times[v][_ii] if new_parts_times[v][_ii] > 0 else -1. for _ii in range(len(new_parts_times[v]))]

    if trim_uneven_last:
        if not all([_pt[-1] == -1 for _pt in new_parts_times]):
            sum_lasts = -1
            part_sums = []
            part_cuts = []
            for v in range(len(new_parts_times)):
                minuses = np.where(np.array(new_parts_times[v]) == -1)[0]
                last_minus = minuses[-1]
                part_cuts.append(last_minus)
                part_s = sum(new_parts_times[v][last_minus + 1:])
                if part_s == measure_quarters:
                    part_sums.append(True)
                else:
                    part_sums.append(False)

            if all(part_sums):
                l = None
                for v in range(len(new_parts_times)):
                    if new_parts[v][-1] != 99:
                        new_parts[v].append(99)
                        new_parts_times[v].append(-1.)
                        new_parts_cumulative_times[v].append(new_parts_cumulative_times[-1])
                        new_parts_velocities[v].append(0)
                    else:
                        raise ValueError("part sums -1 is 99, but all _pt[-1] == -1... shouldn't get here in RowRaster corpus")

                    if l is not None:
                        assert len(new_parts[v]) == l
                        assert len(new_parts_times[v]) == l
                        assert len(new_parts_cumulative_times[v]) == l
                        assert len(new_parts_velocities[v]) == l
                    else:
                        l = len(new_parts[v])
                        assert len(new_parts_times[v]) == l
                        assert len(new_parts_cumulative_times[v]) == l
                        assert len(new_parts_velocities[v]) == l
            else:
                for v in range(len(new_parts_times)):
                   new_parts[v] = new_parts[v][:part_cuts[v] + 1]
                   new_parts_times[v] = new_parts_times[v][:part_cuts[v] + 1]
                   new_parts_cumulative_times[v] = new_parts_cumulative_times[v][:part_cuts[v] + 1]
                   new_parts_velocities[v] = new_parts_velocities[v][:part_cuts[v] + 1]
        assert all([_pt[-1] == -1 for _pt in new_parts_times])
        # check they all sum to the right value
        for v in range(len(new_parts)):
            _s = 0
            for el in new_parts_times[v]:
                if el != -1:
                    _s += el
                elif el == -1:
                    assert _s == measure_quarters
                    _s = 0
    return new_parts, new_parts_times, new_parts_velocities


def convert_voice_roll_to_music_json(voice_roll, quantization_rate=.25, onsets_boundary=100, verbose=True):
    """
    take in voice roll and turn it into a pitch, duration thing again

    currently assume onsets are any notes > 100 , 0 is rest

    example input, where 170, 70, 70, 70 is an onset of pitch 70 (noted as 170), followed by a continuation for 4 steps
    array([[170.,  70.,  70.,  70.],
           [165.,  65.,  65.,  65.],
           [162.,  62.,  62.,  62.],
           [158.,  58.,  58.,  58.]])
    """
    duration_step = quantization_rate
    voice_data = {}
    voice_data["parts"] = []
    voice_data["parts_times"] = []
    voice_data["parts_cumulative_times"] = []
    for v in range(voice_roll.shape[0]):
        voice_data["parts"].append([])
        voice_data["parts_times"].append([])
        voice_data["parts_cumulative_times"].append([])
    for v in range(voice_roll.shape[0]):
        ongoing_duration = duration_step
        note_held = 0
        for t in range(len(voice_roll[v])):
            token = int(voice_roll[v][t])
            if voice_roll[v][t] > onsets_boundary:
                voice_data["parts"][v].append(note_held)
                voice_data["parts_times"][v].append(ongoing_duration)
                ongoing_duration = duration_step
                note_held = token - onsets_boundary
            elif token != 0:
                if token != note_held:
                    # make it an onset?
                    if verbose:
                        print("WARNING: got non-onset pitch change, forcing onset token at step {}, voice {}".format(t, v))
                    voice_data["parts"][v].append(note_held)
                    voice_data["parts_times"][v].append(ongoing_duration)
                    note_held = token
                    ongoing_duration = duration_step
                else:
                    ongoing_duration += duration_step
            else:
                # just adding 16th note silences?
                ongoing_duration = duration_step
                note_held = 0
                voice_data["parts"][v].append(note_held)
                voice_data["parts_times"][v].append(ongoing_duration)
        voice_data["parts_cumulative_times"][v] = [e for e in np.cumsum(voice_data["parts_times"][v])]
    spq = .5
    ppq = 220
    qbpm = 120
    voice_data["seconds_per_quarter"] = spq
    voice_data["quarter_beats_per_minute"] = qbpm
    voice_data["pulses_per_quarter"] = ppq
    voice_data["parts_names"] = ["Soprano", "Alto", "Tenor", "Bass"]
    j = json.dumps(voice_data, indent=4)
    return j


def convert_voice_lists_to_music_json(pitch_lists, duration_lists, velocity_lists=None, voices_list=None,
                                      default_velocity=120,
                                      measure_value=99,
                                      onsets_boundary=100):
    """
    can either work by providing a list of lists input for pitch_lists and duration_lists (optionally velocity lists)

    or

    1 long list for pitch, 1 long list for duration (optionally velocity), and the voices_list argument which has
    indicators for each voice and how it maps
    """
    voice_data = {}
    voice_data["parts"] = []
    voice_data["parts_times"] = []
    voice_data["parts_cumulative_times"] = []
    if voices_list is not None:
        assert len(pitch_lists) == len(duration_lists)
        voices = sorted(list(set(voices_list)))
        for v in voices:
            selector = [v_i == v for v_i in voices_list]
            parts = [pitch_lists[i] for i in range(len(pitch_lists)) if selector[i]]
            parts_times = [duration_lists[i] for i in range(len(pitch_lists)) if selector[i]]
            if velocity_lists is not None:
                parts_velocities = [velocity_lists[i] for i in range(len(pitch_lists)) if selector[i]]
            else:
                parts_velocities = [default_velocity for i in range(len(pitch_lists)) if selector[i]]
            # WE ASSUME MEASURE SELECTOR IS A UNIQUE ONE
            if any([p == measure_value for p in parts]):
                continue
            else:
                voice_data["parts"].append(parts)
                voice_data["parts_times"].append(parts_times)
                voice_data["parts_cumulative_times"].append([0.] + [e for e in np.cumsum(parts_times)])
    else:
        from IPython import embed; embed(); raise ValueError()
    spq = .5
    ppq = 220
    qbpm = 120
    voice_data["seconds_per_quarter"] = spq
    voice_data["quarter_beats_per_minute"] = qbpm
    voice_data["pulses_per_quarter"] = ppq
    voice_data["parts_names"] = ["Soprano", "Alto", "Tenor", "Bass"]
    j = json.dumps(voice_data, indent=4)
    return j


def convert_sampled_pkl_sequence_to_music_json_data(np_arr, corpus, measure_quarters=4, force_column=False, no_measure_mark=True, add_const=0,
                                                    verbose=True):
    # 16 * 4 = 64
    # 4 16ths per quarter, 4 voices
    shape_bound = int(measure_quarters * 4 * 4)
    all_data = []
    for i in range(np_arr.shape[1]):
        tmp = np_arr[:, i]

        if force_column:
            if no_measure_mark:
                if len(tmp) % shape_bound != 0:
                    tmp = tmp[:len(tmp) - (len(tmp) % shape_bound)]
                r_tmp = tmp.reshape(-1, shape_bound)
                voices_rolls = [[] for i in range(4)]
                for mi in range(len(r_tmp)):
                    # no skip, no measure marks
                    measure_tmp = r_tmp[mi, :]
                    notes_measure_tmp = [corpus.dictionary.idx2word[m] for m in measure_tmp]
                    # 4 voices, X events each
                    voices_measure = np.array(notes_measure_tmp).reshape(-1, 4).transpose()
                    # final array will be 4 lists of N steps, no measure marks
                    # can check the conversion with:
                    # convert_voice_roll_to_music_json(voices_measure)
                    for v in range(len(voices_measure)):
                        voices_rolls[v].extend([m for m in voices_measure[v]])
            else:
                tmp = np.concatenate(([corpus.dictionary.word2idx[999]], tmp))
                boundaries = np.where(tmp == corpus.dictionary.word2idx[999])[0]
                r_tmp = tmp[:boundaries[-1]].reshape(-1, shape_bound + 1)
                voices_rolls = [[] for i in range(4)]
                for mi in range(len(r_tmp)):
                    # 1: to skip the initial marker
                    assert r_tmp[mi, 0] == corpus.dictionary.word2idx[999]
                    measure_tmp = r_tmp[mi, 1:]
                    notes_measure_tmp = [corpus.dictionary.idx2word[m] for m in measure_tmp]
                    # 4 voices, X events each
                    voices_measure = np.array(notes_measure_tmp).reshape(-1, 4).transpose()
                    # final array will be 4 lists of N steps, no measure marks
                    # can check the conversion with:
                    # convert_voice_roll_to_music_json(voices_measure)
                    for v in range(len(voices_measure)):
                        voices_rolls[v].extend([m for m in voices_measure[v]])
        else:
            # append on 1 value to make it uniform
            tmp = np.concatenate(([corpus.dictionary.word2idx[999]], tmp))
            # last multiple?
            boundaries = np.where(tmp == corpus.dictionary.word2idx[999])[0]
            l = 0
            for b in boundaries:
                if b % (shape_bound + 1) == 0:
                    l = b
            # dont use l for now
            measure_chunk_shp = shape_bound

            # + 1 for measure mark

            r_tmp = tmp[:l].reshape(-1, measure_chunk_shp + 1)

            voices_rolls = [[] for i in range(4)]
            for mi in range(len(r_tmp)):
                # 1: to skip the initial marker
                assert r_tmp[mi, 0] == corpus.dictionary.word2idx[999]
                measure_tmp = r_tmp[mi, 1:]
                notes_measure_tmp = [corpus.dictionary.idx2word[m] for m in measure_tmp]
                # 4 voices, X events each
                voices_measure = np.array(notes_measure_tmp).reshape(4, -1)
                # final array will be 4 lists of N steps, no measure marks
                # can check the conversion with:
                # convert_voice_roll_to_music_json(voices_measure)
                for v in range(len(voices_measure)):
                    voices_rolls[v].extend([m for m in voices_measure[v]])
        vr = np.array(voices_rolls)
        for ii in range(vr.shape[1]):
            nz = np.where(vr[:, ii] > 0)[0]
            vr[nz, ii] += add_const
        data = convert_voice_roll_to_music_json(vr, verbose=verbose)
        all_data.append(data)
    return all_data


class MusicJSONCorpus(object):
    def __init__(self, train_data_file_paths, valid_data_file_paths=None, test_data_file_paths=None,
                 max_vocabulary_size=-1,
                 voices=[0,1,2,3]):
        """
        """
        self.dictionary = LookupDictionary()
        self.max_vocabulary_size = max_vocabulary_size
        self.voices = voices
        self.fill_symbol = (-2, -2)
        self.measure_symbol = (-1, -1)
        self.dictionary.add_word(self.fill_symbol)

        self.train_data_file_paths = train_data_file_paths
        self.valid_data_file_paths = valid_data_file_paths

        train_pitches, train_durations, train_voices = self._load_music_json(train_data_file_paths)

        self.build_vocabulary(train_data_file_paths)
        if valid_data_file_paths != None:
            valid_pitches, valid_durations, valid_voices = self._load_music_json(valid_data_file_paths)
            self.build_vocabulary(valid_data_file_paths)
        if test_data_file_paths != None:
            test_pitches, test_durations, test_voices = self._load_music_json(test_data_file_paths)

        self.train, self.train_offsets = self.pre_tokenize(train_data_file_paths)
        if valid_data_file_paths is not None:
            self.valid, self.valid_offsets = self.pre_tokenize(valid_data_file_paths)
        if test_data_file_paths is not None:
            self.test, self.test_offsets = self.pre_tokenize(test_data_file_paths)

    def _load_music_json(self, json_file_paths):
        all_pitches = []
        all_durations = []
        all_voices = []
        for path in json_file_paths:
            pitches, durations, voices = pitch_duration_velocity_lists_from_music_json_file(path)
            all_pitches.append([p for n, p in enumerate(pitches) if n in self.voices])
            all_durations.append([d for n, d in enumerate(durations) if n in self.voices])
            all_voices.append([v for n, v in enumerate(voices) if n in self.voices])
        return all_pitches, all_durations, all_voices

    def build_vocabulary(self, json_file_paths):
        for path in json_file_paths:
            pitches, durations, voices = pitch_duration_velocity_lists_from_music_json_file(path)
            all_pitches = []
            all_durations = []
            if self.raster_scan:
                joint = []
                v = 0
                voice_counter = [0, 0, 0, 0]
                while True:
                    if voice_counter[v] < len(pitches[v]):
                        if all([voice_counter[i] >= len(pitches[i]) for i in range(len(voice_counter))]):
                            break
                        all_pitches.append(pitches[voice_counter[v]])
                        all_durations.append(durations[voice_counter[v]])
                    v = v + 1
                    v = v % len(pitches)
                for p, d in zip(all_pitches, all_durations):
                    if d == -1:
                        continue
                    else:
                        joint.append((p, d))

                for j in joint:
                    if j not in self.dictionary.word2idx:
                        self.dictionary.add_word(j)
            else:
                for v in self.voices:
                    _pitches = [p for n, p in enumerate(pitches) if n == v]
                    _durations = [d for n, d in enumerate(durations) if n == v]
                    _pitches = _pitches[0]
                    _durations = _durations[0]
                    assert len(_pitches) == len(_durations)
                    joint = []
                    for p, d in zip(_pitches, _durations):
                        if d == -1:
                            continue
                        else:
                            joint.append((p, d))
                    for j in joint:
                        if j not in self.dictionary.word2idx:
                            self.dictionary.add_word(j)

    def pre_tokenize(self, json_file_paths):
        joints = []
        joints_offsets = []
        for path in json_file_paths:
            pitches, durations, voices = pitch_duration_velocity_lists_from_music_json_file(path)
            all_pitches = []
            all_durations = []
            if self.raster_scan:
                joint = []
                v = 0
                voice_counter = [0, 0, 0, 0]
                while True:
                    if voice_counter[v] < len(pitches[v]):
                        if all([voice_counter[i] >= len(pitches[i]) for i in range(len(voice_counter))]):
                            break
                        all_pitches.append(pitches[voice_counter[v]])
                        all_durations.append(durations[voice_counter[v]])
                    v = v + 1
                    v = v % len(pitches)
                for p, d in zip(all_pitches, all_durations):
                    if d == -1:
                        continue
                    else:
                        joint.append((p, d))
                joints.extend(joint)
            else:
                for v in self.voices:
                    _pitches = [p for n, p in enumerate(pitches) if n == v]
                    _durations = [d for n, d in enumerate(durations) if n == v]
                    _pitches = _pitches[0]
                    _durations = _durations[0]
                    assert len(_pitches) == len(_durations)
                    joint = []
                    for p, d in zip(_pitches, _durations):
                        if d == -1:
                            continue
                        else:
                            joint.append((p, d))
                    joints.extend(joint)
        return joints, joints_offsets

    def get_iterator(self, batch_size, random_seed, sequence_len=64, context_len=32, sample_percent=.15,
                     max_n_gram=8, _type="train"):
        random_state = np.random.RandomState(random_seed)
        if _type == "train":
            content = self.train
            offsets = self.train_offsets
        elif _type == "valid":
            content = self.valid
            offsets = self.valid_offsets
        elif _type == "test":
            content = self.test
            offsets = self.test_offsets

        fill_symbol = self.fill_symbol

        def sample_minibatch(batch_size=batch_size, content=content, random_state=random_state,
                             sequence_len=sequence_len,
                             context_len=context_len,
                             fill_symbol=fill_symbol,
                             sample_percent=sample_percent,
                             max_n_gram=max_n_gram):
            while True:
                cur_batch = []
                for i in range(batch_size):
                    # sample place to start element from the dataset
                    el = random_state.choice(len(content) - sequence_len - 1)
                    cur = content[el:el + sequence_len]
                    cur_batch.append(cur)
                # make transformer masks for constructed batch
                max_len = max([len(b) for b in cur_batch])
                cur_batch_masks = [[0] * len(b) + [1] * (max_len - len(b)) for b in cur_batch]
                cur_batch = [b + [fill_symbol] * (max_len - len(b)) for b in cur_batch]
                token_batch = [[self.dictionary.word2idx[bi] for bi in b] for b in cur_batch]
                yield np.array(token_batch).T[..., None], np.array(cur_batch_masks).T
        return sample_minibatch()


class MusicJSONPitchDurationCorpus(object):
    def __init__(self, train_data_file_paths, valid_data_file_paths=None, test_data_file_paths=None,
                 max_vocabulary_size=-1,
                 voices=[0,1,2,3],
                 raster_scan=False):
        raise ValueError("deprecated")
        """
        """
        self.dictionary = LookupDictionary()
        self.max_vocabulary_size = max_vocabulary_size
        self.voices = voices
        self.fill_symbol = (-1, -1)
        self.dictionary.add_word(self.fill_symbol)
        self.raster_scan = raster_scan

        self.train_data_file_paths = train_data_file_paths
        self.valid_data_file_paths = valid_data_file_paths

        train_pitches, train_durations, train_voices = self._load_music_json(train_data_file_paths)

        self.build_vocabulary(train_data_file_paths)
        if valid_data_file_paths != None:
            valid_pitches, valid_durations, valid_voices = self._load_music_json(valid_data_file_paths)
            self.build_vocabulary(valid_data_file_paths)
        if test_data_file_paths != None:
            test_pitches, test_durations, test_voices = self._load_music_json(test_data_file_paths)

        self.train, self.train_offsets = self.pre_tokenize(train_data_file_paths)
        if valid_data_file_paths is not None:
            self.valid, self.valid_offsets = self.pre_tokenize(valid_data_file_paths)
        if test_data_file_paths is not None:
            self.test, self.test_offsets = self.pre_tokenize(test_data_file_paths)

    def _load_music_json(self, json_file_paths):
        all_pitches = []
        all_durations = []
        all_voices = []
        for path in json_file_paths:
            pitches, durations, voices = pitch_duration_velocity_lists_from_music_json_file(path)
            all_pitches.append([p for n, p in enumerate(pitches) if n in self.voices])
            all_durations.append([d for n, d in enumerate(durations) if n in self.voices])
            all_voices.append([v for n, v in enumerate(voices) if n in self.voices])
        return all_pitches, all_durations, all_voices

    def build_vocabulary(self, json_file_paths):
        for path in json_file_paths:
            pitches, durations, voices = pitch_duration_velocity_lists_from_music_json_file(path)
            all_pitches = []
            all_durations = []
            if self.raster_scan:
                joint = []
                v = 0
                voice_counter = [0, 0, 0, 0]
                while True:
                    if voice_counter[v] < len(pitches[v]):
                        if all([voice_counter[i] >= len(pitches[i]) for i in range(len(voice_counter))]):
                            break
                        all_pitches.append(pitches[voice_counter[v]])
                        all_durations.append(durations[voice_counter[v]])
                    v = v + 1
                    v = v % len(pitches)
                for p, d in zip(all_pitches, all_durations):
                    if d == -1:
                        continue
                    else:
                        joint.append((p, d))

                for j in joint:
                    if j not in self.dictionary.word2idx:
                        self.dictionary.add_word(j)
            else:
                for v in self.voices:
                    _pitches = [p for n, p in enumerate(pitches) if n == v]
                    _durations = [d for n, d in enumerate(durations) if n == v]
                    _pitches = _pitches[0]
                    _durations = _durations[0]
                    assert len(_pitches) == len(_durations)
                    joint = []
                    for p, d in zip(_pitches, _durations):
                        if d == -1:
                            continue
                        else:
                            joint.append((p, d))
                    for j in joint:
                        if j not in self.dictionary.word2idx:
                            self.dictionary.add_word(j)

    def pre_tokenize(self, json_file_paths):
        joints = []
        joints_offsets = []
        print("NYI handle joint offsets")
        from IPython import embed; embed(); raise ValueError()
        for path in json_file_paths:
            pitches, durations, voices = pitch_duration_velocity_lists_from_music_json_file(path)
            all_pitches = []
            all_durations = []
            if self.raster_scan:
                joint = []
                v = 0
                voice_counter = [0, 0, 0, 0]
                while True:
                    if voice_counter[v] < len(pitches[v]):
                        if all([voice_counter[i] >= len(pitches[i]) for i in range(len(voice_counter))]):
                            break
                        all_pitches.append(pitches[voice_counter[v]])
                        all_durations.append(durations[voice_counter[v]])
                    v = v + 1
                    v = v % len(pitches)
                for p, d in zip(all_pitches, all_durations):
                    if d == -1:
                        continue
                    else:
                        joint.append((p, d))
                joints.extend(joint)
            else:
                for v in self.voices:
                    _pitches = [p for n, p in enumerate(pitches) if n == v]
                    _durations = [d for n, d in enumerate(durations) if n == v]
                    _pitches = _pitches[0]
                    _durations = _durations[0]
                    assert len(_pitches) == len(_durations)
                    joint = []
                    for p, d in zip(_pitches, _durations):
                        if d == -1:
                            continue
                        else:
                            joint.append((p, d))
                    joints.extend(joint)
        return joints, joints_offsets

    def get_iterator(self, batch_size, random_seed, sequence_len=64, context_len=32, sample_percent=.15,
                     max_n_gram=8, _type="train"):
        random_state = np.random.RandomState(random_seed)
        if _type == "train":
            content = self.train
            offsets = self.train_offsets
        elif _type == "valid":
            content = self.valid
            offsets = self.valid_offsets
        elif _type == "test":
            content = self.test
            offsets = self.test_offsets

        fill_symbol = self.fill_symbol

        def sample_minibatch(batch_size=batch_size, content=content, random_state=random_state,
                             sequence_len=sequence_len,
                             context_len=context_len,
                             fill_symbol=fill_symbol,
                             sample_percent=sample_percent,
                             max_n_gram=max_n_gram):
            while True:
                cur_batch = []
                for i in range(batch_size):
                    # sample place to start element from the dataset
                    el = random_state.choice(len(content) - sequence_len - 1)
                    cur = content[el:el + sequence_len]
                    cur_batch.append(cur)
                # make transformer masks for constructed batch
                max_len = max([len(b) for b in cur_batch])
                cur_batch_masks = [[0] * len(b) + [1] * (max_len - len(b)) for b in cur_batch]
                cur_batch = [b + [fill_symbol] * (max_len - len(b)) for b in cur_batch]
                token_batch = [[self.dictionary.word2idx[bi] for bi in b] for b in cur_batch]
                yield np.array(token_batch).T[..., None], np.array(cur_batch_masks).T
        return sample_minibatch()


class MusicJSONInfillCorpus(object):
    def __init__(self, train_data_file_paths, valid_data_file_paths=None, test_data_file_paths=None,
                 max_vocabulary_size=-1,
                 voices=[0,1,2,3],
                 raster_scan=True):
        raise ValueError("deprecated")
        """
        """
        self.dictionary = LookupDictionary()
        self.max_vocabulary_size = max_vocabulary_size
        self.voices = voices
        self.mask_symbol = (-1, -1)
        self.answer_symbol = (-2, -2)
        self.end_context_symbol = (-3, -3)
        self.file_separator_symbol = (-4, -4)
        self.fill_symbol = (-5, -5)
        self.special_symbols = [self.mask_symbol, self.answer_symbol, self.end_context_symbol, self.file_separator_symbol, self.fill_symbol]
        self.raster_scan = raster_scan

        self.train_data_file_paths = train_data_file_paths
        self.valid_data_file_paths = valid_data_file_paths

        train_pitches, train_durations, train_voices = self._load_music_json(train_data_file_paths)

        self.dictionary.add_word(self.mask_symbol)
        self.dictionary.add_word(self.answer_symbol)
        self.dictionary.add_word(self.end_context_symbol)
        self.dictionary.add_word(self.file_separator_symbol)
        self.dictionary.add_word(self.fill_symbol)

        self.build_vocabulary(train_data_file_paths)
        if valid_data_file_paths != None:
            valid_pitches, valid_durations, valid_voices = self._load_music_json(valid_data_file_paths)
            self.build_vocabulary(valid_data_file_paths)
        if test_data_file_paths != None:
            test_pitches, test_durations, test_voices = self._load_music_json(test_data_file_paths)

        self.train, self.train_offsets, self.train_files_attribution = self.pre_tokenize(train_data_file_paths)
        if valid_data_file_paths is not None:
            self.valid, self.valid_offsets, self.valid_files_attribution = self.pre_tokenize(valid_data_file_paths)
        if test_data_file_paths is not None:
            self.test, self.test_offsets, self.test_files_attribution = self.pre_tokenize(test_data_file_paths)

    def _load_music_json(self, json_file_paths):
        all_pitches = []
        all_durations = []
        all_voices = []
        for path in json_file_paths:
            pitches, durations, velocities = pitch_duration_velocity_lists_from_music_json_file(path)
            all_pitches.append([p for n, p in enumerate(pitches)])
            all_durations.append([d for n, d in enumerate(durations)])
            all_voices.append([v for n, v in enumerate(velocities)])
        return all_pitches, all_durations, all_voices

    def build_vocabulary(self, json_file_paths):
        # merge the logic from the other pre_tokenize step
        for path in json_file_paths:
            joint, joint_offsets, joint_files = self.pre_tokenize([path])
            for j in joint:
                if j not in self.dictionary.word2idx:
                    self.dictionary.add_word(j)

    def pre_tokenize(self, json_file_paths):
        joints = []
        joints_offsets = []
        joints_files = []
        for path in json_file_paths:
            pitches, durations, velocities = pitch_duration_velocity_lists_from_music_json_file(path)
            all_pitches = []
            all_durations = []
            all_voice_offsets = []
            if self.raster_scan:
                # need to get groups, then "flatten" SSSAAATTTBBB
                # then add marks between, track voice
                voice_step_counter = [0, 0, 0, 0]

                def split_list(iterable, splitter):
                    is_splitter = [n for n, el in enumerate(iterable) if el == splitter]
                    if is_splitter[0] != 0:
                        is_splitter = [0] + is_splitter
                    if is_splitter[-1] != len(iterable):
                        is_splitter = is_splitter + [len(iterable)]
                    return split_by_index(iterable, is_splitter)

                def split_by_index(iterable, is_splitter):
                    i_s = is_splitter

                    subs = []
                    for i, j in zip(i_s[:-1], i_s[1:]):
                        if i == 0:
                            subs.append(iterable[i:j])
                        else:
                            subs.append(iterable[i + 1:j])

                    # remove empties
                    subs = [s_ for s_ in subs if len(s_) > 0]
                    for s_ in subs:
                        if len(s_) == 0:
                            raise ValueError("Size zero split remains, fix this")
                    return subs, i_s

                def group_split_list(iterables, splitter_0):
                    i_0, splits = split_list(iterables[0], splitter_0)
                    res = [i_0]
                    for iter_ in iterables[1:]:
                        i_i, _ = split_by_index(iter_, splits)
                        res.append(i_i)
                    return res

                this_g = []
                for v in range(len(pitches)):
                    g_i = group_split_list([pitches[v], durations[v]], 99)
                    if v > 0:
                        for el in g_i:
                            assert len(el) == len(this_g[0][0])
                    this_g.append(g_i)
                # the joint thing is now the flattened version of this 99 SSAATB 99 SAATB 99, with voice and offset information added
                joint = []
                joint_offsets = []
                voice_offsets = [0, 0, 0, 0]
                for step in range(len(this_g[0][0])):
                    joint.append((99, 0))
                    joint_offsets.append((0, voice_offsets[0], 0))
                    for v in range(len(pitches)):
                        # 0 is pitch
                        # 1 is duration
                        p = this_g[v][0][step]
                        d = this_g[v][1][step]
                        assert len(p) == len(d)
                        for p_i, d_i in zip(p, d):
                            joint.append((p_i, d_i))
                            joint_offsets.append((v, voice_offsets[v], d_i))
                            voice_offsets[v] += d_i
                joint.append((99, 0))
                # voice -1 special symbols channel
                joint_offsets.append((-1, voice_offsets[0], 0))

                joints_offsets.extend(joint_offsets)
                joints.extend(joint)
                joints_files.extend([(path, _n) for _n, j in enumerate(joint)])
                joints_files.append((path, len(joint)))
            else:
                raise ValueError("Only support raster for now")
                # treat each voice as a separate "file" for now
                for v in self.voices:
                    _pitches = [p for n, p in enumerate(pitches) if n == v]
                    _durations = [d for n, d in enumerate(durations) if n == v]
                    _pitches = _pitches[0]
                    _durations = _durations[0]
                    assert len(_pitches) == len(_durations)
                    joint = []
                    for p, d in zip(_pitches, _durations):
                        if d == -1:
                            continue
                        else:
                            joint.append((p, d))
                    # include last for file separator symbol...
                    joints_offsets.extend([jo for jo in np.cumsum([0] + [j[1] for j in joint])[:-1]] + [0])
                    joint.append(self.file_separator_symbol)
                    joints.extend(joint)
                    joints_files.extend([(path, _n) for _n, j in enumerate(joint)])
                print("NYI: fix joint_offsets")
        assert len(joints) == len(joints_offsets)
        return joints, joints_offsets, joints_files

    def get_answer_groups_from_example(self, batch, batch_offsets):
        batch_return_answers = []
        batch_return_offsets = []
        batch_return_positions = []
        for t in range(batch.shape[1]):
            context_token = self.dictionary.word2idx[self.end_context_symbol]
            answer_token = self.dictionary.word2idx[self.answer_symbol]
            mask_token = self.dictionary.word2idx[self.mask_symbol]

            # +1 to include context boundary
            context_boundary = np.where(batch[:, t, 0] == context_token)[0][0] + 1
            mask_positions = np.where(batch[:context_boundary, t, 0] == mask_token)[0]

            # find last answer symbol
            end_point = np.where(batch[:, t, 0] == self.dictionary.word2idx[self.answer_symbol])[0][-1] + 1

            t1 = batch[context_boundary:end_point]
            t2 = batch_offsets[context_boundary:end_point]
            a1 = [self.dictionary.idx2word[ts] for ts in t1[:, t, 0]]
            assert len(a1) == len(t2)
            # t2 and a1 are off by one because t2 has 1 extra "closure" element, slice it out
            assert np.all(t2[-1, t] == np.array([-1, 0, 0]))
            _a1_durations = [aa1[1] for aa1 in a1]
            _t2_durations = t2[:, t, -1]
            # ensure the sequences line up
            assert all([_a == _t for _a, _t in zip(_a1_durations, _t2_durations) if _a > 0])
            assert all([_t == 0 for _a, _t in zip(_a1_durations, _t2_durations) if _a <= 0])

            # 2 separate checks because < 0 is "special" symbols in the iterator, should have gt duration of 0
            # get expected duration total for each answer, for each voice!
            gt_answers = []
            gt_offsets = []
            _pre = [_aa1 if _aa1 > 0 else None for _aa1 in _a1_durations]
            # collapse consequentive None values?
            split_idx = [_n for _n in range(len(_pre)) if _pre[_n] == None]
            if split_idx[0] != 0:
                split_idx.insert(0, 0)
            # if the last split idx isn't at the last element, add a dummy splitter
            if split_idx[-1] != (len(_pre) - 1):
                 # list grabs aren't inclusive, so we add 1 so that l[a:b] grabs elem b-1 (the last value in our data)
                 split_idx.append(len(_pre))

            boundaries = list(zip(split_idx[:-1], split_idx[1:]))
            for b in boundaries:
                # le or lt
                if b[0] + 1 < b[1]:
                    if b[0] > 0: # +1 to skip the boundary value of None
                        gt_answers.append(a1[b[0] + 1:b[1]])
                        gt_offsets.append(t2[b[0] + 1:b[1], t])
                    else:
                        gt_answers.append(a1[b[0]:b[1]])
                        gt_offsets.append(t2[b[0]:b[1], t])
                else:
                    if b[1] == (len(_pre) - 1):
                        # edge case where the last dummy chunk only has 1 elem
                        # skip it? last value is end indicator
                        #print("example 2")
                        #from IPython import embed; embed(); raise ValueError()
                        pass
                        #gt_answers.append(a1[b[0] + 1:])
                        #gt_offsets.append(t2[b[0] + 1:, t])
                    elif b[0] == 0:
                        gt_answers.append(a1[b[0]:b[1]])
                        gt_offsets.append(t2[b[0]:b[1], t])
                    else:
                        print("boundary issue")
                        from IPython import embed; embed(); raise ValueError()
                        raise ValueError("boundary issue")

            # now that we have gathered the answer groups, make sure there are the same number as mask positions
            try:
                assert len(gt_answers) == len(mask_positions)
                assert len(gt_offsets) == len(mask_positions)
            except:
                print("answers and offsets do not have same length!")
                from IPython import embed; embed(); raise ValueError()
            batch_return_answers.append(gt_answers)
            batch_return_offsets.append(gt_offsets)
            batch_return_positions.append(mask_positions)
        return batch_return_answers, batch_return_offsets, batch_return_positions


    def get_iterator(self, batch_size, random_seed, sequence_len=64, context_len=32, sample_percent=.15,
                     max_n_gram=8, _type="train"):
        # TODO: SIMPLIFY ARGS
        random_state = np.random.RandomState(random_seed)
        if _type == "train":
            content = self.train
            offsets = self.train_offsets
        elif _type == "valid":
            content = self.valid
            offsets = self.valid_offsets
        elif _type == "test":
            content = self.test
            offsets = self.test_offsets

        mask_symbol = self.mask_symbol
        answer_symbol = self.answer_symbol
        end_context_symbol = self.end_context_symbol
        file_separator_symbol = self.file_separator_symbol
        fill_symbol = self.fill_symbol
        fill_offset = (-1, 0, 0)
        def sample_minibatch(batch_size=batch_size, content=content, offsets=offsets,
                             random_state=random_state,
                             sequence_len=sequence_len,
                             context_len=context_len,
                             sample_percent=sample_percent,
                             max_n_gram=max_n_gram,
                             mask_symbol=mask_symbol,
                             answer_symbol=answer_symbol,
                             end_context_symbol=end_context_symbol,
                             file_separator_symbol=file_separator_symbol,
                             fill_offset=fill_offset,
                             fill_symbol=fill_symbol,
                             loader=self):
            while True:
                cur_batch = []
                cur_batch_offsets = []
                cur_batch_indices = []
                for i in range(batch_size):
                    # sample place to start element from the dataset
                    while True:
                        while True:
                            el = random_state.choice(len(content) - sequence_len - 1)
                            if el < (len(content) - 2 * (sequence_len + context_len)):
                                break

                        # find the nearest measure break and set start point
                        while True:
                            if content[el][0] == 99:
                                break
                            el = el + 1

                        cur_masked = content[el:el + sequence_len + context_len]
                        cur_offsets = offsets[el:el + sequence_len + context_len]
                        _times = np.array(cur_offsets)[:, 1]
                        if any([t > _times[-1] for t in _times]):
                            # skip ones that happen on the boundary of 2 files
                            # find it because time is not monotonically increasing
                            continue

                        if len(np.where(np.array([loader.dictionary.word2idx[cm] for cm in cur_masked]) == loader.dictionary.word2idx[file_separator_symbol])[0]) == 0:
                            # extra defensive check, be sure there's at least 1 boundary
                            if len(np.where(np.array([loader.dictionary.word2idx[cm] for cm in cur_masked]) == loader.dictionary.word2idx[(99, 0)])[0]) > 1:
                                break
                        else:
                            pass

                    # has to terminate on measure boundary otherwise we can't re-serialize (is non-causal)
                    boundary_points = np.where(np.array([loader.dictionary.word2idx[cm] for cm in cur_masked]) == loader.dictionary.word2idx[(99, 0)])[0]
                    if boundary_points[-1] != (len(cur_masked) - 1):
                        cur_masked = cur_masked[:boundary_points[-1] + 1]
                        cur_offsets = cur_offsets[:boundary_points[-1] + 1]

                    cur_batch_indices.append(el)

                    # change logic here, use bit mask instead, 2 loops 
                    cur_percent = .0
                    cur_bitmask = [0 for _ in range(len(cur_masked))]
                    n_steps = 0
                    n_steps_limit = 10 * sequence_len
                    while cur_percent < (sample_percent - .01 * sample_percent):
                        n_steps += 1
                        if n_steps >= n_steps_limit:
                            print("WARNING: more than (10 * the number of chunk elements) steps taken during masking/sampling")
                        # sample whether single word chunk or multiple
                        n_gram_type = random_state.choice(2)

                        # choices here:
                        # don't sample any measure marks!

                        if n_gram_type == 0:
                            # if it was 0, just mask a single word
                            mask_out_sz = 1
                        else:
                            mask_out_sz = random_state.choice(np.arange(2, max_n_gram + 1))

                        # can sample "vertical" (all notes in a time slice)
                        # sample horizontal "full voice"
                        # horizontal "groups of notes" (can cross boundary then we clean up)
                        n_gram_style = random_state.choice(2)
                        num_voices = len(np.unique(np.array(cur_offsets)[:, 0]))
                        if n_gram_style == 0:
                            mask_out_sz = random_state.choice(np.arange(2, num_voices))
                            # if there are exactly 2 boundary points, it's just start and end
                            if len(boundary_points) > 2:
                                block_index = random_state.choice(len(boundary_points) - 2)
                            else:
                                block_index = 0
                            l_i = boundary_points[block_index]
                            # +1 to include the measure mark
                            r_i = boundary_points[block_index + 1] + 1
                            sub_masked = cur_masked[l_i:r_i]
                            sub_offsets = cur_offsets[l_i:r_i]
                            sub_time_point = random_state.choice(np.unique(np.array(sub_offsets)[:, 1]))
                            # find entry for each voice that crosses the sub_time_point
                            elements_to_blank = []
                            sub_voices = random_state.choice(np.arange(2, num_voices + 1))
                            which_voices = np.arange(num_voices + 1)
                            random_state.shuffle(which_voices)
                            which_voices = [w for w in which_voices[:sub_voices]]
                            for v in sorted(np.unique(np.array(cur_offsets)[:, 0])):
                                this_voice_idx = np.where(np.array(cur_offsets)[:, 0] == v)[0].astype("int32")
                                all_crossed = []
                                for _ii in this_voice_idx:
                                    # find first point which is not strictly less than the sampled time point
                                    if cur_offsets[_ii][1] < sub_time_point:
                                        pass
                                    else:
                                        if len(all_crossed) > 0 and all_crossed[-1] != cur_offsets[_ii][1]:
                                            all_crossed.append(_ii)
                                            break
                                        else:
                                            all_crossed.append(_ii)
                                # only do it for voices from which_voices, this gives a random subset of all voices
                                if v not in which_voices:
                                    continue
                                # need to get the first one that crossed, which wasn't a special symbol aka 0 duration
                                non_blank_crossings = [ac for ac in all_crossed if cur_offsets[ac][2] != 0]
                                if len(non_blank_crossings) == 0:
                                    # sanity check that this condition happens rarely - can sample the very last measure mark
                                    # thus the only non_crossing is a special symbol, so we end up with 0 non_blank_crossings
                                    #print("empty non_crossings, continue")
                                    continue
                                else:
                                    elements_to_blank.append(non_blank_crossings[0])
                            cur_bitmask = [cb if n not in elements_to_blank else 1 for n, cb in enumerate(cur_bitmask)]
                            cur_percent = sum(cur_bitmask) / float(len(cur_bitmask))
                        elif n_gram_style == 1:
                            # only edge case here is to NEVER blank out 0 duration "special symbols"
                            # horizontal
                            # start point for slice
                            # context len?
                            mask_out_start = random_state.choice(np.arange(0, len(cur_masked) - (mask_out_sz + 1)))
                            mask_range = list(range(mask_out_start, mask_out_start + mask_out_sz))
                            cur_bitmask = [cb if n not in mask_range or cur_masked[n][-1] == 0. else 1 for n, cb in enumerate(cur_bitmask)]
                            cur_percent = sum(cur_bitmask) / float(len(cur_bitmask))

                    # make sure offset durations match data duration values
                    assert len(cur_masked) == len(cur_offsets)
                    for n, (cm, co) in enumerate(zip(cur_masked, cur_offsets)):
                        assert cm[1] == co[2]

                    # now that we have a bitmask, loop over and grab answers, mark chunk boundaries
                    up_flag = False
                    starts = []
                    stops = []
                    for _pos in range(len(cur_bitmask)):
                        if cur_bitmask[_pos] == 1:
                            if up_flag == False:
                                starts.append(_pos)
                            up_flag = True
                        else:
                            if up_flag == True:
                                stops.append(_pos)
                            up_flag = False

                    # if loop ends and we are still "up", close it off
                    if up_flag == True:
                        up_flag = False
                        # slices don't include the step itself
                        stops.append(len(cur_bitmask))
                    assert len(starts) == len(stops)

                    tmp_masked = (-9999, -9999)
                    tmp_offset = (-7777, -7777, -7777)
                    cur_answer_groups = []
                    cur_offset_groups = []
                    # create new answer for masked elements, fill with tmp token
                    for sta, sto in zip(starts, stops):
                        _a = copy.deepcopy([cur_masked[_i] for _i in range(sta, sto)])
                        _o = copy.deepcopy([cur_offsets[_i] for _i in range(sta, sto)])
                        cur_answer_groups.append(_a)
                        cur_offset_groups.append(_o)
                        cur_masked = [cur_masked[_i] if _i not in list(range(sta, sto)) else tmp_masked for _i in range(len(cur_masked))]
                        cur_offsets = [cur_offsets[_i] if _i not in list(range(sta, sto)) else tmp_offset for _i in range(len(cur_offsets))]

                    assert len(cur_answer_groups) == len(cur_offset_groups)
                    for cag, cog in zip(cur_answer_groups, cur_offset_groups):
                        assert len(cag) == len(cog)
                        # voice < 0 means it is a fill token
                        # assert that all fill tokens have 0 duration marks (-1, 0, 0)
                        assert all([cag[_i][1] == cog[_i][2] if cag[_i][0] >= 0 else cog[_i] == fill_offset for _i in range(len(cag))])

                    # check that everything is still aligned
                    for cm, co in zip(cur_masked, cur_offsets):
                        if cm[0] >= 0:
                            assert cm[1] == co[2]
                        else:
                            assert co[0] < 0

                    # now collapse the tmp token chunks to a single value, then replace that single value with single mask token
                    tmp_cur_masked = []
                    tmp_cur_offsets = []
                    assert len(cur_masked) == len(cur_offsets)
                    for _ii in range(len(cur_masked)):
                        if cur_masked[_ii] == tmp_masked:
                            if len(tmp_cur_masked) == 0:
                                tmp_cur_masked.append(tmp_masked)
                                tmp_cur_offsets.append(tmp_offset)
                            elif tmp_cur_masked[-1] != tmp_masked:
                                tmp_cur_masked.append(tmp_masked)
                                tmp_cur_offsets.append(tmp_offset)
                        else:
                            tmp_cur_masked.append(cur_masked[_ii])
                            tmp_cur_offsets.append(cur_offsets[_ii])

                    # check that everything is still aligned after reduction
                    for tcm, tco in zip(tmp_cur_masked, tmp_cur_offsets):
                        if tcm[0] >= 0:
                            assert tcm[1] == tco[2]
                        else:
                            assert tco[0] < 0

                    # check they have the same non-blank values
                    non_blank_offsets_a = [tco for tco in tmp_cur_offsets if tco[0] >= 0]
                    non_blank_offsets_b = [co for co in cur_offsets if co[0] >= 0]
                    assert len(non_blank_offsets_a) == len(non_blank_offsets_b)
                    assert all([a == b for a,b in zip(non_blank_offsets_a, non_blank_offsets_b)])

                    # check that chunk sizes match
                    sum_at_blanks_a = []
                    running_sum_a = 0
                    for _i in range(len(tmp_cur_offsets)):
                        if tmp_cur_offsets[_i][0] < 0:
                            if len(sum_at_blanks_a) >0 and running_sum_a != sum_at_blanks_a[-1]:
                                sum_at_blanks_a.append(running_sum_a)
                            elif len(sum_at_blanks_a) == 0:
                                sum_at_blanks_a.append(running_sum_a)
                        else:
                            running_sum_a += 1

                    sum_at_blanks_b = []
                    running_sum_b = 0
                    for _i in range(len(cur_offsets)):
                        if cur_offsets[_i][0] < 0:
                            if len(sum_at_blanks_b) > 0 and running_sum_b != sum_at_blanks_b[-1]:
                                sum_at_blanks_b.append(running_sum_b)
                            elif len(sum_at_blanks_b) == 0:
                                sum_at_blanks_b.append(running_sum_b)
                        else:
                            running_sum_b += 1

                    # check all chunk lengths are the same, via the cumulative sum over non-blanks
                    assert len(sum_at_blanks_a) == len(sum_at_blanks_b)
                    for a, b in zip(sum_at_blanks_a, sum_at_blanks_b):
                        assert a == b

                    cur_masked = tmp_cur_masked
                    cur_offsets = tmp_cur_offsets
                    special_symbols = [mask_symbol,
                                       answer_symbol,
                                       end_context_symbol,
                                       file_separator_symbol,
                                       tmp_masked]
                    # now loop through and replace the tmp values with the correct "fill" values
                    for _ii in range(len(cur_masked)):
                        # check that offsets and masked are aligned
                        if cur_offsets[_ii] == fill_offset:
                            assert cur_masked[_ii] in special_symbols

                        if cur_masked[_ii] == tmp_masked:
                            assert cur_offsets[_ii] == tmp_offset
                            cur_masked[_ii] = mask_symbol
                            cur_offsets[_ii] = fill_offset

                    # finally, append on the answers and offsets, separating with the answer separator and the offset fill value
                    cur_answer = [end_context_symbol]
                    cur_answer_offsets = [fill_offset]
                    for _ii in range(len(cur_answer_groups)):
                        # rewrite the answers and offsets to be "linear", SSAATTBB order
                        offs = cur_offset_groups[_ii]
                        ans = cur_answer_groups[_ii]
                        for _v in range(4):
                            assert len(offs) == len(ans)
                            v_ans = [a for a, o in zip(ans, offs) if o[0] == _v]
                            v_offs = [o for o in offs if o[0] == _v]
                            cur_answer.extend(v_ans)
                            cur_answer_offsets.extend(v_offs)
                        cur_answer.append(answer_symbol)
                        cur_answer_offsets.append(fill_offset)
                    cur_batch.append(cur_masked + cur_answer)
                    cur_batch_offsets.append(cur_offsets + cur_answer_offsets)
                    assert len(cur_batch[-1]) == len(cur_batch_offsets[-1])

                max_len = max([len(b) for b in cur_batch])
                cur_batch_masks = [[0] * len(b) + [1] * (max_len - len(b)) for b in cur_batch]
                cur_batch = [b + [fill_symbol] * (max_len - len(b)) for b in cur_batch]

                token_batch = [[self.dictionary.word2idx[bi] for bi in b] for b in cur_batch]
                cur_batch_offsets = [b + [fill_offset] * (max_len - len(b)) for b in cur_batch_offsets]
                token_batch = np.array(token_batch).T[..., None]
                cur_batch_masks = np.array(cur_batch_masks).T
                cur_batch_offsets = np.array(cur_batch_offsets).transpose(1, 0, 2)
                yield token_batch, cur_batch_masks, cur_batch_offsets, cur_batch_indices
                #yield np.array(token_batch).T[..., None], np.array(cur_batch_masks).T, np.array(cur_batch_offsets).transpose(1, 0, 2)
        return sample_minibatch()


class MusicPklCorpus(object):
    def __init__(self,
                 pkl_path,
                 tokenization_type="fixed_vocabulary",
                 use_only_these_valid_indices=None,
                 measure_quarters=None,
                 quantization="16th",
                 no_measure_mark=True,
                 force_column=True,
                 preserve_shift_order=False,
                 add_eof_mark=True,
                 eof_mark_token=None,
                 repeat_data_once=True,
                 show_skip_error=False):
        raise ValueError("deprecated")
        if force_column == False:
            if no_measure_mark == True:
                raise ValueError("Row based rasterization requires measure marks! force_column=False, no_measure_mark=True cannot be specified")
        self.pkl_path = pkl_path
        self.no_measure_mark = no_measure_mark
        self.force_column = force_column
        self.measure_quarters = measure_quarters
        self.use_only_these_valid_indices = use_only_these_valid_indices
        self.tokenization_type = tokenization_type
        self.quantization = quantization
        self.add_eof_mark = add_eof_mark
        self.eof_mark_token = eof_mark_token
        self.repeat_data_once = repeat_data_once
        self.show_skip_error = show_skip_error
        self.preserve_shift_order = preserve_shift_order
        if self.quantization != "16th":
            raise ValueError("Only '16th' quantization currently supported")
        self._span_mult = 4
        self._match_type = "pair"
        # match_type can be "single", "pair", "triple", "full"

        with open(pkl_path, 'rb') as p:
            data = pickle.load(p, encoding='latin1')
        train_data = copy.deepcopy(data['train'])
        valid_data = copy.deepcopy(data['valid'])
        test_data = copy.deepcopy(data['test'])

        """
        # theoretical is 21-108 inclusive (range(21, 109))
        # practical is 36-81 inclusive
        all_notes = set()
        for d in [train_data, valid_data, test_data]:
            for di in d:
                try:
                    li = np.array(di)
                    # if it isnt 2d it made an object array
                    li.shape[0]
                    li.shape[1]
                    l = [el for el in li.ravel()]
                    all_notes = all_notes | set(l)
                except:
                    continue
        """

        vocab_opts = ["fixed_vocabulary", "standard_nlp"]
        if self.tokenization_type == "fixed_vocabulary":
            self.dictionary = LookupDictionary()
            self.build_fixed_vocabulary()
        elif self.tokenization_type in vocab_opts:
            self.dictionary = LookupDictionary()
            self.build_vocabulary(train_data, is_valid=False)
            self.build_vocabulary(valid_data, is_valid=True)
        else:
            raise ValueError("Unknown tokenization_type {} given! Options are {}".format(self.tokenization_type, vocab_opts))

        # dont build vocabulary over test data...
        # edit: do we force the same output size as baselines?

        self.train, self.train_file_attribution = self.tokenize(train_data, is_valid=False)
        self.valid, self.valid_file_attribution = self.tokenize(valid_data, is_valid=True)

        logger.info("Number of training tokens: {}".format(len(self.train)))
        logger.info("Number of valid tokens: {}".format(len(self.valid)))
        logger.info("Vocabulary size: {}, using tokenization_type {}".format(len(self.dictionary.idx2word), self.tokenization_type))

    def build_fixed_vocabulary(self):
        for word in range(21, 109):
            self.dictionary.add_word(word)
        if self.no_measure_mark is False:
            # measure mark is 999
            self.dictionary.add_word(999)

    def build_vocabulary(self, list_data, is_valid):
        # always build vocabulary over all values
        words, _ = self.tokenize(list_data, run_pre_tokenization_instead=True, is_valid=False)
        for word in words:
            self.dictionary.add_word(word)

    def tokenize(self, list_data, run_pre_tokenization_instead=False, is_valid=False):
        full_flat_list = []
        # train step index, step into example index, voice value, shift value
        full_flat_index_attribution = []
        # span mult of 4 assumes 16th note quantization
        span_mult = self._span_mult

        full_flat_jj_lists = []
        full_flat_jj_index_attribution = []
        # natural key, then others by half step
        # TODO: "logical" augmentation order?
        shift_order_scheme = [0] + list(range(-6, 0)) + list(range(1, 6))
        for jj in shift_order_scheme:
            full_flat_jj_lists.append([])
            full_flat_jj_index_attribution.append([])

        for ii in range(len(list_data)):
            try:
                this_shifted = np.array(list_data[ii])
                this_shifted.shape[0]
                this_shifted.shape[1]
            except:
                if self.show_skip_error:
                    logger.info("error creating uniform array on index {}".format(ii))
                continue
            if is_valid:
               if self.use_only_these_valid_indices is not None:
                   if ii not in self.use_only_these_valid_indices:
                       continue

            for nn, jj in enumerate(shift_order_scheme):
                """
                tmp = [ee for ee in full_flat_list if ee != 999]
                if len(tmp) > 0 and min(tmp) < 21:
                    print("just got min")
                    from IPython import embed; embed(); raise ValueError()

                if len(tmp) > 0 and max(tmp) > 108:
                    print("just got max")
                    from IPython import embed; embed(); raise ValueError()
                """

                _shift = int(jj)
                _measure_fill = 999 - _shift
                try:
                    this_shifted = np.array(list_data[ii]) + _shift
                except:
                    if self.show_skip_error:
                        logger.info("error creating uniform array on index {}".format(ii))
                    continue

                if this_shifted.ravel().min() < min(self.dictionary.word2idx.keys()) or this_shifted.ravel().max() > max(self.dictionary.word2idx.keys()):
                    if self.show_skip_error:
                        logger.info("data augmentation {} puts this piece {} out of vocabulary bounds, skipping".format(_shift, ii))
                    continue

                if self.force_column:
                    if self.no_measure_mark:
                        full_flat_jj_lists[nn].append([el for el in this_shifted.ravel()])
                        full_flat_jj_index_attribution[nn].append([(ii, _shift) for (n, el) in enumerate(this_shifted.ravel())])
                        #full_flat_list.extend([el for el in this_shifted.ravel()])
                        #full_flat_index_attribution.extend([(ii, _shift) for (n, el) in enumerate(this_shifted.ravel())])
                    elif self.no_measure_mark is False and self.measure_quarters is not None:
                        arr = np.array(list_data[ii])
                        change_0 = np.diff(arr[:, 0], prepend=arr[0, 0]) != 0
                        change_1 = np.diff(arr[:, 1], prepend=arr[0, 1]) != 0
                        change_2 = np.diff(arr[:, 2], prepend=arr[0, 2]) != 0
                        change_3 = np.diff(arr[:, 3], prepend=arr[0, 3]) != 0
                        changes = [a for a in np.where(change_0 & change_1 & change_2 & change_3)[0]]
                        # assume 16th span here
                        theoretical = list(np.arange(0, len(list_data[ii]), span_mult * self.measure_quarters)[1:])
                        last_match = None
                        count = 0
                        for t in theoretical:
                             if t in changes:
                                 last_match = t
                                 count += 1

                        if last_match is None:
                            continue
                        assert self.measure_quarters is not None
                        if len(arr) > last_match + self.measure_quarters:
                            last_match = last_match + self.measure_quarters
                        if count >= 8:
                            # if at least 8 groups matched the theoretical, keep as much as we can and use it

                            # need to add measure marks at each of the change points, and the end? 
                            this_arr = np.array(arr)
                            # fill backwards so putting the values in doesn't confuse the count
                            for t in theoretical[::-1]:
                                if t >= last_match:
                                    continue
                                if len(this_arr[:t]) >= last_match:
                                    continue
                                this_arr = np.concatenate((this_arr[:t], np.array(4 * [_measure_fill])[None], this_arr[t:]), axis=0)
                            this_arr = np.concatenate((np.array(4 * [_measure_fill])[None], this_arr), axis=0)
                            this_flat_list = [el for el in this_arr.ravel() + _shift]
                            # this attribution is wrong for step...
                            this_flat_index_attribution = [(ii, _shift) for (n, el) in enumerate(this_arr.ravel())]
                            out_flat_list = []
                            out_flat_index_attribution = []
                            # remove duplicate measure marks
                            was_mark = False
                            lst = -1
                            for _n in range(len(this_flat_list)):
                                if this_flat_list[_n] == lst:
                                    if this_flat_list[_n] == 999:
                                        last = this_flat_list[_n]
                                        continue
                                lst = this_flat_list[_n]
                                out_flat_list.append(this_flat_list[_n])
                                out_flat_index_attribution.append(this_flat_index_attribution[_n])
                            full_flat_jj_lists[nn].append(out_flat_list)
                            full_flat_jj_index_attribution[nn].append(out_flat_index_attribution)
                            #full_flat_list.extend(out_flat_list)
                            #full_flat_index_attribution.extend([out_flat_index_attribution])
                    elif self.measure_mark is None:
                        print("support auto??")
                        from IPython import embed; embed(); raise ValueError()
                else:
                    # TODO: Backport this logic to columnar
                    flattened = []
                    arr = np.array(list_data[ii])
                    change_0 = np.diff(arr[:, 0], prepend=arr[0, 0]) != 0
                    change_1 = np.diff(arr[:, 1], prepend=arr[0, 1]) != 0
                    change_2 = np.diff(arr[:, 2], prepend=arr[0, 2]) != 0
                    change_3 = np.diff(arr[:, 3], prepend=arr[0, 3]) != 0

                    if self._match_type == "full":
                        changes = [a for a in np.where(change_0 & change_1 & change_2 & change_3)[0]]
                        change_group = [changes]
                    elif self._match_type == "single":
                        changes0 = [a for a in np.where(change_0)[0]]
                        changes1 = [a for a in np.where(change_1)[0]]
                        changes2 = [a for a in np.where(change_2)[0]]
                        changes3 = [a for a in np.where(change_3)[0]]

                        change_group = [changes0, changes1, changes2, changes3]
                    elif self._match_type == "pair":
                        changes01 = [a for a in np.where(change_0 & change_1)[0]]
                        changes02 = [a for a in np.where(change_0 & change_2)[0]]
                        changes03 = [a for a in np.where(change_0 & change_3)[0]]
                        changes12 = [a for a in np.where(change_1 & change_2)[0]]
                        changes13 = [a for a in np.where(change_1 & change_3)[0]]
                        changes23 = [a for a in np.where(change_2 & change_3)[0]]

                        change_group = [changes01, changes02, changes03, changes12, changes13, changes23]
                    elif self._match_type == "triple":
                        changes012 = [a for a in np.where(change_0 & change_1 & change_2)[0]]
                        changes013 = [a for a in np.where(change_0 & change_1 & change_3)[0]]
                        changes023 = [a for a in np.where(change_0 & change_2 & change_3)[0]]
                        changes123 = [a for a in np.where(change_1 & change_2 & change_3)[0]]

                        change_group = [changes012, changes013, changes023, changes123]
                    else:
                        raise ValueError("Unknown self._match_type {}".format(self._match_type))

                    theoretical = list(np.arange(0, len(list_data[ii]), span_mult * self.measure_quarters)[1:])
                    last_match = None
                    count = 0
                    for t in theoretical:
                         any_checks = any([t in c for c in change_group])
                         if any_checks:
                             last_match = t
                             count += 1

                    if last_match is None:
                        continue

                    assert self.measure_quarters is not None
                    if len(arr) > last_match + self.measure_quarters:
                        last_match = last_match + self.measure_quarters
                    if count >= 8:
                        # if at least 8 groups matched the theoretical, keep as much as we can and use it
                        # need to add measure marks at each of the change points, and the end? 
                        this_arr = np.array(arr)
                        # fill backwards so putting the values in doesn't confuse the count
                        for t in theoretical[::-1]:
                            if t >= last_match:
                                continue
                            if len(this_arr[:t]) >= last_match:
                                continue
                            # append all 4 frows, with a measure mark at the end...
                            new_flattened = []
                            new_flattened.extend([e for e in this_arr[t:, 0]])
                            new_flattened.extend([e for e in this_arr[t:, 1]])
                            new_flattened.extend([e for e in this_arr[t:, 2]])
                            new_flattened.extend([e for e in this_arr[t:, 3]])
                            new_flattened.append(_measure_fill)
                            flattened = new_flattened + flattened
                            this_arr = this_arr[:t]
                        # do the last values (0 to first t)
                        new_flattened = []
                        new_flattened.extend([e for e in this_arr[:t, 0]])
                        new_flattened.extend([e for e in this_arr[:t, 1]])
                        new_flattened.extend([e for e in this_arr[:t, 2]])
                        new_flattened.extend([e for e in this_arr[:t, 3]])
                        new_flattened.append(_measure_fill)

                        flattened = new_flattened + flattened
                        this_flat_list = [el + _shift for el in flattened]
                        this_flat_index_attribution = [(ii, _shift) for (n, el) in enumerate(flattened)]

                        full_flat_jj_lists[nn].append(this_flat_list)
                        full_flat_jj_index_attribution[nn].append(this_flat_index_attribution)
                        #full_flat_list.extend(this_flat_list)
                        #full_flat_index_attribution.extend([this_flat_index_attribution])

        if self.preserve_shift_order:
            raise ValueError("TODO: impl preserve_shift_order=True")
        else:
            hash_seed = hash(len(full_flat_jj_lists[0]))
            shuff_rs = np.random.RandomState(hash_seed)

            # create random sample order, shift order pairs
            sample_order = list(range(len(full_flat_jj_lists[0])))
            shuff_rs.shuffle(sample_order)
            sample_shift_pairs = []
            for o in sample_order:
                shift_order = list(range(len(full_flat_jj_lists)))
                shuff_rs.shuffle(shift_order)
                # randomize the shift order seen for every example
                for so in shift_order:
                    sample_shift_pairs.append((so, o))
            # shuffle again, should be totally random example/shift ordering
            shuff_rs.shuffle(sample_shift_pairs)
            for ssp in sample_shift_pairs:
                # do it twice if we have the repeat data flag - hypothesis is that this
                # envourages the model's "memory" to repeat/learn repeat structures
                full_flat_list.extend(full_flat_jj_lists[ssp[0]][ssp[1]])
                full_flat_index_attribution.extend(full_flat_jj_index_attribution[ssp[0]][ssp[1]])
                if self.add_eof_mark:
                    full_flat_list.append(self.eof_mark_token)
                    full_flat_index_attribution.append(self.eof_mark_token)
                if self.repeat_data_once:
                    full_flat_list.extend(full_flat_jj_lists[ssp[0]][ssp[1]])
                    full_flat_index_attribution.extend(full_flat_jj_index_attribution[ssp[0]][ssp[1]])
                    if self.add_eof_mark:
                        full_flat_list.append(self.eof_mark_token)
                        full_flat_index_attribution.append(self.eof_mark_token)
        # now construct full flat list and full flat index attribution in order to avoid any issue with data ordering
        # TODO: add new song token?
        if run_pre_tokenization_instead:
            return [el for el in full_flat_list if el != self.eof_mark_token], [el for el in full_flat_index_attribution if el != self.eof_mark_token]
        else:
            full_token_list = [self.dictionary.word2idx[el] if el != self.eof_mark_token else self.eof_mark_token for el in full_flat_list]
            return full_token_list, full_flat_index_attribution


class MusicJSONRowRasterCorpus(object):
    def __init__(self, train_data_file_paths, valid_data_file_paths=None, test_data_file_paths=None,
                 measure_quarters=8,
                 max_vocabulary_size=-1,
                 add_eos=False,
                 force_column=False,
                 no_measure_mark=False,
                 tokenization_fn="row_flatten",
                 default_velocity=120, quantization_rate=.25, n_voices=4,
                 separate_onsets=True, onsets_boundary=100):
        raise ValueError("deprecated")
        """
        """
        self.dictionary = LookupDictionary()
        self.measure_quarters = measure_quarters
        self.max_vocabulary_size = max_vocabulary_size

        self.add_eos = add_eos
        self.no_measure_mark = no_measure_mark
        self.tokenization_fn = tokenization_fn
        self.default_velocity = default_velocity
        self.quantization_rate = quantization_rate
        self.n_voices = n_voices
        self.separate_onsets = separate_onsets
        self.onsets_boundary = onsets_boundary
        self.force_column = force_column

        if tokenization_fn == "row_flatten":
            def tk(arr):
                print("tokenize")
                from IPython import embed; embed(); raise ValueError()
                t = [el for el in arr.ravel()]
                if add_eos:
                    # 2 measures of silence are the eos
                    return t + [0] * 32
                else:
                    return t
            self.tokenization_fn = tk
        else:
            raise ValueError("Unknown tokenization_fn {}".format(tokenization_fn))

        """
        self.voices = voices
        self.mask_symbol = (-1, -1)
        self.answer_symbol = (-2, -2)
        self.end_context_symbol = (-3, -3)
        self.file_separator_symbol = (-4, -4)
        self.fill_symbol = (-5, -5)
        self.special_symbols = [self.mask_symbol, self.answer_symbol, self.end_context_symbol, self.file_separator_symbol, self.fill_symbol]
        self.raster_scan = raster_scan
        """

        self.train_data_file_paths = train_data_file_paths
        self.valid_data_file_paths = valid_data_file_paths

        train_pitches, train_durations, train_velocities, train_updated_files = self._load_music_json(train_data_file_paths)
        self.train_data_file_paths = train_updated_files

        """
        self.dictionary.add_word(self.mask_symbol)
        self.dictionary.add_word(self.answer_symbol)
        self.dictionary.add_word(self.end_context_symbol)
        self.dictionary.add_word(self.file_separator_symbol)
        self.dictionary.add_word(self.fill_symbol)
        """

        self.build_vocabulary(train_data_file_paths)
        if valid_data_file_paths != None:
            valid_pitches, valid_durations, valid_velocities, valid_updated_files = self._load_music_json(valid_data_file_paths)
            self.valid_data_file_paths = valid_updated_files
            self.build_vocabulary(self.valid_data_file_paths)
        if test_data_file_paths != None:
            test_pitches, test_durations, test_velocities, test_updated_files = self._load_music_json(test_data_file_paths)
            self.test_data_file_paths = test_updated_files

        self.train, self.train_files_attribution = self.tokenize(train_data_file_paths)
        if valid_data_file_paths is not None:
            self.valid, self.valid_files_attribution = self.tokenize(valid_data_file_paths)
        if test_data_file_paths is not None:
            self.test, self.test_files_attribution = self.tokenize(test_data_file_paths)

    def _load_music_json(self, json_file_paths, skip_invalid=True, verbose=True):
        all_pitches = []
        all_durations = []
        all_velocities = []
        skips = 0
        nonskips = 0
        # 8, 2k nonskip 1k skip
        # 16, 3360 nonskips 624 skips
        nonskip_paths = []
        for path in json_file_paths:
            """
            pitches, durations, velocities = pitch_duration_velocity_lists_from_music_json_file(path,
                                                                                                force_quarters_match=True,
                                                                                                trim_uneven_last=True,
                                                                                                measure_quarters=self.measure_quarters)
            for i in range(len(pitches)):
                dij = [durations[i][j] if durations[i][j] != -1 else self.quantization_rate for j in range(len(durations[i]))]
                if not all([dijel % self.quantization_rate == 0. for dijel in dij]):
                    raise ValueError("invalid quantization value")
            all_pitches.append([p for n, p in enumerate(pitches)])
            # set measure marks to minimum interval
            all_durations.append([d for n, d in enumerate(durations)])
            all_velocities.append([v for n, v in enumerate(velocities)])
            nonskip_paths.append(path)
            nonskips += 1
            """
            try:
                pitches, durations, velocities = pitch_duration_velocity_lists_from_music_json_file(path,
                                                                                                    force_quarters_match=True,
                                                                                                    trim_uneven_last=True,
                                                                                                    measure_quarters=self.measure_quarters)
                for i in range(len(pitches)):
                    dij = [durations[i][j] if durations[i][j] != -1 else self.quantization_rate for j in range(len(durations[i]))]
                    if not all([dijel % self.quantization_rate == 0. for dijel in dij]):
                        raise ValueError("invalid quantization value")
                all_pitches.append([p for n, p in enumerate(pitches)])
                # set measure marks to minimum interval
                all_durations.append([d for n, d in enumerate(durations)])
                all_velocities.append([v for n, v in enumerate(velocities)])
                nonskip_paths.append(path)
                nonskips += 1
            except:
                skips += 1
                continue
        if skips > 0:
            if verbose:
                logger.info("Skipped {} files due to force_quarters_match settings...".format(skips))
                logger.info("Using {} files which satisfy force_quarters_match settings...".format(nonskips))

        assert len(all_pitches) == len(nonskip_paths)
        assert len(all_pitches) == len(all_durations)
        assert len(all_pitches) == len(all_velocities)
        for _i in range(len(all_pitches)):
            assert len(all_pitches[_i]) == len(all_durations[_i])
            assert len(all_pitches[_i]) == len(all_velocities[_i])
            for _j in range(len(all_pitches[_i])):
                assert len(all_pitches[_i][_j]) == len(all_durations[_i][_j])
                assert len(all_pitches[_i][_j]) == len(all_velocities[_i][_j])

        # just convert all pitches and durations to piano roll directly...
        return all_pitches, all_durations, all_velocities, nonskip_paths

    def build_vocabulary(self, json_file_paths):
        """Tokenizes a text file."""

        words, _ = self.tokenize(json_file_paths, return_pre_tokenization_instead=True)

        for word in words:
            self.dictionary.add_word(word)

    def tokenize(self, paths, return_pre_tokenization_instead=False, inspect=False):
        """Tokenizes a text file."""
        # just use the load function to get the data...
        pitches, durations, velocities, updated_files = self._load_music_json(paths, verbose=False)
        new_durations = []
        for i in range(len(pitches)):
            # make a piano roll here
            new_durations_i = []
            for j in range(len(pitches[i])):
                assert pitches[i][j][-1] == 99
                dij = [durations[i][j][el] if durations[i][j][el] != -1 else self.quantization_rate for el in range(len(durations[i][j]))]
                new_durations_i.append(dij)
            new_durations.append(new_durations_i)

        rolls = []
        voice_rolls = []
        for i in range(len(pitches)):
            this_roll = []
            this_voice_roll = []
            # find chunk boundaries
            marks = np.where(np.array(pitches[i][0]) == 99 )[0]
            if marks[0] != 0:
                marks = np.concatenate(([0], marks))

            n_marks = len(marks)
            assert n_marks > 1
            # -1 because we pair them off
            for _c in range(n_marks - 1):
                shared_chunk_roll = []
                shared_voice_roll = []
                for j in range(len(pitches[i])):
                    marks = np.where(np.array(pitches[i][j]) == 99)[0]
                    if marks[0] != 0:
                        marks = np.concatenate(([0], marks))
                    bs = list(zip(marks[:-1], marks[1:]))
                    bi = bs[_c][0]
                    bj = bs[_c][1]
                    # write em out in chunks
                    # skip the actual mark, add in manually once all are written
                    chunk_roll = []
                    chunk_voice_roll = []
                    for el in range(bi, bj):
                        if pitches[i][j][el] == 99:
                            continue
                        assert durations[i][j][el] % self.quantization_rate == 0.
                        n_q_steps = int(durations[i][j][el] / self.quantization_rate)
                        assert n_q_steps > 0
                        cr = [pitches[i][j][el]] * n_q_steps
                        cvr = [j] * n_q_steps
                        if self.separate_onsets:
                            cr[0] += self.onsets_boundary
                        chunk_roll.extend(cr)
                        chunk_voice_roll.extend(cvr)
                    shared_chunk_roll.extend(chunk_roll)
                    shared_voice_roll.extend(chunk_voice_roll)
                if self.force_column:
                    assert len(shared_chunk_roll) % self.n_voices == 0
                    chunk_sz = int(len(shared_chunk_roll) / self.n_voices)
                    chunks = []
                    voice_chunks = []
                    for _ii in range(self.n_voices):
                        _ll = _ii * chunk_sz
                        _rr = (_ii + 1) * chunk_sz
                        cc = shared_chunk_roll[_ll:_rr]
                        vv = shared_voice_roll[_ll:_rr]
                        assert all([vv[0] == vvi for vvi in vv])
                        chunks.append(cc)
                        voice_chunks.append(vv)
                    for cc in chunks:
                        assert len(cc) == len(chunks[0])
                    column_shared_chunk_roll = []
                    column_shared_voice_roll = []
                    for _ii in range(len(chunks[0])):
                        for _jj in range(len(chunks)):
                            column_shared_chunk_roll.append(chunks[_jj][_ii])
                            column_shared_voice_roll.append(voice_chunks[_jj][_ii])
                    shared_voice_roll = column_shared_voice_roll
                    shared_chunk_roll = column_shared_chunk_roll
                if self.no_measure_mark:
                    if not self.force_column:
                        raise ValueError("Unable to skip measure marks (no_measure_mark=True) with row rasterization (force_column=False)\n",
                                         "set force_column=True to skip measure marks")
                    # can we just skip it?
                    pass
                else:
                    shared_chunk_roll.append(99)
                    shared_voice_roll.append(-1)
                this_roll.extend(shared_chunk_roll)
                this_voice_roll.extend(shared_voice_roll)
            rolls.append(this_roll)
            voice_rolls.append(this_voice_roll)

        ids = []
        file_attr = []
        for i, roll in enumerate(rolls):
            this_file_attr = updated_files[i]
            for word in roll:
                if return_pre_tokenization_instead:
                    token = word
                else:
                    if word in self.dictionary.word2idx:
                        token = self.dictionary.word2idx[word]
                    else:
                        token = self.dictionary.word2idx["<unk>"]
                ids.append(token)
                file_attr.append(this_file_attr)
        return ids, file_attr


class MusicJSONRasterCorpus(object):
    def __init__(self, train_data_file_paths, valid_data_file_paths=None, test_data_file_paths=None,
                 max_vocabulary_size=-1,
                 add_eos=False,
                 tokenization_fn="flatten",
                 default_velocity=120, quantization_rate=.25, n_voices=4,
                 separate_onsets=True, onsets_boundary=100):
        raise ValueError("deprecated")
        """
        """
        self.dictionary = LookupDictionary()

        self.max_vocabulary_size = max_vocabulary_size
        self.default_velocity = default_velocity
        self.quantization_rate = quantization_rate
        self.n_voices = n_voices
        self.separate_onsets = separate_onsets
        self.onsets_boundary = onsets_boundary
        self.add_eos = add_eos

        if tokenization_fn == "flatten":
            def tk(arr):
                t = [el for el in arr.ravel()]
                if add_eos:
                    # 2 measures of silence are the eos
                    return t + [0] * 32
                else:
                    return t
            self.tokenization_fn = tk
        else:
            raise ValueError("Unknown tokenization_fn {}".format(tokenization_fn))

        base = [fp for fp in train_data_file_paths]
        if valid_data_file_paths is not None:
            base = base + [fp for fp in valid_data_file_paths]
        if test_data_file_paths is not None:
            base = base + [fp for fp in test_data_file_paths]

        self.build_vocabulary(base)

        if self.max_vocabulary_size > -1:
            self.dictionary._prune_to_top_k_counts(self.max_vocabulary_size)

        self.train = self.tokenize(train_data_file_paths)
        if valid_data_file_paths is not None:
            self.valid = self.tokenize(valid_data_file_paths)
        if test_data_file_paths is not None:
            self.test = self.tokenize(test_data_file_paths)

    def build_vocabulary(self, json_file_paths):
        """Tokenizes a text file."""
        for path in json_file_paths:
            assert os.path.exists(path)
            roll = piano_roll_from_music_json_file(path,
                                                   default_velocity=self.default_velocity,
                                                   quantization_rate=self.quantization_rate,
                                                   n_voices=self.n_voices,
                                                   separate_onsets=self.separate_onsets,
                                                   onsets_boundary=self.onsets_boundary,
                                                   as_numpy=True)
            words = self.tokenization_fn(roll)

            for word in words:
                self.dictionary.add_word(word)

    def tokenize(self, paths):
        """Tokenizes a text file."""
        ids = []
        for path in paths:
            assert os.path.exists(path)
            # Add words to the dictionary
            roll = piano_roll_from_music_json_file(path,
                                                   default_velocity=self.default_velocity,
                                                   quantization_rate=self.quantization_rate,
                                                   n_voices=self.n_voices,
                                                   separate_onsets=self.separate_onsets,
                                                   onsets_boundary=self.onsets_boundary,
                                                   as_numpy=True)
            words = self.tokenization_fn(roll)
            for word in words:
                if word in self.dictionary.word2idx:
                    token = self.dictionary.word2idx[word]
                else:
                    token = self.dictionary.word2idx["<unk>"]
                ids.append(token)
        return ids


class MusicJSONFlatKeyframeMeasureCorpus(object):
    def __init__(self, train_data_file_paths, valid_data_file_paths=None, test_data_file_paths=None,
                 max_vocabulary_size=-1,
                 add_measure_marks=True,
                 tokenization_fn="measure_flatten",
                 default_velocity=120, n_voices=4,
                 measure_value=99,
                 transition_value=9999,
                 fill_value=-1,
                 separate_onsets=True, onsets_boundary=100):
        raise ValueError("deprecated")
        """
        """
        self._make_dictionaries()

        self.max_vocabulary_size = max_vocabulary_size
        self.default_velocity = default_velocity
        self.n_voices = n_voices
        self.measure_value = measure_value
        self.fill_value = fill_value
        self.separate_onsets = separate_onsets
        self.onsets_boundary = onsets_boundary
        self.add_measure_marks = add_measure_marks
        self.transition_value = transition_value

        if tokenization_fn != "measure_flatten":
            raise ValueError("Only default tokenization_fn currently supported")

        base = [fp for fp in train_data_file_paths]
        if valid_data_file_paths is not None:
            base = base + [fp for fp in valid_data_file_paths]
        if test_data_file_paths is not None:
            base = base + [fp for fp in test_data_file_paths]

        self.build_vocabulary(base)

        self.train = self.tokenize(train_data_file_paths)
        if valid_data_file_paths is not None:
            self.valid = self.tokenize(valid_data_file_paths)
        if test_data_file_paths is not None:
            self.test = self.tokenize(test_data_file_paths)

    def _make_dictionaries(self):
        self.fingerprint_features_zero_dictionary = LookupDictionary()
        self.fingerprint_features_one_dictionary = LookupDictionary()
        self.duration_features_zero_dictionary = LookupDictionary()
        self.duration_features_mid_dictionary = LookupDictionary()
        self.duration_features_one_dictionary = LookupDictionary()
        self.voices_dictionary = LookupDictionary()

        self.centers_0_dictionary = LookupDictionary()
        self.centers_1_dictionary = LookupDictionary()
        self.centers_2_dictionary = LookupDictionary()
        self.centers_3_dictionary = LookupDictionary()

        self.keypoint_dictionary = LookupDictionary()
        self.keypoint_base_dictionary = LookupDictionary()
        self.keypoint_durations_dictionary = LookupDictionary()

        self.target_0_dictionary = LookupDictionary()
        self.target_1_dictionary = LookupDictionary()
        self.target_2_dictionary = LookupDictionary()
        self.target_3_dictionary = LookupDictionary()

    def _process(self, path):
        assert os.path.exists(path)
        pitch, duration, velocity = pitch_duration_velocity_lists_from_music_json_file(path,
                                                                                       default_velocity=self.default_velocity,
                                                                                       n_voices=self.n_voices,
                                                                                       measure_value=self.measure_value,
                                                                                       fill_value=self.fill_value)
        return self._features_from_lists(pitch, duration, velocity)

    def _features_from_lists(self, pitch, duration, velocity):
        def isplit(iterable, splitters):
            return [list(g) for k,g in itertools.groupby(iterable,lambda x:x in splitters) if not k]

        group = []
        for v in range(len(pitch)):
            # per voice, do split and merge
            s_p = isplit(pitch[v], [self.measure_value])
            s_d = isplit(duration[v], [self.fill_value])
            s_v = isplit(velocity[v], [self.fill_value])
            group.append([s_p, s_d, s_v])

        not_merged = True
        # all should be the same length in terms of measures so we can merge
        try:
            assert len(group[0][0]) == len(group[1][0])
            for g in group:
                assert len(group[0][0]) == len(g[0])
        except:
            raise ValueError("Group check assertion failed in _process of MusicJSONFlatMeasureCorpus")

        # just checked that all have the same number of measures, so now we combine them
        flat_key_zero = []
        flat_key_one = []
        flat_rel_encoded_pitch = []
        flat_lines_encoded = []

        flat_fingerprint_features_zero = []
        flat_fingerprint_features_one = []
        flat_duration_features_zero = []
        flat_duration_features_mid = []
        flat_duration_features_one = []
        flat_voices = []
        flat_centers = []
        flat_targets = []

        flat_key_zero = []
        flat_key_zero_base = []
        flat_key_durations_zero = []

        flat_key_one = []
        flat_key_one_base = []
        flat_key_durations_one = []
        flat_key_indicators = []

        # key frame encoding
        bottom_voice = len(group) - 1
        key_counter = 0
        for i in range(len(group[0][0]) - 1):
            lines_offset_from_frame = []
            centers = []

            key_zero = []
            key_durations_zero = []
            key_one = []
            key_durations_one = []

            for v in range(len(group)):
            # for now only top voice
                curr_pitch_set = group[v][0][i]
                curr_dur_set = group[v][1][i]
                next_pitch_set = group[v][0][i + 1]
                next_dur_set = group[v][1][i + 1]
                if curr_pitch_set[0] == 0:
                    # if currently resting, just use next
                    center = next_pitch_set[0]
                elif next_pitch_set[0] == 0:
                    # if next is rest, just use current
                    center = curr_pitch_set[0]
                elif curr_pitch_set[0] == 0 and next_pitch_set[0] == 0:
                    # both resting doesn't occur in jsb... for now
                    print("both rest")
                    from IPython import embed; embed(); raise ValueError()
                    center = (next_pitch_set[0] + curr_pitch_set[0]) / 2.
                else:
                    # if both active, get the centerline between the two
                    center = (next_pitch_set[0] + curr_pitch_set[0]) / 2.

                line_offset_from_frame = center - group[bottom_voice][0][i][0]

                key_zero.append(curr_pitch_set[0])
                key_durations_zero.append(curr_dur_set[0])
                key_one.append(next_pitch_set[0])
                key_durations_one.append(next_dur_set[0])

                lines_offset_from_frame.append(line_offset_from_frame)
                centers.append(center)


            for v in range(len(group)):
                curr_pitch_set = group[v][0][i]
                next_pitch_set = group[v][0][i + 1]
                curr_dur_set = group[v][1][i]
                cumulative_curr_dur_set = np.cumsum(curr_dur_set)

                fingerprint_features_zero = []
                fingerprint_features_one = []
                duration_features_zero = []
                duration_features_mid = []
                duration_features_one = []
                voices = []
                these_centers = []

                # create "fingerprint" features against left and right keyframes
                # create duration offset features
                # annotate with voices
                for l in range(1, len(curr_pitch_set)):
                    print_set_zero = [curr_pitch_set[l] - group[vv][0][i][0] for vv in range(len(group))]
                    print_set_one = [curr_pitch_set[l] - group[vv][0][i + 1][0] for vv in range(len(group))]

                    fingerprint_features_zero.append(print_set_zero)
                    fingerprint_features_one.append(print_set_one)

                    duration_features_zero.append(cumulative_curr_dur_set[l - 1])
                    duration_features_one.append(cumulative_curr_dur_set[-1] - cumulative_curr_dur_set[l - 1])
                    duration_features_mid.append(curr_dur_set[l])
                    voices.append(v)
                    these_centers.append(centers)

                flat_voices.extend(voices)
                flat_duration_features_zero.extend(duration_features_zero)
                flat_duration_features_mid.extend(duration_features_mid)
                flat_duration_features_one.extend(duration_features_one)
                flat_fingerprint_features_zero.extend(fingerprint_features_zero)
                flat_fingerprint_features_one.extend(fingerprint_features_one)
                flat_centers.extend(these_centers)

                # create prediction targets - distance from each "left" keypoint entry
                # 1: because we skip the keypoint itself
                # key_zero is the UN centered values aka true pitches
                rel_encoded = [[csi - kpz if csi != 0 else 100 for kpz in key_zero] for csi in curr_pitch_set[1:]]
                flat_targets.extend(rel_encoded)

                flat_key_zero_base.extend([key_zero[-1] for l in range(1, len(curr_pitch_set))])
                flat_key_zero.extend([[kz - key_zero[-1] for kz in key_zero] for l in range(1, len(curr_pitch_set))])
                flat_key_durations_zero.extend([key_durations_zero for l in range(1, len(curr_pitch_set))])

                flat_key_one_base.extend([key_one[-1] for l in range(1, len(curr_pitch_set))])
                flat_key_one.extend([[ko - key_one[-1] for ko in key_one] for l in range(1, len(curr_pitch_set))])
                flat_key_durations_one.extend([key_durations_one for l in range(1, len(curr_pitch_set))])

                flat_key_indicators.extend([key_counter for l in range(1, len(curr_pitch_set))])

            if i == 0:
                # add valid cut point marker at the start
                # so insert(0)
                tmp = [flat_fingerprint_features_zero,
                       flat_fingerprint_features_one,
                       flat_duration_features_zero,
                       flat_duration_features_mid,
                       flat_duration_features_one,
                       flat_voices,
                       flat_centers,
                       flat_key_zero_base,
                       flat_key_zero,
                       flat_key_durations_zero,
                       flat_key_one_base,
                       flat_key_one,
                       flat_key_durations_one,
                       flat_key_indicators,
                       flat_targets]

                for z, r in enumerate(tmp):
                    fill = self.transition_value
                    if hasattr(tmp[z][-1], "__len__"):
                        tmp[z].insert(0, [fill for el in tmp[z][-1]])
                    else:
                        tmp[z].insert(0, fill)

            # add valid cut point marker at the boundary
            tmp = [flat_fingerprint_features_zero,
                   flat_fingerprint_features_one,
                   flat_duration_features_zero,
                   flat_duration_features_mid,
                   flat_duration_features_one,
                   flat_voices,
                   flat_centers,
                   flat_key_zero_base,
                   flat_key_zero,
                   flat_key_durations_zero,
                   flat_key_one_base,
                   flat_key_one,
                   flat_key_durations_one,
                   flat_key_indicators,
                   flat_targets]

            for z, r in enumerate(tmp):
                fill = self.transition_value
                if hasattr(tmp[z][-1], "__len__"):
                    tmp[z].append([fill for el in tmp[z][-1]])
                else:
                    tmp[z].append(fill)
            key_counter += 1

        ret = [flat_fingerprint_features_zero,
               flat_fingerprint_features_one,
               flat_duration_features_zero,
               flat_duration_features_mid,
               flat_duration_features_one,
               flat_voices,
               flat_centers,
               flat_key_zero_base,
               flat_key_zero,
               flat_key_durations_zero,
               flat_key_one_base,
               flat_key_one,
               flat_key_durations_one,
               flat_key_indicators,
               flat_targets]
        return ret


    def build_vocabulary(self, json_file_paths):
        """Tokenizes a text file."""
        for path in json_file_paths:
            ret = self._process(path)
            (flat_fingerprint_features_zero,
             flat_fingerprint_features_one,
             flat_duration_features_zero,
             flat_duration_features_mid,
             flat_duration_features_one,
             flat_voices,
             flat_centers,
             flat_key_zero_base,
             flat_key_zero,
             flat_key_durations_zero,
             flat_key_one_base,
             flat_key_one,
             flat_key_durations_one,
             flat_key_indicators,
             flat_targets) = ret

            for fpz in flat_fingerprint_features_zero:
                self.fingerprint_features_zero_dictionary.add_word(tuple(fpz))

            for fpo in flat_fingerprint_features_one:
                self.fingerprint_features_one_dictionary.add_word(tuple(fpo))

            for durfz in flat_duration_features_zero:
                self.duration_features_zero_dictionary.add_word(durfz)

            for durfm in flat_duration_features_mid:
                self.duration_features_mid_dictionary.add_word(durfm)

            for durfo in flat_duration_features_one:
                self.duration_features_one_dictionary.add_word(durfo)

            for v in flat_voices:
                self.voices_dictionary.add_word(v)

            for c in flat_centers:
                self.centers_0_dictionary.add_word(c[0])
                self.centers_1_dictionary.add_word(c[1])
                self.centers_2_dictionary.add_word(c[2])
                self.centers_3_dictionary.add_word(c[3])

            for kzb in flat_key_zero_base:
                self.keypoint_base_dictionary.add_word(kzb)

            for kz in flat_key_zero:
                self.keypoint_dictionary.add_word(tuple(kz))

            for kdz in flat_key_durations_zero:
                self.keypoint_durations_dictionary.add_word(tuple(kdz))

            for kob in flat_key_one_base:
                self.keypoint_base_dictionary.add_word(kob)

            for ko in flat_key_one:
                self.keypoint_dictionary.add_word(tuple(ko))

            for kdo in flat_key_durations_one:
                self.keypoint_durations_dictionary.add_word(tuple(kdo))

            for t in flat_targets:
                self.target_0_dictionary.add_word(t[0])
                self.target_1_dictionary.add_word(t[1])
                self.target_2_dictionary.add_word(t[2])
                self.target_3_dictionary.add_word(t[3])

    def tokenize(self, paths):
        """Tokenizes

        [fingerprint_features_zero,
         fingerprint_features_one,
         duration_features_zero,
         duration_features_one,
         voices,
         centers,
         key_zero_base,
         key_zero,
         key_durations_zero,
         key_one_base,
         key_one,
         key_durations_one,
         targets]
        """
        all_fingerprint_features_zero = []
        all_fingerprint_features_one = []
        all_duration_features_zero = []
        all_duration_features_mid = []
        all_duration_features_one = []
        all_voices = []
        all_centers = []
        all_key_zero_base = []
        all_key_zero = []
        all_key_durations_zero = []
        all_key_one_base = []
        all_key_one = []
        all_key_durations_one = []
        all_key_indicators = []
        all_targets = []

        for path in paths:
            ret = self._process(path)
            tokens = self._tokenize_features(ret)
            (fingerprint_features_zero,
             fingerprint_features_one,
             duration_features_zero,
             duration_features_mid,
             duration_features_one,
             voices,
             centers,
             key_zero_base,
             key_zero,
             key_durations_zero,
             key_one_base,
             key_one,
             key_durations_one,
             key_indicators,
             targets) = tokens
            all_fingerprint_features_zero.extend(fingerprint_features_zero)
            all_fingerprint_features_one.extend(fingerprint_features_one)
            all_duration_features_zero.extend(duration_features_zero)
            all_duration_features_mid.extend(duration_features_mid)
            all_duration_features_one.extend(duration_features_one)
            all_voices.extend(voices)
            all_centers.extend(centers)
            all_key_zero_base.extend(key_zero_base)
            all_key_zero.extend(key_zero)
            all_key_durations_zero.extend(key_durations_zero)
            all_key_one_base.extend(key_one_base)
            all_key_one.extend(key_one)
            all_key_durations_one.extend(key_durations_one)
            all_key_indicators.extend(key_indicators)
            all_targets.extend(targets)

        all_tokens = (all_fingerprint_features_zero,
                      all_fingerprint_features_one,
                      all_duration_features_zero,
                      all_duration_features_mid,
                      all_duration_features_one,
                      all_voices,
                      all_centers,
                      all_key_zero_base,
                      all_key_zero,
                      all_key_durations_zero,
                      all_key_one_base,
                      all_key_one,
                      all_key_durations_one,
                      all_key_indicators,
                      all_targets)
        return all_tokens

    def _tokenize_features(self, features):
        """
            (flat_fingerprint_features_zero,
             flat_fingerprint_features_one,
             flat_duration_features_zero,
             flat_duration_features_mid,
             flat_duration_features_one,
             flat_voices,
             flat_centers,
             flat_key_zero_base,
             flat_key_zero,
             flat_key_durations_zero,
             flat_key_one_base,
             flat_key_one,
             flat_key_durations_one,
             flat_key_indicators,
             flat_targets) = features
        """
        fingerprint_features_zero = []
        fingerprint_features_one = []
        duration_features_zero = []
        duration_features_mid = []
        duration_features_one = []
        voices = []
        centers = []
        key_zero_base = []
        key_zero = []
        key_durations_zero = []
        key_one_base = []
        key_one = []
        key_durations_one = []
        key_indicators = []
        targets = []

        (flat_fingerprint_features_zero,
         flat_fingerprint_features_one,
         flat_duration_features_zero,
         flat_duration_features_mid,
         flat_duration_features_one,
         flat_voices,
         flat_centers,
         flat_key_zero_base,
         flat_key_zero,
         flat_key_durations_zero,
         flat_key_one_base,
         flat_key_one,
         flat_key_durations_one,
         flat_key_indicators,
         flat_targets) = features

        for fpz in flat_fingerprint_features_zero:
            fpz_token = self.fingerprint_features_zero_dictionary.word2idx[tuple(fpz)]
            fingerprint_features_zero.append(fpz_token)

        for fpo in flat_fingerprint_features_one:
            fpo_token = self.fingerprint_features_one_dictionary.word2idx[tuple(fpo)]
            fingerprint_features_one.append(fpo_token)

        for durfz in flat_duration_features_zero:
            durfz_token = self.duration_features_zero_dictionary.word2idx[durfz]
            duration_features_zero.append(durfz_token)

        for durfm in flat_duration_features_mid:
            durfm_token = self.duration_features_mid_dictionary.word2idx[durfm]
            duration_features_mid.append(durfm_token)

        for durfo in flat_duration_features_one:
            durfo_token = self.duration_features_one_dictionary.word2idx[durfo]
            duration_features_one.append(durfo_token)

        for v in flat_voices:
             v_token = self.voices_dictionary.word2idx[v]
             voices.append(v_token)

        for c in flat_centers:
            center_0_token = self.centers_0_dictionary.word2idx[c[0]]
            center_1_token = self.centers_1_dictionary.word2idx[c[1]]
            center_2_token = self.centers_2_dictionary.word2idx[c[2]]
            center_3_token = self.centers_3_dictionary.word2idx[c[3]]
            centers.append([center_0_token,
                            center_1_token,
                            center_2_token,
                            center_3_token])

        for kzb in flat_key_zero_base:
            kzb_token = self.keypoint_base_dictionary.word2idx[kzb]
            key_zero_base.append(kzb_token)

        for kz in flat_key_zero:
            kz_token = self.keypoint_dictionary.word2idx[tuple(kz)]
            key_zero.append(kz_token)

        for kdz in flat_key_durations_zero:
            kdz_token = self.keypoint_durations_dictionary.word2idx[tuple(kdz)]
            key_durations_zero.append(kdz_token)

        for kob in flat_key_one_base:
            kob_token = self.keypoint_base_dictionary.word2idx[kob]
            key_one_base.append(kob_token)

        for ko in flat_key_one:
            ko_token = self.keypoint_dictionary.word2idx[tuple(ko)]
            key_one.append(ko_token)

        for kdo in flat_key_durations_one:
            kdo_token = self.keypoint_durations_dictionary.word2idx[tuple(kdo)]
            key_durations_one.append(kdo_token)

        for fki in flat_key_indicators:
            key_indicators.append(fki)

        for t in flat_targets:
            target_0_token = self.target_0_dictionary.word2idx[t[0]]
            target_1_token = self.target_1_dictionary.word2idx[t[1]]
            target_2_token = self.target_2_dictionary.word2idx[t[2]]
            target_3_token = self.target_3_dictionary.word2idx[t[3]]
            targets.append([target_0_token, target_1_token, target_2_token, target_3_token])

        outs = [fingerprint_features_zero,
                fingerprint_features_one,
                duration_features_zero,
                duration_features_mid,
                duration_features_one,
                voices,
                centers,
                key_zero_base,
                key_zero,
                key_durations_zero,
                key_one_base,
                key_one,
                key_durations_one,
                key_indicators,
                targets]
        assert len(outs[0]) == len(outs[1])
        for i in range(2, len(outs)):
            assert len(outs[0]) == len(outs[i])
        return outs

    def pitch_duration_voice_lists_from_preds_and_features(self, preds, features, features_masks, context_len):
        """
        preds: (len, batch)
        features: list of length 14
                   each feature batch is (len, batch, feature_dim)

        ideal r

        [fingerprint_features_zero, : 0
        fingerprint_features_one, : 1
        duration_features_zero, : 2
        duration_features_mid, : 3
        duration_features_one, : 4
        voices, : 5
        centers, : 6
        key_zero_base, : 7
        key_zero, : 8
        key_durations_zero, : 9
        key_one_base, : 10
        key_one, : 11
        key_durations_one, : 12
        key_indicators, : 13
        targets] : 14

        an ideal reconstruction is
        preds = targets[..., 0]
        preds = preds[hp.context_len-1:-1] ?
        """
        batches_list = features
        batches_masks_list = features_masks
        all_pitches_list = []
        all_durations_list = []
        all_voices_list = []
        all_marked_quarters_context_boundary = []

        # do we just make a function on the original data class?
        for i in range(preds.shape[1]):
            # first get back the "left" keypoint
            # as well as the duration
            key_zero_base = batches_list[7]
            key_zero = batches_list[8]
            key_durations_zero = batches_list[9]
            
            key_one_base = batches_list[10]
            key_one = batches_list[11]
            key_durations_one = batches_list[12]

            key_indicators = batches_list[13]

            # same mask for all of em
            this_mask = batches_masks_list[0][:, i]
            f_m = np.where(this_mask)[0][0]

            key_zero_base = key_zero_base[:f_m, i, 0]
            key_zero = key_zero[:f_m, i, 0]
            key_durations_zero = key_durations_zero[:f_m, i, 0]

            key_one_base = key_one_base[:f_m, i, 0]
            key_one = key_one[:f_m, i, 0]
            key_durations_one = key_durations_one[:f_m, i, 0]
            key_indicators = key_indicators[:f_m, i, 0]

            boundary_points = np.concatenate((np.array([-1]), key_indicators))[:-1] != key_indicators
            s_s = np.concatenate((np.where(boundary_points)[0], np.array([len(key_indicators)])))
            boundary_pairs = list(zip(s_s[:-1], s_s[1:]))

            pitches_list = []
            durations_list = []
            voices_list = []
            voice_step_in_quarters = [0, 0, 0, 0]
            voice_step_in_pred = [0, 0, 0, 0]
            marked_quarters_context_boundary = [-1, -1, -1, -1]
            for s, e in boundary_pairs:
                # in each chunk, do keypoint vector, then rest
                this_key = key_zero[s:e]
                assert all([tk == this_key[0] for tk in this_key])
                this_key = self.keypoint_dictionary.idx2word[this_key[0]]

                this_key_base = key_zero_base[s:e]
                assert all([tkb == this_key_base[0] for tkb in this_key_base])
                this_key = tuple([tk + self.keypoint_base_dictionary.idx2word[this_key_base[0]] for tk in this_key])

                this_key_durations = key_durations_zero[s:e]
                assert all([tkd == this_key_durations[0] for tkd in this_key_durations])
                this_key_durations = self.keypoint_durations_dictionary.idx2word[this_key_durations[0]]

                centers = batches_list[6]
                centers = centers[s:e, i]
                center_0 = self.centers_0_dictionary.idx2word[centers[0][0]]
                center_1 = self.centers_1_dictionary.idx2word[centers[0][1]]
                center_2 = self.centers_2_dictionary.idx2word[centers[0][2]]
                center_3 = self.centers_3_dictionary.idx2word[centers[0][3]]

                targets = copy.deepcopy(batches_list[-1])
                # rewrite targets with preds at the correct point
                # originally shifted by 1 then moved back 1

                # currently writing into 0th entry will need to generalize this
                if len(preds) == len(targets):
                    targets[:, :, 0] = preds
                else:
                    # try to offset by context length
                    targets[context_len:context_len + len(preds), :, 0] = preds

                targets_chunk = targets[s:e, i]

                target_0_values = [self.target_0_dictionary.idx2word[targets_chunk[z][0]] for z in range(len(targets_chunk))]
                target_1_values = [self.target_1_dictionary.idx2word[targets_chunk[z][1]] for z in range(len(targets_chunk))]
                target_2_values = [self.target_2_dictionary.idx2word[targets_chunk[z][2]] for z in range(len(targets_chunk))]
                target_3_values = [self.target_3_dictionary.idx2word[targets_chunk[z][3]] for z in range(len(targets_chunk))]


                # 100 was rest
                remapped_0 = [this_key[0] + t_0 if t_0 != 100 else 0 for t_0 in target_0_values]
                remapped_1 = [this_key[1] + t_1 if t_1 != 100 else 0 for t_1 in target_1_values]
                remapped_2 = [this_key[2] + t_2 if t_2 != 100 else 0 for t_2 in target_2_values]
                remapped_3 = [this_key[3] + t_3 if t_3 != 100 else 0 for t_3 in target_3_values]

                # remapped will come from predictions now
                # drop this assert!
                #assert all([remapped_0[n] == remapped_1[n] for n in range(len(remapped_0))])
                #assert all([remapped_0[n] == remapped_2[n] for n in range(len(remapped_0))])
                #assert all([remapped_0[n] == remapped_3[n] for n in range(len(remapped_0))])

                durations = batches_list[3]
                durations = durations[s:e, i, 0]
                duration_values = [self.duration_features_mid_dictionary.idx2word[d_el] for d_el in durations]

                voices = batches_list[5]
                voices = voices[s:e, i, 0]
                voice_values = [self.voices_dictionary.idx2word[v_el] for v_el in voices]

                final_pitch_chunk = []
                final_duration_chunk = []
                final_voice_chunk = []
                key_itr = 0
                last_v = -1
                for n, v in enumerate(voice_values):
                    # we assume it is SSSSSSAAAAAAAAAAAATTTTTTTBB
                    # style format
                    # need to insert key values at the right spot
                    if v != last_v:
                        final_pitch_chunk.append(this_key[key_itr])
                        final_duration_chunk.append(this_key_durations[key_itr])
                        final_voice_chunk.append(key_itr)

                        try:
                            voice_step_in_quarters[v] += this_key_durations[key_itr]
                        except:
                            from IPython import embed; embed(); raise ValueError()
                        # we don't predict this one
                        #voice_step_in_pred[v] += 1
                        key_itr += 1
                        last_v = v

                    final_pitch_chunk.append(int(remapped_0[n]))
                    final_duration_chunk.append(duration_values[n])
                    final_voice_chunk.append(v)
                    voice_step_in_quarters[v] += duration_values[n]
                    voice_step_in_pred[v] += 1
                # we assume it is SSSSSSAAAAAAAAAAAATTTTTTTBB
                current_aggregate_pred_step = sum(voice_step_in_pred)
                if current_aggregate_pred_step >= context_len:
                    # find voice / index where we cross the context boundary
                    # only mark the first time
                    if all([marked_quarters_context_boundary[_] < 0 for _ in range(len(marked_quarters_context_boundary))]):
                        _ind = np.where((np.cumsum(voice_step_in_pred) >= context_len) == True)[0][0]
                        for _ in range(len(marked_quarters_context_boundary)):
                            marked_quarters_context_boundary[_] = int(voice_step_in_quarters[_ind])
                pitches_list.extend(final_pitch_chunk)
                durations_list.extend(final_duration_chunk)
                voices_list.extend(final_voice_chunk)

            all_marked_quarters_context_boundary.append(marked_quarters_context_boundary)
            all_pitches_list.append(pitches_list)
            all_durations_list.append(durations_list)
            all_voices_list.append(voices_list)
        return all_pitches_list, all_durations_list, all_voices_list, all_marked_quarters_context_boundary


class MusicJSONFlatMeasureCorpus(object):
    def __init__(self, train_data_file_paths, valid_data_file_paths=None, test_data_file_paths=None,
                 max_vocabulary_size=-1,
                 add_measure_marks=True,
                 tokenization_fn="measure_flatten",
                 default_velocity=120, n_voices=4,
                 measure_value=99,
                 fill_value=-1,
                 separate_onsets=True, onsets_boundary=100):
        raise ValueError("deprecated")
        """
        """
        self.pitch_dictionary = LookupDictionary()
        self.duration_dictionary = LookupDictionary()
        self.velocity_dictionary = LookupDictionary()
        self.voice_dictionary = LookupDictionary()
        
        self.max_vocabulary_size = max_vocabulary_size
        self.default_velocity = default_velocity
        self.n_voices = n_voices
        self.measure_value = measure_value
        self.fill_value = fill_value
        self.separate_onsets = separate_onsets
        self.onsets_boundary = onsets_boundary
        self.add_measure_marks = add_measure_marks

        if tokenization_fn != "measure_flatten":
            raise ValueError("Only default tokenization_fn currently supported")

        base = [fp for fp in train_data_file_paths]
        if valid_data_file_paths is not None:
            base = base + [fp for fp in valid_data_file_paths]
        if test_data_file_paths is not None:
            base = base + [fp for fp in test_data_file_paths]

        self.build_vocabulary(base)

        if self.max_vocabulary_size > -1:
            self.dictionary._prune_to_top_k_counts(self.max_vocabulary_size)

        self.train = self.tokenize(train_data_file_paths)
        if valid_data_file_paths is not None:
            self.valid = self.tokenize(valid_data_file_paths)
        if test_data_file_paths is not None:
            self.test = self.tokenize(test_data_file_paths)

    def _process(self, path):
        assert os.path.exists(path)
        pitch, duration, velocity = pitch_duration_velocity_lists_from_music_json_file(path,
                                                                                       default_velocity=self.default_velocity,
                                                                                       n_voices=self.n_voices,
                                                                                       measure_value=self.measure_value,
                                                                                       fill_value=self.fill_value)

        def isplit(iterable, splitters):
            return [list(g) for k,g in itertools.groupby(iterable,lambda x:x in splitters) if not k]

        group = []
        for v in range(len(pitch)):
            # per voice, do split and merge
            s_p = isplit(pitch[v], [self.measure_value])
            s_d = isplit(duration[v], [self.fill_value])
            s_v = isplit(velocity[v], [self.fill_value])
            group.append([s_p, s_d, s_v])

        not_merged = True
        # all should be the same length in terms of measures so we can merge
        try:
            assert len(group[0][0]) == len(group[1][0])
            for g in group:
                assert len(group[0][0]) == len(g[0])
        except:
            raise ValueError("Group check assertion failed in _process of MusicJSONFlatMeasureCorpus")

        # just checked that all have the same number of measures, so now we combine them
        flat_pitch = []
        flat_duration = []
        flat_velocity = []
        flat_voice = []

        flat_pitch.append(self.measure_value)
        flat_duration.append(self.fill_value)
        flat_velocity.append(self.fill_value)
        flat_voice.append(len(pitch))

        for i in range(len(group[0][0])):
            for v in range(len(group)):
                m_p = group[v][0][i]
                m_d = group[v][1][i]
                m_v = group[v][2][i]
                m_vv = [v for el in m_p]

                flat_pitch.extend(m_p)
                flat_duration.extend(m_d)
                flat_velocity.extend(m_v)
                flat_voice.extend(m_vv)
            flat_pitch.append(self.measure_value)
            flat_duration.append(self.fill_value)
            flat_velocity.append(self.fill_value)
            flat_voice.append(len(pitch))
        return flat_pitch, flat_duration, flat_velocity, flat_voice


    def build_vocabulary(self, json_file_paths):
        """Tokenizes a text file."""
        for path in json_file_paths:
            pitch, duration, velocity, voice = self._process(path)

            for p in pitch:
                self.pitch_dictionary.add_word(p)

            for d in duration:
                self.duration_dictionary.add_word(d)

            for v in velocity:
                self.velocity_dictionary.add_word(v)

            for vv in voice:
                self.voice_dictionary.add_word(vv)


    def tokenize(self, paths):
        """Tokenizes a text file."""
        pitches = []
        durations = []
        velocities = []
        voices = []
        for path in paths:
            pitch, duration, velocity, voice = self._process(path)

            # do we even bother to check for unknown words
            for p in pitch:
                p_token = self.pitch_dictionary.word2idx[p]
                pitches.append(p_token)

            for d in duration:
                d_token = self.duration_dictionary.word2idx[d]
                durations.append(d_token)

            for v in velocity:
                v_token = self.velocity_dictionary.word2idx[v]
                velocities.append(v_token)

            for vv in voice:
                vv_token = self.voice_dictionary.word2idx[vv]
                voices.append(vv_token)

        return pitches, durations, velocities, voices


class MusicJSONCorpusDEP(object):
    def __init__(self, train_data_file_paths, valid_data_file_paths=None, test_data_file_paths=None,
                 max_vocabulary_size=-1,
                 add_eos=False,
                 eos_amount=32,
                 eos_symbol=0,
                 tokenization_fn="flatten",
                 default_velocity=120, n_voices=4,
                 separate_onsets=True, onsets_boundary=100):
        """
        """
        raise ValueError("deprecated")
        self.dictionary = LookupDictionary()

        self.max_vocabulary_size = max_vocabulary_size
        self.default_velocity = default_velocity
        self.quantization_rate = quantization_rate
        self.n_voices = n_voices
        self.separate_onsets = separate_onsets
        self.onsets_boundary = onsets_boundary
        self.add_eos = add_eos
        self.eos_amount = eos_amount
        self.eos_symbol = eos_symbol

        if tokenization_fn == "flatten":
            def tk(arr):
                t = [el for el in arr.ravel()]
                if add_eos:
                    # 2 measures of silence are the eos
                    return t + [self.eos_symbol] * self.eos_amount
                else:
                    return t
            self.tokenization_fn = tk
        else:
            raise ValueError("Unknown tokenization_fn {}".format(tokenization_fn))

        base = [fp for fp in train_data_file_paths]
        if valid_data_file_paths is not None:
            base = base + [fp for fp in valid_data_file_paths]
        if test_data_file_paths is not None:
            base = base + [fp for fp in test_data_file_paths]

        self.build_vocabulary(base)

        if self.max_vocabulary_size > -1:
            self.dictionary._prune_to_top_k_counts(self.max_vocabulary_size)

        self.train = self.tokenize(train_data_file_paths)
        if valid_data_file_paths is not None:
            self.valid = self.tokenize(valid_data_file_paths)
        if test_data_file_paths is not None:
            self.test = self.tokenize(test_data_file_paths)

    def build_vocabulary(self, json_file_paths):
        """Tokenizes a text file."""
        for path in json_file_paths:
            assert os.path.exists(path)
            roll = piano_roll_from_music_json_file(path,
                                                   default_velocity=self.default_velocity,
                                                   quantization_rate=self.quantization_rate,
                                                   n_voices=self.n_voices,
                                                   separate_onsets=self.separate_onsets,
                                                   onsets_boundary=self.onsets_boundary,
                                                   as_numpy=True)
            words = self.tokenization_fn(roll)

            for word in words:
                self.dictionary.add_word(word)

    def tokenize(self, paths):
        """Tokenizes a text file."""
        ids = []
        for path in paths:
            assert os.path.exists(path)
            # Add words to the dictionary
            roll = piano_roll_from_music_json_file(path,
                                                   default_velocity=self.default_velocity,
                                                   quantization_rate=self.quantization_rate,
                                                   n_voices=self.n_voices,
                                                   separate_onsets=self.separate_onsets,
                                                   onsets_boundary=self.onsets_boundary,
                                                   as_numpy=True)
            words = self.tokenization_fn(roll)
            for word in words:
                if word in self.dictionary.word2idx:
                    token = self.dictionary.word2idx[word]
                else:
                    token = self.dictionary.word2idx["<unk>"]
                ids.append(token)
        return ids


class MusicJSONRasterIterator(object):
    def __init__(self, list_of_music_json_files,
                 batch_size,
                 max_sequence_length,
                 random_seed,
                 n_voices=4,
                 iterate_once=False,
                 with_clocks=[2, 4, 8, 16, 32, 64],
                 separate_onsets=False,
                 #with_clocks=None,
                 resolution="sixteenth"):
        raise ValueError("deprecated")
        super(MusicJSONRasterIterator, self).__init__()
        self.list_of_music_json_files = list_of_music_json_files
        self.random_seed = random_seed
        self.random_state = np.random.RandomState(random_seed)
        self.file_list_indices_ = list(range(len(self.list_of_music_json_files)))
        self.iterate_once = iterate_once
        self.iterate_at_ = 0
        self.batch_size = batch_size
        self.separate_onsets = separate_onsets
        if self.iterate_once:
            pass
        else:
            self.random_state.shuffle(self.file_list_indices_)
        self.file_list_indices_ = self.file_list_indices_[self.iterate_at_:self.iterate_at_ + self.batch_size]
        self.resolution = resolution
        self.max_sequence_length = max_sequence_length
        self.n_voices = n_voices
        if self.resolution != "sixteenth":
            raise ValueError("Currently only support 16th note resolution")
        if self.n_voices != 4:
            raise ValueError("Currently only support 4 voices")
        self.with_clocks = with_clocks
        # build vocabularies now?

    def next(self):
        return self.__next__()

    def __iter__(self):
        while True:
            yield next(self)

    def __next__(self):
        # -1 value for padding - will convert to 0s but mask in the end
        all_roll_voices = []
        for fli in self.file_list_indices_:
            json_file = self.list_of_music_json_files[fli]
            with open(json_file) as f:
                data = json.load(f)
            ppq = data["pulses_per_quarter"]
            qbpm = data["quarter_beats_per_minute"]
            spq = data["seconds_per_quarter"]

            parts = data["parts"]
            parts_times = data["parts_times"]
            parts_cumulative_times = data["parts_cumulative_times"]
            # https://github.com/cuthbertLab/music21/blob/c6fc39204c16c47d1c540b545d0c9869a9cafa8f/music21/midi/__init__.py#L1471
            if "parts_velocities" not in data:
                default_velocity = 120
                parts_velocities = [[default_velocity] * len(p) for p in parts]
            else:
                parts_velocities = data["parts_velocities"]
            end_in_quarters = max([max(p) for p in parts_cumulative_times])
            # clock is set currently by the fact that "sixteenth" is the only option
            # .25 due to "sixteenth"
            clock = np.arange(0, max(max(parts_cumulative_times)), .25)
            # 4 * for 4 voices
            raster_end_in_steps = 4 * len(clock)
            if raster_end_in_steps > self.max_sequence_length:
                pass
                # need to randomly slice out a chunk that fits? or just take the first steps?
                # let it go for now

            # start with 1 quarter note (4 16ths) worth of pure rest
            roll_voices = [[0] * 4, [0] * 4, [0] * 4, [0] * 4]
            # use these for tracking if we cross a change event
            p_i = [0, 0, 0, 0]
            for c in clock:
                # voice
                for v in range(len(parts)):
                    current_note = parts[v][p_i[v]]
                    next_change_time = parts_cumulative_times[v][p_i[v]]
                    new_onset = False
                    if c >= next_change_time:
                        # we hit a boundary, swap notes
                        p_i[v] += 1
                        current_note = parts[v][p_i[v]]
                        next_change_time = parts_cumulative_times[v][p_i[v]]
                        new_onset = True
                    if c == 0. or new_onset:
                        if current_note != 0:
                            if self.separate_onsets:
                                roll_voices[v].append(current_note + 100)
                            else:
                                roll_voices[v].append(current_note)
                        else:
                            # rests have no "onset"
                            roll_voices[v].append(current_note)
                    else:
                       roll_voices[v].append(current_note)
            all_roll_voices.append(roll_voices)

        raster_roll_voices = np.zeros((self.max_sequence_length, self.batch_size, 1)) - 1.

        if self.with_clocks is not None:
            all_clocks = [np.zeros((self.max_sequence_length, self.batch_size, 1)) for ac in self.with_clocks]

        for n, rv in enumerate(all_roll_voices):
            # transpose from 4 long sequences, to a long sequence of 4 "tuples"
            i_rv = [[t_rv[i] for t_rv in rv] for i in range(len(rv[0]))]
            raster_i_rv = [r for step in i_rv for r in step]
            if self.with_clocks is not None:
                # create clock signals, by taking time index modulo each value
                clock_base = [[t for t_rv in rv] for t in range(len(rv[0]))]
                clock_base = [clk for step in clock_base for clk in step]
                lcl_clocks = []
                for cl_i in self.with_clocks:
                    this_clock = [cb % cl_i for cb in clock_base]
                    lcl_clocks.append(this_clock)

            slicer = 0
            if len(raster_i_rv) > self.max_sequence_length:
                # find a point to start where either all voices rest, or all voices are onsets!
                # guaranteed to have at least 1 start point due to beginning, avoid those if we can
                proposed_cuts = [all([i_rv_ii > 100 or i_rv_ii == 0 for i_rv_ii in i_rv_i]) for i_rv_i in i_rv]
                proposed_cuts_i = [i for i in range(len(proposed_cuts)) if proposed_cuts[i] is True]

                # prune to only the cuts that give us a full self.max_sequence_length values after rasterizing
                proposed_cuts_i = [pci for pci in proposed_cuts_i if pci * self.n_voices + self.max_sequence_length <= len(raster_i_rv)]

                if len(proposed_cuts_i) == 0:
                    # edge case if none qualify
                    proposed_cuts_i = [0]
                # shuffle to get one at random - shuffle is in place so we choose the first one
                self.random_state.shuffle(proposed_cuts_i)

                step_slicer = proposed_cuts_i[0]

                # turn it into a raster pointer instead of a "voice tuple" pointer
                slicer = step_slicer * self.n_voices
            subslice = raster_i_rv[slicer:slicer + self.max_sequence_length]
            raster_roll_voices[:len(subslice), n, 0] = subslice
            # we broadcast it to self.batch_size soon
            if self.with_clocks is not None:
                for _i, ac in enumerate(lcl_clocks):
                    all_clocks[_i][:len(subslice), n, 0] = ac[slicer:slicer + self.max_sequence_length]
        # take off trailing 1 from shape
        mask = np.array(raster_roll_voices >= 0.).astype(np.float32)[..., 0]
        # np.abs to get rid of -0. , is annoying to me
        raster_roll_voices = np.abs(raster_roll_voices * mask[..., None])

        # setup new file_list_indices - use a new song for each batch element
        self.file_list_indices_ = list(range(len(self.list_of_music_json_files)))
        if self.iterate_once:
            self.iterate_at_ += self.batch_size
        else:
            self.random_state.shuffle(self.file_list_indices_)
        self.file_list_indices_ = self.file_list_indices_[self.iterate_at_:self.iterate_at_ + self.batch_size]
        if len(self.file_list_indices_) != self.batch_size:
            if self.iterate_once and len(self.file_list_indices_) > 0:
                # let the last batch through for iterate_once / vocabulary and statistics checks, etc
                pass
            else:
                raise ValueError("Unknown error, not enough file list indices to iterate! Current indices {}".format(self.file_list_indices_))
        if self.with_clocks is None:
            return raster_roll_voices, mask
        else:
            return raster_roll_voices, mask, [ac.astype(np.float32) * mask[..., None] for ac in all_clocks]


class MusicJSONVoiceIterator(object):
    """
            return pitch_batch, time_batch, voice_batch, cumulative_time_batch, mask
        else:
            return pitch_batch, time_batch, voice_batch, cumulative_time_batch, mask, clock_batches
    """
    def __init__(self, list_of_music_json_files,
                 batch_size,
                 max_sequence_length,
                 random_seed,
                 n_voices=4,
                 rest_marked_durations=True,
                 iterate_once=False,
                 with_clocks=[2, 4, 8, 16, 32, 64],
                 resolution="sixteenth"):
        raise ValueError("deprecated")
        super(MusicJSONVoiceIterator, self).__init__()
        self.list_of_music_json_files = list_of_music_json_files
        self.random_seed = random_seed
        self.random_state = np.random.RandomState(random_seed)
        self.file_list_indices_ = list(range(len(self.list_of_music_json_files)))
        self.iterate_once = iterate_once
        self.iterate_at_ = 0
        self.batch_size = batch_size
        self.rest_marked_durations = rest_marked_durations
        if self.iterate_once:
            pass
        else:
            self.random_state.shuffle(self.file_list_indices_)
        self.file_list_indices_ = self.file_list_indices_[self.iterate_at_:self.iterate_at_ + self.batch_size]
        self.resolution = resolution
        self.max_sequence_length = max_sequence_length
        self.n_voices = n_voices
        if self.resolution != "sixteenth":
            raise ValueError("Currently only support 16th note resolution")
        if self.n_voices != 4:
            raise ValueError("Currently only support 4 voices")
        self.with_clocks = with_clocks
        # build vocabularies now?

    def next(self):
        return self.__next__()

    def __iter__(self):
        while True:
            yield next(self)

    def __next__(self):
        # -1 value for padding - will convert to 0s but mask in the end
        all_roll_voice_times = []
        all_roll_voice_pitches = []
        for fli in self.file_list_indices_:
            json_file = self.list_of_music_json_files[fli]
            with open(json_file) as f:
                data = json.load(f)
            ppq = data["pulses_per_quarter"]
            qbpm = data["quarter_beats_per_minute"]
            spq = data["seconds_per_quarter"]

            parts = data["parts"]
            parts_times = data["parts_times"]
            parts_cumulative_times = data["parts_cumulative_times"]
            # https://github.com/cuthbertLab/music21/blob/c6fc39204c16c47d1c540b545d0c9869a9cafa8f/music21/midi/__init__.py#L1471
            if "parts_velocities" not in data:
                default_velocity = 120
                parts_velocities = [[default_velocity] * len(p) for p in parts]
            else:
                parts_velocities = data["parts_velocities"]

            all_roll_voice_times.append(parts_times)
            all_roll_voice_pitches.append(parts)

        all_flat_pitch = []
        all_flat_voice = []
        all_flat_time = []
        all_flat_cumulative_step = []
        all_flat_cumulative_time = []
        for n in range(len(all_roll_voice_times)):
            flat_pitch = []
            flat_voice = []
            flat_time = []
            flat_cumulative_step = []
            flat_cumulative_time = []
            # multi-voice, should be 4
            this_time = all_roll_voice_times[n]
            this_pitch = all_roll_voice_pitches[n]
            this_cumulative_time_start = [[0] + [int(el) for el in np.cumsum(vv)] for vv in all_roll_voice_times[n]]
            finished = False
            # track which step 
            n_voices = len(this_pitch)
            voice_time_counter = [0] * n_voices
            voice_step_counter = [0] * n_voices
            last_event_time = -1
            next_event_time = -1
            keep_voices = [0, 1, 2, 3]
            # semi-dynamic program to make a flat sequence out of a stacked event sequence
            while True:
                if len(keep_voices) == 0:
                    #print("terminal")
                    # we need to be sure we got to the end! but how...
                    break

                if last_event_time < 0:
                    # frist
                    for v in range(n_voices):
                        flat_pitch.append(this_pitch[v][0])
                        print("needs measure support also...")
                        flat_voice.append(v)
                        flat_time.append(this_time[v][0])
                        flat_cumulative_step.append(voice_step_counter[v])
                        flat_cumulative_time.append(voice_time_counter[v])
                        voice_time_counter[v] += this_time[v][0]
                        voice_step_counter[v] += 1
                    last_event_time = 0
                    next_event_time = min([min(cts[1:]) for cts in this_cumulative_time_start])
                    # need to do something about if it was a rest or not?
                else:
                    # now
                    for v in range(n_voices):
                        if v not in keep_voices:
                            continue

                        if this_cumulative_time_start[v][voice_step_counter[v]] == next_event_time:
                            flat_pitch.append(this_pitch[v][voice_step_counter[v]])
                            flat_voice.append(v)
                            flat_time.append(this_time[v][voice_step_counter[v]])
                            flat_cumulative_step.append(voice_step_counter[v])
                            flat_cumulative_time.append(voice_time_counter[v])
                            voice_time_counter[v] += this_time[v][voice_step_counter[v]]
                            voice_step_counter[v] += 1
                    last_event_time = next_event_time
                    next_event_time = min([min(cts[voice_step_counter[vi]:]) for vi, cts in enumerate(this_cumulative_time_start) if vi in keep_voices])
                # check if we hit the end of 1 voice
                keep_voices = []
                for v in range(n_voices):
                    if voice_step_counter[v] >= len(this_time[v]):
                        pass
                        #print("dawhkj")
                        #from IPython import embed; embed(); raise ValueError()
                    else:
                        keep_voices.append(v)
                #print(keep_voices)
            all_flat_pitch.append(flat_pitch)
            all_flat_voice.append(flat_voice)
            all_flat_time.append(flat_time)
            all_flat_cumulative_step.append(flat_cumulative_step)
            all_flat_cumulative_time.append(flat_cumulative_time)

        maxlen = max([len(tv) for tv in all_flat_voice])
        pitch_batch = np.zeros((maxlen, self.batch_size, 1))
        voice_batch = np.zeros((maxlen, self.batch_size, 1))
        time_batch = np.zeros((maxlen, self.batch_size, 1))
        # make this one but it seems poitnless to return it
        cumulative_step_batch = np.zeros((maxlen, self.batch_size, 1))
        cumulative_time_batch = np.zeros((maxlen, self.batch_size, 1))
        mask = np.zeros((maxlen, self.batch_size, 1))

        for i in range(self.batch_size):
            l = len(all_flat_pitch[i])
            pitch_batch[:l, i, 0] = all_flat_pitch[i]
            voice_batch[:l, i, 0] = all_flat_voice[i]
            if self.rest_marked_durations:
                # we give special duration marks to rest durations, adding 500 is a quick hack for that
                tft = all_flat_time[i]
                tfp = all_flat_pitch[i]
                tft = [tft[jj] if tfp[jj] != 0 else tft[jj] + 500 for jj in range(len(tft))]
                time_batch[:l, i, 0] = tft
            else:
                time_batch[:l, i, 0] = all_flat_time[i]
            cumulative_step_batch[:l, i, 0] = all_flat_cumulative_step[i]
            cumulative_time_batch[:l, i, 0] = all_flat_cumulative_time[i]
            mask[:l, i, 0] = 1.

        if self.with_clocks is not None:
            # create clock signals, by taking time index modulo each value
            clock_batches = [np.zeros((maxlen, self.batch_size, 1)) for c in self.with_clocks]
            for ii, c in enumerate(self.with_clocks):
                clock_batches[ii] = cumulative_time_batch % c


        # take off trailing 1 from shape
        mask = mask.astype(np.float32)[..., 0]

        # setup new file_list_indices - use a new song for each batch element
        self.file_list_indices_ = list(range(len(self.list_of_music_json_files)))
        if self.iterate_once:
            self.iterate_at_ += self.batch_size
        else:
            self.random_state.shuffle(self.file_list_indices_)
        self.file_list_indices_ = self.file_list_indices_[self.iterate_at_:self.iterate_at_ + self.batch_size]
        if len(self.file_list_indices_) != self.batch_size:
            if self.iterate_once and len(self.file_list_indices_) > 0:
                # let the last batch through for iterate_once / vocabulary and statistics checks, etc
                pass
            else:
                raise ValueError("Unknown error, not enough file list indices to iterate! Current indices {}".format(self.file_list_indices_))
        if self.with_clocks is None:
            return pitch_batch, time_batch, voice_batch, cumulative_time_batch, mask
        else:
            return pitch_batch, time_batch, voice_batch, cumulative_time_batch, mask, clock_batches
