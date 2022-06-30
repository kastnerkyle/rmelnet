import os
import json
import numpy as np
from .datasets import music21_parse_and_save_json, piano_roll_from_music_json_file
from .checkers import music21_from_midi

def ribbons_from_piano_roll(piano_roll, ribbon_type, quantized_bin_size,
                            interval=12):
    from IPython import embed; embed(); raise ValueErro


def plot_piano_roll(midi_or_musicjson_file_path, quantized_bin_size,
                    n_voices=4,
                    onsets_boundary=100,
                    pitch_bot=1, pitch_top=88,
                    add_legend=False,
                    show_axis_labels=False,
                    colors=["red", "blue", "green", "purple"],
                    ribbons=False,
                    ribbon_type="std",
                    axis_handle=None,
                    autorange=True,
                    autoscale_ratio=0.25,
                    midi_tempo_factor=.5,
                    show=True,
                    show_rest_channel=False,
                    force_length=None):

    f = midi_or_musicjson_file_path

    if f.endswith(".json"):
        piano_roll = piano_roll_from_music_json_file(f, default_velocity=120, quantization_rate=quantized_bin_size, n_voices=n_voices,
                                                     as_numpy=True)
    elif f.endswith(".midi") or f.endswith(".mid"):
        tmp_json_path = "_tmp.json"
        if os.path.exists(tmp_json_path):
            os.remove(tmp_json_path)
        p = music21_from_midi(f)
        # need to account for weird .5 mult in music21 midi conversion
        music21_parse_and_save_json(p, tmp_json_path, tempo_factor=midi_tempo_factor)
        piano_roll = piano_roll_from_music_json_file(tmp_json_path, default_velocity=120, quantization_rate=quantized_bin_size, n_voices=n_voices,
                                                     as_numpy=True)
        if os.path.exists(tmp_json_path):
            os.remove(tmp_json_path)

    piano_roll[piano_roll >= onsets_boundary] = piano_roll[piano_roll >= onsets_boundary] - onsets_boundary
    if force_length is not None:
        if len(piano_roll) > force_length:
             piano_roll = piano_roll[:force_length]
        elif len(piano_roll) < force_length:
             piece = np.zeros((force_length - len(piano_roll), piano_roll.shape[1])).astype(piano_roll.dtype)
             piano_roll = np.concatenate((piano_roll, piece), axis=0)

    # piano roll should be time x voices
    # https://stackoverflow.com/questions/10127284/overlay-imshow-plots-in-matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import colorConverter
    import matplotlib as mpl

    if axis_handle is None:
        ax = plt.gca()
    else:
        ax = axis_handle

    time_len = len(piano_roll)
    n_voices = len(piano_roll[0])
    n_colors = len(colors)
    if n_voices > n_colors:
        raise ValueError("Need as many colors as voices! Only gave {} colors for {} voices".format(n_colors, n_voices))
    if ribbons:
        ribbons_traces = ribbons_from_piano_roll(piano_roll, ribbon_type,
                                                 quantized_bin_size)
        from IPython import embed; embed(); raise ValueError()
    else:
       ribbons_traces = None
    # 0 always rest!
    voice_storage = np.zeros((time_len, pitch_top, n_voices))

    for v in range(n_voices):
        pitch_offset_values = [piano_roll[i][v] for i in range(time_len)]

        for n, pov in enumerate(pitch_offset_values):
            voice_storage[n, pov, v] = 255.

    cmaps = [mpl.colors.LinearSegmentedColormap.from_list("my_cmap_{}".format(v), ["white", c], 256)
             for c, v in zip(colors, list(range(n_voices)))]

    for cmap in cmaps:
        cmap._init()
        # lazy way to make zeros of right size
        alphas = np.linspace(0., 0.6, cmap.N + 3)
        cmap._lut[:, -1] = alphas

    nz = np.where(voice_storage != 0.)[1]
    nz = nz[nz >= pitch_bot]
    nz = nz[nz <= pitch_top]
    if autorange:
        mn = nz.min() - 6
        mx = nz.max() + 1 + 6
    else:
        if show_rest_channel:
            mn = 0
        else:
            mn = pitch_bot
        mx = pitch_top + 1

    for v in range(n_voices):
        ax.imshow(voice_storage[:, :, v].T, cmap=cmaps[v])
    ax.set_ylim([mn, mx])
    if show_axis_labels:
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Pitch")

    def autoaspect(x_range, y_range):
        """
        The aspect to make a plot square with ax.set_aspect in Matplotlib
        """
        mx = max(x_range, y_range)
        mn = min(x_range, y_range)
        if x_range <= y_range:
            return mx / float(mn)
        else:
            return mn / float(mx)
    asp = autoaspect(mx - mn, autoscale_ratio * len(voice_storage))
    ax.set_aspect(asp)

    if add_legend:
        patch_list = [mpatches.Patch(color=c, alpha=.6, label="Voice {}".format(v)) for c, v in zip(colors, list(range(n_voices)))]
        ax.legend(handles=patch_list, bbox_to_anchor=(0., -.3, 1., .102), loc=1,
                  ncol=1, borderaxespad=0.)
    if show:
        plt.show()
    return ribbons_traces
