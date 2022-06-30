from .kkpthlib import softmax_np
import copy
import numpy as np

def top_p_from_logits_np(logits, p=.9, fill_value=-np.inf, return_indices=False):
    """
    temperature can be applied beforehand, simply by sending in logits / temperature rather than logits
    """
    # copy.deepcopy so we don't modify them
    logits = copy.deepcopy(logits)
    sorted_indices = np.argsort(softmax_np(logits), axis=-1)
    shp = logits.shape
    sorted_indices_sqr = sorted_indices.transpose(2, 0, 1).reshape(shp[2], shp[0] * shp[1]).transpose(1, 0)
    logits_sqr = logits.transpose(2, 0, 1).reshape(shp[2], shp[0] * shp[1]).transpose(1, 0)
    mod_logits_sqr = copy.deepcopy(logits_sqr)
    # make it sorted from greatest to least
    sorted_indices_sqr = sorted_indices_sqr[:, ::-1]
    rlen = logits_sqr.shape[0]
    sorted_logits_sqr = copy.deepcopy(logits_sqr)
    for i in range(rlen):
        sorted_logits_sqr[i, :] = logits_sqr[i, sorted_indices_sqr[i, :]]
    cumulative_sorted_probs_sqr = np.cumsum(softmax_np(sorted_logits_sqr), axis=-1)
    # handle edge case where a single element is already above threshold
    mask = (cumulative_sorted_probs_sqr <= p)
    # at a minimum, first element is always true by definition
    mask[:, 0] = mask[:, 0] | True
    # set any values in the "tail" of the distribution (looking at cumulative prob) to fill_value
    for i in range(rlen):
        logits_sqr[i][sorted_indices_sqr[i][~mask[i]]] = fill_value
    logits_reduced = logits_sqr.transpose(1, 0).reshape(shp[2], shp[0], shp[1]).transpose(1, 2, 0)
    if return_indices:
        return logits_reduced, mask
    else:
        return logits_reduced

def top_k_from_logits_np(logits, k=1):
    print("top_k NYI")
    from IPython import embed; embed(); raise ValueError()
