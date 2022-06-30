import numpy as np
import torch
from scipy import linalg
from scipy.stats import truncnorm
import math

import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from scipy.stats import betabinom

from functools import reduce
from operator import add

import copy

from .hparams import HParams
from .utils import space2batch
from .utils import batch2space


def log_sum_exp(x, axis=-1):
    """ numerically stable log_sum_exp implementation that prevents overflow
    **** code for this function from https://github.com/pclucas14/pixel-cnn-pp/blob/master/utils.py
    """
    # TF ordering
    #axis  = len(x.size()) - 1
    m, _  = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))

def log_prob_from_logits(x, axis=-1):
    """ numerically stable log_softmax implementation that prevents overflow
    **** code for this function from https://github.com/pclucas14/pixel-cnn-pp/blob/master/utils.py
    """
    # TF ordering
    #axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis, keepdim=True)
    return x - m - torch.log(torch.sum(torch.exp(x - m), dim=axis, keepdim=True))

def np_zeros(shape):
    """
    Builds a numpy variable filled with zeros
    Parameters
    ----------
    shape, tuple of ints
        shape of zeros to initialize
    Returns
    -------
    initialized_zeros, array-like
        Array-like of zeros the same size as shape parameter
    """
    return np.zeros(shape).astype("float32")


def np_normal(shape, random_state, scale=0.01):
    """
    Builds a numpy variable filled with normal random values
    Parameters
    ----------
    shape, tuple of ints or tuple of tuples
        shape of values to initialize
        tuple of ints should be single shape
        tuple of tuples is primarily for convnets and should be of form
        ((n_in_kernels, kernel_width, kernel_height),
         (n_out_kernels, kernel_width, kernel_height))
    random_state, numpy.random.RandomState() object
    scale, float (default 0.01)
        default of 0.01 results in normal random values with variance 0.01
    Returns
    -------
    initialized_normal, array-like
        Array-like of normal random values the same size as shape parameter
    """
    if type(shape[0]) is tuple:
        shp = (shape[1][0], shape[0][0]) + shape[1][1:]
    else:
        shp = shape
    return (scale * random_state.randn(*shp)).astype("float32")



def np_truncated_normal(shape, random_state, scale=0.075):
    """
    Builds a numpy variable filled with truncated normal random values
    Parameters
    ----------
    shape, tuple of ints or tuple of tuples
        shape of values to initialize
        tuple of ints should be single shape
        tuple of tuples is primarily for convnets and should be of form
        ((n_in_kernels, kernel_width, kernel_height),
         (n_out_kernels, kernel_width, kernel_height))
    random_state, numpy.random.RandomState() object
    scale, float (default 0.075)
        default of 0.075
    Returns
    -------
    initialized_normal, array-like
        Array-like of truncated normal random values the same size as shape parameter
    """
    if type(shape[0]) is tuple:
        shp = (shape[1][0], shape[0][0]) + shape[1][1:]
    else:
        shp = shape

    sigma = scale
    lower = -2 * sigma
    upper = 2 * sigma
    mu = 0
    N = np.prod(shp)
    samples = truncnorm.rvs(
              (lower - mu) / float(sigma), (upper - mu) / float(sigma),
              loc=mu, scale=sigma, size=N, random_state=random_state)
    return samples.reshape(shp).astype("float32")


def np_tanh_fan_normal(shape, random_state, scale=1.):
    """
    Builds a numpy variable filled with random values
    Parameters
    ----------
    shape, tuple of ints or tuple of tuples
        shape of values to initialize
        tuple of ints should be single shape
        tuple of tuples is primarily for convnets and should be of form
        ((n_in_kernels, kernel_width, kernel_height),
         (n_out_kernels, kernel_width, kernel_height))
    random_state, numpy.random.RandomState() object
    scale, float (default 1.)
        default of 1. results in normal random values
        with sqrt(2 / (fan in + fan out)) scale
    Returns
    -------
    initialized_fan, array-like
        Array-like of random values the same size as shape parameter
    References
    ----------
    Understanding the difficulty of training deep feedforward neural networks
        X. Glorot, Y. Bengio
    """
    # The . after the 2 is critical! shape has dtype int...
    if type(shape[0]) is tuple:
        kern_sum = np.prod(shape[0]) + np.prod(shape[1])
        shp = (shape[1][0], shape[0][0]) + shape[1][1:]
    else:
        kern_sum = np.sum(shape)
        shp = shape
    var = scale * np.sqrt(2. / kern_sum)
    return var * random_state.randn(*shp).astype("float32")


def np_variance_scaled_uniform(shape, random_state, scale=1.):
    """
    Builds a numpy variable filled with random values
    Parameters
    ----------
    shape, tuple of ints or tuple of tuples
        shape of values to initialize
        tuple of ints should be single shape
        tuple of tuples is primarily for convnets and should be of form
        ((n_in_kernels, kernel_width, kernel_height),
         (n_out_kernels, kernel_width, kernel_height))
    random_state, numpy.random.RandomState() object
    scale, float (default 1.)
        default of 1. results in uniform random values
        with 1 * sqrt(1 / (n_dims)) scale
    Returns
    -------
    initialized_scaled, array-like
        Array-like of random values the same size as shape parameter
    References
    ----------
    Efficient Backprop
        Y. LeCun, L. Bottou, G. Orr, K. Muller
    """
    if type(shape[0]) is tuple:
        shp = (shape[1][0], shape[0][0]) + shape[1][1:]
        kern_sum = np.prod(shape[0])
    else:
        shp = shape
        kern_sum = shape[0]
    #  Make sure bounds aren't the same
    bound = scale * np.sqrt(3. / float(kern_sum))  # sqrt(3) for std of uniform
    return random_state.uniform(low=-bound, high=bound, size=shp).astype(
        "float32")


def np_glorot_uniform(shape, random_state, scale=1.):
    """
    Builds a numpy variable filled with random values
    Parameters
    ----------
    shape, tuple of ints or tuple of tuples
        shape of values to initialize
        tuple of ints should be single shape
    random_state, numpy.random.RandomState() object
    scale, float (default 1.)
        default of 1. results in uniform random values
        with 1. * sqrt(6 / (n_in + n_out)) scale
    Returns
    -------
    initialized_scaled, array-like
        Array-like of random values the same size as shape parameter
    """
    if type(shape[0]) is tuple:
        from IPython import embed; embed(); raise ValueError()
        shp = (shape[1][0], shape[0][0]) + shape[1][1:]
        flat_shp = (shp[0], np.prod(shp[1:]))
    else:
        shp = shape
        kern_sum = sum(shp)
    bound = scale * np.sqrt(6. / float(kern_sum))
    return random_state.uniform(low=-bound, high=bound, size=shp).astype(
        "float32")


def np_ortho(shape, random_state, scale=1.):
    """
    Builds a numpy variable filled with orthonormal random values
    Parameters
    ----------
    shape, tuple of ints or tuple of tuples
        shape of values to initialize
        tuple of ints should be single shape
        tuple of tuples is primarily for convnets and should be of form
        ((n_in_kernels, kernel_width, kernel_height),
         (n_out_kernels, kernel_width, kernel_height))
    random_state, numpy.random.RandomState() object
    scale, float (default 1.)
        default of 1. results in orthonormal random values sacled by 1.
    Returns
    -------
    initialized_ortho, array-like
        Array-like of random values the same size as shape parameter
    References
    ----------
    Exact solutions to the nonlinear dynamics of learning in deep linear
    neural networks
        A. Saxe, J. McClelland, S. Ganguli
    """
    if type(shape[0]) is tuple:
        shp = (shape[1][0], shape[0][0]) + shape[1][1:]
        flat_shp = (shp[0], np.prod(shp[1:]))
    else:
        shp = shape
        flat_shp = shape
    g = random_state.randn(*flat_shp)
    U, S, VT = linalg.svd(g, full_matrices=False)
    res = U if U.shape == flat_shp else VT  # pick one with the correct shape
    res = res.reshape(shp)
    return (scale * res).astype("float32")


def make_numpy_biases(bias_dims, name=""):
    logger.info("Initializing {} with {} init".format(name, "zero"))
    #return [np.random.randn(dim,).astype("float32") for dim in bias_dims]
    return [np_zeros((dim,)) for dim in bias_dims]


def make_numpy_weights(in_dim, out_dims, random_state, init=None,
                       scale="default", name=""):
    """
    Will return as many things as are in the list of out_dims
    You *must* get a list back, even for 1 element
    blah, = make_weights(...)
    or
    [blah] = make_weights(...)

    linear example:
            weight_values, = make_numpy_weights(input_dim, [output_dim],
                                                random_state=random_state,
                                                init=init, scale=scale, name=name_w)

    conv example:
            shape usually constructed internally as:
            shp = (shape[1][0], shape[0][0]) + shape[1][1:]

            weight_values, = make_numpy_weights((input_channels, input_width, input_height),
                                                [(num_feature_maps, kernel_size[0], kernel_size[1])],
                                                init=init,
                                                scale=scale,
                                                random_state=random_state, name=name_w)

            this means input_width, input_height are ignored for most initializers
    """
    ff = [None] * len(out_dims)
    fs = [scale] * len(out_dims)
    for i, out_dim in enumerate(out_dims):
        if init is None:
            logger.info("Initializing {} with {} init".format(name, "ortho"))
            ff[i] = np_ortho
            fs[i] = 1.
            '''
            if in_dim == out_dim:
                logger.info("Initializing {} with {} init".format(name, "ortho"))
                ff[i] = np_ortho
                fs[i] = 1.
            else:
                logger.info("Initializing {} with {} init".format(name, "variance_scaled_uniform"))
                ff[i] = np_variance_scaled_uniform
                fs[i] = 1.
            '''
        elif init == "ortho":
            logger.info("Initializing {} with {} init".format(name, "ortho"))
            if in_dim != out_dim:
                raise ValueError("Unable to use ortho init for non-square matrices!")
            ff[i] = np_ortho
            fs[i] = 1.
        elif init == "glorot_uniform":
            logger.info("Initializing {} with {} init".format(name, "glorot_uniform"))
            ff[i] = np_glorot_uniform
        elif init == "normal":
            logger.info("Initializing {} with {} init".format(name, "normal"))
            ff[i] = np_normal
            fs[i] = 0.01
        elif init == "truncated_normal":
            logger.info("Initializing {} with {} init".format(name, "truncated_normal"))
            ff[i] = np_truncated_normal
            fs[i] = 0.075
        elif init == "embedding_normal":
            logger.info("Initializing {} with {} init".format(name, "embedding_normal"))
            ff[i] = np_truncated_normal
            fs[i] = 1. / np.sqrt(out_dim)
        else:
            raise ValueError("Unknown init type %s" % init)

    ws = []
    for i, out_dim in enumerate(out_dims):
        if fs[i] == "default":
            wi = ff[i]((in_dim, out_dim), random_state)
            if len(wi.shape) == 4:
                wi = wi.transpose(2, 3, 1, 0)
            ws.append(wi)
        else:
            wi = ff[i]((in_dim, out_dim), random_state, scale=fs[i])
            if len(wi.shape) == 4:
                wi = wi.transpose(2, 3, 1, 0)
            ws.append(wi)
    return ws

from scipy.stats import truncnorm
import sys
import uuid
from .core import get_logger
from collections import OrderedDict

logger = get_logger()

# Storage of internal shared
_lib_shared_params = OrderedDict()
has_warned = {}

def _shape(arr):
    return tuple(arr.shape)

def _ndim(arr):
    return len(_shape(arr))

def _get_name():
    return str(uuid.uuid4())

def get_params_dict():
    return _lib_shared_params

def _get_shared(name):
    if name in _lib_shared_params.keys():
        if name not in has_warned:
            logger.info("Found name %s in shared parameters" % name)
            has_warned[name] = True
        return _lib_shared_params[name]
    else:
        raise NameError("Name not found in shared params!")

def _check_shared(name):
    return name in _lib_shared_params.keys()

def _set_shared(name, variable):
    if name in _lib_shared_params.keys():
        raise ValueError("Trying to set key %s which already exists!" % name)
    _lib_shared_params[name] = variable

weight_norm_default = False
def get_weight_norm_default():
    return weight_norm_default

strict_mode_default = False
def get_strict_mode_default():
    return strict_mode_default


device_default = "cpu"
def get_device_default():
    return device_default

def set_device_default(device):
    global device_default
    device_default = device


dtype_default = "float32"
def get_dtype_default():
    return dtype_default

def set_dtype_default(dtype):
    global dtype_default
    dtype_default = dtype


def sigmoid(x):
    return torch.sigmoid(x)


def tanh(x):
    return torch.tanh(x)


def relu(x):
    return torch.nn.functional.relu(x)


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def softmax(x):
    # should work for both 2D and 3D
    e_x = torch.exp(x - x.max(dim=-1, keepdims=True)[0])
    out = e_x / e_x.sum(dim=-1, keepdims=True)
    return out

def softmax_np(x):
    # should work for both 2D and 3D
    e_x = np.exp(x - x.max(axis=-1, keepdims=True))
    out = e_x / e_x.sum(axis=-1, keepdims=True)
    return out

def _lcl_logsumexp(x):
    c = x.max()[0]
    return c + torch.log(torch.sum(torch.exp(x - c), axis=-1, keepdim=True))

def logsumexp(x, dim=0):
    # http://www.cs.toronto.edu/~rfm/code/logreg.py
    _max, _ = x.max(dim=-1, keepdims=True)
    ds = x - _max
    sum_exp = torch.exp(ds).sum(dim=dim, keepdims=True)
    return _max + torch.log(sum_exp)

def log_softmax(x):
    return x - logsumexp(x)

def make_tensor(arr, dtype, device, requires_grad=True):
    if device == "default":
        device = get_device_default()
    else:
        device = device

    if dtype == "default":
        dtype = get_dtype_default()

    if dtype == "float32":
        tensor = torch.from_numpy(arr.astype("float32")).to(device)
    elif dtype == "float64":
        tensor = torch.from_numpy(arr.astype("float64")).to(device)
    else:
        raise ValueError("Not yet implemented for dtype {}".format(dtype))
    if not requires_grad:
        tensor = tensor.requires_grad_(False)
    return tensor

def dot(a, b):
    # Generalized dot for nd sequences, assumes last axis is projection
    # b must be rank 2
    a_tup = _shape(a)
    b_tup = _shape(b)
    if len(a_tup) == 2 and len(b_tup) == 2:
        return torch.matmul(a, b)
    elif len(a_tup) == 3 and len(b_tup) == 2:
        # more generic, supports multiple -1 axes
        return torch.einsum("ijk,kl->ijl", a, b)
        #a_i = tf.reshape(a, [-1, a_tup[-1]])
        #a_n = tf.matmul(a_i, b)
        #a_nf = tf.reshape(a_n, list(a_tup[:-1]) + [b_tup[-1]])
        #return a_nf
    else:
        raise ValueError("Shapes for arguments to dot() are {} and {}, not supported!".format(a_tup, b_tup))

_scan_infos = {}
def scan(fn, sequences, outputs_info):
    nonepos = [n for n, o in enumerate(outputs_info) if o is None]
    nonnone = [o for o in outputs_info if o is not None]
    sequences_and_nonnone = sequences + nonnone
    sliced = [s[0] for s in sequences] + nonnone
    if (sys.version_info > (3, 0)):
        # need to figure out the code
        sig = (fn.__code__.co_filename,) + (fn.__code__.co_name,) + (fn.__code__.co_firstlineno,) + fn.__code__.co_varnames + (fn.__code__.co_code,)
    else:
        sig = (fn.func_code.co_filename,) + (fn.func_code.co_name,) + (fn.func_code.co_firstlineno,) + fn.func_code.co_varnames + (fn.func_code.co_code,)

    lu = hash(sig)
    global _scan_infos
    if lu not in _scan_infos:
        inf_ret = fn(*sliced)
        _scan_infos[lu] = inf_ret
    else:
        inf_ret = _scan_infos[lu]
    if len(outputs_info) < len(inf_ret):
        raise ValueError("More outputs from `fn` than elements in outputs_info. Expected {} outs, given outputs_info of length {}, but `fn` returns {}. Pass None in outputs_info for returns which don't accumulate".format(len(outputs_info), len(outputs_info), len(inf_ret)))
    initializers = []
    for n in range(len(outputs_info)):
        if outputs_info[n] is not None:
            initializers.append(outputs_info[n])
        else:
            initializers.append(0. * inf_ret[n])

    def wrapwrap(nonepos, initializers):
        type_class = "list" if isinstance(initializers, list) else "tuple"
        def fnwrap(accs, inps):
            inps_then_accs = inps + [a for n, a in enumerate(accs) if n not in nonepos]
            fn_rets = fn(*inps_then_accs)
            return [fr for fr in fn_rets]
        return fnwrap

    this_fn = wrapwrap(nonepos, initializers)
    def _scan(lclfn, seqs, inits):
        all_r = [[] for i in range(len(inits))]
        last_out = inits
        for i in range(len(seqs[0])):
            ri = lclfn(last_out, [seqs[n][i] for n in range(len(seqs))])
            last_out = ri
            if not hasattr(ri, "__len__"):
                ri = [ri]
            else:
                [all_r[j].append(ri[j]) for j in range(len(ri))]
        return all_r

    r = _scan(this_fn, sequences, initializers)
    return [torch.stack(rj) for rj in r]


def clipping_grad_norm_(parameters, rescale, named_parameters=False, named_check=False):
    # is a generator... get a static reference so the second iteration isn't empty
    if named_check:
        for n, p in parameters:
            print("Checking {} grad.data".format(n))
            assert p.grad.data is not None
            print(p.grad.data.sum())
            print("{}, OK".format(n))
        raise ValueError("named_check complete!")
    if not named_parameters:
        _params = [p for p in parameters]
    else:
        _params = [p[1] for p in parameters]

    grad_norm = torch.sqrt(sum([torch.sqrt(torch.pow(p.grad.data, 2).sum()) for p in _params]))
    scaling_num = rescale
    scaling_den = max([1.0 * rescale, grad_norm])
    scaling = scaling_num / scaling_den
    for p in _params:
        p.grad.data.mul_(scaling)


def clipping_grad_value_(parameters, clip_value, named_parameters=False, named_check=False):
    # is a generator... get a static reference so the second iteration isn't empty
    if named_check:
        for n, p in parameters:
            print("Checking {} grad.data".format(n))
            assert p.grad.data is not None
            print(p.grad.data.sum())
            print("{}, OK".format(n))
        raise ValueError("named_check complete!")
    if not named_parameters:
        _params = [p for p in parameters]
    else:
        _params = [p[1] for p in parameters]

    clip_value = float(clip_value)
    for p in _params:
        p.grad.data.clamp_(min=-clip_value, max=clip_value)


class Embedding(torch.nn.Module):
    def __init__(self,
                 n_symbols,
                 output_dim,
                 random_state=None,
                 init="embedding_normal",
                 scale=1.,
                 strict=None,
                 name=None,
                 dtype="default",
                 device="default"):
        """
        Last dimension of indices tensor must be 1!!!!
        """
        super(Embedding, self).__init__()

        if name is None:
            name = _get_name()

        if random_state is None:
            raise ValueError("Must pass random_state argument to Embedding")

        name_w = name + "_embedding_w"

        if strict is None:
            strict = get_strict_mode_default()

        if strict:
            cur_defs = get_params_dict()
            if name_w in cur_defs:
                raise ValueError("Name {} already created in params dict!".format(name_w))

        if init != "embedding_normal":
            raise ValueError("Currently unsupported init type {}".format(init))

        try:
            vectors = _get_shared(name_w)
        except NameError:
            vectors_weight, = make_numpy_weights(n_symbols, [output_dim],
                                                 random_state, init=init,
                                                 scale=scale, name=name_w)
            vectors = make_tensor(vectors_weight, dtype=dtype, device=device)
            #vectors = torch.from_numpy(vectors_weight).to(lcl_device)
            _set_shared(name_w, vectors)
        self.vectors = vectors

        th_embed = torch.nn.Embedding(n_symbols, output_dim)
        th_embed.weight.data.copy_(vectors)
        self.th_embed = th_embed

    def forward(self,
                indices):
        ii = indices.long()
        shp = _shape(ii)
        nd = _ndim(ii)
        if shp[-1] != 1:
            if nd < 3:
                logger.info("Embedding input should have last dimension 1, inferring dimension to 1, from shape {} to {}".format(shp, tuple(list(shp) + [1])))
                ii = ii[..., None]
            else:
                raise ValueError("Embedding layer input must have last dimension 1 for input size > 3D, got {}".format(shp))

        shp = _shape(ii)
        nd = len(shp)
        # force 3d for consistency, then slice
        lu = self.th_embed(ii[..., 0])
        return lu, self.vectors


class EmbeddingDropout(Embedding):
    """
    From ENAS
    https://github.com/carpedm20/ENAS-pytorch/blob/master/models/shared_rnn.py

    Class for dropping out embeddings by zero'ing out parameters in the
    embedding matrix.
    This is equivalent to dropping out particular words, e.g., in the sentence
    'the quick brown fox jumps over the lazy dog', dropping out 'the' would
    lead to the sentence '### quick brown fox jumps over ### lazy dog' (in the
    embedding vector space).
    See 'A Theoretically Grounded Application of Dropout in Recurrent Neural
    Networks', (Gal and Ghahramani, 2016).
    """
    def __init__(self,
                 n_symbols,
                 output_dim,
                 dropout_keep_prob=1.,
                 dropout_scale="default",
                 random_state=None,
                 init="embedding_normal",
                 scale=1.,
                 strict=None,
                 name=None,
                 dtype="default",
                 device="default"):
        """Embedding constructor.
        Args:
            dropout_keep_prob: Dropout probability.
            dropout_scale: Used to scale parameters of embedding weight matrix that are
                not dropped out. Note that this is _in addition_ to the
                `1/dropout_keep_prob scaling.
        See `Embedding` for remaining arguments.
        """
        Embedding.__init__(self,
                           n_symbols=n_symbols,
                           output_dim=output_dim,
                           random_state=random_state,
                           init=init,
                           scale=scale,
                           strict=strict,
                           name=name,
                           dtype=dtype,
                           device=device)
        self.g = torch.Generator(device=device)
        self.g.manual_seed(random_state.randint(100000))

        self.dropout_keep_prob = dropout_keep_prob
        if dropout_scale == "default":
            dropout_scale = output_dim ** 0.5
        self.dropout_scale = dropout_scale

    def forward(self, indices):
        """Embeds `indices` with the dropped out embedding weight matrix."""

        if self.training:
            dropout_keep_prob = self.dropout_keep_prob
        else:
            dropout_keep_prob = 1.

        if dropout_keep_prob != 1.:
            mask = self.th_embed.weight.data.new(self.th_embed.weight.size(0), 1)
            mask.bernoulli_(dropout_keep_prob, generator=self.g)
            mask = mask.expand_as(self.th_embed.weight)
            mask = mask / (dropout_keep_prob)
            masked_weight = self.th_embed.weight * Variable(mask)
        else:
            masked_weight = self.th_embed.weight

        if self.dropout_scale and self.dropout_scale != 1.:
            masked_weight = masked_weight * self.dropout_scale

        ii = indices.long()
        shp = _shape(ii)
        nd = _ndim(ii)
        if shp[-1] != 1:
            if nd < 3:
                logger.info("Embedding input should have last dimension 1, inferring dimension to 1, from shape {} to {}".format(shp, tuple(list(shp) + [1])))
                ii = ii[..., None]
            else:
                raise ValueError("Embedding layer input must have last dimension 1 for input size > 3D, got {}".format(shp))

        shp = _shape(ii)
        nd = len(shp)
        # force 3d for consistency, then slice
        lu = F.embedding(ii[..., 0], masked_weight)
        return lu, masked_weight


class Linear(torch.nn.Module):
    def __init__(self,
                 list_of_input_dims,
                 output_dim,
                 random_state=None,
                 name=None,
                 init=None,
                 scale="default",
                 biases=True,
                 bias_offset=0.,
                 dropout_flag_prob_keep=None,
                 strict=None,
                 dtype="default",
                 device="default"):
        super(Linear, self).__init__()

        if random_state is None:
            raise ValueError("must pass instance of np.random.RandomState!")
        input_dim = sum(list_of_input_dims)

        if name is None:
            name = _get_name()

        name_w = name + "_linear_w"
        name_b = name + "_linear_b"
        name_out = name + "_linear_out"

        if init is None or type(init) is str:
            #logger.info("Linear layer {} initialized using init {}".format(name, init))
            weight_values, = make_numpy_weights(input_dim, [output_dim],
                                                random_state=random_state,
                                                init=init, scale=scale, name=name_w)
        else:
            # rely on announcement from parent class
            weight_values=init[0]

        if strict is None:
            strict = get_strict_mode_default()

        if strict:
            cur_defs = get_params_dict()
            if name_w in cur_defs:
                raise ValueError("Name {} already created in params dict!".format(name_w))

            if name_b in cur_defs:
                raise ValueError("Name {} already created in params dict!".format(name_b))

        try:
            weight = _get_shared(name_w)
        except NameError:
            weight = make_tensor(weight_values, dtype=dtype, device=device)
            _set_shared(name_w, weight)

        self.weight = torch.nn.Parameter(weight)
        self.biases = None

        if biases:
            if (init is None) or (type(init) is str):
                b, = make_numpy_biases([output_dim], name=name_b)
            else:
                b = init[1]
            b = b + bias_offset
            try:
                biases = _get_shared(name_b)
            except NameError:
                biases = make_tensor(b, dtype=dtype, device=device)
                _set_shared(name_b, biases)
            self.biases = torch.nn.Parameter(biases)

    def forward(self,
                list_of_inputs,
                bias_offset=0.,
                dropout_flag_prob_keep=None):

        nd = _ndim(list_of_inputs[0])
        input_var = torch.cat(list_of_inputs, dim=nd - 1)
        if dropout_flag_prob_keep is not None:
            # no seed set here, it might not be repeatable
            input_var = torch.nn.functional.dropout(input_var, p=1. - dropout_flag_prob_keep, inplace=False)

        out = dot(input_var, self.weight)

        if self.biases is not None:
            out = out + self.biases
        return out


class Conv2d(torch.nn.Module):
    def __init__(self,
                 list_of_input_dims,
                 num_feature_maps,
                 kernel_size=(3, 3),
                 dilation=[1, 1],
                 strides=[1, 1],
                 input_height_width_init_tuple=(1, 1),
                 border_mode="same",
                 custom_weight_mask=None,
                 init=None, scale="default",
                 biases=True, bias_offset=0.,
                 name=None,
                 random_state=None, strict=None,
                 dtype="default", device="default"):
        super(Conv2d, self).__init__()

        #if strides != [1, 1]:
        #    raise ValueError("Alternate strides not yet supported in conv2d")

        #if dilation != [1, 1]:
        #    raise ValueError("Alternate dilation not yet supported in conv2d")

        # kernel is H, W
        # input assumption is N C H W
        if name is None:
            name = _get_name()

        if random_state is None:
            raise ValueError("Must pass instance of np.random.RandomState!")

        if strides != [1, 1]:
            if hasattr(strides, "__len__") and len(strides) == 2:
                pass
            else:
                try:
                    int(strides)
                    strides = [int(strides), int(strides)]
                except:
                    raise ValueError("Changing strides by non-int not yet supported")

        if dilation != [1, 1]:
            if hasattr(dilation, "__len__") and len(dilation) == 2:
                pass
            else:
                try:
                    int(dilation)
                    dilation = [int(dilation), int(dilation)]
                except:
                    raise ValueError("Changing dilation by non-int not yet supported")

        input_channels = sum(list_of_input_dims)
        #input_height, input_width = input_height_width_tuple
        # these numbers don't matter
        input_height, input_width = input_height_width_init_tuple

        if type(name) is str:
            name_w = name + "_conv2d_w"
            name_b = name + "_conv2d_b"
            name_out = name + "_conv2d_out"
            name_mask = name + "_conv2d_mask"

        if strict is None:
            strict = get_strict_mode_default()

        if strict:
            cur_defs = get_params_dict()
            if name_w in cur_defs:
                raise ValueError("Name {} already created in params dict!".format(name_w))

            if name_b in cur_defs:
                raise ValueError("Name {} already created in params dict!".format(name_b))

        if init is None or type(init) is str:
            weight_values, = make_numpy_weights((input_channels, input_width, input_height),
                                                [(num_feature_maps, kernel_size[0], kernel_size[1])],
                                                init=init,
                                                scale=scale,
                                                random_state=random_state, name=name_w)
        else:
            weight_values = init[0]
            name_w = name[0]
        weight_values = weight_values.transpose(3, 2, 0, 1)
        #weight_values = weight_values[::-1, ::-1].copy()

        try:
            weight = _get_shared(name_w)
        except NameError:
            weight = make_tensor(weight_values, dtype=dtype, device=device).contiguous()
            _set_shared(name_w, weight)

        self.weight = torch.nn.Parameter(weight)

        if custom_weight_mask is not None:
            """
            try:
                mask = _get_shared(name_mask)
            except NameError:
                mask = tf.Variable(custom_weight_mask, trainable=False, name=name_mask)
                _set_shared(name_mask, mask)
            """
            raise ValueError("custom_weight_mask not yet implemented in conv")
            weight = tf.constant(custom_weight_mask) * weight


        # need to custom handle SAME and VALID
        # rip
        # NCHW input, weights are out_chan, in_chan, H, W
        if biases:
            if (init is None) or (type(init) is str):
                b, = make_numpy_biases([num_feature_maps], name=name_b)
            else:
                b = init[1]
                name_b = name[1]
                name_out = name[2]
            b = b + bias_offset
            try:
                biases = _get_shared(name_b)
            except NameError:
                biases = make_tensor(b, dtype=dtype, device=device).contiguous()
                _set_shared(name_b, biases)
            self.biases = torch.nn.Parameter(biases)
        else:
            self.biases = None

        self.strides = strides
        self.dilation = dilation
        self.input_channels = input_channels
        # these numbers don't matter
        self.input_height = input_height
        self.input_width = input_width
        self.border_mode = border_mode
        self.kernel_size = kernel_size

    def forward(self,
                list_of_inputs):
        dilation = self.dilation
        strides = self.strides
        input_channels = self.input_channels
        input_height = self.input_height
        input_width = self.input_width
        border_mode = self.border_mode
        weight = self.weight
        biases = self.biases
        kernel_size = self.kernel_size

        #if strides != [1, 1]:
        #    raise ValueError("Alternate strides not yet supported in conv2d")

        #if dilation != [1, 1]:
        #    raise ValueError("Alternate dilation not yet supported in conv2d")

        # kernel is H, W
        # input assumption is N C H W

        if strides != [1, 1]:
            if hasattr(strides, "__len__") and len(strides) == 2:
                pass
            else:
                try:
                    int(strides)
                    strides = [int(strides), int(strides)]
                except:
                    raise ValueError("Changing strides by non-int not yet supported")

        if dilation != [1, 1]:
            if hasattr(dilation, "__len__") and len(dilation) == 2:
                pass
            else:
                try:
                    int(dilation)
                    dilation = [int(dilation), int(dilation)]
                except:
                    raise ValueError("Changing dilation by non-int not yet supported")

        input_t = torch.cat(list_of_inputs, dim=-1)

        if border_mode == "same":
            pad = "same"
        elif border_mode == "valid":
            pad = "valid"
        else:
            pad = border_mode
            if hasattr(pad, "__len__") and len(pad) == 2:
                pass
            else:
                try:
                    int(pad)
                    strides = [int(strides[0]), int(strides[1])]
                except:
                    raise ValueError("Pad must be integer, tuple of integer (hpad, wpad), or string 'same', 'valid'")

        # https://github.com/pytorch/pytorch/issues/3867
        # credit to @mirceamironenco
        def conv_outdim(in_dim, padding, ks, stride, dilation):
            if isinstance(padding, int) or isinstance(padding, tuple):
                return conv_outdim_general(in_dim, padding, ks, stride, dilation)
            elif isinstance(padding, str):
                assert padding in ['same', 'valid']
                if padding == 'same':
                    return conv_outdim_samepad(in_dim, stride)
                else:
                    return conv_outdim_general(in_dim, 0, ks, stride, dilation)
            else:
                raise TypeError('Padding can be int/tuple or str=same/valid')

        # https://github.com/pytorch/pytorch/issues/3867
        # credit to @mirceamironenco
        def conv_outdim_general(in_dim, padding, ks, stride, dilation=1):
            # See https://arxiv.org/pdf/1603.07285.pdf, eq (15)
            return ((in_dim + 2 * padding - ks - (ks - 1) * (dilation - 1)) // stride) + 1

        # https://github.com/pytorch/pytorch/issues/3867
        # credit to @mirceamironenco
        def conv_outdim_samepad(in_dim, stride):
            return (in_dim + stride - 1) // stride

        # https://github.com/pytorch/pytorch/issues/3867
        # credit to @mirceamironenco
        def pad_same(in_dim, ks, stride, dilation=1):
            """
            References:
                  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/common_shape_fns.h
                  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/common_shape_fns.cc#L21
            """
            assert stride > 0
            assert dilation >= 1
            effective_ks = (ks - 1) * dilation + 1
            out_dim = (in_dim + stride - 1) // stride
            p = max(0, (out_dim - 1) * stride + effective_ks - in_dim)

            padding_before = p // 2
            padding_after = p - padding_before
            return padding_before, padding_after

        if pad == "same":
            ph = pad_same(input_t.shape[-2], kernel_size[0], strides[-2], dilation[-2])[0]
            pw = pad_same(input_t.shape[-1], kernel_size[1], strides[-1], dilation[-1])[0]
        elif pad == "valid":
            raise ValueError("valid pad NYI")
            from IPython import embed; embed(); raise ValueError()
        else:
            if hasattr(pad, "__len__") and len(pad) == 2:
                ph = pad[0]
                pw = pad[1]
            else:
                int(pad)
                ph = pad
                pw = pad

        # NCHW input, weights are out_chan, in_chan, H, W
        out = torch.nn.functional.conv2d(input_t, weight, stride=strides, dilation=dilation, padding=(ph, pw), bias=biases)
        return out


class Conv2dTranspose(torch.nn.Module):
    def __init__(self,
                 list_of_input_dims,
                 num_feature_maps,
                 kernel_size=(3, 3),
                 dilation=[1, 1],
                 strides=[1, 1],
                 input_height_width_init_tuple=(1, 1),
                 border_mode="same",
                 output_padding=(0, 0),
                 custom_weight_mask=None,
                 init=None, scale="default",
                 biases=True, bias_offset=0.,
                 name=None,
                 random_state=None, strict=None,
                 dtype="default", device="default"):
        super(Conv2dTranspose, self).__init__()

        #if strides != [1, 1]:
        #    raise ValueError("Alternate strides not yet supported in conv2d")

        #if dilation != [1, 1]:
        #    raise ValueError("Alternate dilation not yet supported in conv2dTranspose")

        # kernel is H, W
        # input assumption is N C H W
        if name is None:
            name = _get_name()

        if random_state is None:
            raise ValueError("Must pass instance of np.random.RandomState!")

        if strides != [1, 1]:
            if hasattr(strides, "__len__") and len(strides) == 2:
                pass
            else:
                try:
                    int(strides)
                    strides = [int(strides), int(strides)]
                except:
                    raise ValueError("Changing strides by non-int not yet supported")

        if dilation != [1, 1]:
            if hasattr(dilation, "__len__") and len(dilation) == 2:
                pass
            else:
                try:
                    int(dilation)
                    dilation = [int(dilation), int(dilation)]
                except:
                    raise ValueError("Changing dilation by non-int not yet supported")

        input_channels = sum(list_of_input_dims)
        #input_height, input_width = input_height_width_tuple
        # these numbers don't matter
        input_height, input_width = input_height_width_init_tuple

        if type(name) is str:
            name_w = name + "_conv2d_transpose_w"
            name_b = name + "_conv2d_transpose_b"
            name_out = name + "_conv2d_transpose_out"
            name_mask = name + "_conv2d_transpose_mask"

        if strict is None:
            strict = get_strict_mode_default()

        if strict:
            cur_defs = get_params_dict()
            if name_w in cur_defs:
                raise ValueError("Name {} already created in params dict!".format(name_w))

            if name_b in cur_defs:
                raise ValueError("Name {} already created in params dict!".format(name_b))

        if init is None or type(init) is str:
            weight_values, = make_numpy_weights((input_channels, input_width, input_height),
                                                [(num_feature_maps, kernel_size[0], kernel_size[1])],
                                                init=init,
                                                scale=scale,
                                                random_state=random_state, name=name_w)
        else:
            weight_values = init[0]
            name_w = name[0]
        weight_values = weight_values.transpose(2, 3, 0, 1)
        #weight_values = weight_values[::-1, ::-1].copy()

        try:
            weight = _get_shared(name_w)
        except NameError:
            weight = make_tensor(weight_values, dtype=dtype, device=device).contiguous()
            _set_shared(name_w, weight)

        self.weight = torch.nn.Parameter(weight)

        if custom_weight_mask is not None:
            """
            try:
                mask = _get_shared(name_mask)
            except NameError:
                mask = tf.Variable(custom_weight_mask, trainable=False, name=name_mask)
                _set_shared(name_mask, mask)
            """
            raise ValueError("custom_weight_mask not yet implemented in conv")
            weight = tf.constant(custom_weight_mask) * weight


        # need to custom handle SAME and VALID
        # rip
        # NCHW input, weights are out_chan, in_chan, H, W
        if biases:
            if (init is None) or (type(init) is str):
                b, = make_numpy_biases([num_feature_maps], name=name_b)
            else:
                b = init[1]
                name_b = name[1]
                name_out = name[2]
            b = b + bias_offset
            try:
                biases = _get_shared(name_b)
            except NameError:
                biases = make_tensor(b, dtype=dtype, device=device).contiguous()
                _set_shared(name_b, biases)
            self.biases = torch.nn.Parameter(biases)

        self.strides = strides
        self.dilation = dilation
        self.input_channels = input_channels
        # these numbers don't matter
        self.input_height = input_height
        self.input_width = input_width
        self.border_mode = border_mode
        self.kernel_size = kernel_size
        self.output_padding = output_padding

    def forward(self,
                list_of_inputs):
        dilation = self.dilation
        strides = self.strides
        input_channels = self.input_channels
        input_height = self.input_height
        input_width = self.input_width
        border_mode = self.border_mode
        weight = self.weight
        biases = self.biases
        kernel_size = self.kernel_size
        output_padding = self.output_padding

        #if strides != [1, 1]:
        #    raise ValueError("Alternate strides not yet supported in conv2d")

        #if dilation != [1, 1]:
        #    raise ValueError("Alternate dilation not yet supported in conv2d")

        # kernel is H, W
        # input assumption is N C H W

        if strides != [1, 1]:
            if hasattr(strides, "__len__") and len(strides) == 2:
                pass
            else:
                try:
                    int(strides)
                    strides = [int(strides), int(strides)]
                except:
                    raise ValueError("Changing strides by non-int not yet supported")

        if dilation != [1, 1]:
            if hasattr(dilation, "__len__") and len(dilation) == 2:
                pass
            else:
                try:
                    int(dilation)
                    dilation = [int(dilation), int(dilation)]
                except:
                    raise ValueError("Changing dilation by non-int not yet supported")

        input_t = torch.cat(list_of_inputs, dim=-1)

        if border_mode == "same":
            pad = "same"
        elif border_mode == "valid":
            pad = "valid"
        else:
            pad = border_mode
            if hasattr(pad, "__len__") and len(pad) == 2:
                pass
            else:
                try:
                    int(pad)
                    strides = [int(strides[0]), int(strides[1])]
                except:
                    raise ValueError("Pad must be integer, tuple of integer (hpad, wpad), or string 'same', 'valid'")

        # https://github.com/pytorch/pytorch/issues/3867
        # credit to @mirceamironenco
        def conv_outdim(in_dim, padding, ks, stride, dilation):
            if isinstance(padding, int) or isinstance(padding, tuple):
                return conv_outdim_general(in_dim, padding, ks, stride, dilation)
            elif isinstance(padding, str):
                assert padding in ['same', 'valid']
                if padding == 'same':
                    return conv_outdim_samepad(in_dim, stride)
                else:
                    return conv_outdim_general(in_dim, 0, ks, stride, dilation)
            else:
                raise TypeError('Padding can be int/tuple or str=same/valid')

        # https://github.com/pytorch/pytorch/issues/3867
        # credit to @mirceamironenco
        def conv_outdim_general(in_dim, padding, ks, stride, dilation=1):
            # See https://arxiv.org/pdf/1603.07285.pdf, eq (15)
            return ((in_dim + 2 * padding - ks - (ks - 1) * (dilation - 1)) // stride) + 1

        # https://github.com/pytorch/pytorch/issues/3867
        # credit to @mirceamironenco
        def conv_outdim_samepad(in_dim, stride):
            return (in_dim + stride - 1) // stride

        # https://github.com/pytorch/pytorch/issues/3867
        # credit to @mirceamironenco
        def pad_same(in_dim, ks, stride, dilation=1):
            """
            References:
                  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/common_shape_fns.h
                  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/common_shape_fns.cc#L21
            """
            assert stride > 0
            assert dilation >= 1
            effective_ks = (ks - 1) * dilation + 1
            out_dim = (in_dim + stride - 1) // stride
            p = max(0, (out_dim - 1) * stride + effective_ks - in_dim)

            padding_before = p // 2
            padding_after = p - padding_before
            return padding_before, padding_after

        if pad == "same":
            ph = pad_same(input_t.shape[-2], kernel_size[0], strides[-2], dilation[-2])[0]
            pw = pad_same(input_t.shape[-1], kernel_size[1], strides[-1], dilation[-1])[0]
        elif pad == "valid":
            raise ValueError("valid pad NYI")
            from IPython import embed; embed(); raise ValueError()
        else:
            if hasattr(pad, "__len__") and len(pad) == 2:
                ph = pad[0]
                pw = pad[1]
            else:
                int(pad)
                ph = pad
                pw = pad

        # NCHW input, weights are out_chan, in_chan, H, W
        out = torch.nn.functional.conv_transpose2d(input_t, weight, stride=strides, dilation=dilation, padding=(ph, pw), bias=biases,
                                                   output_padding=output_padding)
        return out


class BatchNorm2d(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 train_test_flag,
                 gamma_init=1., beta_init=0.,
                 decay=0.9,
                 eps=1E-3,
                 strict=None,
                 name=None,
                 dtype="default",
                 device="default"):
        super(BatchNorm2d, self).__init__()
        # https://r2rt.com/implementing-batch-normalization-in-tensorflow.html
        # NCHW convention
        if name is None:
            name = _get_name()

        name_scale = name + "_batchnorm_s"
        name_beta = name + "_batchnorm_b"
        name_out = name + "_batchnorm_out"
        if strict is None:
            strict = get_strict_mode_default()

        if strict:
            cur_defs = get_params_dict()
            if name_scale in cur_defs:
                raise ValueError("Name {} already created in params dict!".format(name_scale))

            if name_beta in cur_defs:
                raise ValueError("Name {} already created in params dict!".format(name_beta))

        try:
            scale = _get_shared(name_scale)
        except NameError:
            scale_values = gamma_init * np.ones((input_dim,))
            scale = make_tensor(scale_values, dtype=dtype, device=device)
            _set_shared(name_scale, scale)

        try:
            beta = _get_shared(name_beta)
        except NameError:
            # init with ones? it's what I did in TF
            beta_values = beta_init * np.ones((input_dim,))
            beta = make_tensor(beta_values, dtype=dtype, device=device)
            _set_shared(name_beta, beta)

        self.beta = torch.nn.Parameter(beta)
        self.scale = torch.nn.Parameter(scale)
        self.decay = decay
        self.eps = eps
        self.dtype = dtype
        self.device = device
        self.is_half = False

    def half(self):
        self.is_half = True

    def forward(self, input_tensor, train_test_flag=None):
        # 0 train, 1 test
        # https://stackoverflow.com/questions/44887446/pytorch-nn-functional-batch-norm-for-2d-input
        if train_test_flag == None:
            train_test_flag = 0.

        scale = self.scale
        beta = self.beta
        eps = self.eps
        decay = self.decay
        dtype = self.dtype
        device = self.device

        pop_mean = make_tensor(np.zeros((input_tensor.shape[1],)), dtype=dtype, device=device, requires_grad=False)
        pop_var = make_tensor(np.ones((input_tensor.shape[1],)), dtype=dtype, device=device, requires_grad=False)
        if self.is_half:
            pop_mean = pop_mean.half()
            pop_var = pop_var.half()

        shp = _shape(input_tensor)
        def left():
            return torch.nn.functional.batch_norm(input_tensor, pop_mean, pop_var, weight=scale, bias=beta, momentum=1. - decay, eps=eps, training=True)

        def right():
            return torch.nn.functional.batch_norm(input_tensor, pop_mean, pop_var, training=False, weight=scale, bias=beta, eps=eps)

        if train_test_flag <= 0.5:
            out = left()
        else:
            out = right()
        return out

class Conv1d(torch.nn.Module):
    def __init__(self,
                 list_of_input_dims,
                 num_feature_maps,
                 kernel_size,
                 border_mode="same",
                 init=None,
                 scale="default",
                 biases=True,
                 bias_offset=0.,
                 name=None,
                 random_state=None,
                 strict=None,
                 dtype="default",
                 device="default"):
        super(Conv1d, self).__init__()

        if name is None:
            name = _get_name()

        # assuming they come in as length, batch, features
        #tlist = [li[:, None].permute((2, 3, 1, 0)) for li in list_of_inputs]
        # now N C H W, height of 1 (so laid out along width dim)
        self.conv = Conv2d(list_of_input_dims, num_feature_maps,
                           kernel_size=kernel_size,
                           name=name + "_conv1d", random_state=random_state,
                           border_mode=border_mode, init=init, scale=scale, biases=biases,
                           bias_offset=bias_offset, strict=strict, dtype=dtype, device=device)

    def forward(self,
                list_of_inputs):
        # assuming they come in as length, batch, features
        tlist = [li[:, None].permute((2, 3, 1, 0)) for li in list_of_inputs]
        # now N C H W, height of 1 (so laid out along width dim)
        pre = self.conv(tlist)
        return pre[:, :, 0].permute(2, 0, 1)


class SequenceConv1dStack(torch.nn.Module):
    def __init__(self,
                 list_of_input_dims,
                 num_feature_maps,
                 batch_norm_flag=None,
                 n_stacks=1,
                 residual=True,
                 activation="relu",
                 kernel_sizes=[(1, 1), (3, 3), (5, 5)],
                 border_mode="same",
                 init=None,
                 scale="default",
                 biases=True,
                 bias_offset=0.,
                 name=None,
                 random_state=None,
                 strict=None,
                 dtype="default",
                 device="default"):
        super(SequenceConv1dStack, self).__init__()

        if name is None:
            name = _get_name()

        # assuming they come in as length, batch, features
        #tlist = [li[:, None].permute((2, 3, 1, 0)) for li in list_of_inputs]
        # now N C H W, height of 1 (so laid out along width dim)
        c = Conv2d(list_of_input_dims, len(kernel_sizes) * num_feature_maps,
                   kernel_size=(1, 1),
                   name=name + "_convpre", random_state=random_state,
                   border_mode=border_mode, init=init, scale=scale, biases=biases,
                   bias_offset=bias_offset, strict=strict, dtype=dtype, device=device)
        layers = torch.nn.ModuleList()
        layers.append(torch.nn.ModuleList([c,]))

        for ii in range(n_stacks):
            cs = torch.nn.ModuleList()
            for jj, ks in enumerate(kernel_sizes):
                c = Conv2d([len(kernel_sizes) * num_feature_maps], num_feature_maps,
                           kernel_size=ks,
                           name=name + "_conv{}_ks{}".format(ii, jj), random_state=random_state,
                           border_mode=border_mode, init=init, scale=scale, biases=biases,
                           bias_offset=bias_offset, strict=strict, dtype=dtype, device=device)
                cs.append(c)
            # cat along channel axis
            bn_l = BatchNorm2d(len(cs) * num_feature_maps, batch_norm_flag, name="bn_conv{}".format(ii), dtype=dtype, device=device)
            cs.append(bn_l)
            # ????
            #r_l = ReLU(bn_l)
            layers.append(cs)
            #prev_layer = prev_layer + r_l

        post = Conv2d([len(kernel_sizes) * num_feature_maps], num_feature_maps,
                       kernel_size=(1, 1),
                       name=name + "_convpost", random_state=random_state,
                       border_mode=border_mode, init=init, scale=scale, biases=biases,
                       bias_offset=bias_offset, strict=strict,
                       dtype=dtype, device=device)

        li = torch.nn.ModuleList()
        li.append(post)
        layers.append(li)
        self.layers = layers
        self.n_stacks = n_stacks
        self.kernel_sizes = kernel_sizes

    def forward(self,
                list_of_inputs,
                batch_norm_flag=None):

        # assuming they come in as length, batch, features
        tlist = [li[:, None].permute((2, 3, 1, 0)) for li in list_of_inputs]
        # now N C H W, height of 1 (so laid out along width dim)
        pre = self.layers[0][0](tlist)
        n_stacks = self.n_stacks
        kernel_sizes = self.kernel_sizes
        prev_layer = pre
        for ii in range(n_stacks):
            cs = []
            for jj, ks in enumerate(kernel_sizes):
                # off by one to account for pre
                c = self.layers[ii + 1][jj]([prev_layer])
                cs.append(c)
            c_layer = torch.cat(cs, dim=1)
            # cat along channel axis
            # off by one to account for bn layer last
            bn_l = self.layers[ii + 1][jj + 1](c_layer, batch_norm_flag)
            r_l = relu(bn_l)
            prev_layer = prev_layer + r_l
        post = self.layers[ii + 2][0]([prev_layer])
        return post[:, :, 0].permute(2, 0, 1)


class LSTMCell(torch.nn.Module):
    def __init__(self,
                 list_of_input_dims,
                 num_units,
                 output_dim=None,
                 input_mask=None,
                 random_state=None,
                 name=None, init=None, scale="default",
                 forget_bias=1.,
                 strict=None,
                 device="default"):
        super(LSTMCell, self).__init__()
        # cell_dropout should be a value in [0., 1.], or None
        # output is the thing to use in following layers, state is a tuple that feeds into the next call
        if random_state is None:
            raise ValueError("Must pass random_state")

        if name is None:
            name = _get_name()

        input_dim = sum(list_of_input_dims)
        hidden_dim = 4 * num_units

        if init is None:
            inp_init = None
            h_init = None
            out_init = None
        elif init == "truncated_normal":
            inp_init = "truncated_normal"
            h_init = "truncated_normal"
            out_init = "truncated_normal"
        elif init == "glorot_uniform":
            inp_init = "glorot_uniform"
            h_init = "glorot_uniform"
            out_init = "glorot_uniform"
        elif init == "normal":
            inp_init = "normal"
            h_init = "normal"
            out_init = "normal"
        else:
            raise ValueError("Unknown init argument {}".format(init))

        name_proj = name + "_lstm_proj"
        name_w = name + "_lstm_proj_w"
        name_b = name + "_lstm_proj_b"
        comb_w_np, = make_numpy_weights(input_dim + num_units, [hidden_dim],
                                        random_state=random_state,
                                        init=inp_init, name=name_w)
        comb_b_np, = make_numpy_biases([hidden_dim], name=name_b)

        logger.info("LSTMCell {} input to hidden initialized using init {}".format(name, inp_init))
        logger.info("LSTMCell {} hidden to hidden initialized using init {}".format(name, h_init))

        lstm_proj_obj = Linear(list_of_input_dims + [hidden_dim],
                               hidden_dim,
                               random_state=random_state,
                               name=name_proj,
                               init=(comb_w_np, comb_b_np), strict=strict,
                               device=device)

        if output_dim is not None:
            name_out = name + "_lstm_h_to_out",
            name_out_w = name + "_lstm_h_to_out_w",
            name_out_b = name + "_lstm_h_to_out_b",
            h_to_out_w_np, = make_numpy_weights(num_units, [output_dim],
                                                random_state=random_state,
                                                init=out_init, name=name_out_w)
            h_to_out_b_np, = make_numpy_biases([output_dim], name=name_out_b)
            h_to_out_obj = Linear([num_units], output_dim, random_state=random_state,
                              name=name_out,
                              init=(h_to_out_w_np, h_to_out_b_np), strict=strict,
                              device=device)
            self.h_to_out_obj = h_to_out_obj
        self.lstm_proj_obj = lstm_proj_obj
        self.num_units = num_units
        self.input_dim = input_dim
        self.forget_bias = forget_bias

    def forward(self,
                list_of_inputs,
                previous_hidden, previous_cell,
                output_dim=None,
                input_mask=None,
                cell_dropout=None):
        # cell_dropout should be a value in [0., 1.], or None
        # output is the thing to use in following layers, state is a tuple that feeds into the next call

        input_dim = self.input_dim
        num_units = self.num_units
        forget_bias = self.forget_bias

        ph = previous_hidden
        pc = previous_cell

        lstm_proj = self.lstm_proj_obj(list_of_inputs + [ph])

        i, j, f, o = torch.split(lstm_proj, num_units, dim=-1)


        if cell_dropout is not None:
            pj = torch.nn.functional.dropout(tanh(j), 1. - cell_dropout)
        else:
            pj = tanh(j)

        c = sigmoid(f + forget_bias) * pc + sigmoid(i) * pj
        if input_mask is not None:
            c = input_mask[:, None] * c + (1. - input_mask[:, None]) * pc

        h = sigmoid(o) * tanh(c)
        if input_mask is not None:
            # this line was bugged in released / trained version!
            # https://github.com/kastnerkyle/representation_mixing/blob/master/code/lib/tfbldr/nodes/nodes.py#L1554
            # fixed here but will mean things are different
            # when masks are used
            h = input_mask[:, None] * h + (1. - input_mask[:, None]) * ph

        if output_dim is not None:
            h_to_out = self.h_to_out_obj([h])
            final_out = h_to_out
        else:
            final_out = h
        return final_out, (h, c)


class LSTMLayer(torch.nn.Module):
    def __init__(self,
                 list_of_input_dims,
                 num_units,
                 output_dim=None,
                 random_state=None,
                 name=None,
                 init=None,
                 scale="default",
                 forget_bias=1.,
                 strict=None,
                 device="default"):
        super(LSTMLayer, self).__init__()
        if name is None:
            name = _get_name()
        name = name + "_lstm_layer"
        name_proj = name + "_proj"
        hidden_dim = 4 * num_units
        in_proj_obj = Linear(list_of_input_dims,
                             hidden_dim,
                             random_state=random_state,
                             name=name_proj,
                             init=init,
                             scale=scale,
                             strict=strict,
                             device=device)

        fwd_cell_obj = LSTMCell([hidden_dim],
                                num_units,
                                random_state=random_state,
                                name=name + "_forward_rnn",
                                init=init,
                                scale=scale,
                                device=device)

        self.in_proj_obj = in_proj_obj
        self.fwd_cell_obj = fwd_cell_obj
        self.num_units = num_units

    def forward(self, list_of_inputs,
                previous_forward_hidden=None, previous_forward_cell=None,
                input_mask=None,
                cell_dropout=None,
                strict=None):

        num_units = self.num_units

        in_proj = self.in_proj_obj(list_of_inputs)
        if input_mask is None:
            input_mask = 0. * in_proj[..., 0] + 1.

        if previous_forward_hidden == None:
            h1_f_init = 0. * in_proj[0, :, :num_units].detach()
        else:
            h1_f_init = previous_forward_hidden
        if previous_forward_cell == None:
            c1_f_init = 0. * in_proj[0, :, :num_units].detach()
        else:
            c1_f_init = previous_forward_cell

        def step(inp_t, inp_mask_t,
                 h1_f_tm1, c1_f_tm1):
            output, s = self.fwd_cell_obj([inp_t],
                                          h1_f_tm1, c1_f_tm1,
                                          input_mask=inp_mask_t,
                                          cell_dropout=cell_dropout)
            h1_f_t = s[0]
            c1_f_t = s[1]
            return h1_f_t, c1_f_t

        # should this be a "proper" flip with mask on the end
        r = scan(step,
                 [in_proj, input_mask],
                 [h1_f_init, c1_f_init])
        return r[0], r[0], r[1]


class BiLSTMLayer(torch.nn.Module):
    def __init__(self,
                 list_of_input_dims,
                 num_units,
                 output_dim=None,
                 random_state=None,
                 name=None, init=None, scale="default",
                 forget_bias=1.,
                 strict=None,
                 device="default"):
        super(BiLSTMLayer, self).__init__()
        if name is None:
            name = _get_name()
        name = name + "_bidirlstm_layer"
        name_proj = name + "_proj"
        hidden_dim = 4 * num_units
        in_proj_obj = Linear(list_of_input_dims,
                             hidden_dim,
                             random_state=random_state,
                             name=name_proj,
                             init=init, strict=strict,
                             device=device)

        fwd_cell_obj = LSTMCell([hidden_dim],
                                num_units,
                                random_state=random_state,
                                name=name + "forward_rnn",
                                init=init,
                                device=device)

        rev_cell_obj = LSTMCell([hidden_dim],
                                 num_units,
                                 random_state=random_state,
                                 name=name + "reverse_rnn",
                                 init=init,
                                 device=device)

        self.in_proj_obj = in_proj_obj
        self.fwd_cell_obj = fwd_cell_obj
        self.rev_cell_obj = rev_cell_obj
        self.num_units = num_units

    def forward(self, list_of_inputs,
                previous_forward_hidden=None, previous_forward_cell=None,
                previous_reverse_hidden=None, previous_reverse_cell=None,
                input_mask=None,
                cell_dropout=None,
                strict=None):

        num_units = self.num_units
        if input_mask is None:
            raise ValueError("No input mask currently unsupported")

        in_proj = self.in_proj_obj(list_of_inputs)

        if previous_forward_hidden == None:
            h1_f_init = 0. * in_proj[0, :, :num_units].detach()
        else:
            h1_f_init = previous_forward_hidden
        if previous_reverse_hidden == None:
            h1_b_init = 0. * in_proj[0, :, :num_units].detach()
        else:
            h1_b_init = previous_reverse_hidden
        if previous_forward_cell == None:
            c1_f_init = 0. * in_proj[0, :, :num_units].detach()
        else:
            c1_f_init = previous_forward_cell
        if previous_reverse_cell == None:
            c1_b_init = 0. * in_proj[0, :, :num_units].detach()
        else:
            c1_b_init = previous_reverse_cell

        def step(inp_t, inp_mask_t,
                 rev_inp_t, rev_inp_mask_t,
                 h1_f_tm1, c1_f_tm1, h1_b_tm1, c1_b_tm1):
            output, s = self.fwd_cell_obj([inp_t],
                                          h1_f_tm1, c1_f_tm1,
                                          input_mask=inp_mask_t,
                                          cell_dropout=cell_dropout)
            h1_f_t = s[0]
            c1_f_t = s[1]

            output, s = self.rev_cell_obj([rev_inp_t],
                                          h1_b_tm1, c1_b_tm1,
                                          input_mask=rev_inp_mask_t,
                                          cell_dropout=cell_dropout)
            h1_b_t = s[0]
            c1_b_t = s[1]
            return h1_f_t, c1_f_t, h1_b_t, c1_b_t

        # should this be a "proper" flip with mask on the end
        r = scan(step,
                 [in_proj, input_mask, torch.flip(in_proj, (0,)), torch.flip(input_mask, (0,))],
                 [h1_f_init, c1_f_init, h1_b_init, c1_b_init])
        return torch.cat([r[0], torch.flip(r[2], (0,))], dim=-1)


class GRUCell(torch.nn.Module):
    def __init__(self,
                 list_of_input_dims,
                 num_units,
                 output_dim=None,
                 input_mask=None,
                 random_state=None,
                 name=None, init=None, scale="default",
                 forget_bias=1.,
                 strict=None,
                 device="default"):
        super(GRUCell, self).__init__()
        # cell_dropout should be a value in [0., 1.], or None
        # output is the thing to use in following layers, state is a tuple that feeds into the next call
        if random_state is None:
            raise ValueError("Must pass random_state")

        if name is None:
            name = _get_name()

        input_dim = sum(list_of_input_dims)
        hidden_dim = 3 * num_units

        if init is None:
            inp_init = None
            h_init = None
            out_init = None
        elif init == "truncated_normal":
            inp_init = "truncated_normal"
            h_init = "truncated_normal"
            out_init = "truncated_normal"
        elif init == "glorot_uniform":
            inp_init = "glorot_uniform"
            h_init = "glorot_uniform"
            out_init = "glorot_uniform"
        elif init == "normal":
            inp_init = "normal"
            h_init = "normal"
            out_init = "normal"
        else:
            raise ValueError("Unknown init argument {}".format(init))

        name_gate = name + "_gru_gate"
        name_gate_w = name + "_gru_gate_w"
        name_gate_b = name + "_gru_gate_b"
        gate_w_np, = make_numpy_weights(input_dim + num_units, [2 * num_units],
                                        random_state=random_state,
                                        init=inp_init, name=name_gate_w)
        gate_b_np, = make_numpy_biases([2 * num_units], name=name_gate_b)
        # for now leave them 0 then swap to URGRU

        logger.info("GRUCell {} input to hidden initialized using init {}".format(name, inp_init))
        logger.info("GRUCell {} hidden to hidden initialized using init {}".format(name, h_init))

        gru_gate_obj = Linear(list_of_input_dims + [hidden_dim],
                              2 * num_units,
                              random_state=random_state,
                              name=name_gate,
                              init=(gate_w_np, gate_b_np), strict=strict,
                              device=device)

        name_proj = name + "_gru_proj"
        name_proj_w = name + "_gru_proj_w"
        name_proj_b = name + "_gru_proj_b"
        proj_w_np, = make_numpy_weights(input_dim + num_units, [num_units],
                                        random_state=random_state,
                                        init=inp_init, name=name_proj_w)
        proj_b_np, = make_numpy_biases([num_units], name=name_proj_b)

        gru_proj_obj = Linear(list_of_input_dims + [hidden_dim],
                              num_units,
                              random_state=random_state,
                              name=name_proj,
                              init=(proj_w_np, proj_b_np), strict=strict,
                              device=device)



        if output_dim is not None:
            raise ValueError("GRU out proj not yet implemented")
            name_out = name + "_gru_h_to_out",
            name_out_w = name + "_gru_h_to_out_w",
            name_out_b = name + "_gru_h_to_out_b",
            h_to_out_w_np, = make_numpy_weights(num_units, [output_dim],
                                                random_state=random_state,
                                                init=out_init, name=name_out_w)
            h_to_out_b_np, = make_numpy_biases([output_dim], name=name_out_b)
            h_to_out_obj = Linear([num_units], output_dim, random_state=random_state,
                              name=name_out,
                              init=(h_to_out_w_np, h_to_out_b_np), strict=strict,
                              device=device)
            self.h_to_out_obj = h_to_out_obj
        self.gru_gate_obj = gru_gate_obj
        self.gru_proj_obj = gru_proj_obj
        self.num_units = num_units
        self.input_dim = input_dim

    def forward(self,
                list_of_inputs,
                previous_hidden, previous_cell=None,
                output_dim=None,
                input_mask=None,
                cell_dropout=None):
        # cell_dropout should be a value in [0., 1.], or None
        # output is the thing to use in following layers, state is a tuple that feeds into the next call

        input_dim = self.input_dim
        num_units = self.num_units

        ph = previous_hidden

        pc = previous_cell
        # pc should be None
        # ignore previous cell, return None for it since this is GRU, only for compat with LSTM

        gru_gate = sigmoid(self.gru_gate_obj(list_of_inputs + [ph]))
        r, z = gru_gate[..., :self.num_units], gru_gate[..., self.num_units:]

        # TODO: UR mods?
        # https://arxiv.org/pdf/1910.09890.pdf

        r_state = r * ph
        gru_proj = tanh(self.gru_proj_obj(list_of_inputs + [r_state]))

        h = z * gru_proj + (1. - z) * ph
        if input_mask is not None:
            h = input_mask[:, None] * h + (1. - input_mask[:, None]) * ph

        final_out = h
        return final_out, (h, None)


class GRULayer(torch.nn.Module):
    def __init__(self,
                 list_of_input_dims,
                 num_units,
                 output_dim=None,
                 random_state=None,
                 name=None,
                 init=None,
                 scale="default",
                 forget_bias=1.,
                 strict=None,
                 device="default"):
        super(GRULayer, self).__init__()
        if name is None:
            name = _get_name()
        name = name + "_gru_layer"
        name_proj = name + "_proj"
        hidden_dim = 4 * num_units
        in_proj_obj = Linear(list_of_input_dims,
                             hidden_dim,
                             random_state=random_state,
                             name=name_proj,
                             init=init,
                             scale=scale,
                             strict=strict,
                             device=device)

        fwd_cell_obj = GRUCell([hidden_dim],
                                num_units,
                                random_state=random_state,
                                name=name + "_forward_rnn",
                                init=init,
                                scale=scale,
                                device=device)

        self.in_proj_obj = in_proj_obj
        self.fwd_cell_obj = fwd_cell_obj
        self.num_units = num_units

    def forward(self, list_of_inputs,
                previous_forward_hidden=None, previous_forward_cell=None,
                input_mask=None,
                cell_dropout=None,
                strict=None):

        num_units = self.num_units

        in_proj = self.in_proj_obj(list_of_inputs)
        if input_mask is None:
            input_mask = 0. * in_proj[..., 0] + 1.

        if previous_forward_hidden == None:
            h1_f_init = 0. * in_proj[0, :, :num_units].detach()
        else:
            h1_f_init = previous_forward_hidden
        if previous_forward_cell == None:
            c1_f_init = 0. * in_proj[0, :, :num_units].detach()
        else:
            c1_f_init = previous_forward_cell

        # GRU doesn't use cell!
        def step(inp_t, inp_mask_t,
                 h1_f_tm1):
            output, s = self.fwd_cell_obj([inp_t],
                                          h1_f_tm1, None,
                                          input_mask=inp_mask_t,
                                          cell_dropout=cell_dropout)
            h1_f_t = s[0]
            #c1_f_t = s[1]
            return [h1_f_t,]

        # should this be a "proper" flip with mask on the end
        r = scan(step,
                 [in_proj, input_mask],
                 [h1_f_init])
        return r[0], r[0], None #r[1]


class BiGRULayer(torch.nn.Module):
    def __init__(self,
                 list_of_input_dims,
                 num_units,
                 output_dim=None,
                 random_state=None,
                 name=None, init=None, scale="default",
                 forget_bias=1.,
                 strict=None,
                 device="default"):
        super(BiLSTMLayer, self).__init__()
        if name is None:
            name = _get_name()
        name = name + "_bidirgru_layer"
        name_proj = name + "_proj"
        hidden_dim = 4 * num_units
        in_proj_obj = Linear(list_of_input_dims,
                             hidden_dim,
                             random_state=random_state,
                             name=name_proj,
                             init=init, strict=strict,
                             device=device)

        fwd_cell_obj = GRUCell([hidden_dim],
                                num_units,
                                random_state=random_state,
                                name=name + "forward_rnn",
                                init=init,
                                device=device)

        rev_cell_obj = GRUCell([hidden_dim],
                                 num_units,
                                 random_state=random_state,
                                 name=name + "reverse_rnn",
                                 init=init,
                                 device=device)

        self.in_proj_obj = in_proj_obj
        self.fwd_cell_obj = fwd_cell_obj
        self.rev_cell_obj = rev_cell_obj
        self.num_units = num_units

    def forward(self, list_of_inputs,
                previous_forward_hidden=None, previous_forward_cell=None,
                previous_reverse_hidden=None, previous_reverse_cell=None,
                input_mask=None,
                cell_dropout=None,
                strict=None):

        num_units = self.num_units
        if input_mask is None:
            raise ValueError("No input mask currently unsupported")

        in_proj = self.in_proj_obj(list_of_inputs)

        if previous_forward_hidden == None:
            h1_f_init = 0. * in_proj[0, :, :num_units].detach()
        else:
            h1_f_init = previous_forward_hidden
        if previous_reverse_hidden == None:
            h1_b_init = 0. * in_proj[0, :, :num_units].detach()
        else:
            h1_b_init = previous_reverse_hidden
        if previous_forward_cell == None:
            c1_f_init = 0. * in_proj[0, :, :num_units].detach()
        else:
            c1_f_init = previous_forward_cell

        if previous_reverse_cell == None:
            c1_b_init = 0. * in_proj[0, :, :num_units].detach()
        else:
            c1_b_init = previous_reverse_cell

        # GRU doesn't use cell
        def step(inp_t, inp_mask_t,
                 rev_inp_t, rev_inp_mask_t,
                 h1_f_tm1, h1_b_tm1):
            output, s = self.fwd_cell_obj([inp_t],
                                          h1_f_tm1, None,
                                          input_mask=inp_mask_t,
                                          cell_dropout=cell_dropout)
            h1_f_t = s[0]

            output, s = self.rev_cell_obj([rev_inp_t],
                                          h1_b_tm1, None,
                                          input_mask=rev_inp_mask_t,
                                          cell_dropout=cell_dropout)
            h1_b_t = s[0]
            return h1_f_t, h1_b_t

        # should this be a "proper" flip with mask on the end
        r = scan(step,
                 [in_proj, input_mask, torch.flip(in_proj, (0,)), torch.flip(input_mask, (0,))],
                 [h1_f_init, h1_b_init])
        return torch.cat([r[0], torch.flip(r[1], (0,))], dim=-1)


class GaussianAttentionCell(torch.nn.Module):
    def __init__(self, list_of_step_input_dims,
                 full_conditioning_tensor_dim,
                 num_units,
                 att_dim=10,
                 attention_scale=1.,
                 step_op="exp",
                 cell_type="lstm",
                 name=None,
                 random_state=None,
                 strict=None, init=None):
        super(GaussianAttentionCell, self).__init__()
        #returns w_t, k_t, phi_t, state
        # where state is the state tuple returned by the inner cell_type

        if name is None:
            name = _get_name()

        name = name + "_gaussian_attention"

        #check = any([len(_shape(si)) != 2 for si in list_of_step_inputs])
        #if check:
        #    raise ValueError("Unable to support step_input with n_dims != 2")

        if init is None or init == "truncated_normal":
            rnn_init = "truncated_normal"
            forward_init = "truncated_normal"
        else:
            raise ValueError("init != None not supported")

        random_state = np.random.RandomState(1442)
        if cell_type == "gru":
            raise ValueError("NYI")
        elif cell_type == "lstm":
            self.attn_rnn_cell = LSTMCell(list_of_step_input_dims + [full_conditioning_tensor_dim],
                                          num_units,
                                          random_state=random_state,
                                          name=name + "_gauss_att_lstm",
                                          init=rnn_init)
        else:
            raise ValueError("Unsupported cell_type %s" % cell_type)

        random_state = np.random.RandomState(1442)
        self.ret_obj = Linear(
            list_of_input_dims=[num_units],
            output_dim=3 * att_dim, name=name + "_group",
            random_state=random_state,
            strict=strict, init=forward_init)
        self.att_dim = att_dim
        self.full_conditioning_tensor_dim = full_conditioning_tensor_dim
        self.step_op = step_op
        self.attention_scale = attention_scale

    def forward(self,
                list_of_step_inputs,
                previous_state_list,
                previous_attention_position,
                full_conditioning_tensor,
                previous_attention_weight,
                input_mask=None,
                conditioning_mask=None,
                cell_dropout=None):

        att_dim = self.att_dim
        full_conditioning_tensor_dim = self.full_conditioning_tensor_dim
        step_op = self.step_op
        attention_scale = self.attention_scale

        attn_rnn_out, state = self.attn_rnn_cell(list_of_step_inputs + [previous_attention_weight],
                                                 previous_state_list[0],
                                                 previous_state_list[1],
                                                 input_mask=input_mask,
                                                 cell_dropout=cell_dropout)

        ret = self.ret_obj([attn_rnn_out])
        a_t = ret[:, :att_dim]
        b_t = ret[:, att_dim:2 * att_dim]
        k_t = ret[:, 2 * att_dim:]

        k_tm1 = previous_attention_position
        cond_dim = full_conditioning_tensor_dim
        ctx = full_conditioning_tensor
        ctx_mask = conditioning_mask

        """
        ctx = Linear(
            list_of_inputs=[full_conditioning_tensor],
            list_of_input_dims=[full_conditioning_tensor_dim],
            output_dim=next_proj_dim, name=name + "_proj_ctx",
            weight_norm=weight_norm,
            random_state=random_state,
            strict=strict, init=ctx_forward_init)
        """
        if step_op == "exp":
            a_t = torch.exp(a_t)
            b_t = torch.exp(b_t)
            step_size = attention_scale * torch.exp(k_t)
            k_t = k_tm1 + step_size
        elif step_op == "softplus":
            a_t = torch.exp(a_t)
            b_t = torch.exp(b_t)
            step_size = attention_scale * torch.nn.functional.softplus(k_t)
            k_t = k_tm1 + step_size
        elif step_op == "relu":
            a_t = torch.exp(a_t)
            b_t = torch.exp(b_t)
            step_size = attention_scale * relu(k_t)
            k_t = k_tm1 + step_size
        else:
            raise ValueError("{} not a known step_op".format(step_op))
        u = torch.arange(0, full_conditioning_tensor.shape[0], dtype=torch.float32)
        u = u[None, None]

        def calc_phi(lk_t, la_t, lb_t, lu):
            phi = torch.exp(-torch.pow(lk_t[..., None] - lu, 2) * lb_t[..., None]) * la_t[..., None]
            phi = torch.sum(phi, dim=1)[:, None]
            return phi

        phi_t = calc_phi(k_t, a_t, b_t, u)
        if conditioning_mask is not None:
            w_t_pre = phi_t * ctx.permute(1, 2, 0)
            w_t_masked = w_t_pre * ctx_mask.permute(1, 0)[:, None]
            w_t = torch.sum(w_t_masked, dim=-1)[:, None]
        else:
            raise ValueError("Non-masked conditional context NYI")
            w_t = tf.matmul(phi_t, tf.transpose(ctx, (1, 0, 2)))
        phi_t = phi_t[:, 0]
        w_t = w_t[:, 0]
        return w_t, k_t, phi_t, state


class BernoulliCrossEntropyFromLogits(torch.nn.Module):
    """
    Multinomial negative log likelihood of sigmoid logits predicted compared to
    binary (0. or 1.) true_values

    Arguments to forward
    prediction : tensor, shape 2D or 3D
        The predicted class probabilities out of some layer,
        normally the output of softmax_layer

    targets : tensor, shape 2D or 3D
        One hot ground truth values. Must be the same shape as
        predicted_values. One hot representations can be achieved using
        dagbldr.utils.convert_to_one_hot
    eps : float, default 0
        Epsilon to be added during log calculation to avoid NaN values.

    Returns
    -------
    categorical_crossentropy : tensor, shape predicted_values.shape[1:]
        The cost per sample, or per sample per step if 3D
    """
    def __init__(self):
        super(BernoulliCrossEntropyFromLogits, self).__init__()

    def forward(self, prediction, target, eps=0.):
        if target.size(-1) != 1:
            raise ValueError("Last dimension of target must be 1")

        if len(prediction.size()) != len(target.size()):
            raise ValueError("prediction and target must have the same number of dimensions! Got dimensions {} and {}".format(prediction.shape, target.shape))
        if len(prediction.size()) not in [2, 3]:
            raise ValueError("BernoulliCrossEntropy only supports 2D or 3D inputs, got prediction size {}".format(prediction.size()))

        if len(target.size()) not in [2, 3]:
            raise ValueError("BernoulliCrossEntropy only supports 2D or 3D inputs, got target size {}".format(target.size()))

        shp = prediction.size()
        if len(shp) == 3:
            raise ValueError("NYI BernoulliCrossEntropy 3D inputs!")
            # seq_length, batch, 1 -> seq_length * batch
            target_t = target.permute(2, 1, 0).reshape((shp[0] * shp[1],))
            # seq_length, batch, classes -> seq_length * batch, classes
            prediction_t = prediction.permute(2, 1, 0).reshape((shp[2], shp[1] * shp[0],)).transpose(1, 0)
            prediction_c = torch.gather(prediction_t, 1, target_t.long()[..., None])
            per_step_batch_gathered = -torch.log(prediction_c.reshape((shp[1], shp[0])).transpose(1, 0))
            return per_step_batch_gathered
        else:
            # https://github.com/pytorch/pytorch/pull/1792/commits/45d47cc9ad2b02bf71eeb4bc16457dbda0d70f35
            neg_abs = -prediction.abs()
            loss = prediction.clamp(min=0) - prediction * target + (1 + neg_abs.exp()).log()
            return loss


class CategoricalCrossEntropyFromSoftmax(torch.nn.Module):
    """
    Multinomial negative log likelihood of softmax predicted compared to one hot
    true_values

    Arguments to forward
    prediction : tensor, shape 2D or 3D
        The predicted class probabilities out of some layer,
        normally the output of softmax_layer

    targets : tensor, shape 2D or 3D
        One hot ground truth values. Must be the same shape as
        predicted_values. One hot representations can be achieved using
        dagbldr.utils.convert_to_one_hot
    eps : float, default 0
        Epsilon to be added during log calculation to avoid NaN values.

    Returns
    -------
    categorical_crossentropy : tensor, shape predicted_values.shape[1:]
        The cost per sample, or per sample per step if 3D
    """
    def __init__(self):
        super(CategoricalCrossEntropyFromSoftmax, self).__init__()

    def forward(self, prediction, target, eps=0.):
        if target.size(-1) != 1:
            raise ValueError("Last dimension of target must be 1")

        if len(prediction.size()) != len(target.size()):
            raise ValueError("prediction and target must have the same number of dimensions! Got dimensions {} and {}".format(prediction.shape, target.shape))
        if len(prediction.size()) not in [2, 3]:
            raise ValueError("CategoricalCrossEntropy only supports 2D or 3D inputs, got prediction size {}".format(prediction.size()))

        if len(target.size()) not in [2, 3]:
            raise ValueError("CategoricalCrossEntropy only supports 2D or 3D inputs, got target size {}".format(target.size()))

        shp = prediction.size()
        if len(shp) == 3:
            # seq_length, batch, 1 -> seq_length * batch
            target_t = target.permute(2, 1, 0).reshape((shp[0] * shp[1],))
            # seq_length, batch, classes -> seq_length * batch, classes
            prediction_t = prediction.permute(2, 1, 0).reshape((shp[2], shp[1] * shp[0],)).transpose(1, 0)
            prediction_c = torch.gather(prediction_t, 1, target_t.long()[..., None])
            per_step_batch_gathered = -torch.log(prediction_c.reshape((shp[1], shp[0])).transpose(1, 0))
            return per_step_batch_gathered
        else:
            raise ValueError("NYI CategoricalCrossEntropy 2D inputs!")


class CategoricalCrossEntropyFromLogits(torch.nn.Module):
    """
    Multinomial negative log likelihood of logits compared to one hot
    true_values

    Arguments to forward
    prediction : tensor, shape 2D or 3D or 4D
        The predicted class probabilities out of some layer,
        normally the output of softmax_layer
        Last dimension shold always be the category size! So N H W C in terms of image ordering

    targets : tensor, shape 1D or 2D
        One hot ground truth values. Must be same dimension as predicted values, but last axis should be size 1
    eps : float, default 0
        Epsilon to be added during log calculation to avoid NaN values.

    Returns
    -------
    categorical_crossentropy : tensor, shape predicted_values.shape[1:]
        The cost per sample, or per sample per step if 3D, or per sample per dimension if 4D
    """
    def __init__(self):
        super(CategoricalCrossEntropyFromLogits, self).__init__()

    def forward(self, prediction, target, threshold=1.):
        if target.size(-1) != 1:
            raise ValueError("Last dimension of target must be 1")

        if len(prediction.size()) != len(target.size()):
            raise ValueError("prediction and target must have the same number of dimensions! Got dimensions {} and {}".format(prediction.shape, target.shape))
        if len(prediction.size()) not in [2, 3, 4]:
            raise ValueError("CategoricalCrossEntropy only supports 2D or 3D or 4D inputs, got prediction size {}".format(prediction.size()))

        if len(target.size()) not in [2, 3, 4]:
            raise ValueError("CategoricalCrossEntropy only supports 2D or 3D or 4D inputs, got target size {}".format(target.size()))

        shp = prediction.size()
        was_4d = False
        if len(shp) == 4:
            pred_h = prediction.shape[1]
            pred_w = prediction.shape[2]
            prediction = prediction.reshape(prediction.shape[0], prediction.shape[1] * prediction.shape[2], prediction.shape[3])
            target = target.reshape(target.shape[0], target.shape[1] * target.shape[2], target.shape[3])
            was_4d = True

        shp = prediction.size()
        if len(shp) == 3:
            # seq_length, batch, 1 -> seq_length * batch
            target_t = target.permute(2, 1, 0).reshape((shp[0] * shp[1],))
            # seq_length, batch, classes -> seq_length * batch, classes
            bot = logsumexp(prediction, dim=-1)
            if bot.shape[-1] != 1:
                raise ValueError("logsumexp calculation returned non singleton last axis! prediction {}, bot {}".format(prediction.shape, bot.shape))
            # calculate logsoftmax internally
            log_softmax_vals = prediction - bot
            prediction_t = log_softmax_vals.permute(2, 1, 0).reshape((shp[2], shp[1] * shp[0],)).transpose(1, 0)
            prediction_c = torch.gather(prediction_t, 1, target_t.long()[..., None])
            p = prediction_c.reshape((shp[1], shp[0])).transpose(1, 0)
            # https://medium.com/@zhang_yang/understanding-cross-entropy-implementation-in-pytorch-softmax-log-softmax-nll-cross-entropy-416a2b200e34
            # https://github.com/AliAbbasi/Numerically-Stable-Cross-Entropy-Loss-Function-Tensorflow/blob/master/Numerically-Stable-Cross-Entropy-MultiLabel.py
            large_approx = -p

            softmax_vals = softmax(prediction)
            prediction_t2 = softmax_vals.permute(2, 1, 0).reshape((shp[2], shp[1] * shp[0],)).transpose(1, 0)
            prediction_c2 = torch.gather(prediction_t2, 1, target_t.long()[..., None])
            p2 = prediction_c2.reshape((shp[1], shp[0])).transpose(1, 0)
            small_approx = -torch.log(p2)

            buff = 0. * prediction[..., 0]
            buff[large_approx > threshold] = large_approx[large_approx > threshold]
            buff[large_approx <= threshold] = small_approx[large_approx <= threshold]
            if was_4d:
                buff = buff.reshape((buff.shape[0], pred_h, pred_w, -1))
            return buff
        else:
            raise ValueError("NYI CategoricalCrossEntropy 2D inputs!")


class DiscretizedMixtureOfLogisticsCrossEntropyFromLogits(torch.nn.Module):
    """
    Discretized mixture of logistics negative log likelihood of logits compared to scaled -1, 1 targets 
    true_values
     
    both in NHWC format, with C being (2 * n_mix + n_mix) * n_channels, where n_channels is the number of output maps (3 for RGB images for example)

    Arguments to forward
    prediction : tensor, shape 4D
        The predicted class probabilities out of some layer,

    targets : tensor, shape 4D
        Must be same dimension as predicted values, but last axis should be size 1

    Returns
    -------
   crossentropy : tensor, shape predicted_values.shape
        The cost per sample per dimension if 4D
    """
    def __init__(self):
        super(DiscretizedMixtureOfLogisticsCrossEntropyFromLogits, self).__init__()

    def forward(self, prediction, target, n_mix=10, n_bins=255.):
        if target.size(-1) != 1:
            raise ValueError("Last dimension of target must be 1")

        if len(prediction.size()) != len(target.size()):
            raise ValueError("prediction and target must have the same number of dimensions! Got dimensions {} and {}".format(prediction.shape, target.shape))
        if len(prediction.size()) not in [4]:
            raise ValueError("DiscretizedMixtureOfLogisticsCrossEntropy only supports 4D inputs, got prediction size {}".format(prediction.size()))

        if len(target.size()) not in [4]:
            raise ValueError("DiscretizedMixtureOfLogisticsCrossEntropy only supports 4D inputs, got target size {}".format(target.size()))

        #def discretized_mix_logistic_loss(prediction, target, nr_mix=10, reduction='mean'):
        """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval
        Args:
            prediction: model prediction. channels of model prediction should be mean
                        and scale for each channel and weighting bt components --> (2*nr_mix+nr_mix)*num_channels
            target: min/max should be -1 and 1
        **** code for this function from https://github.com/pclucas14/pixel-cnn-pp/blob/master/utils.py
        """
        chan = prediction.shape[-1]
        nr_mix = n_mix
        #assert (prediction.max()<=1 and prediction.min()>=-1)
        assert (target.max()<=1 and target.min()>=-1)
        device = target.device
        l = prediction
        x = target

        # Pytorch ordering
        # N C H W TO N H W C
        #x = x.permute(0, 2, 3, 1)
        #l = l.permute(0, 2, 3, 1)

        xs = [int(y) for y in x.size()]
        #ls = [int(y) for y in l.size()]

        # here and below: unpacking the params of the mixture of logistics
        #nr_mix = int(ls[-1] / 10)
        # l is prediction
        logit_probs = l[:, :, :, :nr_mix]

        l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix*2]) # 3--changed to 1 for mean, scale, coef
        means = l[:, :, :, :, :nr_mix]
        log_scales = torch.clamp(l[:, :, :, :, nr_mix:2 * nr_mix], min=-7.)

        # here and below: getting the means and adjusting them based on preceding
        # sub-pixels
        x = x.contiguous()
        #x = x.unsqueeze(-1) + torch.Variable(torch.zeros(xs + [nr_mix]).to(device), requires_grad=False)
        x = x.unsqueeze(-1) + torch.zeros(xs + [nr_mix], requires_grad=False).to(device)

        centered_x = x - means
        inv_stdv = torch.exp(-log_scales)
        plus_in = inv_stdv * (centered_x + 1. / float(n_bins))
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inv_stdv * (centered_x - 1. / float(n_bins))
        cdf_min = torch.sigmoid(min_in)
        # log probability for edge case of 0 (before scaling)
        log_cdf_plus = plus_in - F.softplus(plus_in)
        # log probability for edge case of 255 (before scaling)
        log_one_minus_cdf_min = -F.softplus(min_in)
        cdf_delta = cdf_plus - cdf_min  # probability for all other cases
        mid_in = inv_stdv * centered_x
        # log probability in the center of the bin, to be used in extreme cases
        # (not actually used in our code)
        log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)

        # now select the right output: left edge case, right edge case, normal
        # case, extremely low prob case (doesn't actually happen for us)

        # this is what we are really doing, but using the robust version below for extreme cases in other applications and to avoid NaN issue with tf.select()
        # log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999, log_one_minus_cdf_min, tf.log(cdf_delta)))

        # robust version, that still works if probabilities are below 1e-5 (which never happens in our code)
        # tensorflow backpropagates through tf.select() by multiplying with zero instead of selecting: this requires use to use some ugly tricks to avoid potential NaNs
        # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as output, it's purely there to get around the tf.select() gradient issue
        # if the probability on a sub-pixel is below 1e-5, we use an approximation
        # based on the assumption that the log-density is constant in the bin of
        # the observed sub-pixel value

        inner_inner_cond = (cdf_delta > 1e-5).float()
        inner_inner_out  = inner_inner_cond * torch.log(torch.clamp(cdf_delta, min=1e-12)) + (1. - inner_inner_cond) * (log_pdf_mid - np.log(float(n_bins) / 2))
        inner_cond       = (x > 0.999).float()
        inner_out        = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
        cond             = (x < -0.999).float()
        log_probs        = cond * log_cdf_plus + (1. - cond) * inner_out
        log_probs        = torch.sum(log_probs, dim=3) + self.log_prob_from_logits(logit_probs)
        lse = self.log_sum_exp(log_probs)
        return -lse


class MixtureOfGaussiansNegativeLogLikelihood(torch.nn.Module):
    """
    Discretized mixture of logistics negative log likelihood of logits compared to scaled -1, 1 targets 
    true_values
     
    both in NHWC format, with C being (2 * n_mix + n_mix) * n_channels, where n_channels is the number of output maps (3 for RGB images for example)

    Arguments to forward
    prediction : tensor, shape 4D
        The predicted class probabilities out of some layer,

    targets : tensor, shape 4D
        Must be same dimension as predicted values, but last axis should be size 1

    Returns
    -------
   crossentropy : tensor, shape predicted_values.shape
        The cost per sample per dimension if 4D
    """
    def __init__(self):
        super(MixtureOfGaussiansNegativeLogLikelihood, self).__init__()

    def log_sum_exp(self, x):
        """ numerically stable log_sum_exp implementation that prevents overflow
        **** code for this function from https://github.com/pclucas14/pixel-cnn-pp/blob/master/utils.py
        """
        # TF ordering
        axis  = len(x.size()) - 1
        m, _  = torch.max(x, dim=axis)
        m2, _ = torch.max(x, dim=axis, keepdim=True)
        return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))

    def log_prob_from_logits(self, x):
        """ numerically stable log_softmax implementation that prevents overflow
        **** code for this function from https://github.com/pclucas14/pixel-cnn-pp/blob/master/utils.py
        """
        # TF ordering
        axis = len(x.size()) - 1
        m, _ = torch.max(x, dim=axis, keepdim=True)
        return x - m - torch.log(torch.sum(torch.exp(x - m), dim=axis, keepdim=True))

    def forward(self, prediction, target, n_mix=10):
        if target.size(-1) != 1:
            raise ValueError("Last dimension of target must be 1")

        if len(prediction.size()) != len(target.size()):
            raise ValueError("prediction and target must have the same number of dimensions! Got dimensions {} and {}".format(prediction.shape, target.shape))
        if len(prediction.size()) not in [4]:
            raise ValueError("MixtureOfGaussiansNegativeLogLikelihood only supports 4D inputs, got prediction size {}".format(prediction.size()))

        if len(target.size()) not in [4]:
            raise ValueError("MixtureOfGaussiansNegativeLogLikelihood only supports 4D inputs, got target size {}".format(target.size()))

        print("hjkdwakl;")
        from IPython import embed; embed(); raise ValueError()
        #def discretized_mix_logistic_loss(prediction, target, nr_mix=10, reduction='mean'):
        """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval
        Args:
            prediction: model prediction. channels of model prediction should be mean
                        and scale for each channel and weighting bt components --> (2*nr_mix+nr_mix)*num_channels
            target: min/max should be -1 and 1
        **** code for this function from https://github.com/pclucas14/pixel-cnn-pp/blob/master/utils.py
        """
        chan = prediction.shape[-1]
        nr_mix = n_mix
        #assert (prediction.max()<=1 and prediction.min()>=-1)
        assert (target.max()<=1 and target.min()>=-1)
        device = target.device
        l = prediction
        x = target

        # Pytorch ordering
        # N C H W TO N H W C
        #x = x.permute(0, 2, 3, 1)
        #l = l.permute(0, 2, 3, 1)

        xs = [int(y) for y in x.size()]
        #ls = [int(y) for y in l.size()]

        # here and below: unpacking the params of the mixture of logistics
        #nr_mix = int(ls[-1] / 10)
        # l is prediction
        logit_probs = l[:, :, :, :nr_mix]

        l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix*2]) # 3--changed to 1 for mean, scale, coef
        means = l[:, :, :, :, :nr_mix]
        log_scales = torch.clamp(l[:, :, :, :, nr_mix:2 * nr_mix], min=-7.)

        # here and below: getting the means and adjusting them based on preceding
        # sub-pixels
        x = x.contiguous()
        #x = x.unsqueeze(-1) + torch.Variable(torch.zeros(xs + [nr_mix]).to(device), requires_grad=False)
        x = x.unsqueeze(-1) + torch.zeros(xs + [nr_mix], requires_grad=False).to(device)

        centered_x = x - means
        inv_stdv = torch.exp(-log_scales)
        plus_in = inv_stdv * (centered_x + 1. / float(n_bins))
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inv_stdv * (centered_x - 1. / float(n_bins))
        cdf_min = torch.sigmoid(min_in)
        # log probability for edge case of 0 (before scaling)
        log_cdf_plus = plus_in - F.softplus(plus_in)
        # log probability for edge case of 255 (before scaling)
        log_one_minus_cdf_min = -F.softplus(min_in)
        cdf_delta = cdf_plus - cdf_min  # probability for all other cases
        mid_in = inv_stdv * centered_x
        # log probability in the center of the bin, to be used in extreme cases
        # (not actually used in our code)
        log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)

        # now select the right output: left edge case, right edge case, normal
        # case, extremely low prob case (doesn't actually happen for us)

        # this is what we are really doing, but using the robust version below for extreme cases in other applications and to avoid NaN issue with tf.select()
        # log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999, log_one_minus_cdf_min, tf.log(cdf_delta)))

        # robust version, that still works if probabilities are below 1e-5 (which never happens in our code)
        # tensorflow backpropagates through tf.select() by multiplying with zero instead of selecting: this requires use to use some ugly tricks to avoid potential NaNs
        # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as output, it's purely there to get around the tf.select() gradient issue
        # if the probability on a sub-pixel is below 1e-5, we use an approximation
        # based on the assumption that the log-density is constant in the bin of
        # the observed sub-pixel value

        inner_inner_cond = (cdf_delta > 1e-5).float()
        inner_inner_out  = inner_inner_cond * torch.log(torch.clamp(cdf_delta, min=1e-12)) + (1. - inner_inner_cond) * (log_pdf_mid - np.log(float(n_bins) / 2))
        inner_cond       = (x > 0.999).float()
        inner_out        = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
        cond             = (x < -0.999).float()
        log_probs        = cond * log_cdf_plus + (1. - cond) * inner_out
        log_probs        = torch.sum(log_probs, dim=3) + self.log_prob_from_logits(logit_probs)
        lse = self.log_sum_exp(log_probs)
        return -lse

class NoamOpt(object):
    """
    def get_std_opt(model):
        return NoamOpt(192, 1, 4000,
                torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    """
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()


class RampOpt(object):
    """
    Similar to NoamOpt but with specified "max" learning rate rather than derived from model size

    Factor specifies whether ramp linearly or to the X power
    Warmup describest the number of steps to ramp up learning rate

    Decay to 0 by default, using cosine decay
        terminates at decay_to_zero_at_steps

    RampOpt(target_learning_rate, ramp_power, steps, opt)
    return RampOpt(.0001, 1, 4000, 4000 * 100,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    can set decay_to_zero_at_steps to -1 to disable decay
    """
    def __init__(self, target_learning_rate, factor, warmup, decay_to_zero_at_steps, optimizer, min_decay_learning_rate=None):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.target_learning_rate = target_learning_rate
        self.decay_to_zero_at_steps = decay_to_zero_at_steps
        self.min_decay_learning_rate = min_decay_learning_rate
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        if step <= self.warmup:
            return self.target_learning_rate * ((step / float(self.warmup)) ** self.factor)

        if self.decay_to_zero_at_steps == -1:
            return self.target_learning_rate

        new_rate = self.target_learning_rate * np.cos((float(step - self.warmup) / (self.decay_to_zero_at_steps - self.warmup)) * (np.pi / 2.))

        if self.min_decay_learning_rate is not None:
            if new_rate < self.min_decay_learning_rate:
                new_rate = self.min_decay_learning_rate

        if step > self.decay_to_zero_at_steps:
            if self.min_decay_learning_rate is None:
                logger.info("WARNING: RampOpt optimizer has decayed to LR 0! Current step {}, so no more learning happening!".format(step))
                new_rate = 0.
        # warmup is 0 on cos curve
        # infinity is pi/2?
        return new_rate

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

class Dropout(nn.Module):
    def __init__(self, dropout_keep_prob=1.,
                 name=None,
                 random_state=None,
                 strict=None,
                 dtype="default",
                 device="default"):
        super(Dropout, self).__init__()
        self.dropout = 1. - dropout_keep_prob
        if random_state is None:
            raise ValueError("Must pass random_state to LockedDropout")
        if device == "default":
            device = get_device_default()
        self.g = torch.Generator(device=device)
        self.g.manual_seed(random_state.randint(100000))

    def forward(self, x):
        if not self.training or self.dropout == 0.:
            return x
        pm = x.data.new(*x.size()).zero_()
        pm = 0. * pm + (1. - self.dropout)
        m = torch.bernoulli(pm, generator=self.g)
        mask = Variable(m, requires_grad=False) / (1. - self.dropout)
        mask = mask.expand_as(x)
        return mask * x


class LockedDropout(nn.Module):
    def __init__(self, dropout_keep_prob=1.,
                 name=None,
                 random_state=None,
                 strict=None,
                 dtype="default",
                 device="default"):
        super(LockedDropout, self).__init__()
        self.dropout = 1. - dropout_keep_prob
        if random_state is None:
            raise ValueError("Must pass random_state to LockedDropout")
        if device == "default":
            device = get_device_default()
        self.g = torch.Generator(device=device)
        self.g.manual_seed(random_state.randint(100000))

    def forward(self, x):
        if not self.training or self.dropout == 0.:
            return x
        pm = x.data.new(1, *x.size()[1:]).zero_()
        pm = 0. * pm + (1. - self.dropout)
        m = torch.bernoulli(pm, generator=self.g)
        mask = Variable(m, requires_grad=False) / (1. - self.dropout)
        mask = mask.expand_as(x)
        return mask * x


class PositionalEmbedding(nn.Module):
    # credit to transformer xl
    def __init__(self, embedding_dimension,
                 name=None,
                 strict=None,
                 dtype="default",
                 device="default"):
        super(PositionalEmbedding, self).__init__()

        self.embedding_dimension = embedding_dimension
        d = embedding_dimension

        inv_freq = 1. / (10000. ** (torch.arange(0.0, self.embedding_dimension, 2.0) / float(self.embedding_dimension)))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, position_sequence, batch_size=None):
        sinusoid_inp = torch.ger(position_sequence.float(), self.inv_freq)
        pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)

        if batch_size is not None:
            return pos_emb[:, None, :].expand(-1, batch_size, -1)
        else:
            return pos_emb[:, None, :]


class EventPositionalEmbedding(nn.Module):
    # credit to transformer xl
    def __init__(self, embedding_dimension,
                 name=None,
                 strict=None,
                 dtype="default",
                 device="default"):
        super(EventPositionalEmbedding, self).__init__()

        self.embedding_dimension = embedding_dimension
        d = embedding_dimension

        inv_freq = 1. / (10000. ** (torch.arange(0.0, self.embedding_dimension, 2.0) / float(self.embedding_dimension)))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, event_position_sequence, batch_size=None):
        assert event_position_sequence.shape[1] == batch_size
        all_pos_emb = []
        for i in range(batch_size):
            torch.ger(event_position_sequence[:, 0].float(), self.inv_freq)
            sinusoid_inp_i = torch.ger(event_position_sequence[:, i].float(), self.inv_freq)
            pos_emb_i = torch.cat([torch.sin(sinusoid_inp_i), torch.cos(sinusoid_inp_i)], dim=-1)
            all_pos_emb.append(pos_emb_i[:, None, :])

        pos_emb = torch.cat(all_pos_emb, dim=1)
        assert pos_emb.shape[1] == batch_size
        assert len(pos_emb.shape) == 3
        return pos_emb


class LayerNorm(torch.nn.Module):
    """
    https://nlp.seas.harvard.edu/2018/04/03/attention.html
    """
    def __init__(self,
                 input_dim,
                 eps=1E-6,
                 name=None,
                 strict=None,
                 dtype="default",
                 device="default"):
        super(LayerNorm, self).__init__()
        if name is None:
            name = _get_name()

        self.input_dim = input_dim
        self.eps = eps

        name_w = name + "_layer_norm_w"
        name_b = name + "_layer_norm_b"

        if strict is None:
            strict = get_strict_mode_default()

        if strict:
            cur_defs = get_params_dict()
            if name_w in cur_defs:
                raise ValueError("Name {} already created in params dict!".format(name_w))

            if name_b in cur_defs:
                raise ValueError("Name {} already created in params dict!".format(name_b))
        try:
            weight = _get_shared(name_w)
        except NameError:
            weight_values = np.ones((input_dim,)).astype(np.float32)
            bias_values = np.zeros((input_dim,)).astype(np.float32)
            weight = make_tensor(weight_values, dtype=dtype, device=device)
            bias = make_tensor(bias_values, dtype=dtype, device=device)
            _set_shared(name_w, weight)
            _set_shared(name_b, bias)

        self.weight = torch.nn.Parameter(weight)
        self.bias = torch.nn.Parameter(bias)

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        # eps trick wont work here - std has issues if every element is 0 on that axis
        # std = x.std(-1, keepdim=True)
        # want to use sqrt of var + eps instead
        var = x.var(-1, keepdim=True)
        return self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias
        """
        rms_x = x.norm(2, dim=-1, keepdim=True)
        x_normed = x / (rms_x + self.eps)
        return self.weight * x_normed
        """


class PositionwiseFeedforward(nn.Module):
    def __init__(self, list_of_input_dims,
                 projection_dim,
                 dropout_keep_prob=1.0,
                 name=None,
                 random_state=None,
                 strict=None, init=None,
                 scale="default",
                 device="default",
                 dtype="default"):
        super(PositionwiseFeedforward, self).__init__()

        if name is None:
            name = _get_name()

        if random_state is None:
            raise ValueError("Must pass random_state to PositionwiseFeedforward")

        name_i = name + "_positionwise_input_projection"
        name_o = name + "_positionwise_ouput_projection"
        o_dim = sum(list_of_input_dims)
        self.i = Linear(list_of_input_dims,
                        projection_dim,
                        biases=True,
                        random_state=random_state,
                        name=name_i,
                        strict=strict,
                        init=init,
                        scale=scale,
                        device=device,
                        dtype=dtype)

        self.o = Linear([projection_dim],
                         o_dim,
                         biases=True,
                         random_state=random_state,
                         name=name_o,
                         strict=strict,
                         init=init,
                         scale=scale,
                         device=device,
                         dtype=dtype)

        self.ln = LayerNorm(o_dim,
                            name=name + "_ln",
                            device=device,
                            dtype=dtype)

        self.ld1 = LockedDropout(dropout_keep_prob=dropout_keep_prob,
                                 device=device,
                                 random_state=random_state)
        self.ld2 = LockedDropout(dropout_keep_prob=dropout_keep_prob,
                                 device=device,
                                 random_state=random_state)

    def forward(self, list_of_inputs):
        inp = torch.cat(list_of_inputs, dim=-1)
        s1 = relu(self.i([inp]))
        ds1 = self.ld1(s1)

        s2 = self.o([ds1])
        ds2 = self.ld2(s2)
        return self.ln(ds2 + inp)


def _rel_shift(x, klen=-1):
    x_padded = x.reshape(x.size(1), x.size(0), *x.size()[2:])
    x = x_padded[1:].reshape(x.size(0), x.size(1) - 1, *x.size()[2:])
    if klen != -1:
        x = x[:, :klen, :, :]
    return x


class RelativeMultiHeadAttention(nn.Module):
     def __init__(self, list_of_input_dims,
                  n_heads=10, head_dim=38, model_dim=380,
                  attention_dropout_keep_prob=1.0,
                  name=None,
                  random_state=None,
                  strict=None, init=None,
                  scale="default",
                  device="default",
                  dtype="default"):
         super(RelativeMultiHeadAttention, self).__init__()

         if name is None:
             name = _get_name()

         if random_state is None:
             raise ValueError("Must pass random_state to RelativeMultiHeadAttention")

         self.n_heads = n_heads
         self.head_dim = head_dim
         self.model_dim = model_dim
         self.attention_dropout_keep_prob = attention_dropout_keep_prob

         qkv_name = name + "_qkv"
         o_name = name + "_out"

         self.drop = nn.Dropout(1. - attention_dropout_keep_prob)
         self.locked_drop = LockedDropout(attention_dropout_keep_prob, random_state=random_state, device=device)

         # no biases in transformer XL code
         self.qkv_net = Linear(list_of_input_dims,
                               3 * n_heads * head_dim,
                               biases=False,
                               random_state=random_state,
                               name=qkv_name,
                               strict=strict,
                               init=init,
                               scale=scale,
                               device=device,
                               dtype=dtype)

         self.o_net = Linear([head_dim * n_heads],
                             model_dim,
                             biases=False,
                             random_state=random_state,
                             name=o_name,
                             strict=strict,
                             init=init,
                             scale=scale,
                             device=device,
                             dtype=dtype)

         self.ln = LayerNorm(model_dim,
                             name=name + "_ln",
                             device=device,
                             dtype=dtype)
         self.scale = 1. / (head_dim ** .5)

     def forward(self, list_of_inputs, relative_positional_embedding, local_bias_ac, local_bias_bd, attention_mask=None, memory=None):
         i = torch.cat(list_of_inputs, dim=-1)
         r = relative_positional_embedding
         qlen = i.size(0)
         rlen = r.size(0)
         batch_size = i.size(1)
         if memory is not None:
             i_heads = self.qkv_net([torch.cat([memory, i], 0)])
         else:
             i_heads = self.qkv_net([i])

         r_heads = self.qkv_net([r])
         i_head_q, i_head_k, i_head_v = torch.chunk(i_heads, 3, dim=-1)
         r_head_q, r_head_k, r_head_v = torch.chunk(r_heads, 3, dim=-1)
         if memory is not None:
             # slice out the memory part for query
             i_head_q = i_head_q[-qlen:]
         klen = i_head_k.size(0)
         i_head_q = i_head_q.view(qlen, batch_size, self.n_heads, self.head_dim)
         # this could be much longer
         i_head_k = i_head_k.view(klen, batch_size, self.n_heads, self.head_dim)
         i_head_v = i_head_v.view(klen, batch_size, self.n_heads, self.head_dim)

         r_head_q = r_head_q.view(rlen, batch_size, self.n_heads, self.head_dim)
         r_head_k = r_head_k.view(rlen, batch_size, self.n_heads, self.head_dim)

         # attention
         # [qlen x bsz x n_head x d_head]
         ir_head_q = i_head_q + local_bias_ac #+ r_head_q[-1] # bias term from pos embed
         # [klen x bsz x n_head x d_head]
         # i_head_k
         # [qlen x klen x bsz x n_head]
         AC = torch.einsum('ibnd,jbnd->ijbn', (ir_head_q, i_head_k))

         ir2_head_q = i_head_q + local_bias_bd # bias term
         BD = torch.einsum('ibnd,jbnd->ijbn', (ir2_head_q, r_head_k))
         # rel shift effectively removes 1 along the 1st dim
         BD = _rel_shift(BD)
         attention_score = AC + BD
         # [qlen x klen x bsz x n_head]
         attention_score *= self.scale

         #need to find and do something about rows with all masked
         if attention_mask is not None and attention_mask.any().item():
             # fill 1s with neg ing, leave 0s alone!
             if attention_mask.dim() == 2:
                 attention_score.masked_fill_(attention_mask[None, :, :, None], -float('inf'))
             # can define either 2d or 3d mask here, but will need to be very careful about what is 0 and what is 1
             elif attention_mask.dim() == 3:
                 row_mask = (attention_mask.sum(axis=1) == attention_mask.shape[1])
                 attention_score.masked_fill_(attention_mask[:, :, :, None], -float('inf'))
                 if row_mask.sum() > 0:
                     #logger.info("WARNING: Found 3D mask containing rows with all masked values! Filling blank rows with -1E9")
                     # do it in two steps, fill all with -inf then the blank ones with -1E9
                     attention_score.masked_fill_(row_mask[:, None, :, None], -1E9)
             else:
                 raise ValueError("Attention_mask dim not handled in relative multihead attention!")

         faker = 0. * attention_score + 1.
         faker_mask = self.drop(faker)
         # find rows with all -inf
         # can't fill with neginf since could all be 0 - hence all neginf, leading to nan in softmax
         attention_score.masked_fill_(faker_mask == 0, -1E9)
         attention_prob = F.softmax(attention_score, dim=1)

         # technically it's not a prob distribution if I force a completely blanked out row to be all 0

         # as for dropout, this is how it is done in the PTB code but... not normalized anymore!
         # question is, will other method freak out at test time? whereas this is more "normal" - basically same as dropping pieces of i_head_v
         #attention_prob = self.drop(attention_prob)
         # isn't this just dropout on the thing you are attending, in disguise?
         # https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py#L1822

         attention_weighted_values = torch.einsum('ijbn,jbnd->ibnd', (attention_prob, i_head_v))

         # [qlen x bsz x n_head x d_head]
         attention_weighted_values = attention_weighted_values.contiguous().view(
                attention_weighted_values.size(0),
                attention_weighted_values.size(1),
                self.n_heads * self.head_dim)

         o = self.o_net([attention_weighted_values])
         o = self.locked_drop(o)

         output = self.ln(i + o)
         return output


class RelativeDecoderLayer(nn.Module):
    def __init__(self, list_of_input_dims,
                 n_heads=10,
                 head_dim=38,
                 model_dim=380,
                 inner_dim=900,
                 attention_dropout_keep_prob=0.8,
                 inner_dropout_keep_prob=0.8,
                 name=None,
                 random_state=None,
                 strict=None, init=None,
                 scale="default",
                 device="default",
                 dtype="default"):
        super(RelativeDecoderLayer, self).__init__()

        if name is None:
            name = _get_name()

        if random_state is None:
            raise ValueError("Must pass random_state to AWDTransformerXLDecoder")

        attention_name = name + "_multihead_attention"
        feedforward_name = name + "_positionwise_ff"
        self.attention = RelativeMultiHeadAttention(list_of_input_dims,
                                                    n_heads=n_heads,
                                                    head_dim=head_dim,
                                                    model_dim=model_dim,
                                                    attention_dropout_keep_prob=attention_dropout_keep_prob,
                                                    name=attention_name,
                                                    random_state=random_state,
                                                    strict=strict,
                                                    init=init,
                                                    scale=scale,
                                                    device=device,
                                                    dtype=dtype)

        self.position_ff = PositionwiseFeedforward(list_of_input_dims,
                                                   inner_dim,
                                                   dropout_keep_prob=inner_dropout_keep_prob,
                                                   name=feedforward_name,
                                                   random_state=random_state,
                                                   strict=strict,
                                                   init=init,
                                                   scale=scale,
                                                   device=device,
                                                   dtype=dtype)

    def forward(self, decoder_input, relative_positional_embedding, local_bias_ac, local_bias_bd, decoder_attention_mask=None, memory=None):
        output = self.attention([decoder_input], relative_positional_embedding, local_bias_ac=local_bias_ac, local_bias_bd=local_bias_bd, attention_mask=decoder_attention_mask,
                                memory=memory)
        output = self.position_ff([output])
        return output


class AWDTransformerXLBaseBlock(nn.Module):
    def __init__(self,
                 list_of_input_dims,
                 remove_context=False,
                 n_layers=16, n_heads=10, head_dim=38, model_dim=380, inner_dim=900,
                 input_dropout_keep_prob=0.4,
                 attention_dropout_keep_prob=0.8,
                 inner_dropout_keep_prob=0.8,
                 hidden_dropout_keep_prob=1.0,
                 output_dropout_keep_prob=0.5,
                 event_based_positions=False,
                 name=None,
                 random_state=None,
                 memory_len=0,
                 context_len=0,
                 strict=None,
                 init=None,
                 scale="default",
                 device="default",
                 dtype="default"):
        super(AWDTransformerXLBaseBlock, self).__init__()

        self.remove_context = remove_context

        if name is None:
            name = _get_name()

        if random_state is None:
            raise ValueError("Must pass random_state to AWDTransformerXLDecoderBlock")

        self.layers = nn.ModuleList()
        input_dim = sum(list_of_input_dims)
        if input_dim != model_dim:
            raise ValueError("sum of list_of_input_dims should match model_dim due to residual architecture, if this is not the case project the data or change dims! Have {}, sum = {}, model_dim = {}".format(list_of_input_dims, input_dim, model_dim))
        if n_heads * head_dim != model_dim:
            raise ValueError("head_dim * n_heads should == model_dim, have {} * {} != {}".format(head_dim, n_heads, model_dim))
        self.n_layers = n_layers
        self.device = device
        self.dtype = dtype
        self.event_based_positions = event_based_positions
        self.local_bias_ac = nn.Parameter(torch.zeros(n_heads, head_dim))
        self.local_bias_bd = nn.Parameter(torch.zeros(n_heads, head_dim))

        for i in range(n_layers):
            layer_name = name + "_relative_decoder_layer{}".format(i)
            l = RelativeDecoderLayer(list_of_input_dims,
                                     n_heads=n_heads,
                                     head_dim=head_dim,
                                     model_dim=model_dim,
                                     inner_dim=inner_dim,
                                     attention_dropout_keep_prob=attention_dropout_keep_prob,
                                     inner_dropout_keep_prob=inner_dropout_keep_prob,
                                     name=layer_name,
                                     random_state=random_state,
                                     strict=strict,
                                     init=init,
                                     scale=scale,
                                     device=device,
                                     dtype=dtype)
            self.layers.append(l)

        self.memory_len = memory_len
        self.context_len = context_len
        if self.event_based_positions:
            # ? get rid of this?
            self.pos_emb = EventPositionalEmbedding(model_dim,
                                                    device=device,
                                                    dtype=dtype)
        else:
            self.pos_emb = PositionalEmbedding(model_dim,
                                               device=device,
                                               dtype=dtype)
        self.locked_drop_i = LockedDropout(input_dropout_keep_prob,
                                           random_state=random_state,
                                           device=device,
                                           dtype=dtype)
        self.locked_drop_h = LockedDropout(hidden_dropout_keep_prob,
                                           random_state=random_state,
                                           device=device,
                                           dtype=dtype)
        self.locked_drop_o = LockedDropout(output_dropout_keep_prob,
                                           random_state=random_state,
                                           device=device,
                                           dtype=dtype)

    def init_list_of_mems(self):
        if self.device == "default":
            device = get_device_default()
        else:
            device = self.device

        if self.dtype == "default":
            dtype = get_dtype_default()
        if dtype == "float32":
            dtype = torch.float32
        else:
            dtype = torch.float64

        # returns None if mem_len is 0 else
        if self.memory_len > 0:
            mems = []

            for i in range(self.n_layers):
                empty = torch.empty(0, dtype=dtype, device=torch.device(device))
                mems.append(empty)
            return mems
        else:
            return None

    def update_list_of_mems(self, hiddens, list_of_mems, query_len, memory_len):
        # mlen and qlen were swapped in call vs signature in original code!
        # https://github.com/kimiyoung/transformer-xl/issues/96
        # effectively, would mean memory_len= len query
        # and query_len= 0 for PTB experiment
        # where self.context_len was 70
        # self.mem_len 0
        # we swap the call to be correct, and set hyperparameters to actually use memory
        if list_of_mems is None:
            return None

        if self.memory_len == 0:
            return None

        qlen = query_len
        mlen = memory_len

        assert len(hiddens) == len(list_of_mems), "len(list_of_mems) != len(hiddens)"

        # Copied from transformer XL main code
        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen - 0 - self.context_len)
            beg_idx = max(0, end_idx - self.memory_len)
            for i in range(len(hiddens)):
                cat = torch.cat([list_of_mems[i], hiddens[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())
        return new_mems


    def forward(self, input_tensor, input_mask_tensor=None, input_event_positions=None, list_of_mems=None):
        if not list_of_mems:
            list_of_mems = self.init_list_of_mems()

        shp = input_tensor.shape

        qlen = shp[0]
        mlen = list_of_mems[0].size(0) if list_of_mems is not None else 0
        klen = mlen + qlen
        # masking works internally by setting 1s to neg inf, 0s are left alone! This is slightly different than expected
        # attention mask shows the position each tgt word (row) is allowed to look at (column).
        # 0 1 1 1 1
        # 0 0 1 1 1
        # 0 0 0 1 1
        # 0 0 0 0 1
        attn_mask = torch.triu(input_tensor.new_ones(qlen, klen), diagonal=1 + mlen).bool()[:, :, None]
        if input_mask_tensor is None:
            input_mask_tensor = (0. * input_tensor[:, :, 0]).long()
        input_mask_tensor_dtype = input_mask_tensor.dtype
        attn_mask = (attn_mask.type(input_mask_tensor_dtype) + input_mask_tensor[:, None, :] > 0).bool()

        #? skip embed?
        if self.event_based_positions:
            if input_event_positions is None:
                raise ValueError("event_based_positions enabled in init, but not passed in forward pass!")
            # reverse the input positions?, and append 0 at the end
            #pos_seq = torch.flip(input_event_positions, dims=[0])
            pos_seq = input_event_positions
            pos_seq = torch.cat([(pos_seq[-1, :] * 0.)[None, :], pos_seq], dim=0)
            # duplicate 
            pe = self.pos_emb(pos_seq, batch_size=shp[1])
        else:
            # relative positional embedding
            pos_seq = torch.arange(klen, -1, -1.0, device=input_tensor.device)
            pe = self.pos_emb(pos_seq, batch_size=shp[1])
            # one longer than expected because _rel_shift reduces size by 1

        hids = []
        core_out = self.locked_drop_i(input_tensor)
        pe = self.locked_drop_i(pe)
        for i, this_layer in enumerate(self.layers):
            hids.append(core_out)
            mems_i = list_of_mems[i] if list_of_mems is not None else None
            core_out = this_layer(core_out, pe,
                                  local_bias_ac=self.local_bias_ac[None, None],
                                  local_bias_bd=self.local_bias_bd[None, None],
                                  decoder_attention_mask=attn_mask, memory=mems_i)
            if i < len(self.layers) - 1:
                core_out = self.locked_drop_h(core_out)

        # update memory
        # mlen = 0
        # qlen = len(inpt)
        #new_mems = self.update_list_of_mems(hids, list_of_mems, mlen, qlen)
        # original code had a bug, see comments for detail
        new_mems = self.update_list_of_mems(hids, list_of_mems, qlen, mlen)

        # slice according to context_len, normally set to 0
        # in original code they do this via target size, but we don't have that information
        if self.remove_context == True:
            core_out = core_out[self.context_len:]
        core_out = self.locked_drop_o(core_out)
        return core_out, new_mems


class AWDTransformerXLEncoderBlock(nn.Module):
    def __init__(self,
                 list_of_input_dims,
                 remove_context=False,
                 n_layers=16, n_heads=10, head_dim=38, model_dim=380, inner_dim=900,
                 input_dropout_keep_prob=0.4,
                 attention_dropout_keep_prob=0.8,
                 inner_dropout_keep_prob=0.8,
                 hidden_dropout_keep_prob=1.0,
                 output_dropout_keep_prob=0.5,
                 event_based_positions=False,
                 name=None,
                 random_state=None,
                 memory_len=0,
                 context_len=0,
                 strict=None,
                 init=None,
                 scale="default",
                 device="default",
                 dtype="default"):
        super(AWDTransformerXLEncoderBlock, self).__init__()
        if name is None:
            name = _get_name()

        if random_state is None:
            raise ValueError("Must pass random_state to AWDTransformerXLDecoderBlock")

        self.transformer = AWDTransformerXLBaseBlock(list_of_input_dims,
                                                     remove_context=remove_context,
                                                     n_layers=n_layers,
                                                     n_heads=n_heads,
                                                     head_dim=head_dim,
                                                     model_dim=model_dim,
                                                     inner_dim=inner_dim,
                                                     input_dropout_keep_prob=input_dropout_keep_prob,
                                                     attention_dropout_keep_prob=attention_dropout_keep_prob,
                                                     inner_dropout_keep_prob=inner_dropout_keep_prob,
                                                     hidden_dropout_keep_prob=hidden_dropout_keep_prob,
                                                     output_dropout_keep_prob=output_dropout_keep_prob,
                                                     event_based_positions=event_based_positions,
                                                     name=name,
                                                     random_state=random_state,
                                                     memory_len=memory_len,
                                                     context_len=context_len,
                                                     strict=strict,
                                                     init=init,
                                                     scale=scale,
                                                     device=device,
                                                     dtype=dtype)

    def init_list_of_mems(self):
        return self.transformer.init_list_of_mems()

    def update_list_of_mems(self, hiddens, list_of_mems, query_len, memory_len):
        return self.transformer.update_list_of_mems(hiddens, list_of_mems, query_len, memory_len)


    def forward(self, input_tensor, input_mask_tensor=None, input_event_positions=None, list_of_mems=None):
        """
        if not list_of_mems:
            list_of_mems = self.init_list_of_mems()

        shp = input_tensor.shape
        qlen = shp[0]
        mlen = list_of_mems[0].size(0) if list_of_mems is not None else 0
        klen = mlen + qlen
        # masking works internally by setting 1s to neg inf, 0s are left alone! This is slightly different than expected
        # attention mask shows the position each tgt word (row) is allowed to look at (column).
        # 0 1 1 1 1
        # 0 0 1 1 1
        # 0 0 0 1 1
        # 0 0 0 0 1
        attn_mask = torch.triu(input_tensor.new_ones(qlen, klen), diagonal=1 + mlen).bool()[:, :, None]
        if input_mask_tensor is None:
            input_mask_tensor = (0. * input_tensor[:, :, 0]).long()
        input_mask_tensor_dtype = input_mask_tensor.dtype
        attn_mask = (attn_mask.type(input_mask_tensor_dtype) + input_mask_tensor[:, None, :] > 0).bool()
        """
        return self.transformer(input_tensor, input_mask_tensor, input_event_positions, list_of_mems)


class AWDTransformerXLDecoderBlock(nn.Module):
    def __init__(self,
                 list_of_input_dims,
                 remove_context=False,
                 n_layers=16, n_heads=10, head_dim=38, model_dim=380, inner_dim=900,
                 input_dropout_keep_prob=0.4,
                 attention_dropout_keep_prob=0.8,
                 inner_dropout_keep_prob=0.8,
                 hidden_dropout_keep_prob=1.0,
                 output_dropout_keep_prob=0.5,
                 event_based_positions=False,
                 name=None,
                 random_state=None,
                 memory_len=0,
                 context_len=0,
                 strict=None,
                 init=None,
                 scale="default",
                 device="default",
                 dtype="default"):
        super(AWDTransformerXLEncoderBlock, self).__init__()
        if name is None:
            name = _get_name()

        if random_state is None:
            raise ValueError("Must pass random_state to AWDTransformerXLDecoderBlock")

        self.transformer = AWDTransformerXLBaseBlock(list_of_input_dims,
                                                     remove_context=remove_context,
                                                     n_layers=n_layers,
                                                     n_heads=n_heads,
                                                     head_dim=head_dim,
                                                     model_dim=model_dim,
                                                     inner_dim=inner_dim,
                                                     input_dropout_keep_prob=input_dropout_keep_prob,
                                                     attention_dropout_keep_prob=attention_dropout_keep_prob,
                                                     inner_dropout_keep_prob=inner_dropout_keep_prob,
                                                     hidden_dropout_keep_prob=hidden_dropout_keep_prob,
                                                     output_dropout_keep_prob=output_dropout_keep_prob,
                                                     event_based_positions=event_based_positions,
                                                     name=name,
                                                     random_state=random_state,
                                                     memory_len=memory_len,
                                                     context_len=context_len,
                                                     strict=strict,
                                                     init=init,
                                                     scale=scale,
                                                     device=device,
                                                     dtype=dtype)

    def init_list_of_mems(self):
        return self.transformer.init_list_of_mems()

    def update_list_of_mems(self, hiddens, list_of_mems, query_len, memory_len):
        return self.transformer.update_list_of_mems(hiddens, list_of_mems, query_len, memory_len)


    def forward(self, input_tensor, input_mask_tensor=None, input_event_positions=None, list_of_mems=None):
        """
        if not list_of_mems:
            list_of_mems = self.init_list_of_mems()

        shp = input_tensor.shape
        qlen = shp[0]
        mlen = list_of_mems[0].size(0) if list_of_mems is not None else 0
        klen = mlen + qlen
        # masking works internally by setting 1s to neg inf, 0s are left alone! This is slightly different than expected
        # attention mask shows the position each tgt word (row) is allowed to look at (column).
        # 0 1 1 1 1
        # 0 0 1 1 1
        # 0 0 0 1 1
        # 0 0 0 0 1
        attn_mask = torch.triu(input_tensor.new_ones(qlen, klen), diagonal=1 + mlen).bool()[:, :, None]
        if input_mask_tensor is None:
            input_mask_tensor = (0. * input_tensor[:, :, 0]).long()
        input_mask_tensor_dtype = input_mask_tensor.dtype
        attn_mask = (attn_mask.type(input_mask_tensor_dtype) + input_mask_tensor[:, None, :] > 0).bool()
        """
        return self.transformer(input_tensor, input_mask_tensor, input_event_positions, list_of_mems)


def _xlnet_make_sample_mask(seq, goal_n_to_mask, random_state, n_spans=None, max_n_gram=5,
                      alpha=6, beta=1, start_check_function=None):
    """
    seq is typically the INPUT to the XLNet model
    not the target (which would be shifted by 1, in ar context)

    can be target if input and target domains are different

    n_spans will override goal_n_to_mask, forcing a specfic number of spans in the sampled mask

    understanding and reimplementing _sample_mask from XLNet

    0 / False means UNMASKED
    1 / True means MASKED

    add one trick - default code is biased towards masking near the start, if goal_n_to_mask is fairly low
    all the masked tokens happen around the start - nothing in the code really prevents this

    instead, we "overmask" the sequence. then check if we have >= goal_n_to_mask
    if >= goal_n_to_mask, randomly "undelete" certain segments
    otherwise, randomly mask single elements until we hit the condition (as in the original code)

    start_check_function should be a function handle that takes the full sequence element, along with a position pos, and returns True if it is a valid start position
    False otherwise

    def scf(seq, pos):
        return True
    """
    seq_len = len(seq)
    mask = [False for s in seq]

    n_masked_so_far = 0

    if start_check_function is None:
        def scf(seq, pos):
            return True
    else:
        scf = start_check_function

    # setup ngrams, weighted according to length so that longer selectionsa are less probable
    # since they will mask out more tokens
    ngrams = np.arange(1, max_n_gram + 1)
    pvals = 1. / np.arange(1, max_n_gram + 1)
    pvals = pvals / np.sum(pvals)

    cur_step = 0
    masked_bounds = []
    while cur_step < seq_len:
        # don't break if we went past mask, need to "overmask"
        # then randomly undo so as to avoid biasing toward the start
        # this is different than the original code
        # in original code https://github.com/zihangdai/xlnet/blob/master/data_utils.py#L360
        #if n_masked_so_far >= goal_n_to_mask:
        #    break

        # choose an n gram at random
        n = random_state.choice(ngrams, p=pvals)

        # be sure that if we are close to the goal, we only select smaller ones
        # take this OUT, and handle after the loop
        #n = min(n, goal_n_to_mask - n_masked_so_far)

        # set a surrounding context to preserve
        ctx_size = int((n * alpha) // beta)
        # choose the left extent for the context (effectively, context is shifted around a center)
        l_ctx = random_state.choice(ctx_size)
        # choose right extent to complement the left
        r_ctx = ctx_size - l_ctx

        # in original code https://github.com/zihangdai/xlnet/blob/master/data_utils.py#L360
        # only happened on start boundaries
        # here we don't worry about start boundaries (for now)
        b = cur_step + l_ctx
        # scoot over to start on a start boundary... in original code
        while b < seq_len and not scf(seq, b):
            b += 1

        if b >= seq_len:
            break

        # now find a valid end spot, by looking for next start point
        e = b + 1
        _n = 1
        while e < seq_len:
            _n += 1
            if scf(seq, e):
                if _n > n:
                    break
            e += 1

        if e >= seq_len:
            break

        for i in range(b, e):
            mask[i] = True
        masked_bounds.append((b, e))
        n_masked_so_far += e - b
        cur_step = e + r_ctx

    # got too many masked values, lets randomly delete "blocks" until back under
    if n_spans is None:
        while n_masked_so_far > goal_n_to_mask:
            ii = random_state.choice(range(len(masked_bounds)))
            b, e = masked_bounds[ii]
            for i in range(b, e):
                mask[i] = False
            n_masked_so_far -= (e - b)
            masked_bounds = [ma for n, ma in enumerate(masked_bounds) if n != ii]
    else:
        while len(masked_bounds) > n_spans:
            ii = random_state.choice(range(len(masked_bounds)))
            b, e = masked_bounds[ii]
            for i in range(b, e):
                mask[i] = False
            n_masked_so_far -= (e - b)
            masked_bounds = [ma for n, ma in enumerate(masked_bounds) if n != ii]

    # this part would basically just add speckle noise to get to exactly goal_n_mask, but I prefer structure
    # NECESSARY for batch prediction however due to target_mapping used inside XLNet via einsum... so we add it back :|
    while n_masked_so_far < goal_n_to_mask:
        ii = random_state.choice(range(seq_len))
        if mask[ii] == False:
            mask[ii] = True
            n_masked_so_far += 1
    return mask


def _xlnet_make_ar_perm_mask(inp, tgt, is_masked, random_state, sequential_order=False):
    """
    creates valid permutation mask

    a target mask

    and a convenient input stream (q in the paper)
    q is a "blanked out" mask

    k is just a copy of the input, we omit it here

    returns perm_mask_0, target_mask_0, input_k_0, input_q_0

    perm_mask[:, i] tells the connectivity of the ith element
    if target_mask[i] is False, all elements of perm_mask[:, i] will be False, aka allowed to connect

    if target_mask[i] is True, perm_mask[:, i] will be True, EXCEPT for values where self_rev_index[j] > self_rev_index[i]

    basically, this means a target_token[i] for which self_rev_index[i] is a high value, will get very little extra context, whereas
    a target_token[i] for which self_rev_index[i] is a low value, will get a lot of extra context. perm_mask[:, i] will reflect this

    sequential_order will just create a "normal" sequential mask
    """
    seq_len = len(tgt)
    shuffle_inds = np.arange(seq_len)
    if not sequential_order:
        random_state.shuffle(shuffle_inds)
    else:
        # l to r order
        shuffle_inds = shuffle_inds

    index = np.arange(seq_len)
    index = index[shuffle_inds]
    index = index.ravel()

    # fully random permutation
    non_mask_tokens = [not(imt) for imt in is_masked]
    mask_tokens = [not(nmt) for nmt in non_mask_tokens]

    # Set the permutation indices of non tokens to the
    # smallest index (-1):
    # (1) they can be seen by all other positions
    # (2) they cannot see masked positions, so there won"t be information leak
    rev_index = np.where(non_mask_tokens, -1 * np.ones((seq_len)), index)

    target_mask = mask_tokens

    # Create `perm_mask`
    # `target_tokens` cannot see themselves
    self_rev_index = np.where(target_mask, rev_index, rev_index + 1)

    # 1: cannot attend if i <= j and j is not non-masked (masked_or_func_tokens)
    # 0: can attend if i > j or j is non-masked
    perm_mask = (self_rev_index[:, None] <= rev_index[None, :]) & np.array(mask_tokens, dtype=np.bool)
    return perm_mask, rev_index


def _xlnet_make_target_mapping(tgt_mask):
    """
    make target mapping function
    from target sequence and a mask of values to predict

    returned one_hots has dimension
    (num_predict, seq_len)
    where num_predict = sum(tgt_mask == True) aka number of values masked
    """
    indices = np.arange(len(tgt_mask))
    indices = indices[np.where(tgt_mask)]
    one_hots = np.zeros((len(indices), len(tgt_mask)))
    one_hots[np.arange(len(indices), dtype="int32"), indices] = 1
    return one_hots


class TwoStreamRelativeDecoderLayer(nn.Module):
    def __init__(self, list_of_input_dims,
                 n_heads=10,
                 head_dim=38,
                 model_dim=380,
                 inner_dim=900,
                 attention_dropout_keep_prob=0.8,
                 inner_dropout_keep_prob=0.8,
                 residual=True,
                 disable_h_layer_norm=False,
                 name=None,
                 random_state=None,
                 strict=None, init=None,
                 scale="default",
                 device="default",
                 dtype="default"):
        super(TwoStreamRelativeDecoderLayer, self).__init__()

        if name is None:
            name = _get_name()

        if random_state is None:
            raise ValueError("Must pass random_state to TwoStreamRelativeDecoderLayer")

        attention_name = name + "_multihead_attention"
        feedforward_name = name + "_positionwise_ff"

        self.residual = residual
        self.disable_h_layer_norm = disable_h_layer_norm
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.model_dim = model_dim
        self.attention_dropout_keep_prob = attention_dropout_keep_prob

        self.drop = nn.Dropout(1. - attention_dropout_keep_prob)
        self.locked_drop_h = LockedDropout(attention_dropout_keep_prob, random_state=random_state, device=device)
        self.locked_drop_g = LockedDropout(attention_dropout_keep_prob, random_state=random_state, device=device)

        # no biases in transformer XL code
        self.k_net = Linear(list_of_input_dims,
                            n_heads * head_dim,
                            biases=False,
                            random_state=random_state,
                            name=name + "_k",
                            strict=strict,
                            init=init,
                            scale=scale,
                            device=device,
                            dtype=dtype)

        self.v_net = Linear(list_of_input_dims,
                            n_heads * head_dim,
                            biases=False,
                            random_state=random_state,
                            name=name + "_v",
                            strict=strict,
                            init=init,
                            scale=scale,
                            device=device,
                            dtype=dtype)

        self.r_net = Linear(list_of_input_dims,
                            n_heads * head_dim,
                            biases=False,
                            random_state=random_state,
                            name=name + "_r",
                            strict=strict,
                            init=init,
                            scale=scale,
                            device=device,
                            dtype=dtype)

        self.q_net = Linear(list_of_input_dims,
                            n_heads * head_dim,
                            biases=False,
                            random_state=random_state,
                            name=name + "_q",
                            strict=strict,
                            init=init,
                            scale=scale,
                            device=device,
                            dtype=dtype)

        self.o_net = Linear([head_dim * n_heads],
                            model_dim,
                            biases=False,
                            random_state=random_state,
                            name=name + "_o",
                            strict=strict,
                            init=init,
                            scale=scale,
                            device=device,
                            dtype=dtype)

        self.ln = LayerNorm(model_dim,
                            name=name + "_ln",
                            device=device,
                            dtype=dtype)

        self.scale = 1. / (head_dim ** .5)

        self.position_ff = PositionwiseFeedforward(list_of_input_dims,
                                                   inner_dim,
                                                   dropout_keep_prob=inner_dropout_keep_prob,
                                                   name=feedforward_name,
                                                   random_state=random_state,
                                                   strict=strict,
                                                   init=init,
                                                   scale=scale,
                                                   device=device,
                                                   dtype=dtype)

    def _rel_attn_core(self, q_head, k_head_h, v_head_h, k_head_r,
                       local_bias_ac, local_bias_bd, attention_mask,
                       scale, debug=False):

        """Core relative positional attention operations."""
        # [qlen x bsz x n_head x d_head]
        # q_head

        # [klen x bsz x n_head x d_head]
        # i_head_k

        # [qlen x klen x bsz x n_head]
        # content based attention score
        AC = torch.einsum('ibnd,jbnd->ijbn', (q_head + local_bias_ac, k_head_h))

        # Q_HEAD MUST be target MASKED FOR BD calcs?
        # without this, we get grad leaks into the inputs
        # TODO: KEEP EXPLORING THIS
        # this is a mask where if ANY element is masked over timesteps, it is masked in the output
        a = torch.sum(attention_mask, dim=0) > 0
        if attention_mask.shape[0] != a.shape[0]:
            # query part only
            a = a[-attention_mask.shape[0]:]
        q_head.masked_fill_(a[:, :, :, None], 0.)

        # position based attention score
        BD = torch.einsum('ibnd,jbnd->ijbn', (q_head + local_bias_bd, k_head_r))

        # rel shift effectively removes 1 along the 1st dim
        def _rel_shift2(x, klen):
            x_size = x.shape
            x = torch.reshape(x, (x_size[1], x_size[0], x_size[2], x_size[3]))
            x = x[1:]
            x = torch.reshape(x, (x_size[0], x_size[1] - 1, x_size[2], x_size[3]))
            x = x[:, :klen]
            return x

        #BD = _rel_shift(BD, klen=AC.shape[1])
        BD = _rel_shift2(BD, klen=AC.shape[1])

        if attention_mask is not None:
            attention_mask = attention_mask > 0 # cast to bool

        attention_score = AC + BD
        # [qlen x klen x bsz x n_head]
        attention_score *= self.scale


        if attention_mask is not None and attention_mask.any().item():
            # fill 1s with neg inf, leave 0s alone!
            filler = -float('inf')
            #filler = -1E30
            if attention_mask.dim() == 2:
                attention_score.masked_fill_(attention_mask[None, :, :, None], filler)
            # can define either 2d or 3d mask here, but will need to be very careful about what is 0 and what is 1
            elif attention_mask.dim() == 3:
                attention_score.masked_fill_(attention_mask[:, :, :, None], filler)
            elif attention_mask.dim() == 4:
                attention_score.masked_fill_(attention_mask, filler)
            else:
                raise ValueError("Attention mask dim unhandled in _rel_attn_core")

        # alternative method which can preserve attn summing to 1
        #faker = 0. * attention_score + 1.
        #faker_mask = self.drop(faker)
        # can't fill with neginf since could all be 0 - hence all neginf, leading to nan in softmax
        #attention_score.masked_fill_(faker_mask == 0, -1E9)

        attention_prob = F.softmax(attention_score, dim=1)

        # mask out 0 probs, no longer normalized but we are guaranteed not to leak info
        # breaks for any non-4 dim mask but whatever
        attention_prob = attention_prob * (1. - attention_mask.float())

        # this is how it is done in the PTB code but... not normalized anymore!
        # question is, will other method freak out at test time? whereas this is more "normal" - basically same as dropping pieces of i_head_v
        attention_prob = self.drop(attention_prob)

        # isn't this just dropout on the thing you are attending, in disguise?
        # https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py#L1822
        attention_weighted_values = torch.einsum('ijbn,jbnd->ibnd', (attention_prob, v_head_h))

        # [qlen x bsz x n_head x d_head]
        attention_weighted_values = attention_weighted_values.contiguous().view(
               attention_weighted_values.size(0),
               attention_weighted_values.size(1),
               self.n_heads * self.head_dim)
        return attention_weighted_values

    def forward(self, decoder_input_h, decoder_input_g, relative_positional_embedding,
                      local_bias_ac,
                      local_bias_bd,
                      decoder_attention_mask_h=None,
                      decoder_attention_mask_g=None,
                      target_mappings=None,
                      memory=None):

        r = relative_positional_embedding
        h = decoder_input_h
        g = decoder_input_g.type(h.dtype)

        qlen = h.size(0)
        rlen = r.size(0)
        batch_size = h.size(1)

        if memory is not None:
            cat = torch.cat([memory, h], 0)
        else:
            cat = h

        k_head_h = self.k_net([cat])

        v_head_h = self.v_net([cat])

        k_head_r = self.r_net([r])

        q_head_h = self.q_net([h])

        klen = k_head_h.size(0)
        k_head_h = k_head_h.view(klen, batch_size, self.n_heads, self.head_dim)
        v_head_h = v_head_h.view(klen, batch_size, self.n_heads, self.head_dim)

        k_head_r = k_head_r.view(rlen, batch_size, self.n_heads, self.head_dim)

        q_head_h = q_head_h.view(qlen, batch_size, self.n_heads, self.head_dim)

        attention_weighted_values_h = self._rel_attn_core(q_head_h, k_head_h, v_head_h, k_head_r,
                                                          local_bias_ac,
                                                          local_bias_bd,
                                                          decoder_attention_mask_h, self.scale)
        o_h = self.o_net([attention_weighted_values_h])
        o_h = self.locked_drop_h(o_h)
        if self.residual:
            output_h = self.ln(h + o_h)
        else:
            output_h = self.ln(o_h)

        if decoder_input_g is not None:
            q_head_g = self.q_net([g])
            q_head_g = q_head_g.view(g.size(0), batch_size, self.n_heads, self.head_dim)
            if target_mappings is not None:
                q_head_g = torch.einsum('mbnd,mlb->lbnd', (q_head_g, target_mappings))
                attention_weighted_values_g = self._rel_attn_core(q_head_g, k_head_h, v_head_h, k_head_r,
                                                                  local_bias_ac,
                                                                  local_bias_bd,
                                                                  decoder_attention_mask_g, self.scale, debug=True)
                attention_weighted_values_g = attention_weighted_values_g.view(attention_weighted_values_g.size(0), batch_size, self.n_heads, self.head_dim).contiguous()
                attention_weighted_values_g = torch.einsum('lbnd,mlb->mbnd', (attention_weighted_values_g, target_mappings))
                attention_weighted_values_g = attention_weighted_values_g.view(attention_weighted_values_g.size(0), batch_size, self.n_heads * self.head_dim).contiguous()
            else:
                attention_weighted_values_g = self._rel_attn_core(q_head_g, k_head_h, v_head_h, k_head_r,
                                                                  local_bias_ac,
                                                                  local_bias_bd,
                                                                  decoder_attention_mask_g, self.scale)
            o_g = self.o_net([attention_weighted_values_g])
            o_g = self.locked_drop_g(o_g)
            if self.residual:
                output_g = self.ln(g + o_g)
            else:
                output_g = self.ln(o_g)

            output_g = self.position_ff([output_g])
        else:
            output_g = None

        output_h = self.position_ff([output_h])
        return output_h, output_g


class AWDXLNetDecoderBlock(nn.Module):
    """
    def scf(seq, pos):
        if seq[pos] == " ":
            return True
        else:
            return False

    sent = "The purple balloon has a five foot leg."
    mask = _make_sample_mask(sent, goal_n_to_mask=40, n_spans=2, random_state=random_state, max_n_gram=5, start_check_function=scf)
    masked_sent = "".join(["-" if mask[n] else s for n, s in enumerate(sent)])
    pieces = "".join([s for n, s in enumerate(sent) if mask[n] == True])

    inp = sent[:-1]
    tgt = sent[1:]

    # no partial cuts - rather, partial predictions are autoregressive over random permutation. Still only predict a 1 / K subset, but no "cutting points" like mentioned in the paper... https://github.com/zihangdai/xlnet/issues/54
    perm_mask_0, target_mask_0, input_k_0, input_q_0 = _xlnet_make_ar_perm_mask(inp, tgt, mask[1:], random_state=random_state, sequential_order=False)
    target_mapping_0 = _xlnet_make_target_mapping(tgt, target_mask_0)
    """
    def __init__(self,
                 list_of_input_dims,
                 n_layers=16, n_heads=10, head_dim=38, model_dim=380, inner_dim=900,
                 input_dropout_keep_prob=0.4,
                 attention_dropout_keep_prob=0.8,
                 inner_dropout_keep_prob=0.8,
                 hidden_dropout_keep_prob=1.0,
                 output_dropout_keep_prob=0.5,
                 name=None,
                 random_state=None,
                 memory_len=0,
                 context_len=0,
                 strict=None,
                 init=None,
                 scale="default",
                 device="default",
                 dtype="default"):
        super(AWDXLNetDecoderBlock, self).__init__()

        if name is None:
            name = _get_name()

        if random_state is None:
            raise ValueError("Must pass random_state to AWDTransformerXLDecoderBlock")

        self.layers = nn.ModuleList()
        input_dim = sum(list_of_input_dims)
        if input_dim != model_dim:
            raise ValueError("sum of list_of_input_dims should match model_dim due to residual architecture, if this is not the case project the data or change dims! Have {}, sum = {}, model_dim = {}".format(list_of_input_dims, input_dim, model_dim))
        if n_heads * head_dim != model_dim:
            raise ValueError("head_dim * n_heads should == model_dim, have {} * {} != {}".format(head_dim, n_heads, model_dim))
        self.n_layers = n_layers
        self.device = device
        self.dtype = dtype
        self.local_bias_ac = nn.Parameter(torch.zeros(n_heads, head_dim))
        self.local_bias_bd = nn.Parameter(torch.zeros(n_heads, head_dim))

        for i in range(n_layers):
            layer_name = name + "_relative_decoder_layer{}".format(i)
            if i > 0:
                residual = True
            else:
                # could set this to False to force first layer a bit
                residual = True

            if i < len(self.layers) - 1:
                disable_h_layer_norm = False #True
            else:
                # last layer H layer norm never gets trained
                # so just let them be the same params
                disable_h_layer_norm = False

            l = TwoStreamRelativeDecoderLayer(list_of_input_dims,
                                              n_heads=n_heads,
                                              head_dim=head_dim,
                                              model_dim=model_dim,
                                              inner_dim=inner_dim,
                                              attention_dropout_keep_prob=attention_dropout_keep_prob,
                                              inner_dropout_keep_prob=inner_dropout_keep_prob,
                                              name=layer_name,
                                              residual=residual,
                                              disable_h_layer_norm=disable_h_layer_norm,
                                              random_state=random_state,
                                              strict=strict,
                                              init=init,
                                              scale=scale,
                                              device=device,
                                              dtype=dtype)
            self.layers.append(l)

        self.pos_emb = PositionalEmbedding(model_dim,
                                           device=device,
                                           dtype=dtype)
        self.memory_len = memory_len
        self.context_len = context_len
        self.locked_drop_i = LockedDropout(input_dropout_keep_prob,
                                           random_state=random_state,
                                           device=device,
                                           dtype=dtype)
        self.locked_drop_h_h = LockedDropout(hidden_dropout_keep_prob,
                                             random_state=random_state,
                                             device=device,
                                             dtype=dtype)
        self.locked_drop_h_g = LockedDropout(hidden_dropout_keep_prob,
                                             random_state=random_state,
                                             device=device,
                                             dtype=dtype)
        self.locked_drop_o = LockedDropout(output_dropout_keep_prob,
                                           random_state=random_state,
                                           device=device,
                                           dtype=dtype)

    def make_inputs_targets_masks_and_mappings(self, numpy_sequence_array, context_cut, start_check_function=None,
                                               K=6, max_n_gram=5, sequential_order=False, random_state=None):
        """
        """
        if random_state is None:
            raise ValueError("Random state necessary")

        if start_check_function is None:
            def scf(seq, pos):
                return True
        else:
            scf = start_check_function

        if len(numpy_sequence_array.shape) < 2:
            raise ValueError("Sequence array should be (length, examples) in shape)")

        # actual target size matters here
        # so account for context_len / context_cut
        l = len(numpy_sequence_array[context_cut:]) - 1
        goal_n_to_mask = int(l // K)

        agg_input_q = []
        agg_input_k = []
        agg_target = []

        agg_perm_mask = []
        agg_target_mask = []
        agg_target_mapping = []
        agg_perm_orders = []
        for i in range(numpy_sequence_array.shape[1]):
            # only care about targets AFTER context
            assert context_cut >= 1
            #seq = numpy_sequence_array[context_cut:, i]
            #
            full_inp = numpy_sequence_array[:-1, i]
            full_tgt = numpy_sequence_array[1:, i]
            inp = full_inp[context_cut:]
            tgt = full_tgt[context_cut:]

            # tgt is a shift of the input by 1 step as in "classic" AR language models
            mask = _xlnet_make_sample_mask(inp, goal_n_to_mask=goal_n_to_mask, random_state=random_state, max_n_gram=max_n_gram, start_check_function=scf)
            # inpt goes through xlnet_make_ar_perm_mask basically untouched...
            # we take it out of the function definition for generality

            # no partial cuts - rather, partial predictions are autoregressive over random permutation. Still only predict a 1 / K subset, but no "cutting points" like mentioned in the paper... https://github.com/zihangdai/xlnet/issues/54
            # the cutting point is effectively handled by use of context_len in training, as in the transformer XL paper
            #perm_mask_0, target_0, target_mask_0, input_k_0, input_q_0, perm_order_0 = _xlnet_make_ar_perm_mask(inp, tgt, mask, random_state=random_state, sequential_order=sequential_order)
            perm_mask_0, perm_order_0 = _xlnet_make_ar_perm_mask(inp, tgt, mask, random_state=random_state, sequential_order=sequential_order)

            perm_mask_0 = perm_mask_0.astype("float32")

            # all the places where we want to block target info from leaking
            blocked_ind = np.where(np.array(mask) == False)[0][0]
            pad_u = np.zeros((context_cut, perm_mask_0.shape[1]))
            pad_u = np.copy(perm_mask_0[blocked_ind][None]) + pad_u
            # pad_u should respect the default "don't look at any targets" setting to avoid data leaks
            perm_mask_0 = np.concatenate((pad_u, perm_mask_0), axis=0)

            # pad_l is covering context info, which is all valid to be looked at
            pad_l = np.zeros((perm_mask_0.shape[0], context_cut))
            perm_mask_0 = np.concatenate((pad_l, perm_mask_0), axis=1)

            # target_mapping should be the length of the full sequence - due to the mechanics of the attention inside TwoStreamRelativeDecoder target_mask_0 = np.array(mask).astype("float32")
            pad_l = np.zeros((len(full_tgt) - len(tgt),))
            target_mask_0 = np.concatenate((pad_l, target_mask_0))
            target_mapping_0 = _xlnet_make_target_mapping(target_mask_0)

            input_q_0 = np.copy(target_mask_0)
            input_k_0 = np.copy(full_inp)

            # See
            # https://github.com/zihangdai/xlnet/issues/104
            # masking prevents the current step from "seeing itself"
            # basically, new_target becomes full_inp again
            new_target = np.concatenate((full_inp[0:1], full_tgt[:-1]), axis=0)
            target_0 = np.copy(new_target)

            agg_input_q.append(input_q_0)
            agg_input_k.append(input_k_0)

            agg_target.append(target_0)

            agg_perm_mask.append(perm_mask_0)
            agg_target_mask.append(target_mask_0)
            agg_target_mapping.append(target_mapping_0)

            agg_perm_orders.append(perm_order_0)

        perm_masks = np.array(agg_perm_mask).transpose(1, 2, 0).astype("float32")
        target_masks = np.array(agg_target_mask).transpose(1, 0).astype("float32")
        targets = np.array(agg_target).transpose(1, 0).astype("float32")
        input_qs = np.array(agg_input_q).transpose(1, 0).astype("float32")
        input_ks = np.array(agg_input_k).transpose(1, 0).astype("float32")
        # num_predict, tgt_len, bsz?
        target_mappings = np.array(agg_target_mapping).transpose(1, 2, 0).astype("float32")
        perm_orders = np.array(agg_perm_orders).transpose(1, 0).astype("float32")
        return perm_masks, target_mappings, target_masks, input_ks, input_qs, targets, perm_orders

    def init_list_of_mems(self):
        if self.device == "default":
            device = get_device_default()
        else:
            device = self.device

        if self.dtype == "default":
            dtype = get_dtype_default()
        if dtype == "float32":
            dtype = torch.float32
        else:
            dtype = torch.float64

        # returns None if mem_len is 0 else
        if self.memory_len > 0:
            mems = []

            for i in range(self.n_layers):
                empty = torch.empty(0, dtype=dtype, device=torch.device(device))
                mems.append(empty)
            return mems
        else:
            return None

    def update_list_of_mems(self, hiddens, list_of_mems, query_len, memory_len):
        # mlen and qlen were swapped in call vs signature in original code!
        # https://github.com/kimiyoung/transformer-xl/issues/96
        # effectively, would mean memory_len= len query
        # and query_len= 0 for PTB experiment
        # where self.context_len was 70
        # self.mem_len 0
        # we swap the call to be correct, and set hyperparameters to actually use memory
        if list_of_mems is None:
            return None

        if self.memory_len == 0:
            return None

        qlen = query_len
        mlen = memory_len

        assert len(hiddens) == len(list_of_mems), "len(list_of_mems) != len(hiddens)"

        # Copied from transformer XL main code
        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen - 0 - self.context_len)
            beg_idx = max(0, end_idx - self.memory_len)
            for i in range(len(hiddens)):
                cat = torch.cat([list_of_mems[i], hiddens[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())
        return new_mems

    def forward(self, input_ks, input_qs, perm_masks, target_mappings, target_masks, list_of_mems=None):
        if not list_of_mems:
            list_of_mems = self.init_list_of_mems()

        # input_mask = None and perm_mask is not None case from 
        # https://github.com/zihangdai/xlnet/blob/master/modeling.py#L490
        # attn_mask = None for attn_type = 'bi'
        data_mask = perm_masks

        qlen = input_ks.shape[0]
        mlen = list_of_mems[0].size(0) if list_of_mems is not None else 0
        klen = qlen + mlen
        context_pad = input_ks.new_zeros(data_mask.shape[0], input_ks.shape[0] - data_mask.shape[0], data_mask.shape[2])
        mem_pad = input_ks.new_zeros(data_mask.shape[0], mlen, data_mask.shape[2])

        data_mask = torch.cat((mem_pad, context_pad, data_mask), 1)
        attn_mask = 1. * data_mask[:, :, :, None].bool()

        excess_pad = input_ks.new_zeros((data_mask.shape[0], input_ks.shape[0] - data_mask.shape[0] + mlen))
        non_tgt_mask = -1 * torch.eye(data_mask.shape[0], dtype=excess_pad.dtype, device=excess_pad.device)
        non_tgt_mask = torch.cat((excess_pad, non_tgt_mask), -1)

        non_tgt_mask = attn_mask + non_tgt_mask[:, :, None, None]
        non_tgt_mask = 1. * (non_tgt_mask > 0)

        # relative positional embedding
        #pos_seq = torch.arange(klen, -1 if mlen == 0 else -qlen, -1.0, device=input_ks.device)
        #pos_seq = torch.arange(klen, -1, -1.0, device=input_ks.device)

        # from xlnet - "bidirectional attention" attn type
        pos_seq = torch.arange(klen, -qlen, -1.0, device=input_ks.device)
        pe = self.pos_emb(pos_seq, batch_size=input_ks.shape[1])

        hids = []
        output_h = self.locked_drop_i(input_ks)
        if input_qs is not None:
            output_g = self.locked_drop_i(input_qs)
        else:
            output_g = None

        pe = self.locked_drop_i(pe)
        for i, this_layer in enumerate(self.layers):
            hids.append(output_h)
            mems_i = list_of_mems[i] if list_of_mems is not None else None
            new_output_h, new_output_g = this_layer(output_h, output_g, pe,
                                                    local_bias_ac=self.local_bias_ac[None, None],
                                                    local_bias_bd=self.local_bias_bd[None, None],
                                                    decoder_attention_mask_h=non_tgt_mask,
                                                    decoder_attention_mask_g=attn_mask,
                                                    target_mappings=target_mappings,
                                                    memory=mems_i)
            if i < (len(self.layers) - 1):
                output_h = self.locked_drop_h_h(new_output_h)
                if input_qs is not None:
                    output_g = self.locked_drop_h_g(new_output_g)
                else:
                    output_g = None
            else:
                output_h = new_output_h
                if input_qs is not None:
                    output_g = new_output_g
                else:
                    output_g = None

        # update memory
        # mlen = 0
        # qlen = len(inpt)
        #new_mems = self.update_list_of_mems(hids, list_of_mems, mlen, qlen)
        # original code had a bug, see comments for detail
        new_mems = self.update_list_of_mems(hids, list_of_mems, qlen, mlen)

        # slice according to context_len, normally set to 0
        # in original code they do this via target size, but we don't have that information
        output_h = self.locked_drop_o(output_h)
        if input_qs is not None:
            output_g = self.locked_drop_o(output_g)
        else:
            output_g = None
        return output_h, output_g, new_mems


class MelNetLayer(torch.nn.Module):
    def __init__(self,
                 list_of_input_dims,
                 output_dims,
                 init=None, scale="default",
                 biases=True, bias_offset=0.,
                 name=None,
                 cell_dropout=1.,
                 use_centralized_stack=True,
                 random_state=None, strict=None,
                 dtype="default", device="default"):
        super(MelNetLayer, self).__init__()
        if name is None:
            name = _get_name()

        if random_state is None:
            raise ValueError("Must pass instance of np.random.RandomState!")

        if type(name) is str:
            name_td_d = name + "_rnn_td_d"
            name_td_u = name + "_rnn_td_u"
            name_td_f = name + "_rnn_td_f"
            name_td_c = name + "_rnn_td_c"
            name_fd_f = name + "_rnn_fd_f"
            name_proj = name + "_td_conv_proj"


        if strict is None:
            strict = get_strict_mode_default()

        self.hidden_size = list_of_input_dims[0]
        self.cell_dropout = cell_dropout
        self.use_centralized_stack = use_centralized_stack
        self.rnn_time_delayed_down = LSTMCell([self.hidden_size],
                                               output_dims,
                                               name=name_td_d,
                                               init=init,
                                               random_state=random_state)
        self.rnn_time_delayed_up = LSTMCell([self.hidden_size],
                                             output_dims,
                                             name=name_td_u,
                                             init=init,
                                             random_state=random_state)

        self.rnn_time_delayed_fwd = LSTMCell([self.hidden_size],
                                              output_dims,
                                              name=name_td_f,
                                              init=init,
                                              random_state=random_state)
        if self.use_centralized_stack:
            self.rnn_time_delayed_cent = LSTMCell([self.hidden_size],
                                                  output_dims,
                                                  name=name_td_c,
                                                  init=init,
                                                  random_state=random_state)

        self.rnn_freq_delayed_fwd = LSTMCell([self.hidden_size],
                                              output_dims,
                                              name=name_fd_f,
                                              init=init,
                                              random_state=random_state)

        self.td_proj_conv = Conv2d([3 * self.hidden_size], self.hidden_size, kernel_size=(1, 1), strides=(1, 1), border_mode=(0, 0),
                                   random_state=random_state, name=name_proj)


    # unconditional first tier
    def forward(self, list_of_inputs):
        # condidering axis 2 time 3 frequency
        # do time
        tier_temp_base_t = list_of_inputs[0]
        tier_temp_base_f = list_of_inputs[1]
        tier_temp_base_c = list_of_inputs[2]

        tier_temp_base_t_result = 0. * tier_temp_base_t + tier_temp_base_t

        tier_temp_t = space2batch(tier_temp_base_t_result, axis=2)
        # flipped in freq
        tier_temp_t_r = torch.flip(tier_temp_t, dims=(0,))

        # make these learnable?
        batch_size = tier_temp_t.shape[1]
        inp_h_init = torch.Tensor(np.zeros((batch_size, self.hidden_size)).astype("float32")).to(tier_temp_base_t.device)
        inp_c_init = torch.Tensor(np.zeros((batch_size, self.hidden_size)).astype("float32")).to(tier_temp_base_t.device)
        inp_r_h_init = torch.Tensor(np.zeros((batch_size, self.hidden_size)).astype("float32")).to(tier_temp_base_t.device)
        inp_r_c_init = torch.Tensor(np.zeros((batch_size, self.hidden_size)).astype("float32")).to(tier_temp_base_t.device)
        def tier_step_time_freq(inp_t, inp_r_t, inp_h_tm1, inp_c_tm1, inp_r_h_tm1, inp_r_c_tm1):
            output, s = self.rnn_time_delayed_down([inp_t],
                                                   inp_h_tm1,
                                                   inp_c_tm1,
                                                   cell_dropout=self.cell_dropout)
            inp_h_t = s[0]
            inp_c_t = s[1]

            output, s = self.rnn_time_delayed_up([inp_r_t],
                                                  inp_r_h_tm1,
                                                  inp_r_c_tm1,
                                                  cell_dropout=self.cell_dropout)
            inp_r_h_t = s[0]
            inp_r_c_t = s[1]
            return [inp_h_t, inp_c_t, inp_r_h_t, inp_r_c_t]

        r = scan(tier_step_time_freq, [tier_temp_t, tier_temp_t_r], [inp_h_init, inp_c_init, inp_r_h_init, inp_r_c_init])
        tier_temp_t_f_res = r[0]
        # unflip
        tier_temp_t_f_r_res = torch.flip(r[2], dims=(0,))

        tier_temp_t_t = space2batch(tier_temp_base_t_result, axis=3)
        batch_size = tier_temp_t_t.shape[1]
        inp_h_init = torch.Tensor(np.zeros((batch_size, self.hidden_size)).astype("float32")).to(tier_temp_base_t.device)
        inp_c_init = torch.Tensor(np.zeros((batch_size, self.hidden_size)).astype("float32")).to(tier_temp_base_t.device)
        def tier_step_time_time(inp_t, inp_h_tm1, inp_c_tm1):
            output, s = self.rnn_time_delayed_fwd([inp_t],
                                                  inp_h_tm1,
                                                  inp_c_tm1,
                                                  cell_dropout=self.cell_dropout)
            inp_h_t = s[0]
            inp_c_t = s[1]
            return [inp_h_t, inp_c_t]
        r = scan(tier_step_time_time, [tier_temp_t_t], [inp_h_init, inp_c_init])
        tier_temp_t_t_res = r[0]

        # some model here
        tier_temp_revert_t_f = batch2space(tier_temp_t_f_res, n_batch=tier_temp_base_t.shape[0], axis=2)
        tier_temp_revert_t_f_r = batch2space(tier_temp_t_f_r_res, n_batch=tier_temp_base_t.shape[0], axis=2)
        tier_temp_revert_t_t = batch2space(tier_temp_t_t_res, n_batch=tier_temp_base_t.shape[0], axis=3)

        # down proj after concat
        tier_temp_revert_t = self.td_proj_conv([torch.cat((tier_temp_revert_t_f, tier_temp_revert_t_f_r, tier_temp_revert_t_t), axis=1)])
        # skip connection
        tier_temp_revert_t_merge = tier_temp_base_t + tier_temp_revert_t

        # do it like this so we don't modify the input
        tier_temp_base_f_result = 0. * tier_temp_base_f + tier_temp_base_f
        # do frequency, noting freq conditions on time...
        tier_temp_base_f_result[:, :, :, :-1] = tier_temp_base_f_result[:, :, :, :-1] + tier_temp_revert_t_merge[:, :, :-1, :]

        if tier_temp_base_c is not None:
            tier_temp_base_c_result = 0. * tier_temp_base_c + tier_temp_base_c

            batch_size = tier_temp_base_c_result.shape[1]
            inp_h_init = torch.Tensor(np.zeros((batch_size, self.hidden_size)).astype("float32")).to(tier_temp_base_t.device)
            inp_c_init = torch.Tensor(np.zeros((batch_size, self.hidden_size)).astype("float32")).to(tier_temp_base_t.device)
            def tier_step_cent(inp_t, inp_h_tm1, inp_c_tm1):
                output, s = self.rnn_time_delayed_cent([inp_t],
                                                       inp_h_tm1,
                                                       inp_c_tm1,
                                                       cell_dropout=self.cell_dropout)
                inp_h_t = s[0]
                inp_c_t = s[1]
                return [inp_h_t, inp_c_t]

            tier_temp_c = tier_temp_base_c_result
            r = scan(tier_step_cent, [tier_temp_c], [inp_h_init, inp_c_init])
            # post proj?
            tier_temp_c_res = r[0]

            # skip connected
            tier_temp_revert_c_merge = tier_temp_c_res + tier_temp_c

            tier_temp_revert_c_mod = tier_temp_revert_c_merge.permute(1, 2, 0)[..., None].contiguous()

            # all 3 are summed now
            tier_temp_base_f_result[:, :, :, :-1] = tier_temp_base_f_result[:, :,  :, :-1] + tier_temp_revert_c_mod[:, :, :-1, :]

        tier_temp_f = space2batch(tier_temp_base_f_result, axis=2)
        batch_size = tier_temp_f.shape[1]
        inp_h_init = torch.Tensor(np.zeros((batch_size, self.hidden_size)).astype("float32")).to(tier_temp_base_t.device)
        inp_c_init = torch.Tensor(np.zeros((batch_size, self.hidden_size)).astype("float32")).to(tier_temp_base_t.device)
        def tier_step_freq(inp_t, inp_h_tm1, inp_c_tm1):
            output, s = self.rnn_freq_delayed_fwd([inp_t],
                                                  inp_h_tm1,
                                                  inp_c_tm1,
                                                  cell_dropout=self.cell_dropout)
            inp_h_t = s[0]
            inp_c_t = s[1]
            return [inp_h_t, inp_c_t]

        r = scan(tier_step_freq, [tier_temp_f], [inp_h_init, inp_c_init])
        tier_temp_f_res = r[0]

        tier_temp_revert_f = batch2space(tier_temp_f_res, n_batch=tier_temp_base_f.shape[0], axis=2)
        # skip connected

        # add proj?

        tier_temp_revert_f_merge = tier_temp_revert_f + tier_temp_base_f
        if tier_temp_base_c is not None:
            return tier_temp_revert_t_merge, tier_temp_revert_f_merge, tier_temp_revert_c_merge
        else:
            return tier_temp_revert_t_merge, tier_temp_revert_f_merge


class MelNetFullContextLayer(torch.nn.Module):
    def __init__(self,
                 list_of_input_dims,
                 output_dims,
                 init=None, scale="default",
                 biases=True, bias_offset=0.,
                 name=None,
                 cell_dropout=1.,
                 random_state=None, strict=None,
                 dtype="default", device="default"):
        super(MelNetFullContextLayer, self).__init__()
        if name is None:
            name = _get_name()

        if random_state is None:
            raise ValueError("Must pass instance of np.random.RandomState!")

        if type(name) is str:
            name_td_d = name + "_rnn_d"
            name_td_u = name + "_rnn_u"
            name_td_f = name + "_rnn_f"
            name_td_b = name + "_rnn_b"
            name_proj = name + "_td_conv_proj"

        if strict is None:
            strict = get_strict_mode_default()

        self.hidden_size = list_of_input_dims[0]
        self.cell_dropout = cell_dropout
        self.rnn_down = LSTMCell([self.hidden_size],
                                               output_dims,
                                               name=name_td_d,
                                               init=init,
                                               random_state=random_state)
        self.rnn_up = LSTMCell([self.hidden_size],
                                             output_dims,
                                             name=name_td_u,
                                             init=init,
                                             random_state=random_state)

        self.rnn_fwd = LSTMCell([self.hidden_size],
                                 output_dims,
                                 name=name_td_f,
                                 init=init,
                                 random_state=random_state)

        self.rnn_bwd = LSTMCell([self.hidden_size],
                                 output_dims,
                                 name=name_td_b,
                                 init=init,
                                 random_state=random_state)

        self.proj_conv = Conv2d([4 * self.hidden_size], self.hidden_size, kernel_size=(1, 1), strides=(1, 1), border_mode=(0, 0),
                                 random_state=random_state, name=name_proj)


    # unconditional first tier
    def forward(self, list_of_inputs):
        # condidering axis 2 time 3 frequency
        # do time
        tier_temp_base_t = list_of_inputs[0]

        tier_temp_base_t_result = 0. * tier_temp_base_t + tier_temp_base_t

        tier_temp_t = space2batch(tier_temp_base_t_result, axis=2)
        # flipped in freq
        tier_temp_t_r = torch.flip(tier_temp_t, dims=(0,))

        # make these learnable?
        batch_size = tier_temp_t.shape[1]
        inp_h_init = torch.Tensor(np.zeros((batch_size, self.hidden_size)).astype("float32")).to(tier_temp_base_t.device)
        inp_c_init = torch.Tensor(np.zeros((batch_size, self.hidden_size)).astype("float32")).to(tier_temp_base_t.device)
        inp_r_h_init = torch.Tensor(np.zeros((batch_size, self.hidden_size)).astype("float32")).to(tier_temp_base_t.device)
        inp_r_c_init = torch.Tensor(np.zeros((batch_size, self.hidden_size)).astype("float32")).to(tier_temp_base_t.device)
        def tier_step_time_freq(inp_t, inp_r_t, inp_h_tm1, inp_c_tm1, inp_r_h_tm1, inp_r_c_tm1):
            output, s = self.rnn_down([inp_t],
                                      inp_h_tm1,
                                      inp_c_tm1,
                                      cell_dropout=self.cell_dropout)
            inp_h_t = s[0]
            inp_c_t = s[1]

            output, s = self.rnn_up([inp_r_t],
                                    inp_r_h_tm1,
                                    inp_r_c_tm1,
                                    cell_dropout=self.cell_dropout)
            inp_r_h_t = s[0]
            inp_r_c_t = s[1]
            return [inp_h_t, inp_c_t, inp_r_h_t, inp_r_c_t]

        r = scan(tier_step_time_freq, [tier_temp_t, tier_temp_t_r], [inp_h_init, inp_c_init, inp_r_h_init, inp_r_c_init])
        tier_temp_t_f_res = r[0]
        # unflip
        tier_temp_t_f_r_res = torch.flip(r[2], dims=(0,))

        tier_temp_t_t = space2batch(tier_temp_base_t_result, axis=3)
        tier_temp_t_t_r = torch.flip(tier_temp_t_t, dims=(0,))

        batch_size = tier_temp_t_t.shape[1]
        inp_h_init = torch.Tensor(np.zeros((batch_size, self.hidden_size)).astype("float32")).to(tier_temp_base_t.device)
        inp_c_init = torch.Tensor(np.zeros((batch_size, self.hidden_size)).astype("float32")).to(tier_temp_base_t.device)
        inp_r_h_init = torch.Tensor(np.zeros((batch_size, self.hidden_size)).astype("float32")).to(tier_temp_base_t.device)
        inp_r_c_init = torch.Tensor(np.zeros((batch_size, self.hidden_size)).astype("float32")).to(tier_temp_base_t.device)
        def tier_step_time_time(inp_t, inp_r_t, inp_h_tm1, inp_c_tm1, inp_r_h_tm1, inp_r_c_tm1):
            output, s = self.rnn_fwd([inp_t],
                                     inp_h_tm1,
                                     inp_c_tm1,
                                     cell_dropout=self.cell_dropout)
            inp_h_t = s[0]
            inp_c_t = s[1]

            output, s = self.rnn_bwd([inp_r_t],
                                     inp_r_h_tm1,
                                     inp_r_c_tm1,
                                     cell_dropout=self.cell_dropout)
            inp_r_h_t = s[0]
            inp_r_c_t = s[1]
            return [inp_h_t, inp_c_t, inp_r_h_t, inp_r_c_t]
        r = scan(tier_step_time_time, [tier_temp_t_t, tier_temp_t_t_r], [inp_h_init, inp_c_init, inp_r_h_init, inp_r_c_init])
        tier_temp_t_t_res = r[0]
        # unflip
        tier_temp_t_t_r_res = torch.flip(r[2], dims=(0,))

        # some model here
        tier_temp_revert_t_f = batch2space(tier_temp_t_f_res, n_batch=tier_temp_base_t.shape[0], axis=2)
        tier_temp_revert_t_f_r = batch2space(tier_temp_t_f_r_res, n_batch=tier_temp_base_t.shape[0], axis=2)
        tier_temp_revert_t_t = batch2space(tier_temp_t_t_res, n_batch=tier_temp_base_t.shape[0], axis=3)
        tier_temp_revert_t_t_r = batch2space(tier_temp_t_t_r_res, n_batch=tier_temp_base_t.shape[0], axis=3)

        # down proj after concat
        tier_temp_revert_t = self.proj_conv([torch.cat((tier_temp_revert_t_f, tier_temp_revert_t_f_r, tier_temp_revert_t_t, tier_temp_revert_t_t_r), axis=1)])
        return tier_temp_revert_t


class MelNetTier(torch.nn.Module):
    def __init__(self,
                 list_of_input_symbol_sizes,
                 n_vert,
                 n_horiz,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 has_centralized_stack=False,
                 has_spatial_condition=False,
                 conditional_layers=2,
                 init=None, scale="default",
                 biases=True, bias_offset=0.,
                 name=None,
                 cell_dropout=1.,
                 use_centralized_stack=False,
                 random_state=None, strict=None,
                 dtype="default", device="default"):
        super(MelNetTier, self).__init__()
        if name is None:
            name = _get_name()

        if random_state is None:
            raise ValueError("Must pass instance of np.random.RandomState!")

        if strict is None:
            strict = get_strict_mode_default()

        self.input_symbols = list_of_input_symbol_sizes[0]
        self.hidden_size = hidden_dim
        self.output_size = output_dim
        self.has_centralized_stack = has_centralized_stack
        self.has_spatial_condition = has_spatial_condition
        self.conditional_layers = conditional_layers

        self.cell_dropout = cell_dropout
        self.n_layers = n_layers
        self.n_vert = n_vert
        self.n_horiz = n_horiz
        self.output_dim = output_dim

        self.embed_td = Embedding(self.input_symbols, self.hidden_size, random_state=random_state, name=name + "_embed_td")
        self.embed_fd = Embedding(self.input_symbols, self.hidden_size, random_state=random_state, name=name + "_embed_fd")

        self.tds_lstms_time_fw = nn.ModuleList()
        self.tds_lstms_freq_fw = nn.ModuleList()
        self.tds_lstms_freq_bw = nn.ModuleList()
        self.tds_projs = nn.ModuleList()
        self.fds_lstms_freq_fw = nn.ModuleList()
        if self.has_centralized_stack:
            self.tds_centralized_lstms = nn.ModuleList()

        if self.has_spatial_condition:
            self.cond_mn = MelNetFullContextSubTier([self.input_symbols], n_vert, n_horiz, self.hidden_size, self.conditional_layers,
                                       random_state=random_state,
                                       init=init,
                                       name=name + "cond_mn")

        for _i in range(self.n_layers):
            self.tds_lstms_time_fw.append(LSTMLayer([self.hidden_size,],
                                                    self.hidden_size,
                                                    random_state=random_state,
                                                    init=init,
                                                    scale=scale,
                                                    name=name + "_tds_lstm_time_fw_{}".format(_i)))

            self.tds_lstms_freq_fw.append(LSTMLayer([self.hidden_size,],
                                                    self.hidden_size,
                                                    random_state=random_state,
                                                    init=init,
                                                    scale=scale,
                                                    name=name + "_tds_lstm_freq_fw_{}".format(_i)))

            self.tds_lstms_freq_bw.append(LSTMLayer([self.hidden_size,],
                                                    self.hidden_size,
                                                    random_state=random_state,
                                                    init=init,
                                                    scale=scale,
                                                    name=name + "_tds_lstm_freq_bw_{}".format(_i)))

            self.tds_projs.append(Linear([3 * self.hidden_size,],
                                          self.hidden_size,
                                          random_state=random_state,
                                          init=init,
                                          scale=scale,
                                          name=name + "_tds_projs_{}".format(_i)))


            if self.has_centralized_stack:
                self.tds_centralized_lstms.append(LSTMLayer([self.n_horiz * self.hidden_size,],
                                                             self.hidden_size,
                                                             random_state=random_state,
                                                             init=init,
                                                             scale=scale,
                                                             name=name + "_tds_centralized_lstm_{}".format(_i)))
                self.fds_lstms_freq_fw.append(LSTMLayer([3 * self.hidden_size,],
                                                        self.hidden_size,
                                                        random_state=random_state,
                                                        init=init,
                                                        scale=scale,
                                                        name=name + "_fds_lstm_freq_fw_{}".format(_i)))
            else:
                self.fds_lstms_freq_fw.append(LSTMLayer([2 * self.hidden_size,],
                                                        self.hidden_size,
                                                        random_state=random_state,
                                                        init=init,
                                                        scale=scale,
                                                        name=name + "_fds_lstm_freq_fw_{}".format(_i)))
        self.out_proj = Linear([self.hidden_size,], self.output_size,
                               random_state=random_state,
                               init=init,
                               scale=scale,
                               name=name + "_output_proj")

    def _time2freq(self, inp):
        inp = inp.reshape((self.n_vert, self.batch_size, self.n_horiz, -1))
        inp = inp.permute(2, 1, 0, 3)
        return inp.reshape((self.n_horiz, self.batch_size * self.n_vert, -1))

    def _freq2time(self, inp):
        # batch size set in forward!
        inp = inp.reshape((self.n_horiz, self.batch_size, self.n_vert, -1))
        inp = inp.permute(2, 1, 0, 3)
        return inp.reshape((self.n_vert, self.batch_size * self.n_horiz, -1))

    def _td_stack(self, tds, layer):
        freq_lstm_fw_h, _, __ = self.tds_lstms_freq_fw[layer]([tds])
        freq_lstm_bw_h, _, __ = self.tds_lstms_freq_bw[layer]([torch.flip(tds, [0])])
        freq_lstm_h = torch.cat((freq_lstm_fw_h, torch.flip(freq_lstm_bw_h, [0])), dim=-1)
        freq_lstm_h = self._freq2time(freq_lstm_h)

        tds_time = self._freq2time(tds)
        time_lstm_h, _, __ = self.tds_lstms_time_fw[layer]([tds_time])
        combined_h = torch.cat((freq_lstm_h, time_lstm_h), dim=-1)
        res = self.tds_projs[layer]([combined_h])
        res = self._time2freq(res)
        return (0.5 ** 0.5) * (tds + res)

    def _td_centralized_stack(self, tds, layer):
        t_tds = self._freq2time(tds)
        t_tds = t_tds.reshape((self.n_vert, self.batch_size, self.n_horiz, -1))
        r_tds = t_tds.reshape((self.n_vert, self.batch_size, -1))
        cent_lstm_h, _, __ = self.tds_centralized_lstms[layer]([r_tds])
        # stack / broadcast the compressed thing back to the combined batch size
        cent_lstm_h_re = 0. * t_tds[:, :, :, 0][..., None] + cent_lstm_h[:, :, None, :]
        res = cent_lstm_h_re.reshape((self.n_vert, self.batch_size, self.n_horiz, -1))
        res = self._time2freq(res)
        return (0.5 ** 0.5) * (tds + res)

    def _fd_stack(self, tds, fds, layer, tds_cent=None):
        if tds_cent is not None:
            freq_lstm_stack = torch.cat((tds, fds, tds_cent), axis=-1)
        else:
            freq_lstm_stack = torch.cat((tds, fds), axis=-1)
        freq_lstm_h, _, __ = self.fds_lstms_freq_fw[layer]([freq_lstm_stack])
        return (0.5 ** 0.5) * (fds + freq_lstm_h)

    # unconditional first tier
    def forward(self, list_of_inputs, list_of_spatial_conditions=None, bypass_td=None, bypass_fd=None, skip_input_embed=False):
        # by default embed the inputs, otherwise bypass
        # condidering axis 2 time 3 frequency

        # shift and project the input
        if len(list_of_inputs) > 1:
            raise ValueError("Only support list_of_inputs length 1 for now")

        x = list_of_inputs[0]
        td_x = torch.cat((0 * x[:, 0][:, None], x[:, :-1]), dim=1)
        fd_x = torch.cat((0 * x[:, :, 0][:, :, None], x[:, :, :-1]), dim=2)
        if not skip_input_embed:
            td_x, td_e = self.embed_td(td_x)
            fd_x, fd_e = self.embed_fd(fd_x)

        if bypass_td is not None:
            td_x = bypass_td
        if bypass_fd is not None:
            fd_x = bypass_fd

        if self.has_spatial_condition:
            cond_info = self.cond_mn([list_of_spatial_conditions[0]])
            td_x = td_x + cond_info
            fd_x = fd_x + cond_info

        batch_size = td_x.shape[0]
        self.batch_size = batch_size
        td_x = td_x.reshape((batch_size * self.n_vert, self.n_horiz, -1))
        fd_x = fd_x.reshape((batch_size * self.n_vert, self.n_horiz, -1))
        td_x = td_x.permute(1, 0, 2)
        fd_x = fd_x.permute(1, 0, 2)
        for _i in range(self.n_layers):
            td_x = self._td_stack(td_x, _i)
            if self.has_centralized_stack:
                td_cent_x = self._td_centralized_stack(td_x, _i)
                fd_x = self._fd_stack(td_x, fd_x, _i, tds_cent=td_cent_x)
            else:
                fd_x = self._fd_stack(td_x, fd_x, _i)
        out = self.out_proj([fd_x])
        out = out.reshape((self.n_horiz, self.batch_size, self.n_vert, self.output_size))
        out = out.permute((1, 2, 0, 3))
        return out


class MelNetFullContextSubTier(torch.nn.Module):
    def __init__(self,
                 list_of_input_symbol_sizes,
                 n_vert,
                 n_horiz,
                 hidden_dim,
                 n_layers,
                 cell_type="lstm",
                 init=None, scale="default",
                 biases=True, bias_offset=0.,
                 name=None,
                 cell_dropout=1.,
                 use_centralized_stack=False,
                 random_state=None, strict=None,
                 dtype="default", device="default"):
        super(MelNetFullContextSubTier, self).__init__()
        if name is None:
            name = _get_name()

        if random_state is None:
            raise ValueError("Must pass instance of np.random.RandomState!")

        if strict is None:
            strict = get_strict_mode_default()

        self.cell_type = cell_type

        if self.cell_type == "lstm":
            bidir_rnn_fn = BiLSTMLayer
            rnn_fn = LSTMLayer
            rnn_cell_fn = LSTMCell
        elif self.cell_type == "gru":
            bidir_rnn_fn = BiGRULayer
            rnn_fn = GRULayer
            rnn_cell_fn = GRUCell
        else:
            raise ValueError("Unknown cell_type, self.cell_type was {}".format(self.cell_type))

        self.input_symbols = list_of_input_symbol_sizes[0]
        self.hidden_size = hidden_dim

        self.cell_dropout = cell_dropout
        self.n_layers = n_layers
        self.n_vert = n_vert
        self.n_horiz = n_horiz

        self.input_proj = Linear([1],
                                            self.hidden_size,
                                             random_state=random_state,
                                             init=init,
                                             scale=scale,
                                             name=name + "_input_proj",
                                             device=device)

        self.rnn_time_fw = nn.ModuleList()
        self.rnn_time_bw = nn.ModuleList()
        self.rnn_freq_fw = nn.ModuleList()
        self.rnn_freq_bw = nn.ModuleList()
        self.projs = nn.ModuleList()

        for _i in range(self.n_layers):
            self.rnn_time_fw.append(rnn_fn([self.hidden_size,],
                                                    self.hidden_size,
                                                    random_state=random_state,
                                                    init=init,
                                                    scale=scale,
                                                    name=name + "_rnn_time_fw_{}".format(_i),
                                                    device=device))

            self.rnn_time_bw.append(rnn_fn([self.hidden_size,],
                                                    self.hidden_size,
                                                    random_state=random_state,
                                                    init=init,
                                                    scale=scale,
                                                    name=name + "_rnn_time_bw_{}".format(_i),
                                                    device=device))

            self.rnn_freq_fw.append(rnn_fn([self.hidden_size,],
                                                    self.hidden_size,
                                                    random_state=random_state,
                                                    init=init,
                                                    scale=scale,
                                                    name=name + "_rnn_freq_fw_{}".format(_i),
                                                    device=device))

            self.rnn_freq_bw.append(rnn_fn([self.hidden_size,],
                                                    self.hidden_size,
                                                    random_state=random_state,
                                                    init=init,
                                                    scale=scale,
                                                    name=name + "_rnn_freq_bw_{}".format(_i),
                                                    device=device))

            self.projs.append(Linear([4 * self.hidden_size,],
                                      self.hidden_size,
                                      random_state=random_state,
                                      init=init,
                                      scale=scale,
                                      name=name + "_projs_{}".format(_i),
                                      device=device))

    def _time2freq(self, inp):
        inp = inp.reshape((self.n_vert, self.batch_size, self.n_horiz, -1))
        inp = inp.permute(2, 1, 0, 3)
        return inp.reshape((self.n_horiz, self.batch_size * self.n_vert, -1))

    def _freq2time(self, inp):
        # batch size set in forward!
        inp = inp.reshape((self.n_horiz, self.batch_size, self.n_vert, -1))
        inp = inp.permute(2, 1, 0, 3)
        return inp.reshape((self.n_vert, self.batch_size * self.n_horiz, -1))

    def _stack(self, nds, layer):
        freq_rnn_fw_h, _, __ = self.rnn_freq_fw[layer]([nds])
        freq_rnn_bw_h, _, __ = self.rnn_freq_bw[layer]([torch.flip(nds, [0])])
        freq_rnn_h = torch.cat((freq_rnn_fw_h, torch.flip(freq_rnn_bw_h, [0])), dim=-1)
        freq_rnn_h = self._freq2time(freq_rnn_h)

        nds_time = self._freq2time(nds)
        time_rnn_fw_h, _, __ = self.rnn_time_fw[layer]([nds_time])
        time_rnn_bw_h, _, __ = self.rnn_time_bw[layer]([torch.flip(nds_time, [0])])
        time_rnn_h = torch.cat((time_rnn_fw_h, torch.flip(time_rnn_bw_h, [0])), dim=-1)
        combined_h = torch.cat((freq_rnn_h, time_rnn_h), dim=-1)
        res = self.projs[layer]([combined_h])
        res = self._time2freq(res)
        return (0.5 ** 0.5) * (nds + res)

    def forward(self, list_of_inputs):
        # by default embed the inputs, otherwise bypass
        # condidering axis 2 time 3 frequency

        # shift and project the input
        if len(list_of_inputs) > 1:
            raise ValueError("Only support list_of_inputs length 1 for now")

        x = list_of_inputs[0]

        batch_size = x.shape[0]
        self.batch_size = batch_size

        c_x = x.reshape((batch_size * self.n_vert, self.n_horiz, -1))
        c_x = self.input_proj([c_x])
        c_x = c_x.permute(1, 0, 2)
        for _i in range(self.n_layers):
            c_x = self._stack(c_x, _i)
        out = c_x.reshape((self.n_horiz, self.batch_size, self.n_vert, self.hidden_size))
        out = out.permute((1, 2, 0, 3))
        return out


class AttentionMelNetTier(torch.nn.Module):
    def __init__(self,
                 list_of_input_symbol_sizes,
                 n_vert,
                 n_horiz,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 has_centralized_stack=False,
                 has_spatial_condition=False,
                 has_attention=False,
                 cell_type="lstm",
                 attention_type="sigmoid_logistic",
                 attention_mixture_components=10,
                 conditional_layers=2,
                 init=None, scale="default",
                 biases=True, bias_offset=0.,
                 name=None,
                 cell_dropout=1.,
                 use_centralized_stack=False,
                 random_state=None, strict=None,
                 dtype="default", device="default"):
        super(AttentionMelNetTier, self).__init__()
        if name is None:
            name = _get_name()

        if random_state is None:
            raise ValueError("Must pass instance of np.random.RandomState!")

        if strict is None:
            strict = get_strict_mode_default()

        self.input_symbols = list_of_input_symbol_sizes[0]
        self.cell_type = cell_type
        self.hidden_size = hidden_dim
        self.output_size = output_dim
        self.has_centralized_stack = has_centralized_stack
        self.has_spatial_condition = has_spatial_condition
        self.attention_mixture_components = attention_mixture_components
        self.has_attention = has_attention
        self.attention_type = attention_type
        self.conditional_layers = conditional_layers

        self.cell_dropout = cell_dropout
        self.n_layers = n_layers
        self.n_vert = n_vert
        self.n_horiz = n_horiz
        self.output_dim = output_dim

        if self.cell_type == "lstm":
            bidir_rnn_fn = BiLSTMLayer
            rnn_fn = LSTMLayer
            rnn_cell_fn = LSTMCell
        elif self.cell_type == "gru":
            bidir_rnn_fn = BiGRULayer
            rnn_fn = GRULayer
            rnn_cell_fn = GRUCell
        else:
            raise ValueError("Unknown cell_type, self.cell_type was {}".format(self.cell_type))

        #self.embed_td = Embedding(self.input_symbols, self.hidden_size, random_state=random_state, name=name + "_embed_td")
        #self.embed_fd = Embedding(self.input_symbols, self.hidden_size, random_state=random_state, name=name + "_embed_fd")
        self.td_input_proj = Linear([1],
                                             self.hidden_size,
                                             random_state=random_state,
                                             init=init,
                                             scale=scale,
                                             name=name + "_td_input_proj",
                                             device=device)
        self.fd_input_proj = Linear([1],
                                             self.hidden_size,
                                             random_state=random_state,
                                             init=init,
                                             scale=scale,
                                             name=name + "_fd_input_proj",
                                             device=device)

        self.tds_lstms_time_fw = nn.ModuleList()
        self.tds_lstms_freq_fw = nn.ModuleList()
        self.tds_lstms_freq_bw = nn.ModuleList()
        self.tds_projs = nn.ModuleList()
        self.fds_projs = nn.ModuleList()
        self.fds_lstms_freq_fw = nn.ModuleList()
        if self.has_centralized_stack:
            self.centralized_input_proj = Linear([self.n_horiz],
                                                 self.hidden_size,
                                                 random_state=random_state,
                                                 init=init,
                                                 scale=scale,
                                                 name=name + "_centralized_input_proj",
                                                 device=device)
            self.cds_centralized_lstms = nn.ModuleList()
            self.cds_projs = nn.ModuleList()

        if self.has_spatial_condition:
            self.cond_mn = MelNetFullContextSubTier([1], n_vert, n_horiz, self.hidden_size, self.conditional_layers,
                                       random_state=random_state,
                                       init=init,
                                       cell_type=self.cell_type,
                                       name=name + "cond_mn",
                                       device=device)

        if self.has_centralized_stack:
            if self.has_attention:
                self.attn_lstm_cell = rnn_cell_fn([self.hidden_size + self.hidden_size],
                                                   self.hidden_size,
                                                   random_state=random_state,
                                                   init=init,
                                                   scale=scale,
                                                   name=name + "_attn_rnn_cell",
                                                   device=device)
                """
                self.attn_reduction_proj = Linear([self.hidden_size,],
                                                  self.n_horiz,
                                                  random_state=random_state,
                                                  init=init,
                                                  scale=scale,
                                                  name=name + "_attention_reduction_proj",
                                                  device=device)
                """
                if self.attention_type == "logistic":
                    # for logistic sigmoid attention ala melnet
                    self.attn_proj = Linear([3 * self.hidden_size],
                                             3 * self.attention_mixture_components,
                                             random_state=random_state,
                                             init=init,
                                             scale=scale,
                                             name=name + "_attn_proj",
                                             device=device)
                elif self.attention_type == "sigmoid_logistic":
                    self.attn_proj = Linear([3 * self.hidden_size],
                                             3 * self.attention_mixture_components,
                                             random_state=random_state,
                                             init=init,
                                             scale=scale,
                                             name=name + "_attn_proj",
                                             device=device)
                elif self.attention_type == "sigmoid_logistic_alt":
                    self.attn_proj = Linear([3 * self.hidden_size],
                                             3 * self.attention_mixture_components,
                                             random_state=random_state,
                                             init=init,
                                             scale=scale,
                                             name=name + "_attn_proj",
                                             device=device)
                elif self.attention_type == "gaussian":
                    # for logistic sigmoid attention ala melnet
                    self.attn_proj = Linear([3 * self.hidden_size],
                                             3 * self.attention_mixture_components,
                                             random_state=random_state,
                                             init=init,
                                             scale=scale,
                                             name=name + "_attn_proj",
                                             device=device)
                elif self.attention_type == "dca":
                    # for convolutional attention
                    # https://github.com/bshall/Tacotron/blob/6fee34a7c3a9d4ceb9215ed3063771a9287010e1/tacotron/model.py
                    prior_length = 11
                    alpha = .1
                    beta = .9
                    # expected average step size (encoder steps per decoder step) is 1 here
                    # alpha * prior_length / (alpha + beta)
                    # makes sense because of the extreme downsampling
                    # roughly on 1 to 1 scale but may need to be tuned
                    # realistically more like .8 encoder steps per decoder steps on datasets
                    P = betabinom.pmf(np.arange(prior_length), prior_length - 1, alpha, beta)
                    self.register_buffer("P", torch.FloatTensor(P).flip(0))
                    # note - W_g in paper
                    self.W_g = Linear([self.hidden_size],
                                     self.hidden_size,
                                     random_state=random_state,
                                     init=init,
                                     scale=scale,
                                     name=name + "_attn_W_term",
                                     device=device)
                    dynamic_channels = 8
                    dynamic_kernel_size = 21
                    # note - V_g in paper
                    self.V_g = Linear([self.hidden_size],
                                     dynamic_channels * dynamic_kernel_size,
                                     random_state=random_state,
                                     init=init,
                                     scale=scale,
                                     name=name + "_attn_V_term",
                                     biases=False,
                                     device=device)
                    static_channels = 8
                    static_kernel_size = 21
                    self.attn_prior_length = prior_length
                    self.attn_alpha = alpha
                    self.attn_beta = beta
                    self.attn_dynamic_channels = dynamic_channels
                    self.attn_dynamic_kernel_size = dynamic_kernel_size
                    self.attn_static_channels = static_channels
                    self.attn_static_kernel_size = static_kernel_size
                    self.F = Conv1d([1], static_channels, kernel_size=(static_kernel_size, static_kernel_size), random_state=random_state,
                                    name=name + "_attn_F_term", biases=False, device=device)
                    self.U = Linear([static_channels],
                                     self.hidden_size,
                                     random_state=random_state,
                                     init=init,
                                     scale=scale,
                                     name=name + "_attn_U_term",
                                     biases=False,
                                     device=device)
                    self.T = Linear([dynamic_channels],
                                     self.hidden_size,
                                     random_state=random_state,
                                     init=init,
                                     scale=scale,
                                     name=name + "_attn_T_term",
                                     device=device)
                    self.v = Linear([self.hidden_size],
                                     1,
                                     random_state=random_state,
                                     init=init,
                                     scale=scale,
                                     name=name + "_attn_v_term",
                                     biases=False,
                                     device=device)
                elif self.attention_type == "lsa":
                    # https://github.com/NVIDIA/tacotron2/blob/master/model.py#L29
                    # hardcode temporarily
                    attention_dim = 128
                    self.query_layer = Linear([self.hidden_size],
                                               attention_dim,
                                               random_state=random_state,
                                               init=init,
                                               scale=scale,
                                               name=name + "_attn_query_term",
                                               biases=False,
                                               device=device)
                    # unused?
                    self.memory_layer = Linear([self.hidden_size],
                                               attention_dim,
                                               random_state=random_state,
                                               init=init,
                                               scale=scale,
                                               name=name + "_attn_memory_term",
                                               biases=False,
                                               device=device)
                    self.v_layer = Linear([attention_dim],
                                           1,
                                           random_state=random_state,
                                           init=init,
                                           scale=scale,
                                           name=name + "_attn_v_term",
                                           biases=False,
                                           device=device)
                    attention_n_filters = 32
                    attention_kernel_size = 31
                    padding = int((attention_kernel_size - 1) / 2)
                    self.location_conv = Conv1d([2], attention_n_filters, kernel_size=(attention_kernel_size, attention_kernel_size),
                                                random_state=random_state,
                                                name=name + "_attn_location_conv", biases=False, device=device)
                    self.location_dense = Linear([attention_n_filters],
                                                  attention_dim,
                                                  random_state=random_state,
                                                  init=init,
                                                  scale=scale,
                                                  name=name + "_attn_location_dense",
                                                  biases=False,
                                                  device=device)
                else:
                    raise ValueError("Unknown value in initialization for attention_type in AttnMelNetTier, got {}".format(attention_type))


        for _i in range(self.n_layers):
            self.tds_lstms_time_fw.append(rnn_fn([self.hidden_size,],
                                                    self.hidden_size,
                                                    random_state=random_state,
                                                    init=init,
                                                    scale=scale,
                                                    name=name + "_tds_rnn_time_fw_{}".format(_i),
                                                    device=device))

            self.tds_lstms_freq_fw.append(rnn_fn([self.hidden_size,],
                                                    self.hidden_size,
                                                    random_state=random_state,
                                                    init=init,
                                                    scale=scale,
                                                    name=name + "_tds_rnn_freq_fw_{}".format(_i),
                                                    device=device))

            self.tds_lstms_freq_bw.append(rnn_fn([self.hidden_size,],
                                                    self.hidden_size,
                                                    random_state=random_state,
                                                    init=init,
                                                    scale=scale,
                                                    name=name + "_tds_rnn_freq_bw_{}".format(_i),
                                                    device=device))

            self.tds_projs.append(Linear([3 * self.hidden_size,],
                                          self.hidden_size,
                                          random_state=random_state,
                                          init=init,
                                          scale=scale,
                                          name=name + "_tds_projs_{}".format(_i),
                                          device=device))

            self.fds_projs.append(Linear([self.hidden_size,],
                                          self.hidden_size,
                                          random_state=random_state,
                                          init=init,
                                          scale=scale,
                                          name=name + "_fds_projs_{}".format(_i),
                                          device=device))

            self.fds_lstms_freq_fw.append(rnn_fn([self.hidden_size,],
                                                    self.hidden_size,
                                                    random_state=random_state,
                                                    init=init,
                                                    scale=scale,
                                                    name=name + "_fds_rnn_freq_fw_{}".format(_i),
                                                    device=device))


            if self.has_centralized_stack:
                self.cds_centralized_lstms.append(rnn_fn([self.hidden_size,],
                                                             self.hidden_size,
                                                             random_state=random_state,
                                                             init=init,
                                                             scale=scale,
                                                             name=name + "_cds_centralized_rnn_{}".format(_i),
                                                             device=device))
                self.cds_projs.append(Linear([self.hidden_size,],
                                              self.hidden_size,
                                              random_state=random_state,
                                              init=init,
                                              scale=scale,
                                              name=name + "_cds_projs_{}".format(_i),
                                              device=device))
        self.out_proj = Linear([self.hidden_size,], self.output_size,
                               random_state=random_state,
                               init=init,
                               scale=scale,
                               name=name + "_output_proj",
                               device=device)
        # used for hook registering alternative softplus for numerical stability
        self._softplus = None

    def _time2freq(self, inp):
        inp = inp.reshape((self.n_vert, self.batch_size, self.n_horiz, -1))
        inp = inp.permute(2, 1, 0, 3)
        return inp.reshape((self.n_horiz, self.batch_size * self.n_vert, -1))

    def _freq2time(self, inp):
        # batch size set in forward!
        inp = inp.reshape((self.n_horiz, self.batch_size, self.n_vert, -1))
        inp = inp.permute(2, 1, 0, 3)
        return inp.reshape((self.n_vert, self.batch_size * self.n_horiz, -1))

    def _td_stack(self, tds, layer):
        freq_lstm_fw_h, _, __ = self.tds_lstms_freq_fw[layer]([tds])
        freq_lstm_bw_h, _, __ = self.tds_lstms_freq_bw[layer]([torch.flip(tds, [0])])
        freq_lstm_h = torch.cat((freq_lstm_fw_h, torch.flip(freq_lstm_bw_h, [0])), dim=-1)
        freq_lstm_h = self._freq2time(freq_lstm_h)

        tds_time = self._freq2time(tds)
        time_lstm_h, _, __ = self.tds_lstms_time_fw[layer]([tds_time])
        combined_h = torch.cat((freq_lstm_h, time_lstm_h), dim=-1)
        res = self.tds_projs[layer]([combined_h])
        res = self._time2freq(res)
        #return (0.5 ** 0.5) * (tds + res)
        return res

    def _cd_centralized_stack(self, cds, layer):
        cent_lstm_h, _, __ = self.cds_centralized_lstms[layer]([cds])
        res = self.cds_projs[layer]([cent_lstm_h])
        #return (0.5 ** 0.5) * (cds + cent_lstm_h)
        return res

    def _fd_stack(self, tds, fds, layer, tds_cent=None):
        # broadcast tds_cent across frequency axis
        if tds_cent is not None:
            # need to permute + reshape to match what was done for td_x, fd_x
            # so that batch combined stuff stays contiguous in the right way!
            # time2freq is freq, batch * time, -1
            # time is also self.n_vert

            # nvert batch feat to batch nvert feat
            tds_cent = tds_cent.permute(1, 0, 2)
            ext_tds_cent = tds_cent.reshape((self.batch_size * self.n_vert, -1))
            # now 1, batch * time, hidden
            # broadcasts over frequency, since the cent rnn has puts out a whole freq frame per step dim...
            ext_tds_cent = ext_tds_cent[None]
            # (fds dim 0)
            # broacasts over features, since the cent rnn has effectively seen the whole frequency
            #ext_tds_cent = ext_tds_cent + 0. * fds
            freq_lstm_stack = tds + fds + ext_tds_cent#  torch.cat((tds, fds, ext_tds_cent), axis=-1)
        else:
            #freq_lstm_stack = torch.cat((tds, fds), axis=-1)
            freq_lstm_stack = tds + fds

        freq_lstm_h, _, __ = self.fds_lstms_freq_fw[layer]([freq_lstm_stack])
        res = self.fds_projs[layer]([freq_lstm_h])
        #return (0.5 ** 0.5) * (fds + freq_lstm_h)
        return res

    def _attention_step(self, h_i, memory, memory_mask, previous_attn):
        # TODO: location sensitive attention
        # https://gist.github.com/acetylSv/9dcff15bc0e895c0190c5942b573c28b
        if self.attention_type == "logistic":
            #_attention_step(self, h_i, memory, memory_mask, ksi):
            # condition on input sequence length
            phi_hat = self.attn_proj([h_i])

            #ksi = ksi + torch.exp(phi_hat[:, :self.attention_mixture_components])
            # clamp beta so it doesn't collapse?
            #beta = torch.exp(torch.clamp(phi_hat[:, self.attention_mixture_components:2 * self.attention_mixture_components], min=-4))
            # changes based on GMMv2 of https://arxiv.org/pdf/1910.10288.pdf
            # as well as https://arxiv.org/pdf/1811.07240.pdf
            ksi = previous_attn
            kappa = ksi + (3. * sigmoid(phi_hat[:, :self.attention_mixture_components]) + .05)

            beta = (5. * sigmoid(phi_hat[:, self.attention_mixture_components:2 * self.attention_mixture_components]) + .1)
            # min beta .1
            # max beta 10
            alpha = F.softmax(phi_hat[:, 2 * self.attention_mixture_components:3 * self.attention_mixture_components], dim=1)

            u = memory.new_tensor(np.arange(memory.size(0)), dtype=memory.dtype)
            u_L = u + 0.5
            u_R = u - 0.5
            #termL = torch.sum(alpha[..., None] * (1. / (1. + (torch.exp((kappa[..., None] - u_L) * beta[..., None])))), keepdim=True, dim=1)
            #termR = torch.sum(alpha[..., None] * (1. / (1. + (torch.exp((kappa[..., None] - u_R) * beta[..., None])))), keepdim=True, dim=1)
            #termL = 1. / (1. + (torch.exp((kappa[..., None] - u_L) * beta[..., None])))
            #termR = 1. / (1. + (torch.exp((kappa[..., None] - u_R) * beta[..., None])))
            pL = (u_L - kappa[..., None]) * beta[..., None]
            pR = (u_R - kappa[..., None]) * beta[..., None]
            termL = torch.sigmoid(pL)
            termR = torch.sigmoid(pR)
            alpha_termL = alpha[..., None] * termL
            alpha_termR = alpha[..., None] * termR
            weights = torch.sum(alpha_termL, dim=1) - torch.sum(alpha_termR, dim=1)

            termination = 1. - torch.sum(alpha_termL, keepdim=True, dim=1)[:, 0]
            weights = memory_mask.transpose(0, 1) * weights
            context = torch.bmm(weights[:, None], memory.permute(1, 0, 2))
            # context is B, 1, D
            # weights B, mem_T 
            extras = {}
            extras["termination"] = termination
            extras["kappa"] = kappa
            extras["beta"] = beta
        elif self.attention_type == "logistic_hack":
            #_attention_step(self, h_i, memory, memory_mask, ksi):
            # condition on input sequence length
            phi_hat = self.attn_proj([h_i])

            #ksi = ksi + torch.exp(phi_hat[:, :self.attention_mixture_components])
            # clamp beta so it doesn't collapse?
            #beta = torch.exp(torch.clamp(phi_hat[:, self.attention_mixture_components:2 * self.attention_mixture_components], min=-4))
            # changes based on GMMv2 of https://arxiv.org/pdf/1910.10288.pdf
            # as well as https://arxiv.org/pdf/1811.07240.pdf
            ksi = previous_attn
            kappa = ksi + F.softplus(phi_hat[:, :self.attention_mixture_components])
            #beta = (F.softplus(phi_hat[:, self.attention_mixture_components:2 * self.attention_mixture_components]) + 1E-4)
            # cap beta
            beta = torch.exp(phi_hat[:, self.attention_mixture_components:(2 * self.attention_mixture_components)]) + 1E-2

            logit_alpha = phi_hat[:, (2 * self.attention_mixture_components):(3 * self.attention_mixture_components)]

            #alpha = F.softmax(phi_hat[:, 2 * self.attention_mixture_components:3 * self.attention_mixture_components], dim=1)

            u = memory.new_tensor(np.arange(memory.size(0)), dtype=memory.dtype)
            u_L = u + 0.5
            u_R = u - 0.5
            #u_L = u + 1.5
            #u_R = u + 0.5
            #termL = torch.sum(alpha[..., None] * (1. / (1. + (torch.exp((kappa[..., None] - u_L) * beta[..., None])))), keepdim=True, dim=1)
            #termR = torch.sum(alpha[..., None] * (1. / (1. + (torch.exp((kappa[..., None] - u_R) * beta[..., None])))), keepdim=True, dim=1)
            # could rewrite further by taking softmax out of context
            # softmax(alpha) * exp(-softplus(a)) -> exp(alpha) * exp(-softplus(a)) / sum(exp(alpha)) -> exp(alpha - softplus(a)) / sum(exp(alpha))
            # but would probably be less stable than "stable" softmax due to sum(exp) in denominator
            # softplus(a) = log(1 + exp(a))
            # with a = ((ksi - u) / beta)
            # this overall becomes
            # 1. / exp(softplus(a)) -> exp(-softplus(a)) -> exp(-log(1 + exp(a))) -> 1./(1 + exp(a))
            #termL = torch.sum(alpha[..., None] * (torch.exp(-F.softplus(((kappa[..., None] - u_L)) * beta[..., None]))), keepdim=True, dim=1)
            #termR = torch.sum(alpha[..., None] * (torch.exp(-F.softplus(((kappa[..., None] - u_R)) * beta[..., None]))), keepdim=True, dim=1)

            # possible best derivation (inspired by https://github.com/Deepest-Project/MelNet/blob/master/model/tts.py#L138 although I don't get the 1.5 and .5 instead of -.5 and +.5, perhaps this is setting the default step size to 1. for all ksi?)
            # 1. / (1. + exp((k-u) / b)) -> 1. / (1. + exp(-(u - k) / b)) -> 1. / (1. + exp(-t)), where t is (u - k) / b
            # knowing that sigmoid(t) = 1. / (1. + exp(-t))
            # this results in sigmoid((u - k) / b) for each term
            # this means both terms L and R are bounded btwn 0 and 1, and potentially more stable than exp(-softplus) shenanigans would allow
            #termL = torch.sum(alpha[..., None] * torch.sigmoid((((u_L - kappa[..., None])) * beta[..., None])), keepdim=True, dim=1)
            #termR = torch.sum(alpha[..., None] * torch.sigmoid((((u_R - kappa[..., None])) * beta[..., None])), keepdim=True, dim=1)
            # finally since beta is bounded > 0 due to exp() activation, we note that dividing by beta and multiplying by beta are effectively the same
            # in terms of optimization paths
            # simply swapping "what regime" wrt values exp(x) < 1, and values exp(x) > 1
            # with the *key* difference being a vanishing to 0 of beta (perhaps due to very negative weights for beta or other issues during training), will not explode the whole equation
            # reweight in log space before summation?

            #termL = torch.sum(alpha[..., None] * torch.sigmoid((((u_L - kappa[..., None])) * beta[..., None])), keepdim=True, dim=1)
            #termR = torch.sum(alpha[..., None] * torch.sigmoid((((u_R - kappa[..., None])) * beta[..., None])), keepdim=True, dim=1)

            # combined term
            #termL_R = torch.sum(alpha[..., None] * (torch.sigmoid((((u_L - kappa[..., None])) * beta[..., None])) - torch.sigmoid((((u_R - kappa[..., None])) * beta[..., None]))), keepdim=True, dim=1)
            #weights = termL_R

            # introduce sum(exp(log(alpha * sigmoid))) -> sum(exp(log(alpha) + log(sigmoid)))
            # log(alpha) -> log_softmax
            # logsoftmax = logits - log(reduce_sum(exp(logits), dim))
            # log(sigmoid(q)) -> q - log(exp(q) + 1) aka q - softplus(q)
            # term = log_alpha + q - softplus(q)
            # https://math.stackexchange.com/questions/2320905/obtaining-derivative-of-log-of-sigmoid-function
            #go further beyond, do multiplication in log space (so additive)
            # then sum exps afterward
            #log_alpha = logit_alpha - torch.log(torch.sum(torch.exp(logit_alpha), keepdim=True, dim=1))
            #log_alpha = log_alpha[..., None]
            #q_L = (u_L - kappa[..., None]) * beta[..., None]
            #termL = torch.sum(torch.exp(log_alpha + q_L - F.softplus(q_L)), keepdim=True, dim=1)
            #q_R = (u_R - kappa[..., None]) * beta[..., None]
            #termR = torch.sum(torch.exp(log_alpha + q_R - F.softplus(q_R)), keepdim=True, dim=1)
            #weights = termL - termR

            # even more further beyond...
            log_alpha = log_prob_from_logits(logit_alpha, axis=1)
            log_alpha = log_alpha[..., None]
            q_L = (u_L - kappa[..., None]) * beta[..., None]
            # keep dims
            termL = torch.exp(log_sum_exp(log_alpha + q_L - F.softplus(q_L), axis=1))[:, None]
            q_R = (u_R - kappa[..., None]) * beta[..., None]
            # keep dims
            termR = torch.exp(log_sum_exp(log_alpha + q_R - F.softplus(q_R), axis=1))[:, None]
            weights = termL - termR

            termination = 1. - termL[:, 0]
            weights = memory_mask.transpose(0, 1)[:, None, :] * weights
            context = torch.bmm(weights, memory.permute(1, 0, 2))
            # context is B, 1, D
            # weights B, mem_T 
            extras = {}
            extras["termination"] = termination
            extras["kappa"] = kappa
            extras["beta"] = beta
        elif self.attention_type == "sigmoid_logistic":
            #_attention_step(self, h_i, memory, memory_mask, ksi):
            # condition on input sequence length
            phi_hat = self.attn_proj([h_i])
            # cast to 32 bit
            orig_dtype = phi_hat.dtype
            phi_hat = phi_hat.float()

            #ksi = ksi + torch.exp(phi_hat[:, :self.attention_mixture_components])
            # clamp beta so it doesn't collapse?
            #beta = torch.exp(torch.clamp(phi_hat[:, self.attention_mixture_components:2 * self.attention_mixture_components], min=-4))
            # changes based on GMMv2 of https://arxiv.org/pdf/1910.10288.pdf
            # as well as https://arxiv.org/pdf/1811.07240.pdf
            ksi = previous_attn
            ksi = ksi.float()

            """
            if self._softplus is None:
                # hook it into the main module to keep a reference around
                self._softplus = torch.nn.Softplus()
                '''
                     output z  (grad_output)
                     ___________
                     |         |
                     |  layer  |
                     |_________|

                     input x  (grad_input)
                '''
                def hook(module, grad_input, grad_output):
                    return (sigmoid(grad_output[0]),)

                self._softplus.register_backward_hook(hook)

            alt_softplus = self._softplus
            """

            def alt_log1p(x):
                # https://www.johndcook.com/blog/2012/07/25/trick-for-computing-log1x/
                # we dirty hack this to avoid div by 0 since torch.where has issues with NaN grad
                # if *either* branch has inf
                # https://github.com/pytorch/pytorch/issues/4132
                y = 1. + x
                z = y - 1.
                res = 0. * x
                z_mask = (z == 0)
                res[z_mask] = x[z_mask]
                z_nonmask = (z != 0)
                res[z_nonmask] = x[z_nonmask] * torch.log(y[z_nonmask]) / (z[z_nonmask])
                #return torch.where(z == 0, x, x * torch.log(y) / z)
                return res

            def alt_softplus(x):
                # https://stackoverflow.com/questions/44230635/avoid-overflow-with-softplus-function-in-python
                return alt_log1p(torch.exp(-torch.abs(x))) + torch.where(x > 0, x, 0. * x)

            kappa = ksi + alt_softplus(phi_hat[:, :self.attention_mixture_components]) #+ 1E-2
            #kappa = ksi + swish(phi_hat[:, :self.attention_mixture_components])

            #beta = (F.softplus(phi_hat[:, self.attention_mixture_components:2 * self.attention_mixture_components]) + 1E-4)
            #beta = torch.clamp(torch.exp(phi_hat[:, self.attention_mixture_components:2 * self.attention_mixture_components]), min=.5)
            # fix beta to 1 - hack city but works!
            #beta = 0. * phi_hat[:, self.attention_mixture_components:2 * self.attention_mixture_components] + 1.

            # aggressive clamping here, use softplus to help stability as well
            #beta = torch.clamp(F.softplus(phi_hat[:, self.attention_mixture_components:2 * self.attention_mixture_components]), min=.25, max=2.0)
            #beta = torch.clamp(F.softplus(phi_hat[:, self.attention_mixture_components:2 * self.attention_mixture_components]), min=.01, max=10.0)
            #beta = F.softplus(phi_hat[:, self.attention_mixture_components:2 * self.attention_mixture_components])

            beta = alt_softplus(phi_hat[:, self.attention_mixture_components:2 * self.attention_mixture_components] + 3.)
            # add constant 3 to beta, so the default for beta is "large" at init
            # model can learn biases to overcome this if it wants small beta

            #beta = swish(phi_hat[:, self.attention_mixture_components:2 * self.attention_mixture_components])
            #beta = swish(phi_hat[:, self.attention_mixture_components:2 * self.attention_mixture_components])

            alpha = F.softmax(phi_hat[:, 2 * self.attention_mixture_components:3 * self.attention_mixture_components], dim=1)
            #alpha = F.softplus(phi_hat[:, 2 * self.attention_mixture_components:3 * self.attention_mixture_components]) + 1E-2
            #alpha = F.log_softmax(phi_hat[:, 2 * self.attention_mixture_components:3 * self.attention_mixture_components], dim=1)

            u = memory.new_tensor(np.arange(memory.size(0)), dtype=memory.dtype)
            #u_L = u + 0.5
            #u_R = u - 0.5

            # try like this for the following reason...
            # eqn 22 of paper
            # F(u + .5; g) - F(u - 0.5; g)
            # means F(u; g) = 1./(1 + exp((k - u) / B))
            # we can interpret this as either SUBTITUTING u + 0.5 for u, or simply adding/subtracting on 0.5. This would swap the 2 terms
            # interpreting as simply adding 0.5 to the end means
            # sigm(x) = 1./(1+exp(-x))
            # x = ((-k + u) / B) = ((u - k) / B)
            # sigm((u - .5 - k) / B) as the left hand
            # sigm((u + .5 - k) / B) as the right hand
            # alternately, interpreting as substituting u - .5 for u
            # sigm((u + .5 - k) / B) as the left hand
            # sigm((u - .5 - k) / B) as the right hand
            # noting that we can multiply or divide by beta, if beta is constrained from 0 to inf
            # since / by a number from 0->1 is the same as multiplying by 1->inf
            # aka beta can parameterize the division term, or 1 / division
            # parameterizing the 1 / division means that we don't face edge cases for beta near 0, as 1/inf -> 0 and 1/0. -> inf
            u_L = u + 0.5
            u_R = u - 0.5

            """
            alternative?
            TANH(t) = [1 - exp(-2t)]/[1 + exp(-2t)]  for  t>=0
            and
            TANH(t) = [exp(2t) - 1]/[exp(2t) + 1] for t<0
            """

            # we approximate tanh with x/(1+abs(x))
            def alt_tanh(x):
                return x / (1. + torch.abs(x))
            # logistic can be expressed as 1/2 + 1/2 * tanh((x - u)/(2 * s)) instead of sigmoid
            # if beta is 1/s, this is .5 * beta
            def term(u, k, b):
                return .5 + .5 * alt_tanh((u - k) * .5 * b)

            #termL = torch.sum(alpha[..., None] * torch.sigmoid((((u_L - kappa[..., None])) * beta[..., None])), keepdim=True, dim=1)
            #termR = torch.sum(alpha[..., None] * torch.sigmoid((((u_R - kappa[..., None])) * beta[..., None])), keepdim=True, dim=1)

            termL = torch.sum(alpha[..., None] * term(u_L, kappa[..., None], beta[..., None]), keepdim=True, dim=1)
            #termR = torch.sum(alpha[..., None] * term(u_R, kappa[..., None], beta[..., None]), keepdim=True, dim=1)

            diff = (term(u_L, kappa[..., None], beta[..., None]) - term(u_R, kappa[..., None], beta[..., None]))
            weights = torch.sum(alpha[..., None] * diff, keepdim=True, dim=1)

            #termL = torch.logsumexp(alpha[..., None] + torch.nn.functional.logsigmoid((((u_L - kappa[..., None])) * beta[..., None])), keepdim=True, dim=1)
            #termR = torch.logsumexp(alpha[..., None] + torch.nn.functional.logsigmoid((((u_R - kappa[..., None])) * beta[..., None])), keepdim=True, dim=1)
            #termR_mask = torch.abs(termR < 1).type(termR.dtype)
            #termR = termR * termR_mask + 1 * (1. - termR_mask)

            #weights = termL - termR

            #weights = torch.exp(termL) - torch.exp(termR)
            #weights = torch.exp(termL / termR)

            #termL = alpha[..., None] * torch.sigmoid((((u_L - kappa[..., None])) * beta[..., None]))
            #termR = alpha[..., None] * torch.sigmoid((((u_R - kappa[..., None])) * beta[..., None]))
            #termL = torch.sigmoid((((u_L - kappa[..., None])) * beta[..., None]))
            #termR = torch.sigmoid((((u_R - kappa[..., None])) * beta[..., None]))

            #weights = torch.sum(alpha[..., None] * (termL - termR), keepdim=True, dim=1)

            termination = 1. - torch.exp(termL[:, 0])
            weights = memory_mask.transpose(0, 1)[:, None, :] * weights

            weights = weights.to(orig_dtype)
            kappa = kappa.to(orig_dtype)
            beta = beta.to(orig_dtype)
            alpha = alpha.to(orig_dtype)
            termination = termination.to(orig_dtype)

            context = torch.bmm(weights, memory.permute(1, 0, 2))
            # context is B, 1, D
            # weights B, 1, mem_T 
            extras = {}
            extras["termination"] = termination
            extras["kappa"] = kappa
            extras["beta"] = beta
            extras["alpha"] = alpha
        elif self.attention_type == "sigmoid_logistic_alt":
            #_attention_step(self, h_i, memory, memory_mask, ksi):
            # condition on input sequence length
            phi_hat = self.attn_proj([h_i])
            # cast to 32 bit
            orig_dtype = phi_hat.dtype
            phi_hat = phi_hat.float()

            #ksi = ksi + torch.exp(phi_hat[:, :self.attention_mixture_components])
            # clamp beta so it doesn't collapse?
            #beta = torch.exp(torch.clamp(phi_hat[:, self.attention_mixture_components:2 * self.attention_mixture_components], min=-4))
            # changes based on GMMv2 of https://arxiv.org/pdf/1910.10288.pdf
            # as well as https://arxiv.org/pdf/1811.07240.pdf
            ksi = previous_attn
            ksi = ksi.float()

            """
            if self._softplus is None:
                # hook it into the main module to keep a reference around
                self._softplus = torch.nn.Softplus()
                '''
                     output z  (grad_output)
                     ___________
                     |         |
                     |  layer  |
                     |_________|

                     input x  (grad_input)
                '''
                def hook(module, grad_input, grad_output):
                    return (sigmoid(grad_output[0]),)

                self._softplus.register_backward_hook(hook)

            alt_softplus = self._softplus
            """

            def alt_log1p(x):
                # https://www.johndcook.com/blog/2012/07/25/trick-for-computing-log1x/
                # we dirty hack this to avoid div by 0 since torch.where has issues with NaN grad
                # if *either* branch has inf
                # https://github.com/pytorch/pytorch/issues/4132
                y = 1. + x
                z = y - 1.
                res = 0. * x
                z_mask = (z == 0)
                res[z_mask] = x[z_mask]
                z_nonmask = (z != 0)
                res[z_nonmask] = x[z_nonmask] * torch.log(y[z_nonmask]) / (z[z_nonmask])
                #return torch.where(z == 0, x, x * torch.log(y) / z)
                return res

            def alt_softplus(x):
                # https://stackoverflow.com/questions/44230635/avoid-overflow-with-softplus-function-in-python
                return alt_log1p(torch.exp(-torch.abs(x))) + torch.where(x > 0, x, 0. * x)

            kappa = ksi + alt_softplus(phi_hat[:, :self.attention_mixture_components]) + 1E-3
            #kappa = ksi + swish(phi_hat[:, :self.attention_mixture_components])

            #beta = (F.softplus(phi_hat[:, self.attention_mixture_components:2 * self.attention_mixture_components]) + 1E-4)
            #beta = torch.clamp(torch.exp(phi_hat[:, self.attention_mixture_components:2 * self.attention_mixture_components]), min=.5)
            # fix beta to 1 - hack city but works!
            #beta = 0. * phi_hat[:, self.attention_mixture_components:2 * self.attention_mixture_components] + 1.

            # aggressive clamping here, use softplus to help stability as well
            #beta = torch.clamp(F.softplus(phi_hat[:, self.attention_mixture_components:2 * self.attention_mixture_components]), min=.25, max=2.0)
            #beta = torch.clamp(F.softplus(phi_hat[:, self.attention_mixture_components:2 * self.attention_mixture_components]), min=.01, max=10.0)
            #beta = F.softplus(phi_hat[:, self.attention_mixture_components:2 * self.attention_mixture_components])

            beta = alt_softplus(phi_hat[:, self.attention_mixture_components:2 * self.attention_mixture_components])

            #beta = swish(phi_hat[:, self.attention_mixture_components:2 * self.attention_mixture_components])
            #beta = swish(phi_hat[:, self.attention_mixture_components:2 * self.attention_mixture_components])

            alpha = F.softmax(phi_hat[:, 2 * self.attention_mixture_components:3 * self.attention_mixture_components], dim=1)
            #alpha = F.softplus(phi_hat[:, 2 * self.attention_mixture_components:3 * self.attention_mixture_components]) + 1E-2
            #alpha = F.log_softmax(phi_hat[:, 2 * self.attention_mixture_components:3 * self.attention_mixture_components], dim=1)

            u = memory.new_tensor(np.arange(memory.size(0)), dtype=memory.dtype)
            #u_L = u + 0.5
            #u_R = u - 0.5

            # try like this for the following reason...
            # eqn 22 of paper
            # F(u + .5; g) - F(u - 0.5; g)
            # means F(u; g) = 1./(1 + exp((k - u) / B))
            # we can interpret this as either SUBTITUTING u + 0.5 for u, or simply adding/subtracting on 0.5. This would swap the 2 terms
            # interpreting as simply adding 0.5 to the end means
            # sigm(x) = 1./(1+exp(-x))
            # x = ((-k + u) / B) = ((u - k) / B)
            # sigm((u - .5 - k) / B) as the left hand
            # sigm((u + .5 - k) / B) as the right hand
            # alternately, interpreting as substituting u - .5 for u
            # sigm((u + .5 - k) / B) as the left hand
            # sigm((u - .5 - k) / B) as the right hand
            # noting that we can multiply or divide by beta, if beta is constrained from 0 to inf
            # since / by a number from 0->1 is the same as multiplying by 1->inf
            # aka beta can parameterize the division term, or 1 / division
            # parameterizing the 1 / division means that we don't face edge cases for beta near 0, as 1/inf -> 0 and 1/0. -> inf
            # however, it means that making small beta we have less precision due to floating point
            u_L = u + 0.5
            u_R = u - 0.5

            """
            alternative?
            TANH(t) = [1 - exp(-2t)]/[1 + exp(-2t)]  for  t>=0
            and
            TANH(t) = [exp(2t) - 1]/[exp(2t) + 1] for t<0
            """

            # we approximate tanh with x/(1+abs(x))
            # https://stackoverflow.com/questions/10732027/fast-sigmoid-algorithm
            # see comment about conversion to / from sigmoid
            def alt_tanh(x):
                return x / (1. + torch.abs(x))
            # logistic can be expressed as 1/2 + 1/2 * tanh((x - u)/(2 * s)) instead of sigmoid
            # if beta is 1/s, this is .5 * beta
            def term(u, k, b):
                #return .5 + .5 * alt_tanh(.5 * (u - k) / b)
                # limit min beta to .01
                return .5 + .5 * alt_tanh(.5 * (u - k) * (b + .01))

            #termL = torch.sum(alpha[..., None] * torch.sigmoid((((u_L - kappa[..., None])) * beta[..., None])), keepdim=True, dim=1)
            #termR = torch.sum(alpha[..., None] * torch.sigmoid((((u_R - kappa[..., None])) * beta[..., None])), keepdim=True, dim=1)

            termL = torch.sum(alpha[..., None] * term(u_L, kappa[..., None], beta[..., None]), keepdim=True, dim=1)
            #termR = torch.sum(alpha[..., None] * term(u_R, kappa[..., None], beta[..., None]), keepdim=True, dim=1)

            diff = (term(u_L, kappa[..., None], beta[..., None]) - term(u_R, kappa[..., None], beta[..., None]))
            weights = torch.sum(alpha[..., None] * diff, keepdim=True, dim=1)

            #termL = torch.logsumexp(alpha[..., None] + torch.nn.functional.logsigmoid((((u_L - kappa[..., None])) * beta[..., None])), keepdim=True, dim=1)
            #termR = torch.logsumexp(alpha[..., None] + torch.nn.functional.logsigmoid((((u_R - kappa[..., None])) * beta[..., None])), keepdim=True, dim=1)
            #termR_mask = torch.abs(termR < 1).type(termR.dtype)
            #termR = termR * termR_mask + 1 * (1. - termR_mask)

            #weights = termL - termR

            #weights = torch.exp(termL) - torch.exp(termR)
            #weights = torch.exp(termL / termR)

            #termL = alpha[..., None] * torch.sigmoid((((u_L - kappa[..., None])) * beta[..., None]))
            #termR = alpha[..., None] * torch.sigmoid((((u_R - kappa[..., None])) * beta[..., None]))
            #termL = torch.sigmoid((((u_L - kappa[..., None])) * beta[..., None]))
            #termR = torch.sigmoid((((u_R - kappa[..., None])) * beta[..., None]))

            #weights = torch.sum(alpha[..., None] * (termL - termR), keepdim=True, dim=1)

            termination = 1. - torch.exp(termL[:, 0])
            weights = memory_mask.transpose(0, 1)[:, None, :] * weights

            # grad scaling here
            grad_rescale = 1. / np.sqrt(self.attention_mixture_components)
            weights = (1. - grad_rescale) * weights.detach() + grad_rescale * weights

            weights = weights.to(orig_dtype)
            kappa = kappa.to(orig_dtype)
            beta = beta.to(orig_dtype)
            alpha = alpha.to(orig_dtype)
            termination = termination.to(orig_dtype)

            context = torch.bmm(weights, memory.permute(1, 0, 2))
            # context is B, 1, D
            # weights B, 1, mem_T 
            extras = {}
            extras["termination"] = termination
            extras["kappa"] = kappa
            extras["beta"] = beta
            extras["alpha"] = alpha
        elif self.attention_type == "dca":
            # code ref:
            # https://github.com/bshall/Tacotron
            p = F.conv1d(F.pad(previous_attn[:, None], (self.attn_prior_length - 1, 0)), self.P[None, None])
            p = torch.log(p.clamp_min_(1E-6)[:, 0])

            G = self.V_g([self.W_g([h_i])])

            g = F.conv1d(previous_attn[None], G.view(-1, 1, self.attn_dynamic_kernel_size),
                         padding=(self.attn_dynamic_kernel_size - 1) // 2,
                         groups=h_i.shape[0])
            g = g.view(h_i.size(0), self.attn_dynamic_channels, -1).transpose(1, 2)

            f = self.F([previous_attn.transpose(1, 0)[..., None]])
            e = self.v([torch.tanh(self.U([f]) + self.T([g]).transpose(0, 1))])[..., 0]

            e = e.transpose(1, 0) + p
            # now B, mem_T
            weights = F.softmax(e, dim=1)

            # mask weights here
            # technically don't sum to 1 anymore but don't want to add weight to zero info places...
            # for now, don't mask
            #weights = memory_mask.transpose(0, 1) * weights

            context = torch.bmm(weights[:, None], memory.permute(1, 0, 2))
            # context is B, 1, D
            # TODO: fix the dims to match, should be weights B, 1, mem_T 
            # weights B, mem_T 
            extras = {}
        elif self.attention_type == "lsa":
            processed_query = self.query_layer([h_i])
            processed_memory = self.memory_layer([memory.permute(1, 0, 2)])
            processed_attention = self.location_conv([previous_attn.transpose(1, 0)])
            processed_attention_dense = self.location_dense([processed_attention]).transpose(1, 0)
            # processed_attention_dense is batch, mem_T, attn_dim
            weight_logits = self.v_layer([torch.tanh(processed_attention_dense + processed_memory + processed_query[:, None])])[..., 0]
            weights = F.softmax(weight_logits, dim=1)
            context = torch.bmm(weights[:, None], memory.permute(1, 0, 2))
            weights = weights[:, None]
            # context is B, 1, D
            # weights B, 1, mem_T 
            extras = {}
        elif self.attention_type == "gaussian":
            #_attention_step(self, h_i, memory, memory_mask, ksi):
            # condition on input sequence length
            phi_hat = self.attn_proj([h_i])

            #ksi = ksi + torch.exp(phi_hat[:, :self.attention_mixture_components])
            # clamp beta so it doesn't collapse?
            #beta = torch.exp(torch.clamp(phi_hat[:, self.attention_mixture_components:2 * self.attention_mixture_components], min=-4))
            # changes based on GMMv2 of https://arxiv.org/pdf/1910.10288.pdf
            # as well as https://arxiv.org/pdf/1811.07240.pdf
            ksi = previous_attn
            kappa = ksi + F.softplus(phi_hat[:, :self.attention_mixture_components])

            # don't need capped beta becase we parameterize the inverse
            beta = torch.exp(phi_hat[:, self.attention_mixture_components:2 * self.attention_mixture_components])
            alpha = torch.exp(phi_hat[:, 2 * self.attention_mixture_components:3 * self.attention_mixture_components])

            u = memory.new_tensor(np.arange(memory.size(0)), dtype=memory.dtype)
            #u_L = u + 0.5
            #u_R = u - 0.5
            #u_L = u + 1.5
            #u_R = u + 0.5
            #termL = torch.sum(alpha[..., None] * (1. / (1. + (torch.exp((kappa[..., None] - u_L) * beta[..., None])))), keepdim=True, dim=1)
            #termR = torch.sum(alpha[..., None] * (1. / (1. + (torch.exp((kappa[..., None] - u_R) * beta[..., None])))), keepdim=True, dim=1)
            # could rewrite further by taking softmax out of context
            # softmax(alpha) * exp(-softplus(a)) -> exp(alpha) * exp(-softplus(a)) / sum(exp(alpha)) -> exp(alpha - softplus(a)) / sum(exp(alpha))
            # but would probably be less stable than "stable" softmax due to sum(exp) in denominator
            # softplus(a) = log(1 + exp(a))
            # with a = ((ksi - u) / beta)
            # this overall becomes
            # 1. / exp(softplus(a)) -> exp(-softplus(a)) -> exp(-log(1 + exp(a))) -> 1./(1 + exp(a))
            #termL = torch.sum(alpha[..., None] * (torch.exp(-F.softplus(((kappa[..., None] - u_L)) * beta[..., None]))), keepdim=True, dim=1)
            #termR = torch.sum(alpha[..., None] * (torch.exp(-F.softplus(((kappa[..., None] - u_R)) * beta[..., None]))), keepdim=True, dim=1)

            # possible best derivation (inspired by https://github.com/Deepest-Project/MelNet/blob/master/model/tts.py#L138 although I don't get the 1.5 and .5 instead of -.5 and +.5, perhaps this is setting the default step size to 1. for all ksi?)
            # 1. / (1. + exp((k-u) / b)) -> 1. / (1. + exp(-(u - k) / b)) -> 1. / (1. + exp(-t)), where t is (u - k) / b
            # knowing that sigmoid(t) = 1. / (1. + exp(-t))
            # this results in sigmoid((u - k) / b) for each term
            # this means both terms L and R are bounded btwn 0 and 1, and potentially more stable than exp(-softplus) shenanigans would allow
            #termL = torch.sum(alpha[..., None] * torch.sigmoid((((u_L - kappa[..., None])) * beta[..., None])), keepdim=True, dim=1)
            #termR = torch.sum(alpha[..., None] * torch.sigmoid((((u_R - kappa[..., None])) * beta[..., None])), keepdim=True, dim=1)
            # finally since beta is bounded > 0 due to exp() activation, we note that dividing by beta and multiplying by beta are effectively the same
            # in terms of optimization paths
            # simply swapping "what regime" wrt values exp(x) < 1, and values exp(x) > 1
            # with the *key* difference being a vanishing to 0 of beta (perhaps due to very negative weights for beta or other issues during training), will not explode the whole equation
            # reweight in log space before summation?
            weights = torch.sum(alpha[..., None] * torch.exp(-1. * ((kappa[..., None] - u) ** 2) * beta[..., None]), keepdim=True, dim=1)
            weights = memory_mask.transpose(0, 1)[:, None, :] * weights
            context = torch.bmm(weights, memory.permute(1, 0, 2))
            # context is B, 1, D
            # weights B, mem_T 
            extras = {}
            extras["kappa"] = kappa
            extras["beta"] = beta
            extras["alpha"] = alpha
        return context, weights, extras

    def _attention(self, cds, layer, memory, memory_mask):
        T, B, D = cds.size()
        # make init a function of the mean of the 
        h_i = cds.new_zeros(B, self.hidden_size)
        c_i = cds.new_zeros(B, self.hidden_size)
        context = cds.new_zeros(B, self.hidden_size)
        if self.attention_type == "logistic":
            prev_attn = cds.new_zeros(B, self.attention_mixture_components)
        elif self.attention_type == "sigmoid_logistic":
            prev_attn = cds.new_zeros(B, self.attention_mixture_components)
        elif self.attention_type == "sigmoid_logistic_alt":
            prev_attn = cds.new_zeros(B, self.attention_mixture_components)
        elif self.attention_type == "dca":
            prev_attn = F.one_hot(torch.zeros(B, dtype=torch.long, device=cds.device), memory.size(0)).float()
        elif self.attention_type == "gaussian":
            prev_attn = cds.new_zeros(B, self.attention_mixture_components)
        elif self.attention_type == "lsa":
            # one dim for attention, one for accumulative weights
            prev_attn_base = cds.new_zeros(B, memory.size(0), 1)
            prev_attn_accum = cds.new_zeros(B, memory.size(0), 1)
        else:
            raise ValueError("Unknown self.attention_type {} found".format(self.attention_type))
        contexts = []
        weights = []
        terminations = []
        all_extras = []
        out_hiddens = []
        for _i in range(T):
            x = torch.cat([cds[_i], context.squeeze(1)], dim=-1)
            out, s = self.attn_lstm_cell([x],
                                         h_i, c_i,
                                         input_mask=None)
            h_t, c_t = s[0], s[1]
            out_hiddens.append(h_t[None])

            if self.attention_type == "lsa":
                prev_attn = torch.cat((prev_attn_base, prev_attn_accum), dim=2)

            h_comb = torch.cat([cds[_i], context.squeeze(1), h_t], dim=-1)
            #context, attn_weight, extras = self._attention_step(h_t, memory, memory_mask, prev_attn)
            context, attn_weight, extras = self._attention_step(h_comb, memory, memory_mask, prev_attn)

            if self.attention_type == "logistic":
                prev_attn = extras["kappa"]
            elif self.attention_type == "sigmoid_logistic":
                prev_attn = extras["kappa"]
            elif self.attention_type == "sigmoid_logistic_alt":
                prev_attn = extras["kappa"]
            elif self.attention_type == "dca":
                prev_attn = attn_weight
            elif self.attention_type == "lsa":
                prev_attn_base = attn_weight[:, 0][..., None]
                prev_attn_accum += attn_weight[:, 0][..., None]
            elif self.attention_type == "gaussian":
                prev_attn = extras["kappa"]
            else:
                raise ValueError("Unknown argument to self.attention_type {}".format(self.attention_type))
            contexts.append(context)
            weights.append(attn_weight[None])
            all_extras.append(extras)
            h_i, c_i = h_t, c_t
        # skip hidden? for better attn control?
        if self.attention_type == "sigmoid_logistic":
            out_contexts = torch.cat(contexts, axis=1).permute(1, 0, 2) + torch.cat(out_hiddens, axis=0)
        if self.attention_type == "sigmoid_logistic_alt":
            out_contexts = torch.cat(contexts, axis=1).permute(1, 0, 2) + torch.cat(out_hiddens, axis=0)
        elif self.attention_type == "gaussian":
            out_contexts = torch.cat(contexts, axis=1).permute(1, 0, 2) + torch.cat(out_hiddens, axis=0)
        else:
            out_contexts = torch.cat(contexts, axis=1).permute(1, 0, 2) + torch.cat(out_hiddens, axis=0)
        #contexts = torch.cat(contexts, axis=1).permute(1, 0, 2) + torch.cat(out_hiddens, axis=0)
        #contexts = torch.cat(contexts, axis=1).permute(1, 0, 2) + tds
        # decoder_T, B, D
        # absolutely no bypassing this?
        # decoder_T, B, encoder_T
        alignments = torch.cat(weights, axis=0)
        return out_contexts, alignments, all_extras

    # conditional first tier
    def forward(self, list_of_inputs, list_of_spatial_conditions=None, bypass_td=None, bypass_fd=None, skip_input_embed=False,
                      memory=None, memory_mask=None):
        # by default embed the inputs, otherwise bypass
        # condidering axis 2 time 3 frequency
        # batch, mel_time, mel_freq, feats

        # shift and project the input
        if len(list_of_inputs) > 1:
            raise ValueError("Only support list_of_inputs length 1 for now")

        x = list_of_inputs[0]
        td_x = torch.cat((0 * x[:, 0][:, None], x[:, :-1]), dim=1)
        fd_x = torch.cat((0 * x[:, :, 0][:, :, None], x[:, :, :-1]), dim=2)

        batch_size = td_x.shape[0]
        self.batch_size = batch_size

        if self.has_centralized_stack:
            # x should has dim of size 1 on the last for input
            cd_x = torch.cat((0 * x[:, 0][:, None], x[:, :-1]), dim=1)[..., 0]
            # cd is now t b f
            cd_x = cd_x.permute(1, 0, 2)

            if not skip_input_embed:
                cd_x = self.centralized_input_proj([cd_x])

        if not skip_input_embed:
            #td_x, td_e = self.embed_td(td_x)
            #fd_x, fd_e = self.embed_fd(fd_x)
            # reshape so the dot works
            td_x = td_x.reshape((batch_size * self.n_vert, self.n_horiz, -1))
            fd_x = fd_x.reshape((batch_size * self.n_vert, self.n_horiz, -1))
            td_x = self.td_input_proj([td_x])
            fd_x = self.fd_input_proj([fd_x])
            td_x = td_x.reshape((batch_size, self.n_vert, self.n_horiz, -1))
            fd_x = fd_x.reshape((batch_size, self.n_vert, self.n_horiz, -1))
            # un reshape it?

        if bypass_td is not None:
            td_x = bypass_td
        if bypass_fd is not None:
            fd_x = bypass_fd

        if self.has_attention:
            assert memory is not None
            assert memory_mask is not None

        if self.has_spatial_condition:
            cond_info = self.cond_mn([list_of_spatial_conditions[0]])
            td_x = td_x + cond_info
            fd_x = fd_x + cond_info

        if self.has_attention:
            # t b f to b t f to stretch
            #mem_shp = memory.shape
            #memory_stretch = memory.permute(1, 0, 2)[:, :, None, :] + 0. * td_x[:, :1, :, :1]
            #memory_stretch = memory_stretch.permute(0, 2, 1, 3).reshape((batch_size * self.n_vert, mem_shp[0], mem_shp[2]))
            # back to t b f
            #memory_stretch = memory_stretch.permute(1, 0, 2)
            memory_stretch = memory
        # batch, mel_time, mel_freq, feats -> batch * mel_time, mel_freq, feats
        td_x = td_x.reshape((batch_size * self.n_vert, self.n_horiz, -1))
        fd_x = fd_x.reshape((batch_size * self.n_vert, self.n_horiz, -1))
        # horiz (freq), batch * vert, feat
        td_x = td_x.permute(1, 0, 2)
        fd_x = fd_x.permute(1, 0, 2)
        # cd_x is n_vert, batch, freq
        has_cd_att = False
        td_x_i = td_x
        fd_x_i = fd_x
        if self.has_centralized_stack:
            cd_x_i = cd_x

        def res(l):
            #return l
            return (0.5 ** 0.5) * l
            #return layer_norm(l, eps=1E-3)
            #return (0.5 ** 0.5) * layer_norm(l, eps=1E-4)

        def layer_norm(x, dim=-1, eps=1E-5):
            mean = torch.mean(x, dim=dim, keepdim=True)
            var = torch.square(x - mean).mean(dim=dim, keepdim=True)
            return (x - mean) / torch.sqrt(var + eps)


        for _i in range(self.n_layers):
            td_x_o = self._td_stack(td_x_i, _i)
            if self.has_centralized_stack:
                if _i == (self.n_layers // 2) and self.has_attention:
                    cd_att, alignment, attn_extras = self._attention(cd_x_i, _i, memory, memory_mask)
                    has_cd_att = True
                    # should this just replace the centralized stack here?
                    cd_x_o = self._cd_centralized_stack(res(cd_x_i + cd_att), _i)
                    fd_x_o = self._fd_stack(res(td_x_o + td_x_i), fd_x_i, _i, tds_cent=res(cd_x_o + cd_x_i + cd_att))
                else:
                    if has_cd_att is False:
                        cd_x_o = self._cd_centralized_stack(cd_x_i, _i)
                        fd_x_o = self._fd_stack(res(td_x_o + td_x_i), fd_x_i, _i, tds_cent=res(cd_x_i + cd_x_o))
                    else:
                        cd_x_o = self._cd_centralized_stack(cd_x_i + cd_att, _i)
                        fd_x_o = self._fd_stack(res(td_x_o + td_x_i), fd_x_i, _i, tds_cent=res(cd_x_o + cd_x_i + cd_att))
            else:
                fd_x_o = self._fd_stack(res(td_x_o + td_x_i), fd_x_i, _i)
            fd_x_i = res(fd_x_o + fd_x_i)
            td_x_i = res(td_x_o + td_x_i)
            if self.has_centralized_stack:
                cd_x_i = res(cd_x_o + cd_x_i)
                # don't add in the attention because we manually add it everwhere cd_x_i is used
                #cd_x_i = cd_x_o + cd_x_i + cd_att
            # set to none to be ensure no "carryover" / errors
            td_x_o = None
            fd_x_o = None
            cd_x_o = None
        out = self.out_proj([fd_x_i])
        out = out.reshape((self.n_horiz, self.batch_size, self.n_vert, self.output_size))
        out = out.permute((1, 2, 0, 3))
        if self.has_attention:
            return out, alignment, attn_extras
        else:
            return out

    def _td_stack_sample(self, tds, layer, time_index, freq_index):
        if self._sample_initial:
            self._sample_cache["layer{}".format(layer)]["td_stack"] = {}
            """
            td_x = td_x.reshape((batch_size * self.n_vert, self.n_horiz, -1))
            fd_x = fd_x.reshape((batch_size * self.n_vert, self.n_horiz, -1))
            # horiz (freq), batch * vert, feat
            td_x = td_x.permute(1, 0, 2)
            """
            #tds_unfold = tds.reshape((self.n_horiz, self.batch_size, self.n_vert, -1))
            #tds = tds_unfold[:, :, -1:, :].reshape((self.n_horiz, self.batch_size, -1))
            # reduced dimension thing works!
            # this is what we want for combined 1 step sampling...

            freq_lstm_fw_h, _, freq_lstm_fw_c = self.tds_lstms_freq_fw[layer]([tds])
            freq_lstm_bw_h, _, freq_lstm_bw_c = self.tds_lstms_freq_bw[layer]([torch.flip(tds, [0])])
            freq_lstm_h = torch.cat((freq_lstm_fw_h, torch.flip(freq_lstm_bw_h, [0])), dim=-1)
            self._sample_cache["layer{}".format(layer)]["td_stack"]["freq_lstm_fw_h"] = freq_lstm_fw_h

            # assume GRU!
            if self.cell_type != "gru":
                raise ValueError("Non GRU step sampling NYI")
            #self._sample_cache["layer{}".format(layer)]["td_stack"]["freq_lstm_fw_c"] = freq_lstm_fw_c

            self._sample_cache["layer{}".format(layer)]["td_stack"]["freq_lstm_bw_h"] = freq_lstm_bw_h

            #self._sample_cache["layer{}".format(layer)]["td_stack"]["freq_lstm_bw_c"] = freq_lstm_bw_c

            self._sample_cache["layer{}".format(layer)]["td_stack"]["freq_lstm_h"] = freq_lstm_h

            freq_lstm_h = self._freq2time(freq_lstm_h)
            tds_time = self._freq2time(tds)
            # first output is just the hidden
            time_lstm_h, _, time_lstm_c = self.tds_lstms_time_fw[layer]([tds_time])
            self._sample_cache["layer{}".format(layer)]["td_stack"]["time_lstm_h"] = time_lstm_h

            #self._sample_cache["layer{}".format(layer)]["td_stack"]["time_lstm_c"] = time_lstm_c
            combined_h = torch.cat((freq_lstm_h, time_lstm_h), dim=-1)
            self._sample_cache["layer{}".format(layer)]["td_stack"]["combined_h"] = combined_h

            res = self.tds_projs[layer]([combined_h])
            res = self._time2freq(res)

            if layer == 0:
                self._sample_last_time_index = {}
                self._sample_last_freq_index = {}

            self._sample_last_time_index["layer{}".format(layer)] = {}
            self._sample_last_freq_index["layer{}".format(layer)] = {}

            self._sample_last_time_index["layer{}".format(layer)]["td_stack"] = time_index
            self._sample_last_freq_index["layer{}".format(layer)]["td_stack"] = freq_index
            return res
        else:
            assert self._sample_step
            if self._sample_last_time_index["layer{}".format(layer)]["td_stack"] != time_index:
                freq_lstm_fw_h, _, freq_lstm_fw_c = self.tds_lstms_freq_fw[layer]([tds])
                freq_lstm_bw_h, _, freq_lstm_bw_c = self.tds_lstms_freq_bw[layer]([torch.flip(tds, [0])])
                freq_lstm_h = torch.cat((freq_lstm_fw_h, torch.flip(freq_lstm_bw_h, [0])), dim=-1)
                if self.cell_type != "gru":
                    raise ValueError("Non GRU step sampling NYI")
                prev_freq_lstm_fw_h = self._sample_cache["layer{}".format(layer)]["td_stack"]["freq_lstm_fw_h"]
                #prev_freq_lstm_fw_c = self._sample_cache["layer{}".format(layer)]["td_stack"]["freq_lstm_fw_c"]
                prev_freq_lstm_bw_h = self._sample_cache["layer{}".format(layer)]["td_stack"]["freq_lstm_bw_h"]
                #prev_freq_lstm_bw_c = self._sample_cache["layer{}".format(layer)]["td_stack"]["freq_lstm_bw_c"]
                prev_freq_lstm_h = self._sample_cache["layer{}".format(layer)]["td_stack"]["freq_lstm_h"]

                self._sample_cache["layer{}".format(layer)]["td_stack"]["freq_lstm_fw_h"] = torch.cat((prev_freq_lstm_fw_h, freq_lstm_fw_h), axis=1)
                #self._sample_cache["layer{}".format(layer)]["td_stack"]["freq_lstm_fw_c"] = torch.cat((prev_freq_lstm_fw_c, freq_lstm_fw_c), axis=1)

                self._sample_cache["layer{}".format(layer)]["td_stack"]["freq_lstm_bw_h"] = torch.cat((prev_freq_lstm_bw_h, freq_lstm_bw_h), axis=1)
                #self._sample_cache["layer{}".format(layer)]["td_stack"]["freq_lstm_bw_c"] = torch.cat((prev_freq_lstm_bw_c, freq_lstm_bw_c), axis=1)

                self._sample_cache["layer{}".format(layer)]["td_stack"]["freq_lstm_h"] = torch.cat((prev_freq_lstm_h, freq_lstm_h), axis=1)

                # _freq2time
                #freq_lstm_h = self._freq2time(freq_lstm_h)
                #tds_time = self._freq2time(tds)
                '''
                ref code
                def _time2freq(self, inp):
                    inp = inp.reshape((self.n_vert, self.batch_size, self.n_horiz, -1))
                    inp = inp.permute(2, 1, 0, 3)
                    return inp.reshape((self.n_horiz, self.batch_size * self.n_vert, -1))

                def _freq2time(self, inp):
                    # batch size set in forward!
                    inp = inp.reshape((self.n_horiz, self.batch_size, self.n_vert, -1))
                    inp = inp.permute(2, 1, 0, 3)
                    return inp.reshape((self.n_vert, self.batch_size * self.n_horiz, -1))
                '''
                freq_lstm_h = freq_lstm_h.reshape((self.n_horiz, self.batch_size, 1, -1))
                freq_lstm_h = freq_lstm_h.permute(2, 1, 0, 3)
                freq_lstm_h = freq_lstm_h.reshape((1, self.batch_size * self.n_horiz, -1))

                tds_time = tds.reshape((self.n_horiz, self.batch_size, 1, -1))
                tds_time = tds_time.permute(2, 1, 0, 3)
                tds_time = tds_time.reshape((1, self.batch_size * self.n_horiz, -1))
                # need to manually walk the rnn cell now
                '''
                in_proj = self.in_proj_obj(list_of_inputs)
                if input_mask is None:
                    input_mask = 0. * in_proj[..., 0] + 1.

                if previous_forward_hidden == None:
                    h1_f_init = 0. * in_proj[0, :, :num_units].detach()
                else:
                    h1_f_init = previous_forward_hidden
                if previous_forward_cell == None:
                    c1_f_init = 0. * in_proj[0, :, :num_units].detach()
                else:
                    c1_f_init = previous_forward_cell
                cell_dropout = None

                # GRU doesn't use cell!
                def step(inp_t, inp_mask_t,
                         h1_f_tm1):
                    output, s = self.fwd_cell_obj([inp_t],
                                                  h1_f_tm1, None,
                                                  input_mask=inp_mask_t,
                                                  cell_dropout=cell_dropout)
                '''
                prev_time_lstm_h = self._sample_cache["layer{}".format(layer)]["td_stack"]["time_lstm_h"]
                tds_time_proj = self.tds_lstms_time_fw[layer].in_proj_obj([tds_time])
                _m = 0. * tds_time_proj[..., 0] + 1.
                _m = _m[..., None]
                # 1 step
                out, s = self.tds_lstms_time_fw[layer].fwd_cell_obj([tds_time_proj],
                                                                    prev_time_lstm_h[-1][None], None,
                                                                    input_mask=_m,
                                                                    cell_dropout=None)
                # 1, 1, batch, feat -> batch, feat
                assert out.shape[0] == 1
                assert out.shape[1] == 1
                out = out[0, 0]
                time_lstm_h = s[0][0, 0]
                # time_lstm_c = s[1][0, 0] but s[1] is None for GRU

                # how often should we redo this computation / cache the time proj??????????????????
                combined_h = torch.cat((freq_lstm_h, time_lstm_h[None]), dim=-1)
                res = self.tds_projs[layer]([combined_h])
                res = res.reshape((1, self.batch_size, self.n_horiz, -1))
                res = res.permute(2, 1, 0, 3)
                res = res.reshape((self.n_horiz, self.batch_size * 1, -1))

                self._sample_cache["layer{}".format(layer)]["td_stack"]["time_lstm_h"] = torch.cat((prev_time_lstm_h, time_lstm_h[None]), dim=0)
                self._sample_cache["layer{}".format(layer)]["td_stack"]["res"] = res

                self._sample_last_time_index["layer{}".format(layer)]["td_stack"] = time_index
                self._sample_last_freq_index["layer{}".format(layer)]["td_stack"] = freq_index
            else:
                res = self._sample_cache["layer{}".format(layer)]["td_stack"]["res"]
            return res

    def _cd_centralized_stack_sample(self, cds, layer, time_index, freq_index):
        if self._sample_initial:
            self._sample_cache["layer{}".format(layer)]["cd_stack"] = {}
            cent_lstm_h, _, cent_lstm_c = self.cds_centralized_lstms[layer]([cds])
            self._sample_cache["layer{}".format(layer)]["cd_stack"]["cent_lstm_h"] = cent_lstm_h
            self._sample_cache["layer{}".format(layer)]["cd_stack"]["cent_lstm_c"] = cent_lstm_c
            res = self.cds_projs[layer]([cent_lstm_h])

            self._sample_last_time_index["layer{}".format(layer)]["cd_stack"] = time_index
            self._sample_last_freq_index["layer{}".format(layer)]["cd_stack"] = freq_index
            return res
        else:
            assert self._sample_step
            if self._sample_last_time_index["layer{}".format(layer)]["cd_stack"] != time_index:
                prev_cent_lstm_h = self._sample_cache["layer{}".format(layer)]["cd_stack"]["cent_lstm_h"]
                cds_proj = self.cds_centralized_lstms[layer].in_proj_obj([cds])
                _m = 0. * cds_proj[..., 0] + 1.
                _m = _m[..., None]
                # 1 step
                out, s = self.cds_centralized_lstms[layer].fwd_cell_obj([cds_proj],
                                                                        prev_cent_lstm_h[-1][None], None,
                                                                        input_mask=_m,
                                                                        cell_dropout=None)
                # 1, 1, batch, feat -> batch, feat
                assert out.shape[0] == 1
                assert out.shape[1] == 1
                out = out[0, 0]
                cent_lstm_h = s[0][0, 0]
                res = self.cds_projs[layer]([cent_lstm_h[None]])

                new_cent_lstm_h = torch.cat((prev_cent_lstm_h, cent_lstm_h[None]), dim=0)
                self._sample_cache["layer{}".format(layer)]["cd_stack"]["cent_lstm_h"] = new_cent_lstm_h
                self._sample_cache["layer{}".format(layer)]["cd_stack"]["res"] = res

                # this is tied to td_sample as well - updating this value should mean they BOTH cached successfully
                self._sample_last_time_index["layer{}".format(layer)]["cd_stack"] = time_index
            else:
                # we already cached the activation for this time index
                res = self._sample_cache["layer{}".format(layer)]["cd_stack"]["res"]
            return res

    def _fd_stack_sample(self, tds, fds, layer, time_index, freq_index, tds_cent=None):
        if self._sample_initial:
            self._sample_cache["layer{}".format(layer)]["fd_stack"] = {}
            # broadcast tds_cent across frequency axis
            if tds_cent is not None:
                # need to permute + reshape to match what was done for td_x, fd_x
                # so that batch combined stuff stays contiguous in the right way!
                # time2freq is freq, batch * time, -1
                # time is also self.n_vert

                # nvert batch feat to batch nvert feat
                tds_cent = tds_cent.permute(1, 0, 2)
                ext_tds_cent = tds_cent.reshape((self.batch_size * self.n_vert, -1))
                # now 1, batch * time, hidden
                # broadcasts over frequency, since the cent rnn has put out a whole freq frame per step dim...
                ext_tds_cent = ext_tds_cent[None]
                # (fds dim 0)
                # broacasts over features, since the cent rnn has effectively seen the whole frequency
                #ext_tds_cent = ext_tds_cent + 0. * fds
                freq_lstm_stack = tds + fds + ext_tds_cent#  torch.cat((tds, fds, ext_tds_cent), axis=-1)
            else:
                #freq_lstm_stack = torch.cat((tds, fds), axis=-1)
                freq_lstm_stack = tds + fds

            freq_lstm_h, _, freq_lstm_c = self.fds_lstms_freq_fw[layer]([freq_lstm_stack])
            # we cache the minimal version for step sampling here
            self._sample_cache["layer{}".format(layer)]["fd_stack"]["freq_lstm_h"] = freq_lstm_h[-1, -1][None, None]
            # GRU so cell is None...
            #self._sample_cache["layer{}".format(layer)]["fd_stack"]["freq_lstm_c"] = freq_lstm_c
            res = self.fds_projs[layer]([freq_lstm_h])

            self._sample_last_time_index["layer{}".format(layer)]["fd_stack"] = time_index
            self._sample_last_freq_index["layer{}".format(layer)]["fd_stack"] = freq_index

            return res
        else:
            assert self._sample_step
            if tds_cent is not None:
                tds_cent = tds_cent.permute(1, 0, 2)
                ext_tds_cent = tds_cent.reshape((self.batch_size * 1, -1))
                ext_tds_cent = ext_tds_cent[None]
                freq_lstm_stack = tds + fds + ext_tds_cent
            else:
                # broadcast....
                freq_lstm_stack = tds + fds
            prev_freq_lstm_h = self._sample_cache["layer{}".format(layer)]["fd_stack"]["freq_lstm_h"]
            freq_lstm_stack = freq_lstm_stack[freq_index][None]
            if freq_index == 0:
                prev_freq_lstm_h = 0. * prev_freq_lstm_h
            stack_proj = self.fds_lstms_freq_fw[layer].in_proj_obj([freq_lstm_stack])
            _m = 0. * stack_proj[..., 0] + 1.
            _m = _m[..., None]
            # 1 step
            out, s = self.fds_lstms_freq_fw[layer].fwd_cell_obj([stack_proj],
                                                                 prev_freq_lstm_h, None,
                                                                 input_mask=_m,
                                                                 cell_dropout=None)
            # 1, 1, batch, feat -> batch, feat
            assert out.shape[0] == 1
            assert out.shape[1] == 1
            out = out[0, 0]
            freq_lstm_h = s[0][0, 0]
            freq_lstm_h = freq_lstm_h[None]

            self._sample_last_time_index["layer{}".format(layer)]["fd_stack"] = time_index
            self._sample_last_freq_index["layer{}".format(layer)]["fd_stack"] = freq_index
            self._sample_cache["layer{}".format(layer)]["fd_stack"]["freq_lstm_h"] = freq_lstm_h

            res = self.fds_projs[layer]([freq_lstm_h])

            return res

    def _attention_step_sample(self, h_i, memory, memory_mask, previous_attn):
        min_attention_step = self._sample_min_attention_step
        if self.attention_type == "sigmoid_logistic_alt":
            # condition on input sequence length?
            phi_hat = self.attn_proj([h_i])
            # cast to 32 bit
            orig_dtype = phi_hat.dtype
            phi_hat = phi_hat.float()

            #ksi = ksi + torch.exp(phi_hat[:, :self.attention_mixture_components])
            # clamp beta so it doesn't collapse?
            #beta = torch.exp(torch.clamp(phi_hat[:, self.attention_mixture_components:2 * self.attention_mixture_components], min=-4))
            # changes based on GMMv2 of https://arxiv.org/pdf/1910.10288.pdf
            # as well as https://arxiv.org/pdf/1811.07240.pdf
            ksi = previous_attn
            ksi = ksi.float()

            """
            if self._softplus is None:
                # hook it into the main module to keep a reference around
                self._softplus = torch.nn.Softplus()
                '''
                     output z  (grad_output)
                     ___________
                     |         |
                     |  layer  |
                     |_________|

                     input x  (grad_input)
                '''
                def hook(module, grad_input, grad_output):
                    return (sigmoid(grad_output[0]),)

                self._softplus.register_backward_hook(hook)

            alt_softplus = self._softplus
            """

            def alt_log1p(x):
                # https://www.johndcook.com/blog/2012/07/25/trick-for-computing-log1x/
                # we dirty hack this to avoid div by 0 since torch.where has issues with NaN grad
                # if *either* branch has inf
                # https://github.com/pytorch/pytorch/issues/4132
                y = 1. + x
                z = y - 1.
                res = 0. * x
                z_mask = (z == 0)
                res[z_mask] = x[z_mask]
                z_nonmask = (z != 0)
                res[z_nonmask] = x[z_nonmask] * torch.log(y[z_nonmask]) / (z[z_nonmask])
                #return torch.where(z == 0, x, x * torch.log(y) / z)
                return res

            def alt_softplus(x):
                # https://stackoverflow.com/questions/44230635/avoid-overflow-with-softplus-function-in-python
                return alt_log1p(torch.exp(-torch.abs(x))) + torch.where(x > 0, x, 0. * x)

            kappa_step = alt_softplus(phi_hat[:, :self.attention_mixture_components]) + 1E-3
            kappa_step = torch.max(kappa_step, 0. * kappa_step + min_attention_step)
            kappa = ksi + kappa_step #alt_softplus(phi_hat[:, :self.attention_mixture_components]) + 1E-3
            #kappa = ksi + swish(phi_hat[:, :self.attention_mixture_components])

            #beta = (F.softplus(phi_hat[:, self.attention_mixture_components:2 * self.attention_mixture_components]) + 1E-4)
            #beta = torch.clamp(torch.exp(phi_hat[:, self.attention_mixture_components:2 * self.attention_mixture_components]), min=.5)
            # fix beta to 1 - hack city but works!
            #beta = 0. * phi_hat[:, self.attention_mixture_components:2 * self.attention_mixture_components] + 1.

            # aggressive clamping here, use softplus to help stability as well
            #beta = torch.clamp(F.softplus(phi_hat[:, self.attention_mixture_components:2 * self.attention_mixture_components]), min=.25, max=2.0)
            #beta = torch.clamp(F.softplus(phi_hat[:, self.attention_mixture_components:2 * self.attention_mixture_components]), min=.01, max=10.0)
            #beta = F.softplus(phi_hat[:, self.attention_mixture_components:2 * self.attention_mixture_components])

            beta = alt_softplus(phi_hat[:, self.attention_mixture_components:2 * self.attention_mixture_components])

            #beta = swish(phi_hat[:, self.attention_mixture_components:2 * self.attention_mixture_components])
            #beta = swish(phi_hat[:, self.attention_mixture_components:2 * self.attention_mixture_components])

            alpha = F.softmax(phi_hat[:, 2 * self.attention_mixture_components:3 * self.attention_mixture_components], dim=1)
            #alpha = F.softplus(phi_hat[:, 2 * self.attention_mixture_components:3 * self.attention_mixture_components]) + 1E-2
            #alpha = F.log_softmax(phi_hat[:, 2 * self.attention_mixture_components:3 * self.attention_mixture_components], dim=1)

            u = memory.new_tensor(np.arange(memory.size(0)), dtype=memory.dtype)
            #u_L = u + 0.5
            #u_R = u - 0.5

            # try like this for the following reason...
            # eqn 22 of paper
            # F(u + .5; g) - F(u - 0.5; g)
            # means F(u; g) = 1./(1 + exp((k - u) / B))
            # we can interpret this as either SUBTITUTING u + 0.5 for u, or simply adding/subtracting on 0.5. This would swap the 2 terms
            # interpreting as simply adding 0.5 to the end means
            # sigm(x) = 1./(1+exp(-x))
            # x = ((-k + u) / B) = ((u - k) / B)
            # sigm((u - .5 - k) / B) as the left hand
            # sigm((u + .5 - k) / B) as the right hand
            # alternately, interpreting as substituting u - .5 for u
            # sigm((u + .5 - k) / B) as the left hand
            # sigm((u - .5 - k) / B) as the right hand
            # noting that we can multiply or divide by beta, if beta is constrained from 0 to inf
            # since / by a number from 0->1 is the same as multiplying by 1->inf
            # aka beta can parameterize the division term, or 1 / division
            # parameterizing the 1 / division means that we don't face edge cases for beta near 0, as 1/inf -> 0 and 1/0. -> inf
            # however, it means that making small beta we have less precision due to floating point
            u_L = u + 0.5
            u_R = u - 0.5

            """
            alternative?
            TANH(t) = [1 - exp(-2t)]/[1 + exp(-2t)]  for  t>=0
            and
            TANH(t) = [exp(2t) - 1]/[exp(2t) + 1] for t<0
            """

            # we approximate tanh with x/(1+abs(x))
            def alt_tanh(x):
                return x / (1. + torch.abs(x))
            # logistic can be expressed as 1/2 + 1/2 * tanh((x - u)/(2 * s)) instead of sigmoid
            # if beta is 1/s, this is .5 * beta
            def term(u, k, b):
                #return .5 + .5 * alt_tanh(.5 * (u - k) / b)
                # limit min beta to .01
                return .5 + .5 * alt_tanh(.5 * (u - k) * (b + .01))

            termL = torch.sum(alpha[..., None] * term(u_L, kappa[..., None], beta[..., None]), keepdim=True, dim=1)
            #termR = torch.sum(alpha[..., None] * term(u_R, kappa[..., None], beta[..., None]), keepdim=True, dim=1)

            diff = (term(u_L, kappa[..., None], beta[..., None]) - term(u_R, kappa[..., None], beta[..., None]))
            weights = torch.sum(alpha[..., None] * diff, keepdim=True, dim=1)

            termination = 1. - torch.exp(termL[:, 0])
            weights = memory_mask.transpose(0, 1)[:, None, :] * weights

            # grad scaling here
            grad_rescale = 1. / np.sqrt(self.attention_mixture_components)
            weights = (1. - grad_rescale) * weights.detach() + grad_rescale * weights

            weights = weights.to(orig_dtype)
            kappa = kappa.to(orig_dtype)
            beta = beta.to(orig_dtype)
            alpha = alpha.to(orig_dtype)
            termination = termination.to(orig_dtype)

            context = torch.bmm(weights, memory.permute(1, 0, 2))
            # context is B, 1, D
            # weights B, 1, mem_T 
            extras = {}
            extras["termination"] = termination
            extras["kappa"] = kappa
            extras["beta"] = beta
            extras["alpha"] = alpha
        else:
            raise ValueError("Unknown attention type specified {}".format(self.attention_type))
        return context, weights, extras

    def _attention_sample(self, cds, layer, memory, memory_mask, time_index, freq_index):
        if self._sample_initial:
            self._sample_cache["layer{}".format(layer)]["attention"] = {}
            T, B, D = cds.size()
            # make init a function of the mean of the 
            h_i = cds.new_zeros(B, self.hidden_size)
            c_i = cds.new_zeros(B, self.hidden_size)
            context = cds.new_zeros(B, self.hidden_size)
            if self.attention_type == "sigmoid_logistic_alt":
                prev_attn = cds.new_zeros(B, self.attention_mixture_components)
            else:
                raise ValueError("Unknown self.attention_type {} found".format(self.attention_type))
            contexts = []
            weights = []
            terminations = []
            all_extras = []
            out_hiddens = []
            all_hiddens = [h_i]
            all_cells = [c_i]
            all_attn_pos = [prev_attn]
            for _i in range(T):
                x = torch.cat([cds[_i], context.squeeze(1)], dim=-1)
                out, s = self.attn_lstm_cell([x],
                                             h_i, c_i,
                                             input_mask=None)
                h_t, c_t = s[0], s[1]
                out_hiddens.append(h_t[None])

                h_comb = torch.cat([cds[_i], context.squeeze(1), h_t], dim=-1)
                #context, attn_weight, extras = self._attention_step(h_t, memory, memory_mask, prev_attn)
                context, attn_weight, extras = self._attention_step_sample(h_comb, memory, memory_mask, prev_attn)

                if self.attention_type == "sigmoid_logistic_alt":
                    prev_attn = extras["kappa"]
                else:
                    raise ValueError("Unknown argument to self.attention_type {}".format(self.attention_type))
                contexts.append(context)
                weights.append(attn_weight[None])
                all_extras.append(extras)
                h_i, c_i = h_t, c_t
                all_hiddens.append(h_i)
                all_cells.append(c_i)
                all_attn_pos.append(prev_attn)
            # skip hidden? for better attn control?
            if self.attention_type == "sigmoid_logistic_alt":
                out_contexts = torch.cat(contexts, axis=1).permute(1, 0, 2) + torch.cat(out_hiddens, axis=0)
            else:
                raise ValueError("Unknown argument to self.attention_type {}".format(self.attention_type))
            #contexts = torch.cat(contexts, axis=1).permute(1, 0, 2) + torch.cat(out_hiddens, axis=0)
            #contexts = torch.cat(contexts, axis=1).permute(1, 0, 2) + tds
            # decoder_T, B, D
            # absolutely no bypassing this?
            # decoder_T, B, encoder_T
            alignments = torch.cat(weights, axis=0)
            self._sample_cache["layer{}".format(layer)]["attention"]["attn_h"] = all_hiddens
            # GRU so c is invisible
            #self._sample_cache["layer{}".format(layer)]["attention"]["attn_c"] = all_cells
            self._sample_cache["layer{}".format(layer)]["attention"]["attn_kappa"] = all_attn_pos
            self._sample_cache["layer{}".format(layer)]["attention"]["attn_context"] = contexts

            self._sample_last_time_index["layer{}".format(layer)]["attention"] = time_index
            self._sample_last_freq_index["layer{}".format(layer)]["attention"] = freq_index

            return out_contexts, alignments, all_extras
        else:
            assert self._sample_step
            if self._sample_last_time_index["layer{}".format(layer)]["attention"] != time_index:
                T, B, D = cds.size()
                # make init a function of the mean of the 
                # use h_t
                context = self._sample_cache["layer{}".format(layer)]["attention"]["attn_context"][-1]
                h_i = self._sample_cache["layer{}".format(layer)]["attention"]["attn_h"][-1]

                #h_i = cds.new_zeros(B, self.hidden_size)
                # GRU so cell is nothing
                #c_i = cds.new_zeros(B, self.hidden_size)

                #context = cds.new_zeros(B, self.hidden_size)
                if self.attention_type == "sigmoid_logistic_alt":
                    #prev_attn = cds.new_zeros(B, self.attention_mixture_components)
                    prev_attn = self._sample_cache["layer{}".format(layer)]["attention"]["attn_kappa"][-1]
                else:
                    raise ValueError("Unknown self.attention_type {} found".format(self.attention_type))

                comb = torch.cat([cds[0], context.squeeze(1)], dim=-1)
                out, s = self.attn_lstm_cell([comb],
                                             h_i, None,
                                             input_mask=None)
                h_t, c_t = s[0], s[1]
                h_comb = torch.cat([cds[0], context.squeeze(1), h_t], dim=-1)

                new_context, attn_weight, extras = self._attention_step_sample(h_comb, memory, memory_mask, prev_attn)

                out = new_context + h_t[None]

                self._sample_cache["layer{}".format(layer)]["attention"]["attn_context"].append(new_context)
                self._sample_cache["layer{}".format(layer)]["attention"]["attn_h"].append(h_t)

                prev_attn = extras["kappa"]
                self._sample_cache["layer{}".format(layer)]["attention"]["attn_kappa"].append(prev_attn)

                self._sample_cache["layer{}".format(layer)]["attention"]["out"] = out
                self._sample_cache["layer{}".format(layer)]["attention"]["attn_weight"] = attn_weight
                self._sample_cache["layer{}".format(layer)]["attention"]["extras"] = extras

                self._sample_last_time_index["layer{}".format(layer)]["attention"] = time_index
                self._sample_last_freq_index["layer{}".format(layer)]["attention"] = freq_index
            else:
                out = self._sample_cache["layer{}".format(layer)]["attention"]["out"]
                attn_weight = self._sample_cache["layer{}".format(layer)]["attention"]["attn_weight"]
                extras = self._sample_cache["layer{}".format(layer)]["attention"]["extras"]
            return out, attn_weight, extras


    def sample(self, list_of_inputs,
                     time_index, freq_index, is_initial_step=True,
                     list_of_spatial_conditions=None,
                     bypass_td=None, bypass_fd=None, skip_input_embed=False,
                     memory=None, memory_mask=None,
                     min_attention_step=None):
        if min_attention_step is not None:
            self._sample_min_attention_step = min_attention_step
        else:
            self._sample_min_attention_step = 0.
        if is_initial_step:
            return self._sample_initial_fn(list_of_inputs, time_index, freq_index,
                                        list_of_spatial_conditions=list_of_spatial_conditions,
                                        bypass_td=bypass_td, bypass_fd=bypass_fd,
                                        skip_input_embed=skip_input_embed,
                                        memory=memory, memory_mask=memory_mask)
        else:
            return self._sample_inc_fn(list_of_inputs, time_index, freq_index,
                                    list_of_spatial_conditions=list_of_spatial_conditions,
                                    bypass_td=bypass_td, bypass_fd=bypass_fd,
                                    skip_input_embed=skip_input_embed,
                                    memory=memory, memory_mask=memory_mask)

    def _sample_inc_fn(self, list_of_inputs,
                     time_index, freq_index,
                     list_of_spatial_conditions=None,
                     bypass_td=None, bypass_fd=None, skip_input_embed=False,
                     memory=None, memory_mask=None):
        # by default embed the inputs, otherwise bypass
        # batch, mel_time, mel_freq, feats

        # shift and project the input
        if len(list_of_inputs) > 1:
            raise ValueError("Only support list_of_inputs length 1 for now")

        # we only use the time_index (and potentially, freq_index) step of x
        x = list_of_inputs[0]

        # x[:, time_index-1].sum() == self._sample_cache["input"]["tmp_td_x"][:, time_index].sum() 
        # we need to process only the frame at time_index to extend the sampling
        td_x = torch.cat((0 * x[:, 0][:, None], x), dim=1)
        td_x = td_x[:, time_index][:, None]

        fd_x = torch.cat((0 * x[:, :, 0][:, :, None], x[:, :, :-1]), dim=2)
        fd_x = fd_x[:, time_index][:, None]

        #if freq_index > 0:
        #    freq_index = freq_index - 1
        #    fd_x = x[:, :, freq_index]
        #else:
            # set it to 0s...
        #    freq_index = 0
        #    fd_x = 0. * x[:, :, freq_index]
        #fd_x = fd_x[:, :, None]

        self._sample_initial = False
        self._sample_step = True

        batch_size = td_x.shape[0]
        self.batch_size = batch_size
        self.n_vert = x.shape[1]
        self.n_horiz = x.shape[2]

        if self.has_centralized_stack:
            # x should has dim of size 1 on the last for input
            cd_x = torch.cat((0 * x[:, 0][:, None], x[:, :-1]), dim=1)[..., 0]
            cd_x = cd_x[:, time_index][:, None]
            # cd is now t b f
            cd_x = cd_x.permute(1, 0, 2)

            if not skip_input_embed:
                cd_x = self.centralized_input_proj([cd_x])

        if not skip_input_embed:
            #td_x, td_e = self.embed_td(td_x)
            #fd_x, fd_e = self.embed_fd(fd_x)
            # reshape so the dot works
            td_x = td_x.reshape((batch_size * 1, self.n_horiz, -1))
            td_x = self.td_input_proj([td_x])
            td_x = td_x.reshape((batch_size, 1, self.n_horiz, -1))

            fd_x = fd_x.reshape((batch_size * 1, self.n_horiz, -1))
            fd_x = self.fd_input_proj([fd_x])
            fd_x = fd_x.reshape((batch_size, 1, self.n_horiz, -1))

        if bypass_td is not None:
            td_x = bypass_td
        if bypass_fd is not None:
            fd_x = bypass_fd

        if self.has_attention:
            assert memory is not None
            assert memory_mask is not None

        if self.has_spatial_condition:
            cond_info = self.cond_mn([list_of_spatial_conditions[0]])
            td_x = td_x + cond_info
            fd_x = fd_x + cond_info

        # cached here in initial

        # batch, mel_time, mel_freq, feats -> batch * mel_time, mel_freq, feats
        td_x = td_x.reshape((batch_size * 1, self.n_horiz, -1))
        fd_x = fd_x.reshape((batch_size * 1, self.n_horiz, -1))
        # horiz (freq), batch * vert, feat
        td_x = td_x.permute(1, 0, 2)
        fd_x = fd_x.permute(1, 0, 2)
        # cd_x is n_vert, batch, freq

        if self.has_attention:
            # t b f to b t f to stretch
            #mem_shp = memory.shape
            #memory_stretch = memory.permute(1, 0, 2)[:, :, None, :] + 0. * td_x[:, :1, :, :1]
            #memory_stretch = memory_stretch.permute(0, 2, 1, 3).reshape((batch_size * self.n_vert, mem_shp[0], mem_shp[2]))
            # back to t b f
            #memory_stretch = memory_stretch.permute(1, 0, 2)
            memory_stretch = memory


        has_cd_att = False
        td_x_i = td_x
        fd_x_i = fd_x
        if self.has_centralized_stack:
            cd_x_i = cd_x

        def res(l):
            return (0.5 ** 0.5) * l

        def layer_norm(x, dim=-1, eps=1E-5):
            mean = torch.mean(x, dim=dim, keepdim=True)
            var = torch.square(x - mean).mean(dim=dim, keepdim=True)
            return (x - mean) / torch.sqrt(var + eps)

        for _i in range(self.n_layers):
            td_x_o = self._td_stack_sample(td_x_i, _i, time_index, freq_index)
            if self.has_centralized_stack:
                if _i == (self.n_layers // 2) and self.has_attention:
                    cd_att, alignment, attn_extras = self._attention_sample(cd_x_i, _i, memory, memory_mask, time_index, freq_index)
                    has_cd_att = True
                    # should this just replace the centralized stack here?
                    cd_x_o = self._cd_centralized_stack_sample(res(cd_x_i + cd_att), _i, time_index, freq_index)
                    fd_x_o = self._fd_stack_sample(res(td_x_o + td_x_i), fd_x_i, _i, time_index, freq_index, tds_cent=res(cd_x_o + cd_x_i + cd_att))
                else:
                    if has_cd_att is False:
                        cd_x_o = self._cd_centralized_stack_sample(cd_x_i, _i, time_index, freq_index)
                        fd_x_o = self._fd_stack_sample(res(td_x_o + td_x_i), fd_x_i, _i, time_index, freq_index, tds_cent=res(cd_x_i + cd_x_o))
                    else:
                        cd_x_o = self._cd_centralized_stack_sample(cd_x_i + cd_att, _i, time_index, freq_index)
                        fd_x_o = self._fd_stack_sample(res(td_x_o + td_x_i), fd_x_i, _i, time_index, freq_index, tds_cent=res(cd_x_o + cd_x_i + cd_att))
            else:
                fd_x_o = self._fd_stack_sample(res(td_x_o + td_x_i), fd_x_i, _i, time_index, freq_index)
            fd_x_i = res(fd_x_o + fd_x_i)
            td_x_i = res(td_x_o + td_x_i)
            if self.has_centralized_stack:
                cd_x_i = res(cd_x_o + cd_x_i)
                # don't add in the attention because we manually add it everwhere cd_x_i is used
                #cd_x_i = cd_x_o + cd_x_i + cd_att
            # set to none to be ensure no "carryover" / errors
            td_x_o = None
            fd_x_o = None
            cd_x_o = None
        out = self.out_proj([fd_x_i])
        out = out.reshape((self.n_horiz, self.batch_size, 1, self.output_size))
        #out = out.reshape((self.n_horiz, self.batch_size, self.n_vert, self.output_size))
        out = out.permute((1, 2, 0, 3))
        out = out[:, :, freq_index]
        if self.has_attention:
            return out, alignment, attn_extras
        else:
            return out

    def _sample_initial_fn(self, list_of_inputs,
                        time_index, freq_index,
                        list_of_spatial_conditions=None,
                        bypass_td=None, bypass_fd=None, skip_input_embed=False,
                        memory=None, memory_mask=None):
        # this basically the same as the training method with exta caching to interact with sample when it is not an 
        # by default embed the inputs, otherwise bypass
        # batch, mel_time, mel_freq, feats

        # shift and project the input
        if len(list_of_inputs) > 1:
            raise ValueError("Only support list_of_inputs length 1 for now")

        self._sample_cache = {}
        self._sample_cache["input"] = {}

        x = list_of_inputs[0]
        td_x = torch.cat((0 * x[:, 0][:, None], x[:, :-1]), dim=1)
        fd_x = torch.cat((0 * x[:, :, 0][:, :, None], x[:, :, :-1]), dim=2)

        self._sample_cache["input"]["tmp_x"] = x
        self._sample_cache["input"]["tmp_td_x"] = td_x
        self._sample_cache["input"]["tmp_fd_x"] = fd_x

        batch_size = td_x.shape[0]
        self.batch_size = batch_size

        if self.has_centralized_stack:
            # x should has dim of size 1 on the last for input
            cd_x = torch.cat((0 * x[:, 0][:, None], x[:, :-1]), dim=1)[..., 0]
            # cd is now t b f
            cd_x = cd_x.permute(1, 0, 2)

            if not skip_input_embed:
                cd_x = self.centralized_input_proj([cd_x])
        self.n_vert = x.shape[1]
        self.n_horiz = x.shape[2]

        if not skip_input_embed:
            #td_x, td_e = self.embed_td(td_x)
            #fd_x, fd_e = self.embed_fd(fd_x)
            # reshape so the dot works
            td_x = td_x.reshape((batch_size * self.n_vert, self.n_horiz, -1))
            td_x = self.td_input_proj([td_x])
            td_x = td_x.reshape((batch_size, self.n_vert, self.n_horiz, -1))

        if not skip_input_embed:
            fd_x = fd_x.reshape((batch_size * self.n_vert, self.n_horiz, -1))
            # partial then concatenate
            fd_x = self.fd_input_proj([fd_x])
            fd_x = fd_x.reshape((batch_size, self.n_vert, self.n_horiz, -1))
            # un reshape it?

        if self.has_attention:
            assert memory is not None
            assert memory_mask is not None

        if self.has_spatial_condition:
            cond_info = self.cond_mn([list_of_spatial_conditions[0]])
            td_x = td_x + cond_info
            fd_x = fd_x + cond_info

        if self.has_attention:
            # t b f to b t f to stretch
            #mem_shp = memory.shape
            #memory_stretch = memory.permute(1, 0, 2)[:, :, None, :] + 0. * td_x[:, :1, :, :1]
            #memory_stretch = memory_stretch.permute(0, 2, 1, 3).reshape((batch_size * self.n_vert, mem_shp[0], mem_shp[2]))
            # back to t b f
            #memory_stretch = memory_stretch.permute(1, 0, 2)
            memory_stretch = memory

        self._sample_initial = True
        self._sample_step = False

        self._sample_cache["input"]["td_x"] = td_x
        self._sample_cache["input"]["fd_x"] = fd_x
        # time batch feats
        self._sample_cache["input"]["cd_x"] = cd_x

        # batch, mel_time, mel_freq, feats -> batch * mel_time, mel_freq, feats
        td_x = td_x.reshape((batch_size * self.n_vert, self.n_horiz, -1))
        fd_x = fd_x.reshape((batch_size * self.n_vert, self.n_horiz, -1))
        # horiz (freq), batch * vert, feat
        td_x = td_x.permute(1, 0, 2)
        fd_x = fd_x.permute(1, 0, 2)

        # cd_x is n_vert, batch, freq
        has_cd_att = False
        td_x_i = td_x
        fd_x_i = fd_x
        if self.has_centralized_stack:
            cd_x_i = cd_x

        def res(l):
            return (0.5 ** 0.5) * l

        def layer_norm(x, dim=-1, eps=1E-5):
            mean = torch.mean(x, dim=dim, keepdim=True)
            var = torch.square(x - mean).mean(dim=dim, keepdim=True)
            return (x - mean) / torch.sqrt(var + eps)

        for _i in range(self.n_layers):
            self._sample_cache["layer{}".format(_i)] = {}
            self._sample_cache["layer{}".format(_i)]["fd_stack"] = {}
            self._sample_cache["layer{}".format(_i)]["cd_stack"] = {}
            td_x_o = self._td_stack_sample(td_x_i, _i, time_index, freq_index)
            if self.has_centralized_stack:
                if _i == (self.n_layers // 2) and self.has_attention:
                    cd_att, alignment, attn_extras = self._attention_sample(cd_x_i, _i, memory, memory_mask, time_index, freq_index)
                    has_cd_att = True
                    # should this just replace the centralized stack here?
                    cd_x_o = self._cd_centralized_stack_sample(res(cd_x_i + cd_att), _i, time_index, freq_index)
                    fd_x_o = self._fd_stack_sample(res(td_x_o + td_x_i), fd_x_i, _i, time_index, freq_index, tds_cent=res(cd_x_o + cd_x_i + cd_att))
                else:
                    if has_cd_att is False:
                        cd_x_o = self._cd_centralized_stack_sample(cd_x_i, _i, time_index, freq_index)
                        fd_x_o = self._fd_stack_sample(res(td_x_o + td_x_i), fd_x_i, _i, time_index, freq_index, tds_cent=res(cd_x_i + cd_x_o))
                    else:
                        cd_x_o = self._cd_centralized_stack_sample(cd_x_i + cd_att, _i, time_index, freq_index)
                        fd_x_o = self._fd_stack_sample(res(td_x_o + td_x_i), fd_x_i, _i, time_index, freq_index, tds_cent=res(cd_x_o + cd_x_i + cd_att))
            else:
                fd_x_o = self._fd_stack_sample(res(td_x_o + td_x_i), fd_x_i, _i, time_index, freq_index)
            fd_x_i = res(fd_x_o + fd_x_i)
            td_x_i = res(td_x_o + td_x_i)
            if self.has_centralized_stack:
                cd_x_i = res(cd_x_o + cd_x_i)
                # don't add in the attention because we manually add it everwhere cd_x_i is used
                #cd_x_i = cd_x_o + cd_x_i + cd_att
            # set to none to be ensure no "carryover" / errors
            td_x_o = None
            fd_x_o = None
            cd_x_o = None
        out = self.out_proj([fd_x_i])
        out = out.reshape((self.n_horiz, self.batch_size, self.n_vert, self.output_size))
        out = out.permute((1, 2, 0, 3))
        if self.has_attention:
            return out, alignment, attn_extras
        else:
            return out


class YAMNetFullContextSubTier(torch.nn.Module):
    def __init__(self,
                 list_of_input_sizes,
                 n_vert,
                 n_horiz,
                 hidden_dim,
                 n_layers,
                 # defaults for 380/900 setup
                 n_heads=10,
                 head_dim=38,
                 inner_dim=900,
                 init=None,
                 scale="default",
                 biases=True,
                 bias_offset=0.,
                 name=None,
                 cell_dropout=1.,
                 use_centralized_stack=False,
                 random_state=None, strict=None,
                 dtype="default", device="default"):
        super(YAMNetFullContextSubTier, self).__init__()
        if name is None:
            name = _get_name()

        if random_state is None:
            raise ValueError("Must pass instance of np.random.RandomState!")

        if strict is None:
            strict = get_strict_mode_default()

        self.input_size = sum(list_of_input_sizes)

        self.hidden_size = hidden_dim

        self.cell_dropout = cell_dropout
        self.n_layers = n_layers
        self.n_vert = n_vert
        self.n_horiz = n_horiz

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.inner_dim = inner_dim

        self.input_proj = Linear([self.input_size],
                                 self.hidden_size,
                                 random_state=random_state,
                                 init=init,
                                 scale=scale,
                                 name=name + "_input_proj",
                                 device=device)

        self.seq_time_fw = nn.ModuleList()
        self.seq_time_bw = nn.ModuleList()
        self.seq_freq_fw = nn.ModuleList()
        self.seq_freq_bw = nn.ModuleList()
        self.projs = nn.ModuleList()

        for _i in range(self.n_layers):
            memory_len=10
            context_len=0
            time_fw_layer = AWDTransformerXLDecoderBlock([self.hidden_size,],
                                                          name=name + "_seq_time_fw_{}".format(_i),
                                                          n_layers=self.n_layers,
                                                          n_heads=self.n_heads,
                                                          head_dim=self.head_dim,
                                                          model_dim=self.hidden_size,
                                                          inner_dim=self.inner_dim,
                                                          random_state=random_state,
                                                          memory_len=memory_len,
                                                          context_len=context_len,
                                                          init=init,
                                                          scale=scale,
                                                          device=device)
            self.seq_time_fw.append(time_fw_layer)

            time_bw_layer = AWDTransformerXLDecoderBlock([self.hidden_size,],
                                                          name=name + "_seq_time_bw_{}".format(_i),
                                                          n_layers=self.n_layers,
                                                          n_heads=self.n_heads,
                                                          head_dim=self.head_dim,
                                                          model_dim=self.hidden_size,
                                                          inner_dim=self.inner_dim,
                                                          random_state=random_state,
                                                          memory_len=memory_len,
                                                          context_len=context_len,
                                                          init=init,
                                                          scale=scale,
                                                          device=device)
            self.seq_time_bw.append(time_bw_layer)

            freq_fw_layer = AWDTransformerXLDecoderBlock([self.hidden_size,],
                                                          name=name + "_seq_freq_fw_{}".format(_i),
                                                          n_layers=self.n_layers,
                                                          n_heads=self.n_heads,
                                                          head_dim=self.head_dim,
                                                          model_dim=self.hidden_size,
                                                          inner_dim=self.inner_dim,
                                                          random_state=random_state,
                                                          memory_len=memory_len,
                                                          context_len=context_len,
                                                          init=init,
                                                          scale=scale,
                                                          device=device)
            self.seq_freq_fw.append(freq_fw_layer)

            freq_bw_layer = AWDTransformerXLDecoderBlock([self.hidden_size,],
                                                          name=name + "_seq_freq_bw_{}".format(_i),
                                                          n_layers=self.n_layers,
                                                          n_heads=self.n_heads,
                                                          head_dim=self.head_dim,
                                                          model_dim=self.hidden_size,
                                                          inner_dim=self.inner_dim,
                                                          random_state=random_state,
                                                          memory_len=memory_len,
                                                          context_len=context_len,
                                                          init=init,
                                                          scale=scale,
                                                          device=device)
            self.seq_freq_bw.append(freq_bw_layer)

            self.projs.append(Linear([4 * self.hidden_size,],
                                      self.hidden_size,
                                      random_state=random_state,
                                      init=init,
                                      scale=scale,
                                      name=name + "_projs_{}".format(_i),
                                      device=device))

    def _time2freq(self, inp):
        inp = inp.reshape((self.n_vert, self.batch_size, self.n_horiz, -1))
        inp = inp.permute(2, 1, 0, 3)
        return inp.reshape((self.n_horiz, self.batch_size * self.n_vert, -1))

    def _freq2time(self, inp):
        # batch size set in forward!
        inp = inp.reshape((self.n_horiz, self.batch_size, self.n_vert, -1))
        inp = inp.permute(2, 1, 0, 3)
        return inp.reshape((self.n_vert, self.batch_size * self.n_horiz, -1))

    def _stack(self, nds, layer):
        freq_rnn_fw_h, _ = self.seq_freq_fw[layer](nds)
        freq_rnn_bw_h, _ = self.seq_freq_bw[layer](torch.flip(nds, [0]))
        freq_rnn_h = torch.cat((freq_rnn_fw_h, torch.flip(freq_rnn_bw_h, [0])), dim=-1)
        freq_rnn_h = self._freq2time(freq_rnn_h)

        nds_time = self._freq2time(nds)
        time_rnn_fw_h, _ = self.seq_time_fw[layer](nds_time)
        time_rnn_bw_h, _ = self.seq_time_bw[layer](torch.flip(nds_time, [0]))
        time_rnn_h = torch.cat((time_rnn_fw_h, torch.flip(time_rnn_bw_h, [0])), dim=-1)
        combined_h = torch.cat((freq_rnn_h, time_rnn_h), dim=-1)
        res = self.projs[layer]([combined_h])
        res = self._time2freq(res)
        return (0.5 ** 0.5) * (nds + res)

    def forward(self, list_of_inputs):
        # by default embed the inputs, otherwise bypass
        # condidering axis 2 time 3 frequency
        # batch, mel_time, mel_freq, feats

        # shift and project the input
        if len(list_of_inputs) > 1:
            raise ValueError("Only support list_of_inputs length 1 for now")

        x = list_of_inputs[0]

        batch_size = x.shape[0]
        self.batch_size = batch_size

        c_x = x.reshape((batch_size * self.n_vert, self.n_horiz, -1))
        c_x = self.input_proj([c_x])
        c_x = c_x.permute(1, 0, 2)
        for _i in range(self.n_layers):
            c_x = self._stack(c_x, _i)
        out = c_x.reshape((self.n_horiz, self.batch_size, self.n_vert, self.hidden_size))
        out = out.permute((1, 2, 0, 3))
        return out


class YAMTransformerBlock(torch.nn.Module):
    def __init__(self,
                 list_of_input_symbol_sizes,
                 n_vert,
                 n_horiz,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 input_dim=None,
                 has_centralized_stack=False,
                 has_spatial_condition=False,
                 spatial_condition_input_size=None,
                 spatial_condition_n_layers=2,
                 spatial_condition_n_heads=10,
                 spatial_condition_head_dim=38,
                 spatial_condition_inner_dim=900,
                 has_attention=False,
                 transformer_inner_layers=1,
                 # defaults for hidden_dim=380...
                 transformer_n_heads=10,
                 transformer_head_dim=38,
                 transformer_inner_dim=900,
                 cell_type="lstm",
                 attention_type="sigmoid_logistic",
                 attention_mixture_components=10,
                 init=None, scale="default",
                 biases=True, bias_offset=0.,
                 name=None,
                 cell_dropout=1.,
                 use_centralized_stack=False,
                 random_state=None, strict=None,
                 dtype="default", device="default"):
        super(YAMTransformerBlock, self).__init__()
        if name is None:
            name = _get_name()

        if random_state is None:
            raise ValueError("Must pass instance of np.random.RandomState!")

        if strict is None:
            strict = get_strict_mode_default()

        self.input_symbols = list_of_input_symbol_sizes[0]
        self.cell_type = cell_type
        self.hidden_size = hidden_dim
        self.output_size = output_dim
        self.has_centralized_stack = has_centralized_stack

        self.has_spatial_condition = has_spatial_condition
        self.spatial_condition_n_layers = spatial_condition_n_layers
        self.spatial_condition_n_heads = spatial_condition_n_heads
        self.spatial_condition_head_dim = spatial_condition_head_dim
        self.spatial_condition_inner_dim = spatial_condition_inner_dim

        self.attention_mixture_components = attention_mixture_components
        self.has_attention = has_attention
        self.attention_type = attention_type

        self.transformer_inner_layers = transformer_inner_layers
        self.transformer_n_heads = transformer_n_heads
        self.transformer_head_dim = transformer_head_dim
        self.transformer_inner_dim = transformer_inner_dim

        self.cell_dropout = cell_dropout
        self.n_layers = n_layers
        self.n_vert = n_vert
        self.n_horiz = n_horiz
        self.output_dim = output_dim
        self.input_dim = input_dim

        if self.cell_type == "lstm":
            bidir_rnn_fn = BiLSTMLayer
            rnn_fn = LSTMLayer
            rnn_cell_fn = LSTMCell
        elif self.cell_type == "gru":
            bidir_rnn_fn = BiGRULayer
            rnn_fn = GRULayer
            rnn_cell_fn = GRUCell
        else:
            raise ValueError("Unknown cell_type, self.cell_type was {}".format(self.cell_type))

        #self.embed_td = Embedding(self.input_symbols, self.hidden_size, random_state=random_state, name=name + "_embed_td")
        #self.embed_fd = Embedding(self.input_symbols, self.hidden_size, random_state=random_state, name=name + "_embed_fd")
        proj_dim = self.input_dim if self.input_dim is not None else self.hidden_size
        self.td_input_proj = Linear([proj_dim],
                                    self.hidden_size,
                                    random_state=random_state,
                                    init=init,
                                    scale=scale,
                                    name=name + "_td_input_proj",
                                    device=device)
        self.fd_input_proj = Linear([proj_dim],
                                    self.hidden_size,
                                    random_state=random_state,
                                    init=init,
                                    scale=scale,
                                    name=name + "_fd_input_proj",
                                    device=device)

        self.tds_seq_time_fw = nn.ModuleList()
        self.tds_seq_freq_fw = nn.ModuleList()
        self.tds_seq_freq_bw = nn.ModuleList()
        self.tds_projs = nn.ModuleList()
        self.fds_projs = nn.ModuleList()
        self.fds_seq_freq_fw = nn.ModuleList()
        if self.has_centralized_stack:
            self.centralized_input_proj = Linear([self.n_horiz],
                                                 self.hidden_size,
                                                 random_state=random_state,
                                                 init=init,
                                                 scale=scale,
                                                 name=name + "_centralized_input_proj",
                                                 device=device)
            self.cds_centralized_seq = nn.ModuleList()
            self.cds_projs = nn.ModuleList()

        if self.has_spatial_condition:
            if spatial_condition_input_size is None:
                raise ValueError("has_spatial_condition was True, but no spatial_condition_input_size specified!")
            self.spatial_condition_input_size = spatial_condition_input_size
            self.cond_net = YAMNetFullContextSubTier([spatial_condition_input_size],
                                                     n_vert,
                                                     n_horiz,
                                                     self.hidden_size,
                                                     self.spatial_condition_n_layers,
                                                     n_heads=self.spatial_condition_n_heads,
                                                     head_dim=self.spatial_condition_head_dim,
                                                     inner_dim=self.spatial_condition_inner_dim,
                                                     random_state=random_state,
                                                     init=init,
                                                     name=name + "cond_YAMNet",
                                                     device=device)

        if self.has_centralized_stack:
            if self.has_attention:
                self.attn_lstm_cell = rnn_cell_fn([self.hidden_size + self.hidden_size],
                                                   self.hidden_size,
                                                   random_state=random_state,
                                                   init=init,
                                                   scale=scale,
                                                   name=name + "_attn_rnn_cell",
                                                   device=device)
                """
                self.attn_reduction_proj = Linear([self.hidden_size,],
                                                  self.n_horiz,
                                                  random_state=random_state,
                                                  init=init,
                                                  scale=scale,
                                                  name=name + "_attention_reduction_proj",
                                                  device=device)
                """
                if self.attention_type == "logistic":
                    # for logistic sigmoid attention ala melnet
                    self.attn_proj = Linear([3 * self.hidden_size],
                                             3 * self.attention_mixture_components,
                                             random_state=random_state,
                                             init=init,
                                             scale=scale,
                                             name=name + "_attn_proj",
                                             device=device)
                elif self.attention_type == "sigmoid_logistic":
                    self.attn_proj = Linear([3 * self.hidden_size],
                                             3 * self.attention_mixture_components,
                                             random_state=random_state,
                                             init=init,
                                             scale=scale,
                                             name=name + "_attn_proj",
                                             device=device)
                elif self.attention_type == "sigmoid_logistic_alt":
                    self.attn_proj = Linear([3 * self.hidden_size],
                                             3 * self.attention_mixture_components,
                                             random_state=random_state,
                                             init=init,
                                             scale=scale,
                                             name=name + "_attn_proj",
                                             device=device)
                elif self.attention_type == "gaussian":
                    # for logistic sigmoid attention ala melnet
                    self.attn_proj = Linear([3 * self.hidden_size],
                                             3 * self.attention_mixture_components,
                                             random_state=random_state,
                                             init=init,
                                             scale=scale,
                                             name=name + "_attn_proj",
                                             device=device)
                elif self.attention_type == "dca":
                    # for convolutional attention
                    # https://github.com/bshall/Tacotron/blob/6fee34a7c3a9d4ceb9215ed3063771a9287010e1/tacotron/model.py
                    prior_length = 11
                    alpha = .1
                    beta = .9
                    # expected average step size (encoder steps per decoder step) is 1 here
                    # alpha * prior_length / (alpha + beta)
                    # makes sense because of the extreme downsampling
                    # roughly on 1 to 1 scale but may need to be tuned
                    # realistically more like .8 encoder steps per decoder steps on datasets
                    P = betabinom.pmf(np.arange(prior_length), prior_length - 1, alpha, beta)
                    self.register_buffer("P", torch.FloatTensor(P).flip(0))
                    # note - W_g in paper
                    self.W_g = Linear([self.hidden_size],
                                     self.hidden_size,
                                     random_state=random_state,
                                     init=init,
                                     scale=scale,
                                     name=name + "_attn_W_term",
                                     device=device)
                    dynamic_channels = 8
                    dynamic_kernel_size = 21
                    # note - V_g in paper
                    self.V_g = Linear([self.hidden_size],
                                     dynamic_channels * dynamic_kernel_size,
                                     random_state=random_state,
                                     init=init,
                                     scale=scale,
                                     name=name + "_attn_V_term",
                                     biases=False,
                                     device=device)
                    static_channels = 8
                    static_kernel_size = 21
                    self.attn_prior_length = prior_length
                    self.attn_alpha = alpha
                    self.attn_beta = beta
                    self.attn_dynamic_channels = dynamic_channels
                    self.attn_dynamic_kernel_size = dynamic_kernel_size
                    self.attn_static_channels = static_channels
                    self.attn_static_kernel_size = static_kernel_size
                    self.F = Conv1d([1], static_channels, kernel_size=(static_kernel_size, static_kernel_size), random_state=random_state,
                                    name=name + "_attn_F_term", biases=False, device=device)
                    self.U = Linear([static_channels],
                                     self.hidden_size,
                                     random_state=random_state,
                                     init=init,
                                     scale=scale,
                                     name=name + "_attn_U_term",
                                     biases=False,
                                     device=device)
                    self.T = Linear([dynamic_channels],
                                     self.hidden_size,
                                     random_state=random_state,
                                     init=init,
                                     scale=scale,
                                     name=name + "_attn_T_term",
                                     device=device)
                    self.v = Linear([self.hidden_size],
                                     1,
                                     random_state=random_state,
                                     init=init,
                                     scale=scale,
                                     name=name + "_attn_v_term",
                                     biases=False,
                                     device=device)
                elif self.attention_type == "lsa":
                    # https://github.com/NVIDIA/tacotron2/blob/master/model.py#L29
                    # hardcode temporarily
                    attention_dim = 128
                    self.query_layer = Linear([self.hidden_size],
                                               attention_dim,
                                               random_state=random_state,
                                               init=init,
                                               scale=scale,
                                               name=name + "_attn_query_term",
                                               biases=False,
                                               device=device)
                    # unused?
                    self.memory_layer = Linear([self.hidden_size],
                                               attention_dim,
                                               random_state=random_state,
                                               init=init,
                                               scale=scale,
                                               name=name + "_attn_memory_term",
                                               biases=False,
                                               device=device)
                    self.v_layer = Linear([attention_dim],
                                           1,
                                           random_state=random_state,
                                           init=init,
                                           scale=scale,
                                           name=name + "_attn_v_term",
                                           biases=False,
                                           device=device)
                    attention_n_filters = 32
                    attention_kernel_size = 31
                    padding = int((attention_kernel_size - 1) / 2)
                    self.location_conv = Conv1d([2], attention_n_filters, kernel_size=(attention_kernel_size, attention_kernel_size),
                                                random_state=random_state,
                                                name=name + "_attn_location_conv", biases=False, device=device)
                    self.location_dense = Linear([attention_n_filters],
                                                  attention_dim,
                                                  random_state=random_state,
                                                  init=init,
                                                  scale=scale,
                                                  name=name + "_attn_location_dense",
                                                  biases=False,
                                                  device=device)
                else:
                    raise ValueError("Unknown value in initialization for attention_type in AttnMelNetTier, got {}".format(attention_type))


        for _i in range(self.n_layers):
            memory_len=0
            context_len=0
            # n_layers=16, n_heads=10, head_dim=38, model_dim=380, inner_dim=900,
            # hardcode temporarily, maybe add automatic routine to select heads + head_dim
            time_fw_layer = AWDTransformerXLDecoderBlock([self.hidden_size,],
                                                          name=name + "_tds_seq_time_fw_{}".format(_i),
                                                          n_layers=self.transformer_inner_layers,
                                                          n_heads=self.transformer_n_heads,
                                                          head_dim=self.transformer_head_dim,
                                                          model_dim=self.hidden_size,
                                                          inner_dim=self.transformer_inner_dim,
                                                          random_state=random_state,
                                                          memory_len=memory_len,
                                                          context_len=context_len,
                                                          init=init,
                                                          scale=scale,
                                                          device=device)
            self.tds_seq_time_fw.append(time_fw_layer)
            freq_fw_layer = AWDTransformerXLDecoderBlock([self.hidden_size,],
                                                          name=name + "_tds_seq_freq_fw_{}".format(_i),
                                                          n_layers=self.transformer_inner_layers,
                                                          n_heads=self.transformer_n_heads,
                                                          head_dim=self.transformer_head_dim,
                                                          model_dim=self.hidden_size,
                                                          inner_dim=self.transformer_inner_dim,
                                                          random_state=random_state,
                                                          memory_len=memory_len,
                                                          context_len=context_len,
                                                          init=init,
                                                          scale=scale,
                                                          device=device)

            freq_bw_layer = AWDTransformerXLDecoderBlock([self.hidden_size,],
                                                          name=name + "_tds_seq_freq_bw_{}".format(_i),
                                                          n_layers=self.transformer_inner_layers,
                                                          n_heads=self.transformer_n_heads,
                                                          head_dim=self.transformer_head_dim,
                                                          model_dim=self.hidden_size,
                                                          inner_dim=self.transformer_inner_dim,
                                                          random_state=random_state,
                                                          memory_len=memory_len,
                                                          context_len=context_len,
                                                          init=init,
                                                          scale=scale,
                                                          device=device)

            self.tds_seq_freq_fw.append(freq_fw_layer)
            self.tds_seq_freq_bw.append(freq_bw_layer)

            self.tds_projs.append(Linear([3 * self.hidden_size,],
                                          self.hidden_size,
                                          random_state=random_state,
                                          init=init,
                                          scale=scale,
                                          name=name + "_tds_projs_{}".format(_i),
                                          device=device))

            self.fds_projs.append(Linear([self.hidden_size,],
                                          self.hidden_size,
                                          random_state=random_state,
                                          init=init,
                                          scale=scale,
                                          name=name + "_fds_projs_{}".format(_i),
                                          device=device))

            fd_freq_fw_layer = AWDTransformerXLDecoderBlock([self.hidden_size,],
                                                            name=name + "_fds_seq_freq_fw_{}".format(_i),
                                                            n_layers=self.transformer_inner_layers,
                                                            n_heads=self.transformer_n_heads,
                                                            head_dim=self.transformer_head_dim,
                                                            model_dim=self.hidden_size,
                                                            inner_dim=self.transformer_inner_dim,
                                                            random_state=random_state,
                                                            memory_len=memory_len,
                                                            context_len=context_len,
                                                            init=init,
                                                            scale=scale,
                                                            device=device)

            self.fds_seq_freq_fw.append(fd_freq_fw_layer)


            if self.has_centralized_stack:
                cd_layer = AWDTransformerXLDecoderBlock([self.hidden_size,],
                                                         name=name + "_cds_centralized_rnn_{}".format(_i),
                                                         n_layers=self.transformer_inner_layers,
                                                         n_heads=self.transformer_n_heads,
                                                         head_dim=self.transformer_head_dim,
                                                         model_dim=self.hidden_size,
                                                         inner_dim=self.transformer_inner_dim,
                                                         random_state=random_state,
                                                         memory_len=memory_len,
                                                         context_len=context_len,
                                                         init=init,
                                                         scale=scale,
                                                         device=device)
                self.cds_centralized_seq.append(cd_layer)

                self.cds_projs.append(Linear([self.hidden_size,],
                                              self.hidden_size,
                                              random_state=random_state,
                                              init=init,
                                              scale=scale,
                                              name=name + "_cds_projs_{}".format(_i),
                                              device=device))
        self.out_proj = Linear([self.hidden_size,], self.output_size,
                               random_state=random_state,
                               init=init,
                               scale=scale,
                               name=name + "_output_proj",
                               device=device)
        # used for hook registering alternative softplus for numerical stability
        self._softplus = None

    def _time2freq(self, inp):
        inp = inp.reshape((self.n_vert, self.batch_size, self.n_horiz, -1))
        inp = inp.permute(2, 1, 0, 3)
        return inp.reshape((self.n_horiz, self.batch_size * self.n_vert, -1))

    def _freq2time(self, inp):
        # batch size set in forward!
        inp = inp.reshape((self.n_horiz, self.batch_size, self.n_vert, -1))
        inp = inp.permute(2, 1, 0, 3)
        return inp.reshape((self.n_vert, self.batch_size * self.n_horiz, -1))

    def _td_stack(self, tds, layer):

        freq_lstm_fw_h, _, = self.tds_seq_freq_fw[layer](tds)
        freq_lstm_bw_h, _, = self.tds_seq_freq_bw[layer](torch.flip(tds, [0]))
        freq_lstm_h = torch.cat((freq_lstm_fw_h, torch.flip(freq_lstm_bw_h, [0])), dim=-1)
        freq_lstm_h = self._freq2time(freq_lstm_h)

        tds_time = self._freq2time(tds)
        time_h, _ = self.tds_seq_time_fw[layer](tds_time)

        combined_h = torch.cat((freq_lstm_h, time_h), dim=-1)
        res = self.tds_projs[layer]([combined_h])

        res = self._time2freq(res)
        #return (0.5 ** 0.5) * (tds + res)
        return res

    def _cd_centralized_stack(self, cds, layer):
        cent_lstm_h, _ = self.cds_centralized_seq[layer](cds)
        res = self.cds_projs[layer]([cent_lstm_h])
        #return (0.5 ** 0.5) * (cds + cent_lstm_h)
        return res

    def _fd_stack(self, tds, fds, layer, tds_cent=None):
        # broadcast tds_cent across frequency axis
        if tds_cent is not None:
            # need to permute + reshape to match what was done for td_x, fd_x
            # so that batch combined stuff stays contiguous in the right way!
            # time2freq is freq, batch * time, -1
            # time is also self.n_vert

            # nvert batch feat to batch nvert feat
            tds_cent = tds_cent.permute(1, 0, 2)
            ext_tds_cent = tds_cent.reshape((self.batch_size * self.n_vert, -1))
            # now 1, batch * time, hidden
            # broadcasts over frequency, since the cent rnn has puts out a whole freq frame per step dim...
            ext_tds_cent = ext_tds_cent[None]
            # (fds dim 0)
            # broacasts over features, since the cent rnn has effectively seen the whole frequency
            #ext_tds_cent = ext_tds_cent + 0. * fds
            freq_lstm_stack = tds + fds + ext_tds_cent#  torch.cat((tds, fds, ext_tds_cent), axis=-1)
        else:
            #freq_lstm_stack = torch.cat((tds, fds), axis=-1)
            freq_lstm_stack = tds + fds

        freq_lstm_h, _ = self.fds_seq_freq_fw[layer](freq_lstm_stack)
        res = self.fds_projs[layer]([freq_lstm_h])
        #return (0.5 ** 0.5) * (fds + freq_lstm_h)
        return res

    def _attention_step(self, h_i, memory, memory_mask, previous_attn):
        # TODO: location sensitive attention
        print("end td")
        # https://gist.github.com/acetylSv/9dcff15bc0e895c0190c5942b573c28b
        if self.attention_type == "logistic":
            #_attention_step(self, h_i, memory, memory_mask, ksi):
            # condition on input sequence length
            phi_hat = self.attn_proj([h_i])

            #ksi = ksi + torch.exp(phi_hat[:, :self.attention_mixture_components])
            # clamp beta so it doesn't collapse?
            #beta = torch.exp(torch.clamp(phi_hat[:, self.attention_mixture_components:2 * self.attention_mixture_components], min=-4))
            # changes based on GMMv2 of https://arxiv.org/pdf/1910.10288.pdf
            # as well as https://arxiv.org/pdf/1811.07240.pdf
            ksi = previous_attn
            kappa = ksi + (3. * sigmoid(phi_hat[:, :self.attention_mixture_components]) + .05)

            beta = (5. * sigmoid(phi_hat[:, self.attention_mixture_components:2 * self.attention_mixture_components]) + .1)
            # min beta .1
            # max beta 10
            alpha = F.softmax(phi_hat[:, 2 * self.attention_mixture_components:3 * self.attention_mixture_components], dim=1)

            u = memory.new_tensor(np.arange(memory.size(0)), dtype=memory.dtype)
            u_L = u + 0.5
            u_R = u - 0.5
            #termL = torch.sum(alpha[..., None] * (1. / (1. + (torch.exp((kappa[..., None] - u_L) * beta[..., None])))), keepdim=True, dim=1)
            #termR = torch.sum(alpha[..., None] * (1. / (1. + (torch.exp((kappa[..., None] - u_R) * beta[..., None])))), keepdim=True, dim=1)
            #termL = 1. / (1. + (torch.exp((kappa[..., None] - u_L) * beta[..., None])))
            #termR = 1. / (1. + (torch.exp((kappa[..., None] - u_R) * beta[..., None])))
            pL = (u_L - kappa[..., None]) * beta[..., None]
            pR = (u_R - kappa[..., None]) * beta[..., None]
            termL = torch.sigmoid(pL)
            termR = torch.sigmoid(pR)
            alpha_termL = alpha[..., None] * termL
            alpha_termR = alpha[..., None] * termR
            weights = torch.sum(alpha_termL, dim=1) - torch.sum(alpha_termR, dim=1)

            termination = 1. - torch.sum(alpha_termL, keepdim=True, dim=1)[:, 0]
            weights = memory_mask.transpose(0, 1) * weights
            context = torch.bmm(weights[:, None], memory.permute(1, 0, 2))
            # context is B, 1, D
            # weights B, mem_T 
            extras = {}
            extras["termination"] = termination
            extras["kappa"] = kappa
            extras["beta"] = beta
        elif self.attention_type == "logistic_hack":
            #_attention_step(self, h_i, memory, memory_mask, ksi):
            # condition on input sequence length
            phi_hat = self.attn_proj([h_i])

            #ksi = ksi + torch.exp(phi_hat[:, :self.attention_mixture_components])
            # clamp beta so it doesn't collapse?
            #beta = torch.exp(torch.clamp(phi_hat[:, self.attention_mixture_components:2 * self.attention_mixture_components], min=-4))
            # changes based on GMMv2 of https://arxiv.org/pdf/1910.10288.pdf
            # as well as https://arxiv.org/pdf/1811.07240.pdf
            ksi = previous_attn
            kappa = ksi + F.softplus(phi_hat[:, :self.attention_mixture_components])
            #beta = (F.softplus(phi_hat[:, self.attention_mixture_components:2 * self.attention_mixture_components]) + 1E-4)
            # cap beta
            beta = torch.exp(phi_hat[:, self.attention_mixture_components:(2 * self.attention_mixture_components)]) + 1E-2

            logit_alpha = phi_hat[:, (2 * self.attention_mixture_components):(3 * self.attention_mixture_components)]

            #alpha = F.softmax(phi_hat[:, 2 * self.attention_mixture_components:3 * self.attention_mixture_components], dim=1)

            u = memory.new_tensor(np.arange(memory.size(0)), dtype=memory.dtype)
            u_L = u + 0.5
            u_R = u - 0.5
            #u_L = u + 1.5
            #u_R = u + 0.5
            #termL = torch.sum(alpha[..., None] * (1. / (1. + (torch.exp((kappa[..., None] - u_L) * beta[..., None])))), keepdim=True, dim=1)
            #termR = torch.sum(alpha[..., None] * (1. / (1. + (torch.exp((kappa[..., None] - u_R) * beta[..., None])))), keepdim=True, dim=1)
            # could rewrite further by taking softmax out of context
            # softmax(alpha) * exp(-softplus(a)) -> exp(alpha) * exp(-softplus(a)) / sum(exp(alpha)) -> exp(alpha - softplus(a)) / sum(exp(alpha))
            # but would probably be less stable than "stable" softmax due to sum(exp) in denominator
            # softplus(a) = log(1 + exp(a))
            # with a = ((ksi - u) / beta)
            # this overall becomes
            # 1. / exp(softplus(a)) -> exp(-softplus(a)) -> exp(-log(1 + exp(a))) -> 1./(1 + exp(a))
            #termL = torch.sum(alpha[..., None] * (torch.exp(-F.softplus(((kappa[..., None] - u_L)) * beta[..., None]))), keepdim=True, dim=1)
            #termR = torch.sum(alpha[..., None] * (torch.exp(-F.softplus(((kappa[..., None] - u_R)) * beta[..., None]))), keepdim=True, dim=1)

            # possible best derivation (inspired by https://github.com/Deepest-Project/MelNet/blob/master/model/tts.py#L138 although I don't get the 1.5 and .5 instead of -.5 and +.5, perhaps this is setting the default step size to 1. for all ksi?)
            # 1. / (1. + exp((k-u) / b)) -> 1. / (1. + exp(-(u - k) / b)) -> 1. / (1. + exp(-t)), where t is (u - k) / b
            # knowing that sigmoid(t) = 1. / (1. + exp(-t))
            # this results in sigmoid((u - k) / b) for each term
            # this means both terms L and R are bounded btwn 0 and 1, and potentially more stable than exp(-softplus) shenanigans would allow
            #termL = torch.sum(alpha[..., None] * torch.sigmoid((((u_L - kappa[..., None])) * beta[..., None])), keepdim=True, dim=1)
            #termR = torch.sum(alpha[..., None] * torch.sigmoid((((u_R - kappa[..., None])) * beta[..., None])), keepdim=True, dim=1)
            # finally since beta is bounded > 0 due to exp() activation, we note that dividing by beta and multiplying by beta are effectively the same
            # in terms of optimization paths
            # simply swapping "what regime" wrt values exp(x) < 1, and values exp(x) > 1
            # with the *key* difference being a vanishing to 0 of beta (perhaps due to very negative weights for beta or other issues during training), will not explode the whole equation
            # reweight in log space before summation?

            #termL = torch.sum(alpha[..., None] * torch.sigmoid((((u_L - kappa[..., None])) * beta[..., None])), keepdim=True, dim=1)
            #termR = torch.sum(alpha[..., None] * torch.sigmoid((((u_R - kappa[..., None])) * beta[..., None])), keepdim=True, dim=1)

            # combined term
            #termL_R = torch.sum(alpha[..., None] * (torch.sigmoid((((u_L - kappa[..., None])) * beta[..., None])) - torch.sigmoid((((u_R - kappa[..., None])) * beta[..., None]))), keepdim=True, dim=1)
            #weights = termL_R

            # introduce sum(exp(log(alpha * sigmoid))) -> sum(exp(log(alpha) + log(sigmoid)))
            # log(alpha) -> log_softmax
            # logsoftmax = logits - log(reduce_sum(exp(logits), dim))
            # log(sigmoid(q)) -> q - log(exp(q) + 1) aka q - softplus(q)
            # term = log_alpha + q - softplus(q)
            # https://math.stackexchange.com/questions/2320905/obtaining-derivative-of-log-of-sigmoid-function
            #go further beyond, do multiplication in log space (so additive)
            # then sum exps afterward
            #log_alpha = logit_alpha - torch.log(torch.sum(torch.exp(logit_alpha), keepdim=True, dim=1))
            #log_alpha = log_alpha[..., None]
            #q_L = (u_L - kappa[..., None]) * beta[..., None]
            #termL = torch.sum(torch.exp(log_alpha + q_L - F.softplus(q_L)), keepdim=True, dim=1)
            #q_R = (u_R - kappa[..., None]) * beta[..., None]
            #termR = torch.sum(torch.exp(log_alpha + q_R - F.softplus(q_R)), keepdim=True, dim=1)
            #weights = termL - termR

            # even more further beyond...
            log_alpha = log_prob_from_logits(logit_alpha, axis=1)
            log_alpha = log_alpha[..., None]
            q_L = (u_L - kappa[..., None]) * beta[..., None]
            # keep dims
            termL = torch.exp(log_sum_exp(log_alpha + q_L - F.softplus(q_L), axis=1))[:, None]
            q_R = (u_R - kappa[..., None]) * beta[..., None]
            # keep dims
            termR = torch.exp(log_sum_exp(log_alpha + q_R - F.softplus(q_R), axis=1))[:, None]
            weights = termL - termR

            termination = 1. - termL[:, 0]
            weights = memory_mask.transpose(0, 1)[:, None, :] * weights
            context = torch.bmm(weights, memory.permute(1, 0, 2))
            # context is B, 1, D
            # weights B, mem_T 
            extras = {}
            extras["termination"] = termination
            extras["kappa"] = kappa
            extras["beta"] = beta
        elif self.attention_type == "sigmoid_logistic":
            #_attention_step(self, h_i, memory, memory_mask, ksi):
            # condition on input sequence length
            phi_hat = self.attn_proj([h_i])
            # cast to 32 bit
            orig_dtype = phi_hat.dtype
            phi_hat = phi_hat.float()

            #ksi = ksi + torch.exp(phi_hat[:, :self.attention_mixture_components])
            # clamp beta so it doesn't collapse?
            #beta = torch.exp(torch.clamp(phi_hat[:, self.attention_mixture_components:2 * self.attention_mixture_components], min=-4))
            # changes based on GMMv2 of https://arxiv.org/pdf/1910.10288.pdf
            # as well as https://arxiv.org/pdf/1811.07240.pdf
            ksi = previous_attn
            ksi = ksi.float()

            """
            if self._softplus is None:
                # hook it into the main module to keep a reference around
                self._softplus = torch.nn.Softplus()
                '''
                     output z  (grad_output)
                     ___________
                     |         |
                     |  layer  |
                     |_________|

                     input x  (grad_input)
                '''
                def hook(module, grad_input, grad_output):
                    return (sigmoid(grad_output[0]),)

                self._softplus.register_backward_hook(hook)

            alt_softplus = self._softplus
            """

            def alt_log1p(x):
                # https://www.johndcook.com/blog/2012/07/25/trick-for-computing-log1x/
                # we dirty hack this to avoid div by 0 since torch.where has issues with NaN grad
                # if *either* branch has inf
                # https://github.com/pytorch/pytorch/issues/4132
                y = 1. + x
                z = y - 1.
                res = 0. * x
                z_mask = (z == 0)
                res[z_mask] = x[z_mask]
                z_nonmask = (z != 0)
                res[z_nonmask] = x[z_nonmask] * torch.log(y[z_nonmask]) / (z[z_nonmask])
                #return torch.where(z == 0, x, x * torch.log(y) / z)
                return res

            def alt_softplus(x):
                # https://stackoverflow.com/questions/44230635/avoid-overflow-with-softplus-function-in-python
                return alt_log1p(torch.exp(-torch.abs(x))) + torch.where(x > 0, x, 0. * x)

            kappa = ksi + alt_softplus(phi_hat[:, :self.attention_mixture_components]) #+ 1E-2
            #kappa = ksi + swish(phi_hat[:, :self.attention_mixture_components])

            #beta = (F.softplus(phi_hat[:, self.attention_mixture_components:2 * self.attention_mixture_components]) + 1E-4)
            #beta = torch.clamp(torch.exp(phi_hat[:, self.attention_mixture_components:2 * self.attention_mixture_components]), min=.5)
            # fix beta to 1 - hack city but works!
            #beta = 0. * phi_hat[:, self.attention_mixture_components:2 * self.attention_mixture_components] + 1.

            # aggressive clamping here, use softplus to help stability as well
            #beta = torch.clamp(F.softplus(phi_hat[:, self.attention_mixture_components:2 * self.attention_mixture_components]), min=.25, max=2.0)
            #beta = torch.clamp(F.softplus(phi_hat[:, self.attention_mixture_components:2 * self.attention_mixture_components]), min=.01, max=10.0)
            #beta = F.softplus(phi_hat[:, self.attention_mixture_components:2 * self.attention_mixture_components])

            beta = alt_softplus(phi_hat[:, self.attention_mixture_components:2 * self.attention_mixture_components] + 3.)
            # add constant 3 to beta, so the default for beta is "large" at init
            # model can learn biases to overcome this if it wants small beta

            #beta = swish(phi_hat[:, self.attention_mixture_components:2 * self.attention_mixture_components])
            #beta = swish(phi_hat[:, self.attention_mixture_components:2 * self.attention_mixture_components])

            alpha = F.softmax(phi_hat[:, 2 * self.attention_mixture_components:3 * self.attention_mixture_components], dim=1)
            #alpha = F.softplus(phi_hat[:, 2 * self.attention_mixture_components:3 * self.attention_mixture_components]) + 1E-2
            #alpha = F.log_softmax(phi_hat[:, 2 * self.attention_mixture_components:3 * self.attention_mixture_components], dim=1)

            u = memory.new_tensor(np.arange(memory.size(0)), dtype=memory.dtype)
            #u_L = u + 0.5
            #u_R = u - 0.5

            # try like this for the following reason...
            # eqn 22 of paper
            # F(u + .5; g) - F(u - 0.5; g)
            # means F(u; g) = 1./(1 + exp((k - u) / B))
            # we can interpret this as either SUBTITUTING u + 0.5 for u, or simply adding/subtracting on 0.5. This would swap the 2 terms
            # interpreting as simply adding 0.5 to the end means
            # sigm(x) = 1./(1+exp(-x))
            # x = ((-k + u) / B) = ((u - k) / B)
            # sigm((u - .5 - k) / B) as the left hand
            # sigm((u + .5 - k) / B) as the right hand
            # alternately, interpreting as substituting u - .5 for u
            # sigm((u + .5 - k) / B) as the left hand
            # sigm((u - .5 - k) / B) as the right hand
            # noting that we can multiply or divide by beta, if beta is constrained from 0 to inf
            # since / by a number from 0->1 is the same as multiplying by 1->inf
            # aka beta can parameterize the division term, or 1 / division
            # parameterizing the 1 / division means that we don't face edge cases for beta near 0, as 1/inf -> 0 and 1/0. -> inf
            u_L = u + 0.5
            u_R = u - 0.5

            """
            alternative?
            TANH(t) = [1 - exp(-2t)]/[1 + exp(-2t)]  for  t>=0
            and
            TANH(t) = [exp(2t) - 1]/[exp(2t) + 1] for t<0
            """

            # we approximate tanh with x/(1+abs(x))
            def alt_tanh(x):
                return x / (1. + torch.abs(x))
            # logistic can be expressed as 1/2 + 1/2 * tanh((x - u)/(2 * s)) instead of sigmoid
            # if beta is 1/s, this is .5 * beta
            def term(u, k, b):
                return .5 + .5 * alt_tanh((u - k) * .5 * b)

            #termL = torch.sum(alpha[..., None] * torch.sigmoid((((u_L - kappa[..., None])) * beta[..., None])), keepdim=True, dim=1)
            #termR = torch.sum(alpha[..., None] * torch.sigmoid((((u_R - kappa[..., None])) * beta[..., None])), keepdim=True, dim=1)

            termL = torch.sum(alpha[..., None] * term(u_L, kappa[..., None], beta[..., None]), keepdim=True, dim=1)
            #termR = torch.sum(alpha[..., None] * term(u_R, kappa[..., None], beta[..., None]), keepdim=True, dim=1)

            diff = (term(u_L, kappa[..., None], beta[..., None]) - term(u_R, kappa[..., None], beta[..., None]))
            weights = torch.sum(alpha[..., None] * diff, keepdim=True, dim=1)

            #termL = torch.logsumexp(alpha[..., None] + torch.nn.functional.logsigmoid((((u_L - kappa[..., None])) * beta[..., None])), keepdim=True, dim=1)
            #termR = torch.logsumexp(alpha[..., None] + torch.nn.functional.logsigmoid((((u_R - kappa[..., None])) * beta[..., None])), keepdim=True, dim=1)
            #termR_mask = torch.abs(termR < 1).type(termR.dtype)
            #termR = termR * termR_mask + 1 * (1. - termR_mask)

            #weights = termL - termR

            #weights = torch.exp(termL) - torch.exp(termR)
            #weights = torch.exp(termL / termR)

            #termL = alpha[..., None] * torch.sigmoid((((u_L - kappa[..., None])) * beta[..., None]))
            #termR = alpha[..., None] * torch.sigmoid((((u_R - kappa[..., None])) * beta[..., None]))
            #termL = torch.sigmoid((((u_L - kappa[..., None])) * beta[..., None]))
            #termR = torch.sigmoid((((u_R - kappa[..., None])) * beta[..., None]))

            #weights = torch.sum(alpha[..., None] * (termL - termR), keepdim=True, dim=1)

            termination = 1. - torch.exp(termL[:, 0])
            weights = memory_mask.transpose(0, 1)[:, None, :] * weights

            weights = weights.to(orig_dtype)
            kappa = kappa.to(orig_dtype)
            beta = beta.to(orig_dtype)
            alpha = alpha.to(orig_dtype)
            termination = termination.to(orig_dtype)

            context = torch.bmm(weights, memory.permute(1, 0, 2))
            # context is B, 1, D
            # weights B, 1, mem_T 
            extras = {}
            extras["termination"] = termination
            extras["kappa"] = kappa
            extras["beta"] = beta
            extras["alpha"] = alpha
        elif self.attention_type == "sigmoid_logistic_alt":
            #_attention_step(self, h_i, memory, memory_mask, ksi):
            # condition on input sequence length
            phi_hat = self.attn_proj([h_i])
            # cast to 32 bit
            orig_dtype = phi_hat.dtype
            phi_hat = phi_hat.float()

            #ksi = ksi + torch.exp(phi_hat[:, :self.attention_mixture_components])
            # clamp beta so it doesn't collapse?
            #beta = torch.exp(torch.clamp(phi_hat[:, self.attention_mixture_components:2 * self.attention_mixture_components], min=-4))
            # changes based on GMMv2 of https://arxiv.org/pdf/1910.10288.pdf
            # as well as https://arxiv.org/pdf/1811.07240.pdf
            ksi = previous_attn
            ksi = ksi.float()

            """
            if self._softplus is None:
                # hook it into the main module to keep a reference around
                self._softplus = torch.nn.Softplus()
                '''
                     output z  (grad_output)
                     ___________
                     |         |
                     |  layer  |
                     |_________|

                     input x  (grad_input)
                '''
                def hook(module, grad_input, grad_output):
                    return (sigmoid(grad_output[0]),)

                self._softplus.register_backward_hook(hook)

            alt_softplus = self._softplus
            """

            def alt_log1p(x):
                # https://www.johndcook.com/blog/2012/07/25/trick-for-computing-log1x/
                # we dirty hack this to avoid div by 0 since torch.where has issues with NaN grad
                # if *either* branch has inf
                # https://github.com/pytorch/pytorch/issues/4132
                y = 1. + x
                z = y - 1.
                res = 0. * x
                z_mask = (z == 0)
                res[z_mask] = x[z_mask]
                z_nonmask = (z != 0)
                res[z_nonmask] = x[z_nonmask] * torch.log(y[z_nonmask]) / (z[z_nonmask])
                #return torch.where(z == 0, x, x * torch.log(y) / z)
                return res

            def alt_softplus(x):
                # https://stackoverflow.com/questions/44230635/avoid-overflow-with-softplus-function-in-python
                return alt_log1p(torch.exp(-torch.abs(x))) + torch.where(x > 0, x, 0. * x)

            kappa = ksi + alt_softplus(phi_hat[:, :self.attention_mixture_components]) + 1E-3
            #kappa = ksi + swish(phi_hat[:, :self.attention_mixture_components])

            #beta = (F.softplus(phi_hat[:, self.attention_mixture_components:2 * self.attention_mixture_components]) + 1E-4)
            #beta = torch.clamp(torch.exp(phi_hat[:, self.attention_mixture_components:2 * self.attention_mixture_components]), min=.5)
            # fix beta to 1 - hack city but works!
            #beta = 0. * phi_hat[:, self.attention_mixture_components:2 * self.attention_mixture_components] + 1.

            # aggressive clamping here, use softplus to help stability as well
            #beta = torch.clamp(F.softplus(phi_hat[:, self.attention_mixture_components:2 * self.attention_mixture_components]), min=.25, max=2.0)
            #beta = torch.clamp(F.softplus(phi_hat[:, self.attention_mixture_components:2 * self.attention_mixture_components]), min=.01, max=10.0)
            #beta = F.softplus(phi_hat[:, self.attention_mixture_components:2 * self.attention_mixture_components])

            beta = alt_softplus(phi_hat[:, self.attention_mixture_components:2 * self.attention_mixture_components])

            #beta = swish(phi_hat[:, self.attention_mixture_components:2 * self.attention_mixture_components])
            #beta = swish(phi_hat[:, self.attention_mixture_components:2 * self.attention_mixture_components])

            alpha = F.softmax(phi_hat[:, 2 * self.attention_mixture_components:3 * self.attention_mixture_components], dim=1)
            #alpha = F.softplus(phi_hat[:, 2 * self.attention_mixture_components:3 * self.attention_mixture_components]) + 1E-2
            #alpha = F.log_softmax(phi_hat[:, 2 * self.attention_mixture_components:3 * self.attention_mixture_components], dim=1)

            u = memory.new_tensor(np.arange(memory.size(0)), dtype=memory.dtype)
            #u_L = u + 0.5
            #u_R = u - 0.5

            # try like this for the following reason...
            # eqn 22 of paper
            # F(u + .5; g) - F(u - 0.5; g)
            # means F(u; g) = 1./(1 + exp((k - u) / B))
            # we can interpret this as either SUBTITUTING u + 0.5 for u, or simply adding/subtracting on 0.5. This would swap the 2 terms
            # interpreting as simply adding 0.5 to the end means
            # sigm(x) = 1./(1+exp(-x))
            # x = ((-k + u) / B) = ((u - k) / B)
            # sigm((u - .5 - k) / B) as the left hand
            # sigm((u + .5 - k) / B) as the right hand
            # alternately, interpreting as substituting u - .5 for u
            # sigm((u + .5 - k) / B) as the left hand
            # sigm((u - .5 - k) / B) as the right hand
            # noting that we can multiply or divide by beta, if beta is constrained from 0 to inf
            # since / by a number from 0->1 is the same as multiplying by 1->inf
            # aka beta can parameterize the division term, or 1 / division
            # parameterizing the 1 / division means that we don't face edge cases for beta near 0, as 1/inf -> 0 and 1/0. -> inf
            # however, it means that making small beta we have less precision due to floating point
            u_L = u + 0.5
            u_R = u - 0.5

            """
            alternative?
            TANH(t) = [1 - exp(-2t)]/[1 + exp(-2t)]  for  t>=0
            and
            TANH(t) = [exp(2t) - 1]/[exp(2t) + 1] for t<0
            """

            # we approximate tanh with x/(1+abs(x))
            def alt_tanh(x):
                return x / (1. + torch.abs(x))
            # logistic can be expressed as 1/2 + 1/2 * tanh((x - u)/(2 * s)) instead of sigmoid
            # if beta is 1/s, this is .5 * beta
            def term(u, k, b):
                #return .5 + .5 * alt_tanh(.5 * (u - k) / b)
                # limit min beta to .01
                return .5 + .5 * alt_tanh(.5 * (u - k) * (b + .01))

            #termL = torch.sum(alpha[..., None] * torch.sigmoid((((u_L - kappa[..., None])) * beta[..., None])), keepdim=True, dim=1)
            #termR = torch.sum(alpha[..., None] * torch.sigmoid((((u_R - kappa[..., None])) * beta[..., None])), keepdim=True, dim=1)

            termL = torch.sum(alpha[..., None] * term(u_L, kappa[..., None], beta[..., None]), keepdim=True, dim=1)
            #termR = torch.sum(alpha[..., None] * term(u_R, kappa[..., None], beta[..., None]), keepdim=True, dim=1)

            diff = (term(u_L, kappa[..., None], beta[..., None]) - term(u_R, kappa[..., None], beta[..., None]))
            weights = torch.sum(alpha[..., None] * diff, keepdim=True, dim=1)

            #termL = torch.logsumexp(alpha[..., None] + torch.nn.functional.logsigmoid((((u_L - kappa[..., None])) * beta[..., None])), keepdim=True, dim=1)
            #termR = torch.logsumexp(alpha[..., None] + torch.nn.functional.logsigmoid((((u_R - kappa[..., None])) * beta[..., None])), keepdim=True, dim=1)
            #termR_mask = torch.abs(termR < 1).type(termR.dtype)
            #termR = termR * termR_mask + 1 * (1. - termR_mask)

            #weights = termL - termR

            #weights = torch.exp(termL) - torch.exp(termR)
            #weights = torch.exp(termL / termR)

            #termL = alpha[..., None] * torch.sigmoid((((u_L - kappa[..., None])) * beta[..., None]))
            #termR = alpha[..., None] * torch.sigmoid((((u_R - kappa[..., None])) * beta[..., None]))
            #termL = torch.sigmoid((((u_L - kappa[..., None])) * beta[..., None]))
            #termR = torch.sigmoid((((u_R - kappa[..., None])) * beta[..., None]))

            #weights = torch.sum(alpha[..., None] * (termL - termR), keepdim=True, dim=1)

            termination = 1. - torch.exp(termL[:, 0])
            weights = memory_mask.transpose(0, 1)[:, None, :] * weights

            # grad scaling here
            grad_rescale = 1. / np.sqrt(self.attention_mixture_components)
            weights = (1. - grad_rescale) * weights.detach() + grad_rescale * weights

            weights = weights.to(orig_dtype)
            kappa = kappa.to(orig_dtype)
            beta = beta.to(orig_dtype)
            alpha = alpha.to(orig_dtype)
            termination = termination.to(orig_dtype)

            context = torch.bmm(weights, memory.permute(1, 0, 2))
            # context is B, 1, D
            # weights B, 1, mem_T 
            extras = {}
            extras["termination"] = termination
            extras["kappa"] = kappa
            extras["beta"] = beta
            extras["alpha"] = alpha
        elif self.attention_type == "dca":
            # code ref:
            # https://github.com/bshall/Tacotron
            p = F.conv1d(F.pad(previous_attn[:, None], (self.attn_prior_length - 1, 0)), self.P[None, None])
            p = torch.log(p.clamp_min_(1E-6)[:, 0])

            G = self.V_g([self.W_g([h_i])])

            g = F.conv1d(previous_attn[None], G.view(-1, 1, self.attn_dynamic_kernel_size),
                         padding=(self.attn_dynamic_kernel_size - 1) // 2,
                         groups=h_i.shape[0])
            g = g.view(h_i.size(0), self.attn_dynamic_channels, -1).transpose(1, 2)

            f = self.F([previous_attn.transpose(1, 0)[..., None]])
            e = self.v([torch.tanh(self.U([f]) + self.T([g]).transpose(0, 1))])[..., 0]

            e = e.transpose(1, 0) + p
            # now B, mem_T
            weights = F.softmax(e, dim=1)

            # mask weights here
            # technically don't sum to 1 anymore but don't want to add weight to zero info places...
            # for now, don't mask
            #weights = memory_mask.transpose(0, 1) * weights

            context = torch.bmm(weights[:, None], memory.permute(1, 0, 2))
            # context is B, 1, D
            # TODO: fix the dims to match, should be weights B, 1, mem_T 
            # weights B, mem_T 
            extras = {}
        elif self.attention_type == "lsa":
            processed_query = self.query_layer([h_i])
            processed_memory = self.memory_layer([memory.permute(1, 0, 2)])
            processed_attention = self.location_conv([previous_attn.transpose(1, 0)])
            processed_attention_dense = self.location_dense([processed_attention]).transpose(1, 0)
            # processed_attention_dense is batch, mem_T, attn_dim
            weight_logits = self.v_layer([torch.tanh(processed_attention_dense + processed_memory + processed_query[:, None])])[..., 0]
            weights = F.softmax(weight_logits, dim=1)
            context = torch.bmm(weights[:, None], memory.permute(1, 0, 2))
            weights = weights[:, None]
            # context is B, 1, D
            # weights B, 1, mem_T 
            extras = {}
        elif self.attention_type == "gaussian":
            #_attention_step(self, h_i, memory, memory_mask, ksi):
            # condition on input sequence length
            phi_hat = self.attn_proj([h_i])

            #ksi = ksi + torch.exp(phi_hat[:, :self.attention_mixture_components])
            # clamp beta so it doesn't collapse?
            #beta = torch.exp(torch.clamp(phi_hat[:, self.attention_mixture_components:2 * self.attention_mixture_components], min=-4))
            # changes based on GMMv2 of https://arxiv.org/pdf/1910.10288.pdf
            # as well as https://arxiv.org/pdf/1811.07240.pdf
            ksi = previous_attn
            kappa = ksi + F.softplus(phi_hat[:, :self.attention_mixture_components])

            # don't need capped beta becase we parameterize the inverse
            beta = torch.exp(phi_hat[:, self.attention_mixture_components:2 * self.attention_mixture_components])
            alpha = torch.exp(phi_hat[:, 2 * self.attention_mixture_components:3 * self.attention_mixture_components])

            u = memory.new_tensor(np.arange(memory.size(0)), dtype=memory.dtype)
            #u_L = u + 0.5
            #u_R = u - 0.5
            #u_L = u + 1.5
            #u_R = u + 0.5
            #termL = torch.sum(alpha[..., None] * (1. / (1. + (torch.exp((kappa[..., None] - u_L) * beta[..., None])))), keepdim=True, dim=1)
            #termR = torch.sum(alpha[..., None] * (1. / (1. + (torch.exp((kappa[..., None] - u_R) * beta[..., None])))), keepdim=True, dim=1)
            # could rewrite further by taking softmax out of context
            # softmax(alpha) * exp(-softplus(a)) -> exp(alpha) * exp(-softplus(a)) / sum(exp(alpha)) -> exp(alpha - softplus(a)) / sum(exp(alpha))
            # but would probably be less stable than "stable" softmax due to sum(exp) in denominator
            # softplus(a) = log(1 + exp(a))
            # with a = ((ksi - u) / beta)
            # this overall becomes
            # 1. / exp(softplus(a)) -> exp(-softplus(a)) -> exp(-log(1 + exp(a))) -> 1./(1 + exp(a))
            #termL = torch.sum(alpha[..., None] * (torch.exp(-F.softplus(((kappa[..., None] - u_L)) * beta[..., None]))), keepdim=True, dim=1)
            #termR = torch.sum(alpha[..., None] * (torch.exp(-F.softplus(((kappa[..., None] - u_R)) * beta[..., None]))), keepdim=True, dim=1)

            # possible best derivation (inspired by https://github.com/Deepest-Project/MelNet/blob/master/model/tts.py#L138 although I don't get the 1.5 and .5 instead of -.5 and +.5, perhaps this is setting the default step size to 1. for all ksi?)
            # 1. / (1. + exp((k-u) / b)) -> 1. / (1. + exp(-(u - k) / b)) -> 1. / (1. + exp(-t)), where t is (u - k) / b
            # knowing that sigmoid(t) = 1. / (1. + exp(-t))
            # this results in sigmoid((u - k) / b) for each term
            # this means both terms L and R are bounded btwn 0 and 1, and potentially more stable than exp(-softplus) shenanigans would allow
            #termL = torch.sum(alpha[..., None] * torch.sigmoid((((u_L - kappa[..., None])) * beta[..., None])), keepdim=True, dim=1)
            #termR = torch.sum(alpha[..., None] * torch.sigmoid((((u_R - kappa[..., None])) * beta[..., None])), keepdim=True, dim=1)
            # finally since beta is bounded > 0 due to exp() activation, we note that dividing by beta and multiplying by beta are effectively the same
            # in terms of optimization paths
            # simply swapping "what regime" wrt values exp(x) < 1, and values exp(x) > 1
            # with the *key* difference being a vanishing to 0 of beta (perhaps due to very negative weights for beta or other issues during training), will not explode the whole equation
            # reweight in log space before summation?
            weights = torch.sum(alpha[..., None] * torch.exp(-1. * ((kappa[..., None] - u) ** 2) * beta[..., None]), keepdim=True, dim=1)
            weights = memory_mask.transpose(0, 1)[:, None, :] * weights
            context = torch.bmm(weights, memory.permute(1, 0, 2))
            # context is B, 1, D
            # weights B, mem_T 
            extras = {}
            extras["kappa"] = kappa
            extras["beta"] = beta
            extras["alpha"] = alpha
        return context, weights, extras

    def _attention(self, cds, layer, memory, memory_mask):
        T, B, D = cds.size()
        # make init a function of the mean of the 
        h_i = cds.new_zeros(B, self.hidden_size)
        c_i = cds.new_zeros(B, self.hidden_size)
        context = cds.new_zeros(B, self.hidden_size)
        if self.attention_type == "logistic":
            prev_attn = cds.new_zeros(B, self.attention_mixture_components)
        elif self.attention_type == "sigmoid_logistic":
            prev_attn = cds.new_zeros(B, self.attention_mixture_components)
        elif self.attention_type == "sigmoid_logistic_alt":
            prev_attn = cds.new_zeros(B, self.attention_mixture_components)
        elif self.attention_type == "dca":
            prev_attn = F.one_hot(torch.zeros(B, dtype=torch.long, device=cds.device), memory.size(0)).float()
        elif self.attention_type == "gaussian":
            prev_attn = cds.new_zeros(B, self.attention_mixture_components)
        elif self.attention_type == "lsa":
            # one dim for attention, one for accumulative weights
            prev_attn_base = cds.new_zeros(B, memory.size(0), 1)
            prev_attn_accum = cds.new_zeros(B, memory.size(0), 1)
        else:
            raise ValueError("Unknown self.attention_type {} found".format(self.attention_type))
        contexts = []
        weights = []
        terminations = []
        all_extras = []
        out_hiddens = []
        for _i in range(T):
            x = torch.cat([cds[_i], context.squeeze(1)], dim=-1)
            out, s = self.attn_lstm_cell([x],
                                         h_i, c_i,
                                         input_mask=None)
            h_t, c_t = s[0], s[1]
            out_hiddens.append(h_t[None])

            if self.attention_type == "lsa":
                prev_attn = torch.cat((prev_attn_base, prev_attn_accum), dim=2)

            h_comb = torch.cat([cds[_i], context.squeeze(1), h_t], dim=-1)
            #context, attn_weight, extras = self._attention_step(h_t, memory, memory_mask, prev_attn)
            context, attn_weight, extras = self._attention_step(h_comb, memory, memory_mask, prev_attn)

            if self.attention_type == "logistic":
                prev_attn = extras["kappa"]
            elif self.attention_type == "sigmoid_logistic":
                prev_attn = extras["kappa"]
            elif self.attention_type == "sigmoid_logistic_alt":
                prev_attn = extras["kappa"]
            elif self.attention_type == "dca":
                prev_attn = attn_weight
            elif self.attention_type == "lsa":
                prev_attn_base = attn_weight[:, 0][..., None]
                prev_attn_accum += attn_weight[:, 0][..., None]
            elif self.attention_type == "gaussian":
                prev_attn = extras["kappa"]
            else:
                raise ValueError("Unknown argument to self.attention_type {}".format(self.attention_type))
            contexts.append(context)
            weights.append(attn_weight[None])
            all_extras.append(extras)
            h_i, c_i = h_t, c_t
        # skip hidden? for better attn control?
        if self.attention_type == "sigmoid_logistic":
            out_contexts = torch.cat(contexts, axis=1).permute(1, 0, 2) + torch.cat(out_hiddens, axis=0)
        if self.attention_type == "sigmoid_logistic_alt":
            out_contexts = torch.cat(contexts, axis=1).permute(1, 0, 2) + torch.cat(out_hiddens, axis=0)
        elif self.attention_type == "gaussian":
            out_contexts = torch.cat(contexts, axis=1).permute(1, 0, 2) + torch.cat(out_hiddens, axis=0)
        else:
            out_contexts = torch.cat(contexts, axis=1).permute(1, 0, 2) + torch.cat(out_hiddens, axis=0)
        #contexts = torch.cat(contexts, axis=1).permute(1, 0, 2) + torch.cat(out_hiddens, axis=0)
        #contexts = torch.cat(contexts, axis=1).permute(1, 0, 2) + tds
        # decoder_T, B, D
        # absolutely no bypassing this?
        # decoder_T, B, encoder_T
        alignments = torch.cat(weights, axis=0)
        return out_contexts, alignments, all_extras

    # conditional first tier
    def forward(self, list_of_inputs, list_of_spatial_conditions=None, bypass_td=None, bypass_fd=None,
                      memory=None, memory_mask=None):
        # by default embed the inputs, otherwise bypass
        # condidering axis 2 time 3 frequency

        # shift and project the input
        if len(list_of_inputs) > 1:
            raise ValueError("Only support list_of_inputs length 1 for now")

        x = list_of_inputs[0]
        td_x = torch.cat((0 * x[:, 0][:, None], x[:, :-1]), dim=1)
        fd_x = torch.cat((0 * x[:, :, 0][:, :, None], x[:, :, :-1]), dim=2)
        batch_size = x.shape[0]
        self.batch_size = batch_size

        if self.has_centralized_stack:
            # x should has dim of size 1 on the last for input
            cd_x = torch.cat((0 * x[:, 0][:, None], x[:, :-1]), dim=1)[..., 0]
            # cd is now t b f
            cd_x = cd_x.permute(1, 0, 2)
            cd_x = self.centralized_input_proj([cd_x])

        #td_x, td_e = self.embed_td(td_x)
        #fd_x, fd_e = self.embed_fd(fd_x)
        # reshape so the dot works
        td_x = td_x.reshape((batch_size * self.n_vert, self.n_horiz, -1))
        fd_x = fd_x.reshape((batch_size * self.n_vert, self.n_horiz, -1))
        td_x = self.td_input_proj([td_x])
        fd_x = self.fd_input_proj([fd_x])
        td_x = td_x.reshape((batch_size, self.n_vert, self.n_horiz, -1))
        fd_x = fd_x.reshape((batch_size, self.n_vert, self.n_horiz, -1))
        # un reshape it?

        if bypass_td is not None:
            td_x = bypass_td
        if bypass_fd is not None:
            fd_x = bypass_fd

        if self.has_attention:
            assert memory is not None
            assert memory_mask is not None

        if self.has_spatial_condition:
            cond_info = self.cond_net([list_of_spatial_conditions[0]])
            td_x = td_x + cond_info
            fd_x = fd_x + cond_info

        if self.has_attention:
            # t b f to b t f to stretch
            #mem_shp = memory.shape
            #memory_stretch = memory.permute(1, 0, 2)[:, :, None, :] + 0. * td_x[:, :1, :, :1]
            #memory_stretch = memory_stretch.permute(0, 2, 1, 3).reshape((batch_size * self.n_vert, mem_shp[0], mem_shp[2]))
            # back to t b f
            #memory_stretch = memory_stretch.permute(1, 0, 2)
            memory_stretch = memory
        # b, n_vert, n_horiz, feat
        # batch, mel_time, mel_freq, feats -> batch * mel_time, mel_freq, feats
        td_x = td_x.reshape((batch_size * self.n_vert, self.n_horiz, -1))
        fd_x = fd_x.reshape((batch_size * self.n_vert, self.n_horiz, -1))
        # horiz (freq), batch * vert, feat
        td_x = td_x.permute(1, 0, 2)
        fd_x = fd_x.permute(1, 0, 2)
        # cd_x is n_vert, batch, freq
        has_cd_att = False
        td_x_i = td_x
        fd_x_i = fd_x
        if self.has_centralized_stack:
            cd_x_i = cd_x

        def res(l):
            #return l
            return (0.5 ** 0.5) * l
            #return layer_norm(l, eps=1E-3)
            #return (0.5 ** 0.5) * layer_norm(l, eps=1E-4)

        def layer_norm(x, dim=-1, eps=1E-3):
            # need a low eps for fp16
            mean = torch.mean(x, dim=dim, keepdim=True)
            var = torch.square(x - mean).mean(dim=dim, keepdim=True)
            return (x - mean) / torch.sqrt(var + eps)


        for _i in range(self.n_layers):
            td_x_o = self._td_stack(td_x_i, _i)
            if self.has_centralized_stack:
                if _i == (self.n_layers // 2) and self.has_attention:
                    cd_att, alignment, attn_extras = self._attention(cd_x_i, _i, memory, memory_mask)
                    has_cd_att = True
                    # should this just replace the centralized stack here?
                    cd_x_o = self._cd_centralized_stack(res(cd_x_i + cd_att), _i)
                    fd_x_o = self._fd_stack(res(td_x_o + td_x_i), fd_x_i, _i, tds_cent=res(cd_x_o + cd_x_i + cd_att))
                else:
                    if has_cd_att is False:
                        cd_x_o = self._cd_centralized_stack(cd_x_i, _i)
                        fd_x_o = self._fd_stack(res(td_x_o + td_x_i), fd_x_i, _i, tds_cent=res(cd_x_i + cd_x_o))
                    else:
                        cd_x_o = self._cd_centralized_stack(cd_x_i + cd_att, _i)
                        fd_x_o = self._fd_stack(res(td_x_o + td_x_i), fd_x_i, _i, tds_cent=res(cd_x_o + cd_x_i + cd_att))
            else:
                fd_x_o = self._fd_stack(res(td_x_o + td_x_i), fd_x_i, _i)
            fd_x_i = res(fd_x_o + fd_x_i)
            td_x_i = res(td_x_o + td_x_i)
            if self.has_centralized_stack:
                cd_x_i = res(cd_x_o + cd_x_i)
                # don't add in the attention because we manually add it everwhere cd_x_i is used
                #cd_x_i = cd_x_o + cd_x_i + cd_att
            # set to none to be ensure no "carryover" / errors
            td_x_o = None
            fd_x_o = None
            cd_x_o = None
        out = self.out_proj([fd_x_i])
        out = out.reshape((self.n_horiz, self.batch_size, self.n_vert, self.output_size))
        out = out.permute((1, 2, 0, 3))
        if self.has_attention:
            return out, alignment, attn_extras
        else:
            return out
