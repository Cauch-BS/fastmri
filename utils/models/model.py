# MIT License

# Copyright (c) 2023 Cloudwalker

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# NOTE - This file is sourced from the following repository:
# LINK - https://github.com/pranavphoenix/WavePaint

# SECTION[epic=Imports]
from typing import Union, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function
import pywt
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from numpy.lib.function_base import hamming

MODES = [
    "zero",
    "symmetric",
    "periodization",
    "constant",
    "reflect",
    "replicate",
    "periodic",
]


# SECTION[epic=utils]
def sfb1d(
    lo: torch.tensor,
    hi: torch.tensor,
    g0: Union[torch.tensor, np.ndarray, List[float]],
    g1: Union[torch.tensor, np.ndarray, List[float]],
    mode: str = "zero",
    dim: int = -1,
) -> torch.tensor:
    """
    1D synthesis filter bank of an image tensor.
    This is the inverse operation of the 1D analysis filter bank, afb1d.

    Args:
        lo: low-pass filtered tensor
        hi: high-pass filtered tensor
        g0: low-pass filter (4D input of shape (1, 1, h, 1) or (1, 1, 1, w)). Can be passed as a numpy array or list
        g1: high-pass filter (4D input of shape (1, 1, h, 1) or (1, 1, 1, w)). Can be passed as a numpy array or list
        mode: padding mode. string in {'zero', 'symmetric', 'periodization', 'constant', 'reflect', 'replicate', 'periodic'}
        dim: dimension along which to apply the filter

    Returns:
        y: reconstructed tensor
    """
    C = lo.shape[1]
    d = dim % 4
    # make g0 and g1 tensors
    if not isinstance(g0, torch.Tensor):
        g0 = torch.tensor(
            np.copy(np.array(g0).ravel()), dtype=torch.float, device=lo.device
        )
    if not isinstance(g1, torch.Tensor):
        g1 = torch.tensor(
            np.copy(np.array(g1).ravel()), dtype=torch.float, device=lo.device
        )
    L = g0.numel()
    shape = [1, 1, 1, 1]
    shape[d] = L
    N = 2 * lo.shape[d]
    # if g0 or g1 are not the right shape, make them so
    # If g aren't in the right shape, make them so
    if g0.shape != tuple(shape):
        g0 = g0.reshape(*shape)
    if g1.shape != tuple(shape):
        g1 = g1.reshape(*shape)

    s = (2, 1) if d == 2 else (1, 2)
    g0 = torch.cat([g0] * C, dim=0)
    g1 = torch.cat([g1] * C, dim=0)
    if mode == "per" or mode == "periodization":
        y = F.conv_transpose2d(lo, g0, stride=s, groups=C) + F.conv_transpose2d(
            hi, g1, stride=s, groups=C
        )
        if d == 2:
            y[:, :, : L - 2] = y[:, :, : L - 2] + y[:, :, N : N + L - 2]
            y = y[:, :, :N]
        else:
            y[:, :, :, : L - 2] = y[:, :, :, : L - 2] + y[:, :, :, N : N + L - 2]
            y = y[:, :, :, :N]
        y = torch.roll(y, 1 - L // 2, dim=dim)
    else:
        if (
            mode == "zero"
            or mode == "symmetric"
            or mode == "reflect"
            or mode == "periodic"
        ):
            pad = (L - 2, 0) if d == 2 else (0, L - 2)
            y = F.conv_transpose2d(
                lo, g0, stride=s, padding=pad, groups=C
            ) + F.conv_transpose2d(hi, g1, stride=s, padding=pad, groups=C)
        else:
            raise ValueError("Unkown pad type: {}".format(mode))

    return y


def reflect(x: np.ndarray, minx: float, maxx: float) -> np.ndarray:
    """
    Reflect the values in matrix *x* about the scalar values *minx* and
    *maxx*.  Hence a vector *x* containing a long linearly increasing series is
    converted into a waveform which ramps linearly up and down between *minx*
    and *maxx*.  If *x* contains integers and *minx* and *maxx* are (integers +
    0.5), the ramps will have repeated max and min samples.
    .. codeauthor:: Rich Wareham <rjw57@cantab.net>, Aug 2013
    .. codeauthor:: Nick Kingsbury, Cambridge University, January 1999.

    Args:
        x: input array
        minx: minimum value
        maxx: maximum value

    Returns:
        out: reflected array
    """
    x = np.asanyarray(x)
    rng = maxx - minx
    rng_by_2 = 2 * rng
    mod = np.fmod(x - minx, rng_by_2)
    normed_mod = np.where(mod < 0, mod + rng_by_2, mod)
    out = np.where(normed_mod >= rng, rng_by_2 - normed_mod, normed_mod) + minx
    return np.array(out, dtype=x.dtype)


def mode_to_int(mode) -> int:
    """
    Convert a mode to an integer.
    Args:
        mode: string. One of the following: 'zero', 'symmetric', 'periodization', 'constant', 'reflect', 'replicate', 'periodic'
    Returns:
        int: integer corresponding to the mode
    """
    if mode in MODES:
        return MODES.index(mode)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def int_to_mode(mode: int) -> str:
    """
    Convert an integer to a mode.
    Args:
        mode: integer. One of the following: 0, 1, 2, 3, 4, 5, 6
    Returns:
        str: string corresponding to the mode
    """
    if mode < len(MODES):
        return MODES[mode]
    else:
        raise ValueError(f"Unknown mode: {mode}")


def abf1d(
    x: torch.tensor,
    h0: Union[torch.tensor, np.ndarray, List[float]],
    h1: Union[torch.tensor, np.ndarray, List[float]],
    mode: str = "zero",
    dim: int = -1,
) -> torch.tensor:
    """
    1D analysis filter bank (along one dimension only)
    of an image tensor

    Args:
        x: input tensor (4D input, with last two entries the spatial input)
        h0: low-pass filter (4D input of shape (1, 1, h, 1) or (1, 1, 1, w)). Can be passed as a numpy array or list
        h1: high-pass filter (4D input of shape (1, 1, h, 1) or (1, 1, 1, w)). Can be passed as a numpy array or list
        mode: padding mode. string in {'zero', 'symmetric', 'periodization', 'constant', 'reflect', 'replicate', 'periodic'}
        dim: dimension along which to apply the filter
            d = 2: vertical filter (filter across the rows)
            d = 3: horizontal filter (filter across the columns)

    Returns:
        lohi: low-pass and high-pass filtered tensor subbands concatenated across the channel dimension
    """
