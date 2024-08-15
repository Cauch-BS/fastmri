# MIT License

# Copyright (c) 2019 Kristian Monsen Haug and Mathias Lohne

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

"""
The discrete_wt module contains methods for 1 dimensional, 2 dimensional or the inverse discrete wavelet transform.

Functions:
    dwt1d: 1 dimensional discrete wavelet transform
    dwt2d: 2 dimensional discrete wavelet transform
    idwt1d: 1 dimensional inverse discrete wavelet transform
    idwt2d: 2 dimensional inverse discrete wavelet transform
"""

import torch
from wavelets import Filter, Wavelet


def _cyclic_conv1d(input_node: torch.tensor, filter: Filter):
    """
    Cyclic Convolution 1D
    Args:
      input_node: input tensor

    """
