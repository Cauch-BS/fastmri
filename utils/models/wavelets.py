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
The wavelet module contains functions for creating wavelet filters and performing wavelet transforms.

Classes:
    Filter: A filter object used in wavelet transforms
    DifferentiableFilter: A differentiable filter object used in autograd computations
    Wavelet: A wavelet object containing the filters for decomposition and reconstruction
    DifferentiableWavelet: A differentiable wavelet object used in autograd computations

Functions:
    fetch_db: Fetches the Daubechies wavelet filters of order i
    fetch_haar: Fetches the Haar wavelet filters

Examples: you can define your own wavelet by creating four filters and combining them into a wavelet object.

>>> # First, define the low pass filter for decomposition
>>> decomp_lp = Filter([1 / np.sqrt(2), 1 / np.sqrt(2)]. 0)
>>> # Second, define the high pass filter for decomposition
>>> decomp_hp = Filter([1 / np.sqrt(2), -1 / np.sqrt(2)], 1)
>>> # Now, we define the low pass filter for reconstruction
>>> recon_lp = Filter([1 / np.sqrt(2), 1 / np.sqrt(2)], 0)
>>> # Finally, we define the high pass filter for reconstruction
>>> recon_hp = Filter([-1 / np.sqrt(2), 1 / np.sqrt(2)], 1)
>>> # Combine the filters into a wavelet object
>>> haar = Wavelet(decomp_lp, decomp_hp, recon_lp, recon_hp)
"""

# SECTION[epic=imports] 1: IMPORTS
from typing import Iterable, Union
import numpy as np
import torch


# !SECTION 1: END IMPORTS
# SECTION[epic=utils] 2: UTILITIES
def _adapt_filter(filter: np.ndarray) -> np.ndarray:
    """
    Expands dimensions of a 1d vector to match the required tensor dimensions in a PyTorch
    graph.

    Args:
        filter (np.ndarray):     A 1D vector containing filter coefficients

    Returns:
        np.ndarray: A 3D vector with two empty dimensions as dim 2 and 3.

    """
    # Add empty dimensions for batch size and channel num
    return np.expand_dims(filter, (-1, -2))


def _to_torch_mat(
    matrices: Iterable[np.ndarray], dtype=torch.float32
) -> Iterable[torch.tensor]:
    """
    Expands dimensions of 2D matrices to match the required tensor dimensions in a PyTorch.

    Args:
        matrices (iterable):    A list (or tuple) of 2D numpy arrays.

    Returns:
        torch.tensor: A tensor with the matrices converted to 3D PyTorch tensors, concatenated along the first dimension.
    """

    return (
        [torch.unsqueeze(torch.tensor(matrix, dtype=dtype), 0) for matrix in matrices],
    )


# !SECTION 2: END UTILITIES


# SECTION[epic=classes] 3: CLASSES
# SECTION 3.1. FILTER
class Filter:
    """
    Represents a filter used in wavelet transforms:

    Attributes:
        coeffs (torch.Tensor):  The filter coefficients
        npcoeffs (np.ndarray):  The filter coefficients as a numpy array
        zero (int):             The filter origin index
        edge_matrices (torch.Tensor):   The edge matrices used for the convolution operation. Stored as a 3D tensor.

    Methods:
        num_pos:               Returns the number of positive coefficients in the filter.
        num_neg:               Returns the number of negative coefficients in the filter.
    """

    def __init__(self, coeffs: np.ndarray, zero: int):
        """
        Initializes a filter object.

        Args:
            coeffs (np.ndarray):  The filter coefficients
            zero (int):            The filter origin index
        """
        if not isinstance(coeffs, np.ndarray):
            try:
                np.ndarray(coeffs, dtype=np.float32)
            except Exception as e:
                raise TypeError(
                    f"Coeffs must be a numpy array, tried to convert but failed with {e}"
                )
        self.coeffs = torch.tensor(_adapt_filter(coeffs), dtype=torch.float32)
        self.npcoeffs = coeffs.ravel().astype(np.float32)

        if not isinstance(zero, int):
            raise TypeError("Zero must be an integer")

        self.zero = zero

        self.edge_matrices = _to_torch_mat(self._init_edge())

    def _init_edge(self) -> Iterable[np.ndarray]:
        """
        Computes the submatrices needed at the ends for the circular convolution.

        Returns:
            Iterable[np.ndarray]: A tuple of edge matrices.
            They follow the order:
                - top left
                - top right
                - bottom left
                - bottom right
        """
        (n,) = self.npcoeffs.shape
        self.npcoeffs = np.flip(self.npcoeffs)

        # padding to keep the edge matrices mutually exclusive
        padding = np.max([self.zero, n - self.zero - 1])
        mat_size = n + padding
        filter_mat = np.zeros((mat_size, mat_size), dtype=np.float32)
        neg = self.npcoeffs[
            -(self.zero + 1) :
        ]  # negatively indexed filter coeffs including zero

        pos = self.npcoeffs[
            : -(self.zero + 1)
        ]  # Strictly positively indexed filter coeffs

        num_neg, num_pos = len(neg), len(pos)

        # insert the coefficients if they are non-empty
        filter_mat[0, :num_neg] = neg if num_neg > 0 else 0
        filter_mat[0, -num_pos:] = pos if num_pos > 0 else 0

        # NOTE: writing self.zero as m, and self.npcoeffs as a_i (1<=i<=n)
        # assuming n > 2m,
        # filter_mat should now look like
        # [a_{m+1} a_{m} ... a_{1} a_0 + (0 n -m -1 times) + a_n a_{n-1} ... a_{m+2}]

        # cycle the previous row to the entire matrix
        for i in range(1, mat_size):
            filter_mat[i, :] = np.roll(filter_mat[i - 1, :], 1)

        top_left, top_right, bottom_left, bottom_right = (
            filter_mat[:num_pos, : n - 1],
            filter_mat[:num_pos, -num_pos:],
            filter_mat[-(num_neg - 1) :, : num_neg - 1],
            filter_mat[-(num_neg - 1) :, -(n - 1) :],
        )

        # NOTE[epic=Example] for the case npcoeff = range(5), zero = 2
        # the filter_mat should look like
        #  1  2  .. n-1  0 npos npos-1 ... 1 0
        # [2. 1. 0. 0. | 0. | 4. 3.] 0
        # [3. 2. 1. 0. | 0. | 0. 4.] 1
        # [-  -  -  -  | -  | -  - ] n_pos
        # [4. 3. 2. 1. | 0. | 0. 0.] -
        # [0. 4. 3. 2. | 1. | 0. 0.] -
        # [0. 0. 4. 3. | 2. | 1. 0.] -
        # [- -  -  -   | -  | -  - ] n.zero (= num_neg - 1)
        # [0. 0. | 0. | 4. 3. 2. 1.] n.zero - 1 ... 1
        # [1. 0. | 0. | 0. 4. 3. 2.] 0

        return top_left, top_right, bottom_left, bottom_right

    def __getitem__(self, item: Union[int, slice]) -> np.ndarray:
        """
        Returns the filter coefficients at the requested indices.
        Indices are offset by the filter origin.

        Args:
            item (Union[int, slice]):  The index or slice to get
        Returns:
            np.ndarray: The requested filter coefficients
        """
        if isinstance(item, int):
            return self.npcoeffs[item + self.zero]
        elif isinstance(item, slice):
            return self.npcoeffs[
                item.start + self.zero : item.stop + self.zero : item.step
            ]
        else:
            raise TypeError("Index must be an integer or slice")

    def num_pos(self) -> int:
        """
        Number of mpn-negative coefficients in the filter, including the zero coefficient.

        Hint:
            This is different from the num_pos in the _init_edge method, which excludes the zero coefficient.

        Returns:
            int: Number of positively indexed coefficients in the filter
        """
        return len(self.npcoeffs) - self.zero

    def num_neg(self) -> int:
        """
        Number of negative coefficients in the filter, excluding the zero coefficient.

        Hint:
            This is different from the num_neg in the _init_edge method, which includes the zero coefficient.

        Returns:
            int: Number of negatively indexed coefficients in the filter
        """
        return self.zero


class DifferentiableFilter(Filter):
    """
    A filter object that can be used in autograd computations.

    Attributes:
        init_coeffs (np.ndarray): Initial filter coefficients
        zero (int): The filter origin index
        coeffs (torch.Tensor): The filter coefficients
    """

    def __init__(self, initial_coeffs: np.ndarray, zero: int):
        """
        Initializes a differentiable filter object.

        Args:
            initial_coeffs (np.ndarray): The initial filter coefficients
            zero (int): The filter origin index
        """
        super().__init__(initial_coeffs, zero)
        self.init_coeffs = initial_coeffs
        self.coeffs = self.coeffs.requires_grad_(True)
        del self.npcoeffs
        del self.edge_matrices


# !SECTION 3.1: END


# SECTION 3.2. WAVELET
class Wavelet:
    """
    Represents a wavelet object, containing the filters for decomposition and reconstruction.

    Attributes:
        decomp_lp (Filter): The low pass filter for decomposition
        decomp_hp (Filter): The high pass filter for decomposition
        recon_lp (Filter): The low pass filter for reconstruction
        recon_hp (Filter): The high pass filter for reconstruction
    """

    def __init__(
        self, decomp_lp: Filter, decomp_hp: Filter, recon_lp: Filter, recon_hp: Filter
    ):
        """
        Initializes a wavelet object.

        Args:
            decomp_lp (Filter): The low pass filter for decomposition
            decomp_hp (Filter): The high pass filter for decomposition
            recon_lp (Filter): The low pass filter for reconstruction
            recon_hp (Filter): The high pass filter for reconstruction
        """
        self.decomp_lp = decomp_lp
        self.decomp_hp = decomp_hp
        self.recon_lp = recon_lp
        self.recon_hp = recon_hp

    def __call__(self) -> tuple:
        """
        Returns the wavelet filters as a tuple.

        Returns:
            tuple: The wavelet filters
        """
        return self.decomp_lp, self.decomp_hp, self.recon_lp, self.recon_hp


class DifferentiableWavelet(Wavelet):
    """
    A wavelet object that can be used in autograd computations.

    Attributes:
        decomp_lp (DifferentiableFilter): The low pass filter for decomposition
        decomp_hp (DifferentiableFilter): The high pass filter for decomposition
        recon_lp (DifferentiableFilter): The low pass filter for reconstruction
        recon_hp (DifferentiableFilter): The high pass filter for reconstruction
    """

    def __init__(self, wavelet: Wavelet):
        """
        Initializes a differentiable wavelet object.

        Args:
            wavelet (Wavelet): The initial wavelet object to make differentiable
        """
        super().__init__(
            DifferentiableFilter(wavelet.decomp_lp.npcoeffs, wavelet.decomp_lp.zero),
            DifferentiableFilter(wavelet.decomp_hp.npcoeffs, wavelet.decomp_hp.zero),
            DifferentiableFilter(wavelet.recon_lp.npcoeffs, wavelet.recon_lp.zero),
            DifferentiableFilter(wavelet.recon_hp.npcoeffs, wavelet.recon_hp.zero),
        )

    def __call__(self) -> tuple:
        """
        Returns the wavelet filters as a tuple.

        Returns:
            tuple: The wavelet filters
        """
        return self.decomp_lp, self.decomp_hp, self.recon_lp, self.recon_hp


# !SECTION 3.2. WAVELET
# !SECTION 3: END CLASSES


# SECTION 4: CONSTANTS
# load Daubechies wavelet filters
def fetch_db(i: int) -> Wavelet:
    """
    Fetches the Daubechies wavelet filters of order i. Supported for i < 39.

    Args:
        i (int): The order of the Daubechies wavelet

    Returns:
        Wavelet: The Daubechies wavelet filters of order i
    """
    if i > 38:
        raise IndexError("Order must be less than 39")
    db_coeffs = np.load(f"coeffs/db{i}.npy")
    alt_coeffs = (-1) ** np.arange(1, i + 1)

    return Wavelet(
        Filter(np.flip(db_coeffs), 2 * i - 1),
        Filter(np.flip(db_coeffs) * alt_coeffs, 0),
        Filter(db_coeffs, 0),
        Filter(db_coeffs * alt_coeffs, 2 * i - 1),
    )


# load other wavelet filters
def fetch_haar() -> Wavelet:
    """
    Fetches the Haar wavelet filters.

    Returns:
        Wavelet: The Haar wavelet filters
    """
    return fetch_db(1)


#!SECTION 4: END CONSTANTS
