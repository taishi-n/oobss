"""Collection of utility functions for ICA-based BSS."""

import numpy as np


def tensor_T(A):
    """Compute transpose for tensor."""
    return A.swapaxes(-2, -1)


def tensor_H(A):
    """Compute Hermitian transpose for tensor."""
    return np.conj(A).swapaxes(-2, -1)


def projection_back(Y, W, ref_mic=0):
    """
    Apply projection-back scaling using the demixing filters.

    This is the shared implementation used by both batch and online
    IVA/ILRMA-style algorithms.

    Parameters
    ----------
    Y : ndarray
        Demixed STFT. Supported shapes:
        ``(n_frame, n_freq, n_src)`` or ``(n_freq, n_src)``.
    W : ndarray (n_freq, n_src, n_src)
        Demixing filters.
    ref_mic : int, default=0
        Reference microphone index.

    Returns
    -------
    ndarray
        Projection-back scaled STFT with the same shape as ``Y``.
    """
    if Y.ndim not in {2, 3}:
        raise ValueError(f"Y must be 2-D or 3-D, got ndim={Y.ndim}")
    n_src = Y.shape[-1]
    scale = projection_back_scale(W, ref_mic=ref_mic, n_src=n_src)
    if Y.ndim == 2:
        return Y * scale
    return Y * scale[None, :, :]


def projection_back_scale(W, ref_mic=0, n_src=None):
    """
    Compute projection-back complex scale from demixing filters.

    For each frequency bin, let $W_f$ be the demixing matrix and
    $A_f = W_f^{-1}$ its corresponding mixing matrix. The source image
    for source $k$ at reference microphone $m_{\\mathrm{ref}}$ is
    restored by:

    $$
       d_{k,f} = a_{k,f}[m_{\\mathrm{ref}}], \\quad
       \\hat{y}_{k,f,t} = d_{k,f} y_{k,f,t}
    $$

    In online variants, $W_f$ and $A_f$ become time-dependent
    ($W_{f,t}$, $A_{f,t}$), but the scaling formula is identical.

    Parameters
    ----------
    W : ndarray (n_freq, n_chan, n_chan*(n_tap+1))
        Demixing/dereverberation filters.
    ref_mic : int, default=0
        Reference microphone index.
    n_src : int or None, default=None
        Number of separated sources. If ``None``, ``W.shape[1]`` is used.

    Returns
    -------
    ndarray (n_freq, n_src)
        Complex projection-back scale coefficients.
    """
    n_src_eff = W.shape[1] if n_src is None else int(n_src)
    if ref_mic < 0 or ref_mic >= W.shape[1]:
        raise ValueError(
            f"ref_mic out of range: {ref_mic} for demix channels={W.shape[1]}"
        )
    invW = np.linalg.inv(W[:, :n_src_eff, :n_src_eff])
    return invW[:, ref_mic, :n_src_eff]


def solve_2x2HEAD(V1, V2, method="ono", eig_reverse=False):
    """
    Solve a 2x2 HEAD problem with given two positive semi-definite matrices.

    Parameters
    ----------
    V1: (n_freq, 2, 2)
    V2: (n_freq, 2, 2)
    method: "numpy" or "ono"
        If "numpy", `eigval` is calculated by using `numpy.linalg.eig`.
        If "ono", `eigval` is calculated by the method presented in Ono2012IWAENC.
    eig_reverse: bool
        If True, eigenvalues is sorted in *ascending* order.
        Default is False.
        This parameter will be deprecated in the future.

    Returns
    -------
    eigval: (n_freq, 2)
        eigenvalues, must be real numbers
    eigvec: (2, n_freq, 2)
        eigenvectors corresponding to the eigenvalues
    """
    V_hat = np.array([V1, V2])

    # shape: (n_freq, 2, 2)
    Z = np.zeros(V1.shape, dtype=complex)

    # Z = adj(V1) @ V2
    Z[:, 0, 0] = V2[:, 0, 0] * V1[:, 1, 1] - V2[:, 1, 0] * V1[:, 0, 1]
    Z[:, 0, 1] = V2[:, 0, 1] * V1[:, 1, 1] - V2[:, 1, 1] * V1[:, 0, 1]
    Z[:, 1, 0] = -V2[:, 0, 0] * V1[:, 1, 0] + V2[:, 1, 0] * V1[:, 0, 0]
    Z[:, 1, 1] = -V2[:, 0, 1] * V1[:, 1, 0] + V2[:, 1, 1] * V1[:, 0, 0]

    # shape: (n_freq,)
    tr_Z = np.trace(Z, axis1=1, axis2=2)
    dd = np.sqrt(tr_Z**2 - 4 * np.linalg.det(Z))

    # shape: (n_freq,)
    d1 = (tr_Z + dd).real
    d2 = (tr_Z - dd).real

    # shape: (n_freq, 2)
    eigval = np.array([d1, d2]).T

    D = np.zeros(Z.shape, dtype=Z.dtype)
    D[:, [0, 1], [0, 1]] = np.array([d1, d2]).T

    # shape: (2, n_freq, 2)
    eigvec = (2 * Z - D).transpose([2, 0, 1])
    eigvec[:, :, :, None] /= np.sqrt(
        np.conj(eigvec[:, :, None, :]) @ V_hat @ eigvec[:, :, :, None]
    )

    return eigval, eigvec


def demix(observations, demix_filter):
    """
    Perform the demixing filter into observations.

    Parameters
    ----------
    observations : ndarray of shape (n_frame, n_freq, n_src)
    demix_filter : ndarray of shape (n_freq, n_src, n_src)

    Returns
    -------
    Estimated source
        ndarray of shape (n_frame, n_freq, n_src)
    """
    # shape: (n_freq, n_src, n_frame)
    y = demix_filter @ observations.transpose([1, 2, 0])
    return y.transpose([2, 0, 1])
