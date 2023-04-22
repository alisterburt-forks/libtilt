import torch
from torch import fft as fft

from libtilt.utils.fft import rfft_shape, _distance_from_dc_for_shifted_rfft
from libtilt.rotational_average import _find_shell_indices_1d


def fsc(
    a: torch.Tensor,
    b: torch.Tensor,
    rfft_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Fourier ring/shell correlation between two square/cubic images."""
    # input handling
    dft_shape = rfft_shape(a.shape)
    if a.ndim not in (2, 3):
        raise ValueError('images must be 2D or 3D.')
    elif a.shape != b.shape:
        raise ValueError('images must be the same shape.')
    elif rfft_mask is not None and rfft_mask.shape != dft_shape:
        raise ValueError('valid rfft indices must have same shape as rfft.')

    # fsc calculation
    a, b = fft.rfftn(a), fft.rfftn(b)
    shift_dims = (-3, -2) if a.ndim == 3 else (-2,)
    a, b = fft.fftshift(a, dim=shift_dims), fft.fftshift(b, dim=shift_dims)
    distance_from_dc = _distance_from_dc_for_shifted_rfft(a.shape)
    n_shells = (a.shape[0] // 2) + 1
    if rfft_mask is not None:
        a, b, distance_from_dc = (arr[rfft_mask] for arr in[a, b, distance_from_dc])
    a, b, distance_from_dc = (torch.flatten(arg) for arg in [a, b, distance_from_dc])
    shell_idx = _find_shell_indices_1d(distance_from_dc, n_shells=n_shells)
    fsc = [
        _normalised_cc_complex_1d(a[idx], b[idx])
        for idx in
        shell_idx
    ]
    return torch.real(torch.tensor(fsc))


def _normalised_cc_complex_1d(a: torch.Tensor, b: torch.Tensor):
    correlation = torch.dot(a, torch.conj(b))
    return correlation / (torch.linalg.norm(a) * torch.linalg.norm(b))
