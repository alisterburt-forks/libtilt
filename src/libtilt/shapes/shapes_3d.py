import einops
import torch

from libtilt.grids import coordinate_grid
from libtilt.shapes.geometry_utils import _angle_between_vectors
from libtilt.shapes.soft_edge import add_soft_edge_3d
from libtilt.fft_utils import dft_center


def sphere(
    radius: float,
    image_shape: tuple[int, int, int] | int,
    center: tuple[float, float, float] | None = None,
    smoothing_radius: float = 0,
    device: torch.device | None = None,
) -> torch.Tensor:
    if isinstance(image_shape, int):
        image_shape = (image_shape, image_shape, image_shape)
    if center is None:
        center = dft_center(image_shape, rfft=False, fftshifted=True)
    distances = coordinate_grid(
        image_shape=image_shape,
        center=center,
        norm=True,
        device=device,
    )
    mask = torch.zeros_like(distances, dtype=torch.bool)
    mask[distances < radius] = 1
    return add_soft_edge_3d(mask, smoothing_radius=smoothing_radius)


def cuboid(
    dimensions: tuple[float, float, float],
    image_shape: tuple[int, int, int] | int,
    center: tuple[float, float] | None = None,
    smoothing_radius: float = 0,
    device: torch.device | None = None,
) -> torch.Tensor:
    if isinstance(image_shape, int):
        image_shape = (image_shape, image_shape, image_shape)
    if center is None:
        center = dft_center(image_shape, rfft=False, fftshifted=True)
    coordinates = coordinate_grid(
        image_shape=image_shape,
        center=center,
        device=device,
    )
    dd, dh, dw = dimensions[0] / 2, dimensions[1] / 2, dimensions[2] / 2
    depth_mask = torch.logical_and(coordinates[..., 0] > -dh, coordinates[..., 0] < dh)
    height_mask = torch.logical_and(coordinates[..., 1] > -dw, coordinates[..., 1] < dw)
    width_mask = torch.logical_and(coordinates[..., 2] > -dw, coordinates[..., 2] < dw)
    mask = einops.rearrange([depth_mask, height_mask, width_mask],
                            'dhw d h w -> d h w dhw')
    mask = torch.all(mask, dim=-1)
    return add_soft_edge_3d(mask, smoothing_radius=smoothing_radius)


def cube(
    sidelength: float,
    image_shape: tuple[int, int, int] | int,
    center: tuple[float, float, float] | None = None,
    smoothing_radius: float = 0,
    device: torch.device | None = None,
) -> torch.Tensor:
    cube = cuboid(
        dimensions=(sidelength, sidelength, sidelength),
        image_shape=image_shape,
        center=center,
        smoothing_radius=smoothing_radius,
        device=device,
    )
    return cube


def cone(
    aperture: float,
    image_shape: tuple[int, int, int] | int,
    principal_axis: tuple[float, float, float] = (1, 0, 0),
    smoothing_radius: float = 0,
    device: torch.device | None = None,
) -> torch.Tensor:
    if isinstance(image_shape, int):
        image_shape = (image_shape, image_shape, image_shape)
    center = dft_center(
        image_shape, rfft=False, fftshifted=True
    )
    vectors = coordinate_grid(
        image_shape=image_shape,
        center=center,
        device=device,
    ).float()
    vectors_norm = einops.reduce(vectors ** 2, '... c -> ... 1', reduction='sum') ** 0.5
    vectors /= vectors_norm
    principal_axis = torch.as_tensor(principal_axis, dtype=vectors.dtype, device=device)
    principal_axis_norm = einops.reduce(
        principal_axis ** 2, '... c -> ... 1', reduction='sum'
    ) ** 0.5
    principal_axis /= principal_axis_norm
    angles = _angle_between_vectors(vectors, principal_axis)
    acute_bound = aperture / 2
    obtuse_bound = 180 - acute_bound
    in_cone = torch.logical_or(angles <= acute_bound, angles >= obtuse_bound)
    dc_d, dc_h, dc_w = dft_center(image_shape, rfft=False, fftshifted=True)
    in_cone[dc_d, dc_h, dc_w] = True
    return add_soft_edge_3d(in_cone, smoothing_radius=smoothing_radius)


def gaussian(
    center: tuple[float, float, float] | None = None,
    standard_deviation: float = 1,
    image_shape: tuple[int, int, int] | int = (32, 32, 32),
    device: torch.device | None = None,
) -> torch.Tensor:
    if isinstance(image_shape, int):
        image_shape = (image_shape, image_shape, image_shape)
    if center is None:
        center = dft_center(image_shape, rfft=False, fftshifted=True)
    grid = coordinate_grid(
        image_shape=image_shape,
        center=center,
        device=device,
    )
    sd, sh, sw = standard_deviation, standard_deviation, standard_deviation
    d, h, w = grid[..., 0], grid[..., 1], grid[..., 2]
    d2_norm = (d ** 2) / (2 * sd ** 2)
    h2_norm = (h ** 2) / (2 * sh ** 2)
    w2_norm = (w ** 2) / (2 * sw ** 2)
    gaussian = torch.exp(-(d2_norm + h2_norm + w2_norm))
    sum = einops.reduce(gaussian, pattern='... d h w -> ... 1 1 1', reduction='sum')
    return gaussian / sum
