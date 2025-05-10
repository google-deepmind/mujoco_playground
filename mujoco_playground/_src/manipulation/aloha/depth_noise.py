# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utilities for depth noise."""

import jax
import jax.numpy as jp
import numpy as np


def _bilinear_interpolate(image, y, x):
  """
  Bilinearly interpolate a 2D image at floating-point (y,x) locations,
  using 'nearest' mode behavior for out-of-bound coordinates.

  Parameters:
    image : jp.ndarray of shape (H, W)
    y     : array of y coordinates (any shape)
    x     : array of x coordinates (same shape as y)

  Returns:
    Interpolated values at the provided coordinates.
  """
  height, width = image.shape

  # Clamp coordinates to the valid range.
  y_clamped = jp.clip(y, 0.0, height - 1.0)
  x_clamped = jp.clip(x, 0.0, width - 1.0)

  # Get the integer parts.
  y0 = jp.floor(y_clamped).astype(jp.int32)
  x0 = jp.floor(x_clamped).astype(jp.int32)

  # For the "upper" indices, if we're at the boundary, stay at the same index.
  y1 = jp.where(y0 < height - 1, y0 + 1, y0)
  x1 = jp.where(x0 < width - 1, x0 + 1, x0)

  # Compute the fractional parts.
  dy = y_clamped - y0.astype(y_clamped.dtype)
  dx = x_clamped - x0.astype(x_clamped.dtype)

  # Gather pixel values at the four corners.
  val_tl = image[y0, x0]  # top-left
  val_tr = image[y0, x1]  # top-right
  val_bl = image[y1, x0]  # bottom-left
  val_br = image[y1, x1]  # bottom-right

  # Compute the bilinear interpolated result.
  # Need to be careful to avoid dead pixels in image edges.
  return (
      val_tl * (1.0 - dx) * (1.0 - dy)
      + val_tr * dx * (1.0 - dy)
      + val_bl * (1.0 - dx) * dy
      + val_br * dx * dy
  )


def kinect_noise(key, depth, *, sigma_s=0.5, sigma_d=1 / 6, baseline=35130):
  """
  Apply noise based on the Kinect. Increased noise with distance.

  Parameters:
      depth      : 2D numpy array of ground truth depth values.
      sigma_s    : Std. dev. of spatial shift (in pixels).
      sigma_d    : Std. dev. of the Gaussian noise added in disparity.
      baseline   : Constant used for converting depth to disparity.

  Returns:
      noisy_depth: 2D numpy array with noisy depth.
  """
  if depth.ndim == 3:
    depth = depth[..., 0]

  height, width = depth.shape

  # Create a meshgrid for pixel coordinates.
  grid_y, grid_x = jp.mgrid[0:height, 0:width].astype(jp.float32)

  # Random shifts in x and y (sampled from Gaussian).
  key, key_shift = jax.random.split(key)
  shift_x, shift_y = jax.random.normal(key_shift, (2, height, width)) * sigma_s

  # Shifted coordinates.
  shifted_x = grid_x + shift_x
  shifted_y = grid_y + shift_y

  # Bilinearly interpolate depth at the shifted locations.
  shifted_depth = _bilinear_interpolate(
      depth, shifted_y.ravel(), shifted_x.ravel()
  ).reshape(height, width)

  # Convert depth to disparity.
  eps = 1e-6  # small epsilon to avoid division by zero
  disparity = baseline / (shifted_depth + eps)

  # Add IID Gaussian noise to disparity.
  key, key_noise = jax.random.split(key)
  disparity_noisy = (
      disparity + jax.random.normal(key_noise, (height, width)) * sigma_d
  )

  # Quantise disparity (round to nearest integer).
  disparity_quantized = jp.round(disparity_noisy)

  # Convert quantised disparity back to depth.
  noisy_depth = baseline / (disparity_quantized + eps)

  if depth.ndim == 3:
    noisy_depth = jp.expand_dims(noisy_depth, axis=-1)

  return noisy_depth


def edge_noise(key, depth, *, grad_threshold=0.05, noise_multiplier=10):
  """
  Depth cameras are expected to occasionally lose pixels at edges.
  When the spatial gradient of the depth is greater than threshold, theres a
  chance the pixels are dropped.
  Then, randomly jitter those dropped pixels.
  Note that the proper way to do this requires the surface normals of everything
  in the scene.

  Args:
  grad_threshold: below this, no dropout.
  noise_multiplier: higher values mean more dropout.
  """
  # Compute gradients along the x and y directions.
  # gradient returns [gradient_along_axis0, gradient_along_axis1].
  grad_y, grad_x = jp.gradient(depth)  # each is (H, W)

  # Compute the magnitude of the depth gradient.
  grad_mag = jp.sqrt(grad_x**2 + grad_y**2)

  # Probability that you lose that pixel.
  p_lost = jp.arctan(noise_multiplier * grad_mag)

  p_lost = p_lost * (p_lost > grad_threshold).astype(jp.float32)

  # Sample a mask.
  key_dropout, key = jax.random.split(key)
  mask = (
      jax.random.uniform(key_dropout, depth.shape) < p_lost
  )  # if true, then drop.

  # Scatter the mask.
  height, width = depth.shape
  grid_y, grid_x = jp.mgrid[0:height, 0:width].astype(jp.int32)

  # Random coordinate shifts in x and y, uniformly 0, 1.
  key_shift, key = jax.random.split(key)
  shift_x, shift_y = jax.random.randint(
      key_shift, (2, height, width), minval=0, maxval=2
  )

  # Shifted coordinates.
  shifted_x = grid_x + shift_x
  shifted_y = grid_y + shift_y

  # Ensure the shifted coordinates are within bounds.
  shifted_x = jp.clip(shifted_x, 0, width - 1)
  shifted_y = jp.clip(shifted_y, 0, height - 1)

  # Fancy indexing.
  mask_shifted = mask[shifted_y, shifted_x]

  # Set those values to 0.
  depth_noisy = depth * (1 - mask_shifted).astype(jp.float32)
  return depth_noisy


def random_dropout(key, depth_image, *, p=0.006):
  key_dropout, key = jax.random.split(key)
  mask = jax.random.bernoulli(key_dropout, p, depth_image.shape)
  depth_noisy = depth_image * (1 - mask).astype(jp.float32)
  return depth_noisy


def _np_bresenham_line(x0, y0, x1, y1):
  """
  Compute the list of pixels along a line from (x0,y0) to (x1,y1)
  using Bresenham's algorithm.
  Returns a list of (x, y) tuples.
  """
  points = []
  dx = abs(x1 - x0)
  dy = abs(y1 - y0)
  x, y = x0, y0
  sx = 1 if x0 < x1 else -1
  sy = 1 if y0 < y1 else -1
  if dx > dy:
    err = dx / 2.0
    while x != x1:
      points.append((x, y))
      err -= dy
      if err < 0:
        y += sy
        err += dx
      x += sx
  else:
    err = dy / 2.0
    while y != y1:
      points.append((x, y))
      err -= dx
      if err < 0:
        x += sx
        err += dy
      y += sy
  points.append((x1, y1))
  return points


def _np_draw_line(img, start, end, color):
  """
  Draw a line of thickness 1.
  Start, end are (x, y) tuples.
  """
  height, width = img.shape[:2]
  for x, y in _np_bresenham_line(*start, *end):
    if 0 <= x < width and 0 <= y < height:
      img[y, x] = color
  return img


def np_get_line_bank(height, width, bank_size=100, color_range=None):
  """
  Get a bank of random lines. Not jax-compatible.
  Returns a bank of size:
  (bank_size, H, W)
  where each element is a white image with up to max_lines lines randomly
  drawn on it.
  """
  if color_range is None:
    color_range = [0, 0.4]

  max_lines = 16
  bank = []
  for _ in range(bank_size):
    img = np.zeros((height, width), dtype=np.float32)
    num_lines = np.random.randint(1, max_lines + 1)
    for _ in range(num_lines):
      start = np.random.randint(width), np.random.randint(height)
      theta = np.random.uniform(0, 2 * np.pi)
      length = np.random.randint(2, 6)
      end = (
          start[0] + length * np.cos(theta),
          start[1] + length * np.sin(theta),
      )
      end = int(end[0]), int(end[1])
      color = np.random.uniform(color_range[0], color_range[1])
      img = _np_draw_line(img, start, end, color)
    bank.append(img)
  return np.stack(bank)


def _or_reduce(arr, axis):
  """
  Reduces `arr` along the given axis using an 'or' operator defined as:
    result = y if y != 0 else x
  That is, it returns the last nonzero element along that axis (or 0 if all
  are zero).

  Parameters
  ----------
  arr : np.ndarray
      The input array.
  axis : int
      The axis over which to reduce.

  Returns
  -------
  reduced : np.ndarray
      The array with the specified axis reduced.
  """
  # Flip the array along the specified axis so that the last element comes
  # first.
  arr_rev = jp.flip(arr, axis=axis)  # (H, W)

  # Create a boolean mask of nonzero values.
  mask = arr_rev != 0  # (H, W)

  # Find the index of the first nonzero element in the reversed array.
  # If all are zero, np.argmax will return 0. (This is fine since arr[...0]
  # is 0.)
  idx_rev = jp.argmax(mask, axis=axis)  # (H,)

  # Convert that index back into an index for the original (unflipped) array.
  n = arr.shape[axis]
  idx_orig = n - 1 - idx_rev  # (H,)

  idx_expanded = jp.expand_dims(idx_orig, axis=axis)  # (H, 1)

  # Now take the elements along the axis.
  reduced = jp.take_along_axis(arr, idx_expanded, axis=axis)  # (H, 1)

  # Remove the reduced axis.
  reduced = jp.squeeze(reduced, axis=axis)  # (H,)

  return reduced


def apply_line_noise(img, line_noise):
  return _or_reduce(jp.stack([img, line_noise]), axis=0)
