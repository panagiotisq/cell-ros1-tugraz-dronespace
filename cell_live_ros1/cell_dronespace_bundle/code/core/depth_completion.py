import os
import numpy as np
import cv2

_PARTIAL = ('partial', 'partial_light', 'partial_dc')

def sparse_to_dense(sparse, max_depth=100., fill_level='full', dc_tau=None):
    """Morphological depth completion.

    fill_level='full' (DEFAULT) — ORIGINAL behaviour, byte-identical to before. Do not change.
    fill_level='partial' — conservative: diamond-3 dilate + 3x3 close, SKIP the 7x7 large-gap fill
        (the margin-bridging / hallucination step). ~40-90% fill, only near real points.
    fill_level='partial_light' — 'a bit less' (idea 1): diamond-3 dilate + SKIP close + skip 7x7 fill.
        One fewer growth step => fills fewer pixels than 'partial'.
    fill_level='partial_dc' — depth-consistent (idea 2): like 'partial', then ERASE any filled pixel
        whose local window of ORIGINAL real depths spans more than dc_tau (a depth discontinuity), so we
        never dilate a thin foreground (e.g. a tree trunk) out over the far background. dc_tau default =
        0.15*max_depth. Targets the artifact mechanism directly. GT completion callers must stay 'full'.
    """
    # capture original (meters) BEFORE inversion — only needed for the depth-consistency gate
    if fill_level == 'partial_dc':
        orig = sparse.copy()
        valid0 = orig > 0.1

    ## invert
    valid = sparse > 0.1
    sparse[valid] = max_depth - sparse[valid]

    ## dilate
    if fill_level in _PARTIAL:
        custom_kernel = np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ], dtype=np.uint8)                 # diamond 3x3 — shorter reach, fills only tight gaps
    else:
        custom_kernel = np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
        ], dtype=np.uint8)
    sparse = cv2.dilate(sparse, custom_kernel)

    ## close — skipped entirely for the lightest variant
    if fill_level != 'partial_light':
        _close = 3 if fill_level in _PARTIAL else 5
        custom_kernel = np.ones((_close, _close), np.uint8)
        sparse = cv2.morphologyEx(sparse, cv2.MORPH_CLOSE, custom_kernel)

    ## fill (large-gap bridge) — the margin-hallucination step; skipped for every partial variant
    if fill_level not in _PARTIAL:
        invalid = sparse < 0.1
        custom_kernel = np.ones((7, 7), np.uint8)
        dilated = cv2.dilate(sparse, custom_kernel)
        sparse[invalid] = dilated[invalid]

    ## blur
    sparse = cv2.medianBlur(sparse, 5)
    sparse = cv2.bilateralFilter(sparse, 5, 1.5, 2.0)

    ## invert
    valid = sparse > 0.1
    sparse[valid] = max_depth - sparse[valid]

    ## depth-consistency gate (idea 2): drop fills that straddle a depth discontinuity
    if fill_level == 'partial_dc':
        tau = dc_tau if dc_tau is not None else 0.15 * max_depth
        k = np.ones((7, 7), np.uint8)
        local_max = cv2.dilate(orig, k)                       # far neighbour (invalid=0 never raises max)
        tmp = orig.copy(); tmp[~valid0] = max_depth * 10.0     # invalid -> big so it never wins the min
        local_min = cv2.erode(tmp, k)                          # nearest valid neighbour
        has_valid = cv2.dilate(valid0.astype(np.float32), k) > 0
        local_range = np.where(has_valid, local_max - local_min, 0.0)
        newly_filled = (sparse > 0.1) & (~valid0)              # completed pixels (not original points)
        sparse[newly_filled & (local_range > tau)] = 0.0       # straddles an edge -> leave empty

    return sparse