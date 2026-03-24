"""
Generate visualization video from Bench2Drive closed-loop evaluation results.

Overlays planning trajectory on front camera and BEV images,
combines all 6 camera views + annotated BEV into a single video,
similar to the Bench2DriveZoo analysis visualizations.

Optionally overlays detected 3D bounding boxes (requires detection data
saved by the modified uniad_b2d_agent.py).

Usage:
    python generate_video.py -f <scenario_folder> [-o <output_path>] [--fps 15]
    python generate_video.py -f <scenario_folder> --show-detections [--det-threshold 0.3]
    python generate_video.py -f <scenario_folder> --show-motion
    python generate_video.py -f <scenario_folder> --show-map
    python generate_video.py -f <scenario_folder> --show-detections --show-motion --show-map

The scenario folder is expected to have the following structure:
    <folder>/
        rgb_front/          # Front camera images
        rgb_front_left/     # Front-left camera images
        rgb_front_right/    # Front-right camera images
        rgb_back/           # Back camera images
        rgb_back_left/      # Back-left camera images
        rgb_back_right/     # Back-right camera images
        bev/                # Bird's-eye view images
        meta/               # JSON metadata per frame
        detections/         # (Optional) .npz files with 3D detection results
"""

import cv2
import os
import numpy as np
import json
import argparse
from tqdm import trange
from scipy.interpolate import splprep, splev
import seaborn as sns
from collections import OrderedDict


# ─── Fixed calibration matrices from Bench2Drive agent code ───────────────────
# These are the lidar-to-image projection matrices for each camera,
# identical to those used in uniad_b2d_agent.py / vad_b2d_agent.py.

LIDAR2IMG = {
    'CAM_FRONT': np.array([
        [1.14251841e+03, 8.00000000e+02, 0.00000000e+00, -9.52000000e+02],
        [0.00000000e+00, 4.50000000e+02, -1.14251841e+03, -8.09704417e+02],
        [0.00000000e+00, 1.00000000e+00, 0.00000000e+00, -1.19000000e+00],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
    'CAM_FRONT_LEFT': np.array([
        [6.03961325e-14, 1.39475744e+03, 0.00000000e+00, -9.20539908e+02],
        [-3.68618420e+02, 2.58109396e+02, -1.14251841e+03, -6.47296750e+02],
        [-8.19152044e-01, 5.73576436e-01, 0.00000000e+00, -8.29094072e-01],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
    'CAM_FRONT_RIGHT': np.array([
        [1.31064327e+03, -4.77035138e+02, 0.00000000e+00, -4.06010608e+02],
        [3.68618420e+02, 2.58109396e+02, -1.14251841e+03, -6.47296750e+02],
        [8.19152044e-01, 5.73576436e-01, 0.00000000e+00, -8.29094072e-01],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
    'CAM_BACK': np.array([
        [-5.60166031e+02, -8.00000000e+02, 0.00000000e+00, -1.28800000e+03],
        [5.51091060e-14, -4.50000000e+02, -5.60166031e+02, -8.58939847e+02],
        [1.22464680e-16, -1.00000000e+00, 0.00000000e+00, -1.61000000e+00],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
    'CAM_BACK_LEFT': np.array([
        [-1.14251841e+03, 8.00000000e+02, 0.00000000e+00, -6.84385123e+02],
        [-4.22861679e+02, -1.53909064e+02, -1.14251841e+03, -4.96004706e+02],
        [-9.39692621e-01, -3.42020143e-01, 0.00000000e+00, -4.92889531e-01],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
    'CAM_BACK_RIGHT': np.array([
        [3.60989788e+02, -1.34723223e+03, 0.00000000e+00, -1.04238127e+02],
        [4.22861679e+02, -1.53909064e+02, -1.14251841e+03, -4.96004706e+02],
        [9.39692621e-01, -3.42020143e-01, 0.00000000e+00, -4.92889531e-01],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
}

# Top-down BEV projection matrix (lidar coords -> top-down image coords)
# Matches the coor2topdown used in the agent code.
_topdown_extrinsics = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 50.0],
    [0.0, 0.0, 0.0, 1.0]])
_topdown_intrinsics = np.array([
    [548.993771650447, 0.0, 256.0, 0],
    [0.0, 548.993771650447, 256.0, 0],
    [0.0, 0.0, 1.0, 0],
    [0, 0, 0, 1.0]])
COOR2TOPDOWN = _topdown_intrinsics @ _topdown_extrinsics

# Camera image size
CAM_W, CAM_H = 1600, 900
BEV_SIZE = 512

# Camera folder name mapping
CAM_FOLDER = {
    'CAM_FRONT': 'rgb_front',
    'CAM_FRONT_LEFT': 'rgb_front_left',
    'CAM_FRONT_RIGHT': 'rgb_front_right',
    'CAM_BACK': 'rgb_back',
    'CAM_BACK_LEFT': 'rgb_back_left',
    'CAM_BACK_RIGHT': 'rgb_back_right',
}

# Layout order for the 6-camera grid (top row, bottom row)
CAM_TOP_ROW = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT']
CAM_BOT_ROW = ['CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']


# ─── Detection color mapping ─────────────────────────────────────────────────
# Label colours matching Bench2DriveZoo's seaborn "bright" palette.
# Label indices follow nuScenes: 0=car, 1=truck, 2=trailer, 3=bus,
# 4=construction_vehicle, 5=bicycle, 6=motorcycle, 7=pedestrian,
# 8=traffic_cone, 9=barrier.

def _float_to_uint8_color(float_clr):
    return [int(c * 255.) for c in float_clr]

_COLORS = [_float_to_uint8_color(clr) for clr in sns.color_palette("bright", n_colors=10)]

COLORMAP = OrderedDict({
    6: _COLORS[8],   # motorcycle -> yellow
    4: _COLORS[8],   # construction_vehicle -> yellow
    3: _COLORS[0],   # bus -> blue
    1: _COLORS[6],   # truck -> pink
    0: _COLORS[2],   # car -> green
    8: _COLORS[7],   # traffic_cone -> gray
    7: _COLORS[1],   # pedestrian -> orange
    5: _COLORS[3],   # bicycle -> red
    2: _COLORS[5],   # trailer -> brown
    9: _COLORS[4],   # barrier -> purple (fallback)
})

# Default colour for unknown labels
DEFAULT_DET_COLOR = (0, 255, 0)

# Score threshold for displaying detections
DET_SCORE_THRESHOLD = 0.3

# 3D box edge indices – 12 edges for full 3D box (camera view)
BOX3D_LINE_INDICES = (
    (0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2),
    (3, 7), (4, 5), (4, 7), (2, 6), (5, 6), (6, 7))

# 4 edges for BEV (bottom face only)
BOX3D_BEV_LINE_INDICES = ((0, 3), (3, 7), (4, 7), (0, 4))

# Motion trajectory visualization constants
# Number of trajectory modes to display (top-k by score)
MOTION_TOP_K_MODES = 3
# Minimum track score to show motion trajectories
MOTION_SCORE_THRESHOLD = 0.3
# Only show trajectories for vehicle classes (car, truck, trailer, bus, construction_vehicle)
MOTION_VEHICLE_IDS = [0, 1, 2, 3, 4]
# Motion trajectory line thickness
MOTION_LINE_THICKNESS = 2
# Motion trajectory endpoint radius
MOTION_ENDPOINT_RADIUS = 3

# Map segmentation colours (BEV overlay)
# lane_score channels: 0=divider, 1=crossing, 2=contour
MAP_COLORS = {
    'divider': (0, 165, 255),     # orange
    'crossing': (0, 255, 255),    # yellow
    'contour': (255, 200, 0),     # cyan-ish
    'drivable': (0, 100, 0),      # dark green
}
MAP_ALPHA = 0.35
MAP_THRESHOLD = 0.5


# ─── Drawing helpers ──────────────────────────────────────────────────────────

def draw_trajectory_on_cam(img, plan_pts, lidar2img, canvas_size=(CAM_H, CAM_W),
                           thickness=4, hue_start=120, hue_end=80, add_ego_start=True):
    """
    Project a planning trajectory (in lidar coords) onto a camera image and
    draw it as a colour-gradient polyline.

    Parameters
    ----------
    img : np.ndarray   – BGR image (H, W, 3)
    plan_pts : np.ndarray – (N, 2) waypoints [x, y] in lidar frame
    lidar2img : np.ndarray – (4, 4) projection matrix
    """
    line = plan_pts.copy()
    h, w = canvas_size

    # Build homogeneous 4×N: [x, y, z=ground_plane, 1]
    # z = -1.84 puts the points roughly at ground level in lidar coords
    pts_4d = np.stack([
        line[:, 0],
        line[:, 1],
        np.full(line.shape[0], -1.84),
        np.ones(line.shape[0])
    ])  # (4, N)

    pts_2d = (lidar2img @ pts_4d).T  # (N, 4)
    # Perspective division
    depth = pts_2d[:, 2].copy()
    valid_depth = depth > 0.1
    pts_2d[:, 0] /= np.clip(pts_2d[:, 2], 1e-5, None)
    pts_2d[:, 1] /= np.clip(pts_2d[:, 2], 1e-5, None)

    # Visibility mask
    mask = (valid_depth &
            (pts_2d[:, 0] > 0) & (pts_2d[:, 0] < w) &
            (pts_2d[:, 1] > 0) & (pts_2d[:, 1] < h))
    if not mask.any():
        return img

    pts_2d = pts_2d[mask, :2]

    # Optionally prepend ego vehicle centre at bottom of image
    if add_ego_start:
        pts_2d = np.concatenate([np.array([[w / 2, h]]), pts_2d], axis=0)

    if len(pts_2d) < 2:
        return img

    # Fit a smooth spline through the projected points
    try:
        tck, _ = splprep([pts_2d[:, 0], pts_2d[:, 1]], s=0)
    except Exception:
        return img
    unew = np.linspace(0, 1, 100)
    smoothed = np.stack(splev(unew, tck)).astype(int).T

    # Draw with HSV gradient colour
    out = img.copy()
    n = len(smoothed)
    for i in range(n - 1):
        hue = hue_start + (hue_end - hue_start) * (i / n)
        hsv = np.array([hue, 255, 255], dtype=np.uint8)
        rgb = cv2.cvtColor(hsv[np.newaxis, np.newaxis, :], cv2.COLOR_HSV2RGB).reshape(-1)
        color = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
        cv2.line(out,
                 (smoothed[i, 0], smoothed[i, 1]),
                 (smoothed[i + 1, 0], smoothed[i + 1, 1]),
                 color, thickness, cv2.LINE_AA)
    return out


def draw_trajectory_on_bev(img, plan_pts, proj_matrix, canvas_size=(BEV_SIZE, BEV_SIZE),
                           thickness=3, hue_start=120, hue_end=80, is_ego=True):
    """
    Project a planning trajectory onto the top-down BEV image.
    """
    if is_ego:
        line = np.concatenate([np.zeros((1, 2)), plan_pts], axis=0)
    else:
        line = plan_pts.copy()

    h, w = canvas_size
    pts_4d = np.stack([
        line[:, 0],
        line[:, 1],
        np.zeros(line.shape[0]),
        np.ones(line.shape[0])
    ])  # (4, N)

    pts_2d = (proj_matrix @ pts_4d).T  # (N, 4)
    pts_2d[:, 0] /= np.clip(pts_2d[:, 2], 1e-5, None)
    pts_2d[:, 1] /= np.clip(pts_2d[:, 2], 1e-5, None)

    mask = ((pts_2d[:, 0] > 0) & (pts_2d[:, 0] < w) &
            (pts_2d[:, 1] > 0) & (pts_2d[:, 1] < h))
    if not mask.any():
        return img

    pts_2d = pts_2d[mask, :2]
    if len(pts_2d) < 2:
        return img

    try:
        tck, _ = splprep([pts_2d[:, 0], pts_2d[:, 1]], s=0)
    except Exception:
        return img
    unew = np.linspace(0, 1, 100)
    smoothed = np.stack(splev(unew, tck)).astype(int).T

    out = img.copy()
    n = len(smoothed)
    for i in range(n - 1):
        hue = hue_start + (hue_end - hue_start) * (i / n)
        hsv = np.array([hue, 255, 255], dtype=np.uint8)
        rgb = cv2.cvtColor(hsv[np.newaxis, np.newaxis, :], cv2.COLOR_HSV2RGB).reshape(-1)
        color = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
        p1 = (smoothed[i, 0], smoothed[i, 1])
        p2 = (smoothed[i + 1, 0], smoothed[i + 1, 1])
        if (0 <= p1[0] < w and 0 <= p1[1] < h):
            cv2.line(out, p1, p2, color, thickness, cv2.LINE_AA)
        elif i == 0:
            break
    return out


def draw_ego_box_bev(img, proj_matrix, color=(0, 255, 0), thickness=2):
    """
    Draw ego-vehicle bounding box on BEV image.
    Approximate ego car size: length=4.89m, width=1.84m centred at origin.
    """
    half_l, half_w = 4.89 / 2, 1.84 / 2
    corners = np.array([
        [-half_w, half_l],
        [half_w, half_l],
        [half_w, -half_l],
        [-half_w, -half_l],
    ])  # (4, 2)
    pts_4d = np.stack([
        corners[:, 0],
        corners[:, 1],
        np.zeros(4),
        np.ones(4)
    ])  # (4, 4)
    pts_2d = (proj_matrix @ pts_4d).T
    pts_2d[:, 0] /= np.clip(pts_2d[:, 2], 1e-5, None)
    pts_2d[:, 1] /= np.clip(pts_2d[:, 2], 1e-5, None)
    pts_2d = pts_2d[:, :2].astype(int)

    out = img.copy()
    for i in range(4):
        j = (i + 1) % 4
        cv2.line(out,
                 (pts_2d[i, 0], pts_2d[i, 1]),
                 (pts_2d[j, 0], pts_2d[j, 1]),
                 color, thickness, cv2.LINE_AA)
    # Draw heading indicator (front edge midpoint to centre)
    front_mid = ((pts_2d[0] + pts_2d[1]) / 2).astype(int)
    centre = pts_2d.mean(axis=0).astype(int)
    cv2.line(out, tuple(centre), tuple(front_mid), color, thickness, cv2.LINE_AA)
    return out


def add_label(img, text, position=(10, 30), font_scale=0.8, color=(255, 255, 255), thickness=2, bg=True):
    """Put text label on image with optional dark background."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    if bg:
        cv2.rectangle(img, (x - 2, y - th - 4), (x + tw + 2, y + baseline + 2), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    return img


# ─── Detection drawing helpers ────────────────────────────────────────────────

def load_detections(det_path):
    """
    Load detection data from an .npz file saved by the modified agent.

    Returns
    -------
    dict with keys: 'corners_3d' (N, 8, 3), 'scores_3d' (N,),
                    'labels_3d' (N,), optionally 'track_ids', 'boxes_tensor'.
    Returns None if the file does not exist.
    """
    if not os.path.exists(det_path):
        return None
    data = np.load(det_path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def draw_3d_boxes_on_cam(img, corners_3d, scores, labels, lidar2img,
                         canvas_size=(CAM_H, CAM_W), thickness=2):
    """
    Draw projected 3D bounding boxes on a camera image.

    Parameters
    ----------
    img : np.ndarray – (H, W, 3) BGR image
    corners_3d : np.ndarray – (N, 8, 3) box corners in lidar coords
    scores : np.ndarray – (N,) detection confidence scores
    labels : np.ndarray – (N,) class labels
    lidar2img : np.ndarray – (4, 4) projection matrix
    """
    if corners_3d is None or len(corners_3d) == 0:
        return img

    h, w = canvas_size
    num_bbox = corners_3d.shape[0]

    # Project all corners: (N*8, 4) homogeneous
    pts_4d = np.concatenate([
        corners_3d.reshape(-1, 3),
        np.ones((num_bbox * 8, 1))
    ], axis=-1)  # (N*8, 4)

    lidar2img_mat = np.array(lidar2img).reshape(4, 4)
    pts_2d = (lidar2img_mat @ pts_4d.T).T  # (N*8, 4)

    # Perspective divide
    depth = pts_2d[:, 2].copy()
    pts_2d[:, 0] /= np.clip(pts_2d[:, 2], 1e-5, None)
    pts_2d[:, 1] /= np.clip(pts_2d[:, 2], 1e-5, None)

    imgfov_pts_2d = pts_2d[:, :2].reshape(num_bbox, 8, 2)
    depth = depth.reshape(num_bbox, 8)

    # Validity mask per box: all 8 corners must project within a reasonable range
    # and have positive depth
    mask1 = (
        (imgfov_pts_2d[:, :, 0] > -1e5) & (imgfov_pts_2d[:, :, 0] < 1e5) &
        (imgfov_pts_2d[:, :, 1] > -1e5) & (imgfov_pts_2d[:, :, 1] < 1e5) &
        (depth > -1)
    ).all(axis=-1)  # (N,)

    # Filter out overly large projections (distorted boxes behind camera etc.)
    span = imgfov_pts_2d.reshape(num_bbox, 16).max(axis=-1) - \
           imgfov_pts_2d.reshape(num_bbox, 16).min(axis=-1)
    mask2 = span < 2000

    # Score threshold
    mask3 = scores >= DET_SCORE_THRESHOLD

    mask = mask1 & mask2 & mask3
    if not mask.any():
        return img

    filtered_corners = imgfov_pts_2d[mask]
    filtered_scores = scores[mask]
    filtered_labels = labels[mask]

    out = img.copy()
    for i in range(filtered_corners.shape[0]):
        c = COLORMAP.get(int(filtered_labels[i]), DEFAULT_DET_COLOR)
        corners = filtered_corners[i].astype(int)
        for start, end in BOX3D_LINE_INDICES:
            cv2.line(out,
                     (corners[start, 0], corners[start, 1]),
                     (corners[end, 0], corners[end, 1]),
                     c, thickness, cv2.LINE_AA)
    return out


def draw_3d_boxes_on_bev(img, corners_3d, scores, labels, proj_matrix,
                         canvas_size=(BEV_SIZE, BEV_SIZE), thickness=2):
    """
    Draw projected 3D bounding boxes on the BEV image (bottom face only).
    """
    if corners_3d is None or len(corners_3d) == 0:
        return img

    h, w = canvas_size
    num_bbox = corners_3d.shape[0]

    pts_4d = np.concatenate([
        corners_3d.reshape(-1, 3),
        np.ones((num_bbox * 8, 1))
    ], axis=-1)

    proj_mat = np.array(proj_matrix).reshape(4, 4)
    pts_2d = (proj_mat @ pts_4d.T).T
    pts_2d[:, 0] /= np.clip(pts_2d[:, 2], 1e-5, None)
    pts_2d[:, 1] /= np.clip(pts_2d[:, 2], 1e-5, None)

    imgfov_pts_2d = pts_2d[:, :2].reshape(num_bbox, 8, 2)

    # Validity: check all corners within a reasonable range
    mask1 = (
        (imgfov_pts_2d[:, :, 0] > -1e5) & (imgfov_pts_2d[:, :, 0] < 1e5) &
        (imgfov_pts_2d[:, :, 1] > -1e5) & (imgfov_pts_2d[:, :, 1] < 1e5)
    ).all(axis=-1)

    span = imgfov_pts_2d.reshape(num_bbox, 16).max(axis=-1) - \
           imgfov_pts_2d.reshape(num_bbox, 16).min(axis=-1)
    mask2 = span < 2000
    mask3 = scores >= DET_SCORE_THRESHOLD
    mask = mask1 & mask2 & mask3
    if not mask.any():
        return img

    filtered_corners = imgfov_pts_2d[mask]
    filtered_scores = scores[mask]
    filtered_labels = labels[mask]

    out = img.copy()
    for i in range(filtered_corners.shape[0]):
        c = COLORMAP.get(int(filtered_labels[i]), DEFAULT_DET_COLOR)
        corners = filtered_corners[i].astype(int)
        for start, end in BOX3D_BEV_LINE_INDICES:
            cv2.line(out,
                     (corners[start, 0], corners[start, 1]),
                     (corners[end, 0], corners[end, 1]),
                     c, thickness, cv2.LINE_AA)
    return out


# ─── Motion trajectory drawing helpers ─────────────────────────────────────────

def _parse_traj_array(traj_raw, predict_steps=6):
    """
    Parse the raw trajectory tensor into (N, num_modes, predict_steps, 2+).

    The motion head outputs traj as (N, num_modes, predict_steps*5) where each
    step has [dx, dy, log_sigma_x, log_sigma_y, rho], but only the first 2
    values (dx, dy) are used for visualization.  Alternatively, if the tensor
    was already reshaped to (N, num_modes, predict_steps, 5), just slice xy.
    """
    traj = np.array(traj_raw)
    if traj.ndim == 2:
        # (N, num_modes*predict_steps*5) – unlikely but handle
        return None
    if traj.ndim == 3:
        # (N, num_modes, predict_steps*5)
        N, M, flat = traj.shape
        step_dim = flat // predict_steps
        if step_dim < 2:
            return None
        traj = traj.reshape(N, M, predict_steps, step_dim)
    # Now (N, num_modes, predict_steps, >=2)
    return traj[..., :2]  # (N, M, T, 2)


def draw_motion_trajs_on_cam(img, traj_xy, traj_scores, track_scores, track_labels,
                              lidar2img, canvas_size=(CAM_H, CAM_W),
                              top_k=MOTION_TOP_K_MODES, thickness=MOTION_LINE_THICKNESS):
    """
    Draw predicted motion trajectories on a camera image.

    Parameters
    ----------
    img : np.ndarray – (H, W, 3) BGR image
    traj_xy : np.ndarray – (N, num_modes, T, 2) predicted trajectory waypoints in lidar coords (cumsum xy)
    traj_scores : np.ndarray – (N, num_modes) log-softmax scores per mode
    track_scores : np.ndarray – (N,) detection confidence scores
    track_labels : np.ndarray – (N,) class labels
    lidar2img : np.ndarray – (4, 4) projection matrix
    """
    if traj_xy is None or len(traj_xy) == 0:
        return img

    h, w = canvas_size
    out = img.copy()
    N, M, T, _ = traj_xy.shape

    # traj may include SDC query at the end, which is not present in
    # track_scores / track_labels.  Clamp iteration to the shorter length.
    n_scores = len(track_scores) if track_scores is not None else N
    n_labels = len(track_labels) if track_labels is not None else N
    N_iter = min(N, n_scores, n_labels)

    for i in range(N_iter):
        # Filter by track score and vehicle class
        if track_scores is not None and track_scores[i] < MOTION_SCORE_THRESHOLD:
            continue
        if track_labels is not None and int(track_labels[i]) not in MOTION_VEHICLE_IDS:
            continue

        # Get colour for this object
        label = int(track_labels[i]) if track_labels is not None else 0
        base_color = COLORMAP.get(label, DEFAULT_DET_COLOR)

        # Sort modes by score (descending), take top-k
        mode_scores = traj_scores[i]  # (M,)
        sorted_modes = np.argsort(-mode_scores)[:top_k]

        for rank, mode_idx in enumerate(sorted_modes):
            waypoints = traj_xy[i, mode_idx]  # (T, 2)

            # Build homogeneous points: z = -1.84 (ground plane in lidar coords)
            pts_4d = np.stack([
                waypoints[:, 0],
                waypoints[:, 1],
                np.full(T, -1.84),
                np.ones(T)
            ])  # (4, T)

            pts_2d = (lidar2img @ pts_4d).T  # (T, 4)
            depth = pts_2d[:, 2].copy()
            valid_depth = depth > 0.1
            pts_2d[:, 0] /= np.clip(pts_2d[:, 2], 1e-5, None)
            pts_2d[:, 1] /= np.clip(pts_2d[:, 2], 1e-5, None)

            mask = (valid_depth &
                    (pts_2d[:, 0] > 0) & (pts_2d[:, 0] < w) &
                    (pts_2d[:, 1] > 0) & (pts_2d[:, 1] < h))
            if not mask.any():
                continue

            pts = pts_2d[mask, :2].astype(int)

            # Fade colour for lower-ranked modes
            alpha = 1.0 - 0.25 * rank
            color = tuple(int(c * alpha) for c in base_color)
            line_thick = max(1, thickness - rank)

            for j in range(len(pts) - 1):
                cv2.line(out, tuple(pts[j]), tuple(pts[j + 1]),
                         color, line_thick, cv2.LINE_AA)
            # Draw endpoint circle for the best mode
            if rank == 0 and len(pts) > 0:
                cv2.circle(out, tuple(pts[-1]), MOTION_ENDPOINT_RADIUS + 1,
                           color, -1, cv2.LINE_AA)

    return out


def draw_motion_trajs_on_bev(img, traj_xy, traj_scores, track_scores, track_labels,
                              proj_matrix, canvas_size=(BEV_SIZE, BEV_SIZE),
                              top_k=MOTION_TOP_K_MODES, thickness=MOTION_LINE_THICKNESS):
    """
    Draw predicted motion trajectories on the BEV image.
    """
    if traj_xy is None or len(traj_xy) == 0:
        return img

    h, w = canvas_size
    out = img.copy()
    N, M, T, _ = traj_xy.shape

    # traj may include SDC query at the end, which is not present in
    # track_scores / track_labels.  Clamp iteration to the shorter length.
    n_scores = len(track_scores) if track_scores is not None else N
    n_labels = len(track_labels) if track_labels is not None else N
    N_iter = min(N, n_scores, n_labels)

    for i in range(N_iter):
        if track_scores is not None and track_scores[i] < MOTION_SCORE_THRESHOLD:
            continue
        if track_labels is not None and int(track_labels[i]) not in MOTION_VEHICLE_IDS:
            continue

        label = int(track_labels[i]) if track_labels is not None else 0
        base_color = COLORMAP.get(label, DEFAULT_DET_COLOR)

        mode_scores = traj_scores[i]
        sorted_modes = np.argsort(-mode_scores)[:top_k]

        for rank, mode_idx in enumerate(sorted_modes):
            waypoints = traj_xy[i, mode_idx]  # (T, 2)

            pts_4d = np.stack([
                waypoints[:, 0],
                waypoints[:, 1],
                np.zeros(T),
                np.ones(T)
            ])  # (4, T)

            pts_2d = (proj_matrix @ pts_4d).T
            pts_2d[:, 0] /= np.clip(pts_2d[:, 2], 1e-5, None)
            pts_2d[:, 1] /= np.clip(pts_2d[:, 2], 1e-5, None)

            mask = ((pts_2d[:, 0] > 0) & (pts_2d[:, 0] < w) &
                    (pts_2d[:, 1] > 0) & (pts_2d[:, 1] < h))
            if not mask.any():
                continue

            pts = pts_2d[mask, :2].astype(int)

            alpha = 1.0 - 0.25 * rank
            color = tuple(int(c * alpha) for c in base_color)
            line_thick = max(1, thickness - rank)

            for j in range(len(pts) - 1):
                cv2.line(out, tuple(pts[j]), tuple(pts[j + 1]),
                         color, line_thick, cv2.LINE_AA)
            if rank == 0 and len(pts) > 0:
                cv2.circle(out, tuple(pts[-1]), MOTION_ENDPOINT_RADIUS,
                           color, -1, cv2.LINE_AA)

    return out


# ─── Map segmentation drawing helpers ─────────────────────────────────────────

def draw_map_on_bev(img, lane_score=None, drivable=None, proj_matrix=None,
                    canvas_size=(BEV_SIZE, BEV_SIZE), alpha=MAP_ALPHA, threshold=MAP_THRESHOLD):
    """
    Overlay map segmentation (lane and drivable area) on the BEV image.

    The lane_score and drivable arrays are in BEV grid coordinates
    (typically 200×200) and need to be resized to match the BEV image.

    Parameters
    ----------
    img : np.ndarray – (H, W, 3) BGR BEV image
    lane_score : np.ndarray – (C, H_map, W_map) per-class lane confidence
                 C=3: [divider, crossing, contour]
    drivable : np.ndarray – (H_map, W_map) binary drivable area mask
    """
    h, w = canvas_size
    out = img.copy()
    overlay = np.zeros_like(out)
    has_overlay = False

    if drivable is not None:
        drv = drivable.astype(np.float32)
        # Flip vertically to match BEV orientation (y-axis inversion)
        drv = drv[::-1, :]
        drv_resized = cv2.resize(drv, (w, h), interpolation=cv2.INTER_NEAREST)
        mask = drv_resized > 0.5
        if mask.any():
            overlay[mask] = MAP_COLORS['drivable']
            has_overlay = True

    if lane_score is not None:
        channel_names = ['divider', 'crossing', 'contour']
        C = min(lane_score.shape[0], len(channel_names))
        for c in range(C):
            lane_ch = lane_score[c].astype(np.float32)
            lane_ch = lane_ch[::-1, :]
            lane_resized = cv2.resize(lane_ch, (w, h), interpolation=cv2.INTER_NEAREST)
            mask = lane_resized > threshold
            if mask.any():
                overlay[mask] = MAP_COLORS[channel_names[c]]
                has_overlay = True

    if has_overlay:
        mask_any = np.any(overlay > 0, axis=-1)
        out[mask_any] = cv2.addWeighted(
            out, 1.0 - alpha, overlay, alpha, 0
        )[mask_any]

    return out


# ─── Main video creation ─────────────────────────────────────────────────────

def create_video(images_folder, output_video, fps, show_plan_on_cams=True,
                 show_detections=False, show_motion=False, show_map=False):
    """
    Create a combined visualisation video:
      - Left 2/3:  6 camera views arranged in 2×3 grid
      - Right 1/3: BEV image with ego box + planning trajectory
    All overlaid with planning trajectory on front cam and metadata text.
    Optionally overlays detected 3D bounding boxes, motion prediction
    trajectories, and map segmentation on all views.
    """
    # Discover frames from rgb_front
    front_dir = os.path.join(images_folder, 'rgb_front')
    images = sorted([f for f in os.listdir(front_dir) if f.endswith(('.jpg', '.png'))])
    if not images:
        print(f'No images found in {front_dir}')
        return

    # Detect frame numbering format from first file
    first_name = images[0]
    frame_ext = os.path.splitext(first_name)[1]

    # Check which camera dirs exist
    available_cams = {}
    for cam_key, folder in CAM_FOLDER.items():
        cam_dir = os.path.join(images_folder, folder)
        if os.path.isdir(cam_dir):
            available_cams[cam_key] = cam_dir

    bev_dir = os.path.join(images_folder, 'bev')
    has_bev = os.path.isdir(bev_dir)
    meta_dir = os.path.join(images_folder, 'meta')
    has_meta = os.path.isdir(meta_dir)
    det_dir = os.path.join(images_folder, 'detections')
    needs_det_dir = show_detections or show_motion or show_map
    has_det = os.path.isdir(det_dir) and needs_det_dir
    if needs_det_dir and not os.path.isdir(det_dir):
        print(f'Warning: detection/motion/map overlay requested but {det_dir} not found. '
              'Overlays will be skipped.')

    # Determine output video size
    # Camera panel: 3 columns, 2 rows, each resized to fit
    cam_thumb_w = 533   # ~1600/3
    cam_thumb_h = 300   # ~900/3
    cam_panel_w = cam_thumb_w * 3  # 1599
    cam_panel_h = cam_thumb_h * 2  # 600

    # BEV panel height = cam_panel_h, width proportional
    bev_panel_h = cam_panel_h
    bev_panel_w = bev_panel_h  # square

    total_w = cam_panel_w + bev_panel_w
    total_h = cam_panel_h

    # Info bar at bottom
    info_bar_h = 50
    final_h = total_h + info_bar_h
    final_w = total_w

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, fps, (final_w, final_h))

    print(f'Output resolution: {final_w}x{final_h}, fps={fps}')
    print(f'Writing to: {output_video}')

    for idx in trange(len(images), desc='Generating video'):
        frame_name = images[idx]
        frame_num_str = os.path.splitext(frame_name)[0]

        # ── Load metadata ─────────────────────────────────────────
        meta = {}
        plan_traj = None
        if has_meta:
            meta_path = os.path.join(meta_dir, frame_num_str + '.json')
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                # Extract planning trajectory (cumulative waypoints in lidar frame)
                if 'plan' in meta:
                    plan_traj = np.array(meta['plan'])  # (6, 2)
                    if plan_traj.ndim == 2 and plan_traj.shape[1] >= 2:
                        plan_traj = plan_traj[:, :2]
                    else:
                        plan_traj = None

        # ── Load detection data ───────────────────────────────────
        det = None
        if has_det:
            det_path = os.path.join(det_dir, frame_num_str + '.npz')
            det = load_detections(det_path)

        # ── Load and annotate camera images ───────────────────────
        cam_images = {}
        for cam_key in (CAM_TOP_ROW + CAM_BOT_ROW):
            if cam_key in available_cams:
                img_path = os.path.join(available_cams[cam_key], frame_name)
                if os.path.exists(img_path):
                    cam_img = cv2.imread(img_path)
                else:
                    cam_img = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
            else:
                cam_img = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)

            # Draw planning trajectory on camera views
            if show_plan_on_cams and plan_traj is not None and cam_key in LIDAR2IMG:
                cam_img = draw_trajectory_on_cam(
                    cam_img, plan_traj, LIDAR2IMG[cam_key],
                    canvas_size=(cam_img.shape[0], cam_img.shape[1]),
                    add_ego_start=(cam_key == 'CAM_FRONT'))

            # Draw 3D detection bounding boxes on camera views
            if det is not None and 'corners_3d' in det and cam_key in LIDAR2IMG:
                cam_img = draw_3d_boxes_on_cam(
                    cam_img, det['corners_3d'], det.get('scores_3d', np.ones(len(det['corners_3d']))),
                    det.get('labels_3d', np.zeros(len(det['corners_3d']), dtype=int)),
                    LIDAR2IMG[cam_key],
                    canvas_size=(cam_img.shape[0], cam_img.shape[1]))

            # Draw motion prediction trajectories on camera views
            if show_motion and det is not None and 'traj' in det and cam_key in LIDAR2IMG:
                traj_xy = _parse_traj_array(det['traj'])
                if traj_xy is not None:
                    cam_img = draw_motion_trajs_on_cam(
                        cam_img, traj_xy,
                        det.get('traj_scores', np.zeros((traj_xy.shape[0], traj_xy.shape[1]))),
                        det.get('scores_3d'),
                        det.get('labels_3d'),
                        LIDAR2IMG[cam_key],
                        canvas_size=(cam_img.shape[0], cam_img.shape[1]))

            # Add camera label
            add_label(cam_img, cam_key, position=(10, 30), font_scale=0.7, thickness=2)

            # Resize to thumbnail
            cam_images[cam_key] = cv2.resize(cam_img, (cam_thumb_w, cam_thumb_h))

        # ── Assemble camera grid ──────────────────────────────────
        top_row = np.hstack([cam_images.get(c, np.zeros((cam_thumb_h, cam_thumb_w, 3), dtype=np.uint8))
                             for c in CAM_TOP_ROW])
        bot_row = np.hstack([cam_images.get(c, np.zeros((cam_thumb_h, cam_thumb_w, 3), dtype=np.uint8))
                             for c in CAM_BOT_ROW])
        cam_panel = np.vstack([top_row, bot_row])

        # ── Load and annotate BEV ─────────────────────────────────
        if has_bev:
            bev_path = os.path.join(bev_dir, frame_name)
            if os.path.exists(bev_path):
                bev_img = cv2.imread(bev_path)
            else:
                bev_img = np.zeros((BEV_SIZE, BEV_SIZE, 3), dtype=np.uint8)
        else:
            bev_img = np.zeros((BEV_SIZE, BEV_SIZE, 3), dtype=np.uint8)

        # Draw ego vehicle box on BEV
        bev_img = draw_ego_box_bev(bev_img, COOR2TOPDOWN)

        # Draw planning trajectory on BEV
        if plan_traj is not None:
            bev_img = draw_trajectory_on_bev(bev_img, plan_traj, COOR2TOPDOWN)

        # Draw map segmentation on BEV (draw first so it appears behind boxes/trajs)
        if show_map and det is not None:
            bev_img = draw_map_on_bev(
                bev_img,
                lane_score=det.get('lane_score'),
                drivable=det.get('drivable'),
                proj_matrix=COOR2TOPDOWN,
                canvas_size=(bev_img.shape[0], bev_img.shape[1]))

        # Draw 3D detection bounding boxes on BEV
        if det is not None and 'corners_3d' in det:
            bev_img = draw_3d_boxes_on_bev(
                bev_img, det['corners_3d'], det.get('scores_3d', np.ones(len(det['corners_3d']))),
                det.get('labels_3d', np.zeros(len(det['corners_3d']), dtype=int)),
                COOR2TOPDOWN,
                canvas_size=(bev_img.shape[0], bev_img.shape[1]))

        # Draw motion prediction trajectories on BEV
        if show_motion and det is not None and 'traj' in det:
            traj_xy = _parse_traj_array(det['traj'])
            if traj_xy is not None:
                bev_img = draw_motion_trajs_on_bev(
                    bev_img, traj_xy,
                    det.get('traj_scores', np.zeros((traj_xy.shape[0], traj_xy.shape[1]))),
                    det.get('scores_3d'),
                    det.get('labels_3d'),
                    COOR2TOPDOWN,
                    canvas_size=(bev_img.shape[0], bev_img.shape[1]))

        # Add BEV label
        add_label(bev_img, 'BEV', position=(10, 30), font_scale=0.8, thickness=2)

        # Add command if available
        # CARLA RoadOption remapped in agent: LEFT=0, RIGHT=1, STRAIGHT=2,
        # LANEFOLLOW=3, CHANGELANELEFT=4, CHANGELANERIGHT=5
        cmd_list = ['Left', 'Right', 'Straight',
                    'Lane Follow', 'Change Lane L', 'Change Lane R']
        # navi_embed uses nn.Embedding(3, 256), so only indices 0-2 are valid
        # for the planning head command embedding.
        NAVI_EMBED_DIM = 3
        if 'command' in meta:
            cmd_idx = int(meta['command'])
            if 0 <= cmd_idx < len(cmd_list):
                cmd_str = cmd_list[cmd_idx]
            else:
                cmd_str = f'cmd={cmd_idx}'
            # Show which navi_embed index is used (clamped to valid range)
            embed_idx = min(cmd_idx, NAVI_EMBED_DIM - 1)
            embed_note = '' if cmd_idx < NAVI_EMBED_DIM else f' (embed={embed_idx}!)'
            add_label(bev_img, f'CMD: {cmd_str}{embed_note}',
                      position=(10, BEV_SIZE - 15), font_scale=0.55, thickness=2,
                      color=(0, 255, 255))

        # Resize BEV to fit the panel
        bev_panel = cv2.resize(bev_img, (bev_panel_w, bev_panel_h))

        # ── Compose full frame ────────────────────────────────────
        main_panel = np.hstack([cam_panel, bev_panel])

        # Info bar at bottom
        info_bar = np.zeros((info_bar_h, final_w, 3), dtype=np.uint8)
        speed = meta.get('speed', 0)
        steer = meta.get('steer', 0)
        throttle = meta.get('throttle', 0)
        brake = meta.get('brake', 0)
        agent = meta.get('agent', '?')
        info_text = (f'Speed: {speed:.1f} m/s  |  Steer: {steer:.3f}  |  '
                     f'Throttle: {throttle:.2f}  |  Brake: {brake:.1f}  |  Agent: {agent}')
        cv2.putText(info_bar, info_text, (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)

        # Frame number on the right
        frame_text = f'Frame: {frame_num_str}'
        (tw, _), _ = cv2.getTextSize(frame_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.putText(info_bar, frame_text, (final_w - tw - 15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1, cv2.LINE_AA)

        frame_out = np.vstack([main_panel, info_bar])
        video.write(frame_out)

    video.release()
    print(f'Video saved to: {output_video}')


def main():
    parser = argparse.ArgumentParser(
        description='Generate visualization video from Bench2Drive closed-loop evaluation results')
    parser.add_argument('-f', '--folder', required=True,
                        help='Path to the scenario folder (parent of rgb_front/, meta/, bev/ etc.)')
    parser.add_argument('-o', '--output', default=None,
                        help='Output video path (default: <folder>/output.mp4)')
    parser.add_argument('--fps', type=int, default=15,
                        help='Frames per second (default: 15)')
    parser.add_argument('--no-plan-overlay', action='store_true',
                        help='Disable planning trajectory overlay on camera views')
    parser.add_argument('--show-detections', action='store_true',
                        help='Overlay 3D detection bounding boxes on camera and BEV views '
                             '(requires detections/ folder with .npz files from modified agent)')
    parser.add_argument('--show-motion', action='store_true',
                        help='Overlay motion prediction trajectories on camera and BEV views '
                             '(requires detections/ folder with traj/traj_scores in .npz files)')
    parser.add_argument('--show-map', action='store_true',
                        help='Overlay map segmentation (lane/drivable area) on BEV view '
                             '(requires detections/ folder with lane_score/drivable in .npz files)')
    parser.add_argument('--det-threshold', type=float, default=0.3,
                        help='Score threshold for displaying detections (default: 0.3)')
    args = parser.parse_args()

    # Allow runtime override of detection score threshold
    global DET_SCORE_THRESHOLD
    DET_SCORE_THRESHOLD = args.det_threshold

    images_folder = args.folder
    output_video = args.output if args.output else os.path.join(images_folder, 'output.mp4')

    create_video(images_folder, output_video, args.fps,
                 show_plan_on_cams=(not args.no_plan_overlay),
                 show_detections=args.show_detections,
                 show_motion=args.show_motion,
                 show_map=args.show_map)

if __name__ == '__main__':
    main()
