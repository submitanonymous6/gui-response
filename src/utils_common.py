import json
import logging
import os
import pickle
import shutil

import cv2
import numpy as np
import torch
from pytorch_msssim import ms_ssim

import re
from math import dist
from typing import List, Dict


def calculate_ssim(curr_frame, prev_frame, frame_x, frame_y, frame_w, frame_h, resize_ratio):
    curr_frame_copy = curr_frame.copy()
    prev_frame_copy = prev_frame.copy()

    curr_frame_copy = get_crop_frame(curr_frame_copy, frame_x, frame_y, frame_w, frame_h)
    prev_frame_copy = get_crop_frame(prev_frame_copy, frame_x, frame_y, frame_w, frame_h)

    curr_frame_copy = get_resize_frame(curr_frame_copy, resize_ratio)
    prev_frame_copy = get_resize_frame(prev_frame_copy, resize_ratio)

    # filter_with_threshold(curr_frame_copy, prev_frame_copy, 0)

    # return ssim(curr_frame_copy, prev_frame_copy, channel_axis=-1)
    return torch_msssim(curr_frame_copy, prev_frame_copy)


class Coords:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def __str__(self):
        return "(" + str(self.x) + ", " + str(self.y) + ")"


class ScreenTap:
    def __init__(self, x, y, confidence=None):
        self.loc = Coords(x, y)
        self.touch_confidence = confidence

    def get_x(self):
        return self.loc.get_x()

    def get_y(self):
        return self.loc.get_y()

    def get_touch_confidence(self):
        return self.touch_confidence

    def set_touch_confidence(self, confidence):
        self.touch_confidence = confidence

    def __str__(self):
        return ("tap{" + "x=" + str(self.get_x()) +
                ", y=" + str(self.get_y()) +
                "}")

    def __hash__(self):
        return hash(self.x)

    def asJson(self):
        return dict(x=self.loc.get_x(), y=self.loc.get_y(), confidence=self.touch_confidence)


class Frame:
    def __init__(self, id):
        self.id = id
        self.screen_taps = []

    def get_id(self):
        return self.id

    def set_id(self, id):
        self.id = id

    def get_screen_taps(self):
        return self.screen_taps

    def set_screen_taps(self, taps):
        self.screen_taps = taps

    def add_tap(self, tap):
        self.screen_taps.append(tap)

    def __str__(self):
        return ("frame{" + "id=" + str(self.id) +
                ", taps=" + str([str(tap) for tap in self.screen_taps]) + "}")

    def asJson(self):
        return dict(screenId=self.id, screenTap=self.screen_taps)


def load_variable(filename):
    try:
        f = open(filename, 'rb')
        r = pickle.load(f)
        f.close()
        return r

    except EOFError:
        return ''


def save_variable(v, filename):
    f = open(filename, 'wb')
    pickle.dump(v, f)
    f.close()
    return filename


def image_save(path, image):
    if image is None:
        logging.debug(f"Could not write image: image is None. Path: {path}")
        return

    image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)

    # success = cv2.imencode('.jpg', image)[1].tofile(path)
    success = cv2.imwrite(path, image)  # robust way
    if not success:
        logging.debug(f"Failed to save image to path: {path}")
        # raise Exception(f"Could not write image to {path}")


def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Cannot open video file: {video_path}")
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    cap.release()
    return frame_count, fps


def get_video_fps(video_file):
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def get_video_frames_cv2(video_file):
    frame_list = list()
    cap = cv2.VideoCapture(video_file)
    success, frame = cap.read()
    while success:
        frame_list.append(frame)
        success, frame = cap.read()

    cap.release()
    return frame_list


def get_video_frame_size_and_timestamps(video_file):
    cap = cv2.VideoCapture(video_file)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    timestamps = []
    actual_frame_count = 0

    while True:
        success, _ = cap.read()
        if not success:
            break

        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        timestamps.append(timestamp_ms)
        actual_frame_count += 1

    cap.release()

    return width, height, actual_frame_count, timestamps


# --- Step 1: Parse the raw string ---
def parse_touch_detections(raw_str: str) -> List[Dict]:
    pattern = r"frame\{id=(\d+), taps=\['tap\{x=(.*?), y=(.*?)\}'\]\}"
    matches = re.findall(pattern, raw_str)
    return [{'id': int(fid), 'taps': [f"tap{{x={x}, y={y}}}"]} for fid, x, y in matches]


# --- Step 2: Group consecutive frames ---
def group_consecutive_taps(tap_frames: List[Dict]) -> List[List[Dict]]:
    tap_frames.sort(key=lambda f: f['id'])
    groups = []
    current_group = [tap_frames[0]]

    for i in range(1, len(tap_frames)):
        if tap_frames[i]['id'] == tap_frames[i - 1]['id'] + 1:
            current_group.append(tap_frames[i])
        else:
            groups.append(current_group)
            current_group = [tap_frames[i]]
    groups.append(current_group)
    return groups


# --- Step 3: Classify Tap or Swipe ---
def classify_interaction_type(group: List[Dict]) -> str:
    coords = []
    for frame in group:
        tap_str = frame['taps'][0]
        x = float(tap_str.split("x=")[1].split(",")[0])
        y = float(tap_str.split("y=")[1].replace("}", ""))
        coords.append((x, y))
    if len(coords) >= 2:
        movement = dist(coords[0], coords[-1])
        return 'Swipe' if movement > 10 else 'Tap'
    else:
        return 'Tap'


# --- Step 4: Extract just the interaction types ---
def extract_interaction_types(touch_detections: List[Dict]) -> List[str]:
    groups = group_consecutive_taps(touch_detections)
    return [classify_interaction_type(group) for group in groups]


def get_resize_frame(image, resize_ratio):
    return cv2.resize(image, dsize=None, fx=resize_ratio, fy=resize_ratio)


def get_crop_frame(frame, frame_x, frame_y, frame_w, frame_h):
    return frame[frame_y:frame_y + frame_h, frame_x:frame_x + frame_w]


class ComplexEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'asJson'):
            return obj.asJson()
        return super().default(obj)


def output_data_to_json(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, cls=ComplexEncoder, indent=2, sort_keys=True)


def torch_msssim(npimg1, npimg2):
    img1 = np.array(npimg1).astype(np.float32)
    img2 = np.array(npimg2).astype(np.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    img_1 = torch.from_numpy(img1).unsqueeze(0).permute(0, 3, 1, 2).to(device)  # (1, C, H, W)
    img_2 = torch.from_numpy(img2).unsqueeze(0).permute(0, 3, 1, 2).to(device)

    with torch.no_grad():
        return ms_ssim(img_1, img_2, data_range=255, size_average=False).item()


def is_all_false(hastap_list):
    for hastap in hastap_list:
        if hastap is True:
            return False
    return True


def mkdir(path):
    folder = os.path.exists(path)
    if folder is False:
        os.makedirs(path)
        return True
    return True


def clear_and_mkdir(path):
    folder = os.path.exists(path)
    if folder is True:
        shutil.rmtree(path)
        os.makedirs(path)


def is_all_less_threshod(scores):
    if len(scores) == 0:
        return True

    for score in scores:
        if score >= 0.3:
            return False

    return True
