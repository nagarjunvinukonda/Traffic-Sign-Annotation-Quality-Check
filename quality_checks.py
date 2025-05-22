#!/usr/bin/env python3
"""
quality_checks.py

A framework for running programmatic quality checks on Scale AI traffic-sign annotation tasks.

Features:
- Fetch tasks from Scale API.
- Download each image (once per task) and parse its dimensions.
- Run a suite of annotation checks:
    * Bounds, MinSize, AspectRatio, WholeImage
    * AtLeastOneTrafficControl, TrafficLightBackground
    * OverlapDuplicates (IoU-based duplicate detection)
- Shard output into CSV files of configurable size (default 50k tasks/shard).

Usage:
    1. Populate tasks.json with {"tasks": ["taskid1", "taskid2", ...]}.
    2. Export your API key: export SCALE_API_KEY=live_xxx
    3. Install dependencies: pip install -r requirements.txt
    4. Run: python quality_checks.py
"""

import json
import csv
import io
import os
import requests
from itertools import combinations
from PIL import Image
from scaleapi import ScaleClient
import numpy as np
import cv2

# -- Configuration constants --

# Number of tasks to include per output CSV file
SHARD_SIZE = 50000

# HSV ranges for detecting Hough-circle traffic lights
HSV_RANGES = {
    'red1':   ((0, 100, 100), (10, 255, 255)),
    'red2':   ((170, 100, 100), (180, 255, 255)),
    'yellow': ((20, 100, 100), (30, 255, 255)),
    'green':  ((40,  50,  50), (90, 255, 255)),
}

# -- Load task IDs from tasks.json --

with open("tasks.json") as f:
    TASK_IDS = json.load(f)["tasks"]

# Initialize Scale API client
API_KEY = os.environ.get("SCALE_API_KEY", "live_ebacb4d3144e445a8af4547603b2b78e")  #live_ebacb4d3144e445a8af4547603b2b78e
client = ScaleClient(API_KEY)


def fetch_task(tid):
    """
    Retrieve a single task from Scale AI by task ID.

    Args:
        tid (str): Scale task ID.

    Returns:
        dict: Task payload as a dictionary, including `response.annotations` and `params.attachment`.
    """
    return client.get_task(tid).as_dict()


def get_image_size(url):
    """
    Download an image and return its width and height.

    Args:
        url (str): URL of the image.

    Returns:
        (int, int): (width, height) in pixels.
    """
    resp = requests.get(url)
    resp.raise_for_status()
    img = Image.open(io.BytesIO(resp.content))
    return img.width, img.height


def iou(a, b):
    """
    Compute Intersection-over-Union (IoU) between two bounding boxes.

    Args:
        a, b (dict): Each has keys 'left','top','width','height'.

    Returns:
        float: IoU in [0,1], or 0 if no overlap.
    """
    xA = max(a["left"], b["left"])
    yA = max(a["top"],  b["top"])
    xB = min(a["left"] + a["width"],  b["left"] + b["width"])
    yB = min(a["top"] + a["height"], b["top"] + b["height"])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter = interW * interH
    areaA = a["width"] * a["height"]
    areaB = b["width"] * b["height"]
    union = areaA + areaB - inter

    return inter / union if union > 0 else 0


def detect_light_circles(crop_bgr):
    """
    Detect if a crop contains at least two colored circles (e.g., red/yellow/green traffic lights).

    Uses Hough Circle Transform on morphological HSV masks.

    Args:
        crop_bgr (ndarray): BGR image crop.

    Returns:
        bool: True if >=2 circles detected.
    """
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    circle_count = 0

    for color, (lo, hi) in HSV_RANGES.items():
        # Threshold to color mask
        mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
        # Clean small noise
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern)
        mask = cv2.dilate(mask, kern, iterations=2)

        # Hough Circle detection
        circles = cv2.HoughCircles(
            mask, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
            param1=50, param2=15, minRadius=5, maxRadius=50
        )
        if circles is not None:
            circle_count += len(circles[0])

    return circle_count >= 2


def is_real_traffic_light(full_img, box):
    """
    Crop the full image to the bounding box and detect if it's a real traffic light.

    Args:
        full_img (ndarray): Full BGR image.
        box (dict): Annotation with 'left','top','width','height'.

    Returns:
        bool: True if crop contains >=2 light circles.
    """
    # Convert and clamp to integers
    l = int(round(box['left'])); t = int(round(box['top']))
    w = int(round(box['width'])); h = int(round(box['height']))
    l, t = max(0, l), max(0, t)
    w = min(w, full_img.shape[1] - l)
    h = min(h, full_img.shape[0] - t)

    crop = full_img[t:t+h, l:l+w]
    if crop.size == 0:
        return False

    return detect_light_circles(crop)


def run_checks(task):
    """
    Execute all quality checks on a single task.

    Args:
        task (dict): The task payload from fetch_task().

    Returns:
        list of tuples: (task_id, uuid, check_name, severity, detail)
    """
    issues = []
    tid = task["task_id"]
    anns = task["response"]["annotations"]
    img_url = task["params"]["attachment"]

    # Download & decode image once
    resp = requests.get(img_url)
    resp.raise_for_status()
    full_img = cv2.imdecode(
        np.frombuffer(resp.content, np.uint8),
        cv2.IMREAD_COLOR
    )
    img_h, img_w = full_img.shape[:2]

    # 1) AtLeastOneTrafficControl check
    tcs = [a for a in anns if a["label"] == "traffic_control_sign"]
    if not tcs:
        issues.append((tid, "", "AtLeastOneTrafficControl", "Warning",
                       "no traffic_control_sign in image"))

    # 2) Group annotations by label for duplicate detection
    by_lbl = {}
    for a in anns:
        by_lbl.setdefault(a["label"], []).append(a)

    # 3) Per-annotation checks
    for a in anns:
        uuid = a["uuid"]
        lbl = a["label"]
        left, top = a["left"], a["top"]
        w, h = a["width"], a["height"]
        attrs = a["attributes"]

        # WholeImage: covers entire frame
        if w == img_w and h == img_h:
            issues.append((tid, uuid, "WholeImage", "Error",
                           "box covers entire image"))

        # Bounds: must stay inside frame
        if left < 0 or top < 0 or left + w > img_w or top + h > img_h:
            detail = f"exceeds by {max(0, left+w-img_w)}px horizontal, {max(0, top+h-img_h)}px vertical"
            issues.append((tid, uuid, "Bounds", "Error", detail))

        # MinSize: at least 10x10 px
        if w < 10 or h < 10:
            issues.append((tid, uuid, "MinSize", "Warning", f"{w}x{h}px"))

        # AspectRatio: avoid extreme shapes
        ar = w / h if h else float("inf")
        if ar > 5 or ar < 0.2:
            issues.append((tid, uuid, "AspectRatio", "Warning", f"w/h={ar:.2f}"))

        # TrafficLightBackground: only for traffic_control_sign
        if lbl == "traffic_control_sign":
            if is_real_traffic_light(full_img, a):
                if attrs.get("background_color") != "other":
                    issues.append((tid, uuid, "TrafficLightBackground", "Error",
                                   f"detected real light but bg_color={attrs.get('background_color')}"))

    # 4) OverlapDuplicates: IoU > 0.5 per label
    for lbl, group in by_lbl.items():
        for a1, a2 in combinations(group, 2):
            iou_val = iou(a1, a2)
            if iou_val > 0.5:
                issues.append((tid, a1["uuid"], "OverlapDuplicates", "Warning",
                               f"IoU={iou_val:.2f} between two {lbl}"))

    return issues


def write_shard(shard_idx, task_ids):
    """
    Write a range (shard) of tasks to its own CSV.

    Args:
        shard_idx (int): Zero-based index of the shard.
        task_ids (list): Sub-list of TASK_IDS for this shard.
    """
    start = shard_idx * SHARD_SIZE
    end = min((shard_idx + 1) * SHARD_SIZE, len(TASK_IDS))
    filename = f"results_{start+1}-{end}.csv"

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        for tid in task_ids:
            task = fetch_task(tid)

            # Section header per task
            writer.writerow([tid])
            writer.writerow(["uuid", "label", "check", "severity", "detail"])

            # Run checks and write rows
            for tid2, uuid, check, sev, det in run_checks(task):
                if tid2 == tid:
                    # Look up the label for context
                    lbl = next(
                        (a["label"] for a in task["response"]["annotations"]
                         if a["uuid"] == uuid),
                        ""
                    )
                    writer.writerow([uuid, lbl, check, sev, det])

            writer.writerow([])  # Blank line between tasks

    print(f"Wrote {filename} ({len(task_ids)} tasks)")


if __name__ == "__main__":
    """
    Main entrypoint: shard TASK_IDS into multiple CSVs.
    """
    total = len(TASK_IDS)
    num_shards = (total + SHARD_SIZE - 1) // SHARD_SIZE

    for shard_idx in range(num_shards):
        start = shard_idx * SHARD_SIZE
        end = start + SHARD_SIZE
        write_shard(shard_idx, TASK_IDS[start:end])

    print("✅ All shard results generated.")
