"""
Synthetic Test Image Generator
================================
Generates computer-generated images representing each security threat scenario.
These images are used as test data for the security system pipeline.

Usage:
    python tests/generate_test_images.py

Output:
    tests/test_data/images/<threat_type>/<scene>.jpg
    tests/test_data/labels/<threat_type>/<scene>.txt   (YOLO format)
    tests/test_data/manifest.json                       (ground truth)
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

OUTPUT_DIR = Path(__file__).parent / "test_data"
IMAGE_W, IMAGE_H = 640, 480


# ─── DRAWING HELPERS ──────────────────────────────────────────────────────────

def blank(scene: str = "street") -> np.ndarray:
    img = np.zeros((IMAGE_H, IMAGE_W, 3), dtype=np.uint8)
    if scene == "street":
        img[:220] = [200, 210, 220]           # sky (BGR)
        img[220:] = [110, 110, 110]           # pavement
        cv2.line(img, (0, 220), (IMAGE_W, 220), (80, 80, 80), 3)
    elif scene == "park":
        img[:200] = [200, 210, 220]
        img[200:] = [40, 140, 40]             # grass
    elif scene == "wall":
        img[:] = [175, 165, 155]              # concrete wall
    return img


def draw_person(img, x, y, h=110, color=(50, 50, 200)):
    """Simplified stick figure. (x,y) = bottom-centre of figure."""
    r = max(h // 8, 6)
    head_y = y - h + r
    cv2.circle(img, (x, head_y), r, color, -1)
    bt = head_y + r
    bb = y - h // 3
    cv2.rectangle(img, (x - r, bt), (x + r, bb), color, -1)
    cv2.rectangle(img, (x - r, bb), (x - r // 2, y), color, -1)
    cv2.rectangle(img, (x + r // 2, bb), (x + r, y), color, -1)
    return (x - r, head_y - r, x + r, y)          # rough bbox


def yolo_label(cls_id, bbox, w=IMAGE_W, h=IMAGE_H):
    x1, y1, x2, y2 = bbox
    cx = ((x1 + x2) / 2) / w
    cy = ((y1 + y2) / 2) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


# ─── SCENE GENERATORS ─────────────────────────────────────────────────────────

def scene_loitering():
    img = blank("street")
    draw_person(img, 320, 390, h=130)
    bbox = [255, 255, 385, 395]
    return img, "loitering", [{"class": "person", "cls_id": 0, "bbox": bbox}]


def scene_brawl():
    img = blank("street")
    draw_person(img, 270, 385, color=(50, 50, 200))
    draw_person(img, 320, 378, color=(200, 50, 50))
    draw_person(img, 365, 383, color=(50, 180, 50))
    # Extended arms
    cv2.line(img, (225, 330), (320, 325), (50, 50, 200), 5)
    cv2.line(img, (365, 325), (420, 330), (200, 50, 50), 5)
    bbox = [215, 245, 435, 395]
    return img, "brawl", [{"class": "brawl", "cls_id": 2, "bbox": bbox}]


def scene_fire():
    img = blank("street")
    rng = np.random.default_rng(0)
    cx, base_y = 320, 430
    for y in range(170, base_y):
        for x in range(190, 450):
            dx, dy = x - cx, y - base_y
            d = (dx**2 + dy**2) ** 0.5
            if d < 130:
                t = max(0.0, 1 - d / 130)
                r = int(255 * t)
                g = int(140 * t * t)
                img[y, x] = [0, g, r]
    # flame tips
    for _ in range(25):
        fx = int(rng.integers(220, 420))
        fy = int(rng.integers(160, 320))
        cv2.ellipse(img, (fx, fy), (14, 28), int(rng.integers(0, 35)), 0, 360, (0, 60, 255), -1)
    bbox = [190, 160, 450, 430]
    return img, "fire", [{"class": "fire", "cls_id": 4, "bbox": bbox}]


def scene_smoke():
    img = blank("street")
    rng = np.random.default_rng(1)
    for i in range(14):
        px = int(320 + rng.integers(-80, 80))
        py = int(210 - i * 14 + rng.integers(-15, 15))
        rad = 28 + i * 6
        overlay = img.copy()
        cv2.circle(overlay, (px, py), rad, (175, 175, 175), -1)
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    bbox = [170, 60, 470, 280]
    return img, "smoke", [{"class": "smoke", "cls_id": 5, "bbox": bbox}]


def scene_abandoned_bag():
    img = blank("street")
    # Bag body
    cv2.rectangle(img, (275, 365), (385, 425), (35, 85, 130), -1)
    cv2.rectangle(img, (275, 365), (385, 425), (20, 55, 95), 3)
    # Strap
    cv2.rectangle(img, (300, 352), (360, 370), (25, 65, 110), -1)
    bbox = [270, 348, 390, 430]
    return img, "abandoned_bag", [{"class": "abandoned_bag", "cls_id": 3, "bbox": bbox}]


def scene_crowd_surge():
    img = blank("street")
    positions = [
        (170, 390), (205, 382), (240, 388), (275, 383), (310, 389),
        (345, 384), (380, 390), (415, 385), (450, 389), (485, 383),
        (188, 360), (223, 355), (258, 362), (293, 357), (328, 363),
        (363, 358), (398, 364), (433, 359), (468, 365),
    ]
    colors = [
        (50,  50, 200), (200, 50,  50), (50, 200,  50),
        (200, 200, 50), (200, 50, 200), (50, 200, 200),
    ]
    for i, (px, py) in enumerate(positions):
        draw_person(img, px, py, h=80, color=colors[i % len(colors)])
    bbox = [160, 265, 495, 400]
    return img, "crowd_surge", [{"class": "crowd_surge", "cls_id": 6, "bbox": bbox}]


def scene_weapon():
    img = blank("street")
    draw_person(img, 305, 390, h=125)
    # Weapon (elongated dark shaft + tip)
    cv2.line(img, (315, 305), (400, 235), (25, 25, 25), 7)
    cv2.line(img, (400, 235), (418, 215), (70, 70, 70), 9)
    cv2.circle(img, (418, 215), 5, (120, 120, 120), -1)
    bbox = [285, 210, 425, 395]
    return img, "weapon", [{"class": "weapon", "cls_id": 7, "bbox": bbox}]


def scene_vehicle_intrusion():
    img = blank("street")
    # Car body
    cv2.rectangle(img, (80, 300), (520, 440), (55, 55, 160), -1)
    # Roof
    cv2.rectangle(img, (145, 248), (455, 305), (75, 75, 185), -1)
    # Windows
    cv2.rectangle(img, (165, 258), (280, 300), (180, 220, 255), -1)
    cv2.rectangle(img, (320, 258), (430, 300), (180, 220, 255), -1)
    # Wheels
    cv2.circle(img, (165, 440), 42, (28, 28, 28), -1)
    cv2.circle(img, (440, 440), 42, (28, 28, 28), -1)
    # Pedestrian-zone sign (red circle with X)
    cv2.circle(img, (600, 65), 32, (0, 0, 220), 3)
    cv2.line(img, (578, 43), (622, 87), (0, 0, 220), 4)
    cv2.line(img, (622, 43), (578, 87), (0, 0, 220), 4)
    bbox = [80, 248, 520, 445]
    return img, "vehicle_intrusion", [{"class": "vehicle_intrusion", "cls_id": 8, "bbox": bbox}]


def scene_graffiti():
    img = blank("wall")
    rng = np.random.default_rng(2)
    palette = [(255, 50, 50), (50, 255, 50), (50, 50, 255), (255, 255, 50), (255, 50, 255)]
    for _ in range(35):
        x1 = int(rng.integers(40, 520))
        y1 = int(rng.integers(40, 380))
        x2 = x1 + int(rng.integers(20, 110))
        y2 = y1 + int(rng.integers(10, 65))
        color = palette[int(rng.integers(len(palette)))]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, int(rng.integers(2, 9)))
    for _ in range(22):
        cx = int(rng.integers(80, 560))
        cy = int(rng.integers(60, 400))
        color = palette[int(rng.integers(len(palette)))]
        cv2.circle(img, (cx, cy), int(rng.integers(10, 42)), color, -1)
    bbox = [40, 40, 600, 440]
    return img, "graffiti_vandalism", [{"class": "graffiti_vandalism", "cls_id": 9, "bbox": bbox}]


def scene_suspicious_package():
    img = blank("street")
    # Box
    cv2.rectangle(img, (245, 345), (395, 432), (95, 115, 145), -1)
    cv2.rectangle(img, (245, 345), (395, 432), (45, 55, 75), 3)
    # Tape diagonals
    cv2.line(img, (245, 345), (395, 432), (0, 185, 210), 3)
    cv2.line(img, (395, 345), (245, 432), (0, 185, 210), 3)
    # Wires
    cv2.line(img, (285, 345), (280, 310), (0, 0, 200), 2)
    cv2.line(img, (320, 345), (320, 300), (0, 200, 0), 2)
    cv2.line(img, (355, 345), (360, 310), (200, 0, 0), 2)
    # Small circuit board symbol
    cv2.rectangle(img, (300, 370), (340, 400), (0, 200, 0), 1)
    bbox = [240, 295, 400, 440]
    return img, "suspicious_package", [{"class": "suspicious_package", "cls_id": 10, "bbox": bbox}]


# ─── MULTI-THREAT SCENES ──────────────────────────────────────────────────────

def scene_person_with_fire():
    """Combined: arson scenario — person + fire."""
    img = blank("street")
    draw_person(img, 180, 390, h=120, color=(50, 50, 200))
    # Fire on right side
    rng = np.random.default_rng(3)
    cx, base_y = 440, 430
    for y in range(200, base_y):
        for x in range(360, 520):
            dx, dy = x - cx, y - base_y
            d = (dx**2 + dy**2) ** 0.5
            if d < 90:
                t = max(0.0, 1 - d / 90)
                img[y, x] = [0, int(100 * t * t), int(255 * t)]
    bboxes = [
        {"class": "person",  "cls_id": 0, "bbox": [130, 255, 230, 395]},
        {"class": "fire",    "cls_id": 4, "bbox": [358, 195, 525, 432]},
    ]
    return img, "person_with_fire", bboxes


def scene_multiple_loiterers():
    """Two people loitering together."""
    img = blank("street")
    draw_person(img, 260, 390, h=125, color=(50, 50, 200))
    draw_person(img, 380, 388, h=120, color=(200, 50, 50))
    bboxes = [
        {"class": "person", "cls_id": 0, "bbox": [205, 258, 315, 395]},
        {"class": "person", "cls_id": 0, "bbox": [325, 262, 435, 393]},
    ]
    return img, "multiple_loiterers", bboxes


# ─── GENERATOR ────────────────────────────────────────────────────────────────

SCENES = [
    scene_loitering,
    scene_brawl,
    scene_fire,
    scene_smoke,
    scene_abandoned_bag,
    scene_crowd_surge,
    scene_weapon,
    scene_vehicle_intrusion,
    scene_graffiti,
    scene_suspicious_package,
    scene_person_with_fire,
    scene_multiple_loiterers,
]


def generate_all(output_dir: Path = OUTPUT_DIR):
    manifest = []

    for scene_fn in SCENES:
        img, scene_name, detections = scene_fn()

        # Determine primary threat label (first detection class)
        primary_class = detections[0]["class"]
        img_dir = output_dir / "images" / primary_class
        lbl_dir = output_dir / "labels" / primary_class
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        img_path = img_dir / f"{scene_name}.jpg"
        lbl_path = lbl_dir / f"{scene_name}.txt"

        cv2.imwrite(str(img_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

        with open(lbl_path, "w") as f:
            for det in detections:
                f.write(yolo_label(det["cls_id"], det["bbox"]) + "\n")

        manifest.append({
            "scene":      scene_name,
            "image_path": str(img_path.relative_to(output_dir.parent)),
            "detections": detections,
        })

        print(f"  ✓  {scene_name:30s}  →  {img_path}")

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest written: {manifest_path}")
    print(f"Total scenes: {len(manifest)}")
    return manifest


if __name__ == "__main__":
    print("Generating synthetic test images …\n")
    generate_all()
