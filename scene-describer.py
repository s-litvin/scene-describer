# -*- coding: utf-8 -*-
"""nsfw-scene-describer.ipynb

"""

from google.colab import drive
drive.mount('/content/drive')

import os
project_dir = "/content/drive/MyDrive/ColabData/nsfw-scene-describer"
video_files = [f for f in os.listdir(project_dir) if f.lower().endswith((".mp4", ".mov", ".mkv"))]

!pip install ultralytics transformers torch torchvision pillow -q

import cv2, math, json, random
from ultralytics import YOLO
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from PIL import Image, ImageFilter
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
yolo = YOLO("yolo11x.pt")

ZONES = 15
CANDIDATES_PER_ZONE = 5
RANDOM_OFFSET = 0.2  # ±20% от зоны

def sharpness_score(pil_img):
    # Laplas operator PIL → CV → float variance
    img_cv = np.array(pil_img.convert("L"))
    return cv2.Laplacian(img_cv, cv2.CV_64F).var()

def extract_best_frames(video_path, out_dir, prefix):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total / fps if fps else 0

    zone_frames = np.linspace(0, total - 1, ZONES, dtype=int)
    saved = []

    for zone_idx, center_frame in enumerate(zone_frames):
        candidates = []
        for _ in range(CANDIDATES_PER_ZONE):
            offset = int((random.uniform(-RANDOM_OFFSET, RANDOM_OFFSET)) * (total / ZONES))
            frame_no = np.clip(center_frame + offset, 0, total - 1).astype(int)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
            if not ret: continue
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            score = sharpness_score(pil_img)
            candidates.append((score, pil_img, frame_no))

        if not candidates: continue
        best = max(candidates, key=lambda x: x[0])
        out_fname = f"{prefix}_frame_{zone_idx:02d}.jpg"
        out_path = os.path.join(out_dir, out_fname)
        best[1].save(out_path, quality=100)
        saved.append((out_fname, best[2] / fps))

    cap.release()
    return saved

for video in video_files:
    video_name = os.path.splitext(video)[0]
    video_path = os.path.join(project_dir, video)
    frames_dir = os.path.join(project_dir, video_name, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    print(f"Processing video: {video_name}")
    frames = extract_best_frames(video_path, frames_dir, prefix=video_name)
    print(f"Saved {len(frames)} frames")

video_dirs = [d for d in os.listdir(project_dir) if os.path.isdir(os.path.join(project_dir, d))]

yolo = YOLO("yolo11l.pt")

for video_name in video_dirs:
    frames_dir = os.path.join(project_dir, video_name, "frames")
    if not os.path.exists(frames_dir):
        print(f"⚠️ Skip: folder not found: {frames_dir}")
        continue

    out_dir = os.path.join(project_dir, video_name, "yolo_json")
    os.makedirs(out_dir, exist_ok=True)

    for fname in os.listdir(frames_dir):
        if not fname.endswith(".jpg"): continue
        fpath = os.path.join(frames_dir, fname)
        frame_id = os.path.splitext(fname)[0]

        results = yolo(fpath)
        detections = []
        for r in results:
            for det in r.boxes.data.tolist():
                cls_id = int(det[5])
                label = yolo.model.names[cls_id]
                x1, y1, x2, y2, conf = det[:5]
                detections.append({
                    "label": label,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(conf)
                })

        data = {"video": video_name, "frame": fname, "objects": detections}
        with open(os.path.join(out_dir, f"{frame_id}.json"), "w") as jf:
            json.dump(data, jf, indent=2)

print("Object detecting done.")

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
blip_model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    torch_dtype=torch.float16 if device=="cuda" else torch.float32
)
blip_model.to(device)

for video_name in video_dirs:
    frames_dir = os.path.join(project_dir, video_name, "frames")
    if not os.path.exists(frames_dir):
        print(f"⚠️ Skip: folder not found {frames_dir}")
        continue

    out_dir = os.path.join(project_dir, video_name, "blip_json")
    os.makedirs(out_dir, exist_ok=True)

    for fname in os.listdir(frames_dir):
        if not fname.endswith(".jpg"): continue
        fpath = os.path.join(frames_dir, fname)
        frame_id = os.path.splitext(fname)[0]

        image = Image.open(fpath).convert("RGB")
        #prompt = "Describe the erotic or intimate scene in detail:"
        inputs = processor(images=image, return_tensors="pt").to(device)
        #inputs = processor(images=image, return_tensors="pt").to(device, torch.float16 if device == "cuda" else torch.float32)
        gen = blip_model.generate(**inputs, max_new_tokens=200)
        caption = processor.batch_decode(gen, skip_special_tokens=True)[0]

        data = {"video": video_name, "frame": fname, "caption": caption}
        with open(os.path.join(out_dir, f"{frame_id}.json"), "w") as jf:
            json.dump(data, jf, indent=2)

print("Scene description done")

import os
import json

def load_jsons(folder):
    return {
        os.path.splitext(f)[0]: json.load(open(os.path.join(folder, f)))
        for f in sorted(os.listdir(folder)) if f.endswith(".json")
    }

for video_name in video_dirs:
    yolo_dir = os.path.join(project_dir, video_name, "yolo_json")
    blip_dir = os.path.join(project_dir, video_name, "blip_json")

    if not os.path.exists(yolo_dir) or not os.path.exists(blip_dir):
        print(f"⚠️ Skip: {video_name} — no JSON-folders.")
        continue

    yolo_data = load_jsons(yolo_dir)
    blip_data = load_jsons(blip_dir)

    shared_keys = sorted(set(yolo_data.keys()) & set(blip_data.keys()))

    prompt_lines = []
    section = f"""You are a writer of erotic stories. Based on a visual scene and description of objects, generate an artistic erotic description of what is happening."""
    prompt_lines.append(section)
    for key in shared_keys:
        yolo_objects = yolo_data[key].get("objects", [])
        object_labels = list({obj["label"] for obj in yolo_objects})
        blip_caption = blip_data[key].get("caption", "").strip()

        frame_name = yolo_data[key]["frame"]

        section = f"""
Scene: {frame_name}
NSFW description: {blip_caption}
Objects onscene: {", ".join(object_labels) if object_labels else "not detected"}
"""
        prompt_lines.append(section)

    # Сохраняем в подпапку видео
    prompt_path = os.path.join(project_dir, video_name, "scene_prompts.txt")
    with open(prompt_path, "w") as f:
        f.writelines(prompt_lines)

    print(f"Prompt saved: {prompt_path}")

from PIL import Image
import os
import math

def create_collage_for_video(video_folder, cols=5, thumb_size=(320, 180)):
    frames_path = os.path.join(video_folder, "frames")
    collage_path = os.path.join(video_folder, "collage.jpg")

    if not os.path.exists(frames_path):
        print(f"⏭️ Пропускаю: {video_folder} (нет папки frames/)")
        return

    # Combining all screenshots .jpg 
    frame_files = sorted([
        f for f in os.listdir(frames_path)
        if f.lower().endswith(".jpg")
    ])
    if not frame_files:
        print(f"⚠️ No screenshots found in {frames_path}")
        return


    thumbs = []
    for fname in frame_files:
        img = Image.open(os.path.join(frames_path, fname)).convert("RGB")
        img.thumbnail(thumb_size)
        thumbs.append(img)

    rows = math.ceil(len(thumbs) / cols)
    collage_w = cols * thumb_size[0]
    collage_h = rows * thumb_size[1]

    collage = Image.new("RGB", (collage_w, collage_h), color=(0, 0, 0))

    for idx, thumb in enumerate(thumbs):
        x = (idx % cols) * thumb_size[0]
        y = (idx // cols) * thumb_size[1]
        collage.paste(thumb, (x, y))

    collage.save(collage_path)
    print(f"Collage saved: {collage_path}")


project_dir = "/content/drive/MyDrive/ColabData/nsfw-scene-describer"

for video_name in sorted(os.listdir(project_dir)):
    video_path = os.path.join(project_dir, video_name)
    if os.path.isdir(video_path):
        create_collage_for_video(video_path, cols=3)
