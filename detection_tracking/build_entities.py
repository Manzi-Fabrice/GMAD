
# ********************************* build_entities.py ****************************
# Extracts crops from tracking results and applies optional CLIP relabeling.
# ***********************************************************************

from pathlib import Path
import os, json, cv2
from typing import List, Dict, Any, Optional
import yaml
import torch, clip
from PIL import Image
import argparse

clip_overwrite_threshold = 0.40
clip_min_conf = 0.6
crop_pad = 2
use_clip = True
device = "cuda" if torch.cuda.is_available() else "cpu"

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _safe_crop(img, bbox, pad):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = map(int, bbox[:4])
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)
    if x2 <= x1 or y2 <= y1: 
        return None
    return img[y1:y2, x1:x2]

def _best_frame(frames: List[Dict[str, Any]]):
    if not frames: return {}
    return max(frames, key=lambda f: f.get("det_score", 0.0))

def _collect_entities(cfg: Dict[str, Any]) -> Dict[str, Any]:
    data = json.load(open(cfg["tracks_json"], "r", encoding="utf-8"))
    crops_dir = Path(cfg["crops_dir"]); ensure_dir(crops_dir)
    total, missing = 0, 0
    scenes = []

    for sc in data.get("scenes", []):
        sid = sc["scene_id"]
        ents = []
        for tr in sc.get("tracks", []):
            best = _best_frame(tr.get("frames", []))
            frame_path = best.get("image_path")
            crop_path = None

            if frame_path and os.path.exists(frame_path):
                img = cv2.imread(frame_path)
                if img is not None:
                    crop = _safe_crop(img, best.get("bbox", []), crop_pad)
                    if crop is not None:
                        out_name = f"scene{sid:04d}_{tr['id']}.jpg"
                        out_path = crops_dir / out_name
                        cv2.imwrite(str(out_path), crop)
                        crop_path = str(out_path)
                        total += 1
                    else:
                        missing += 1
                else:
                    missing += 1
            else:
                missing += 1

            ents.append({
                "id": tr["id"],
                "label": tr.get("label", "object"),
                "cls_id": tr.get("cls_id", -1),
                "score": tr.get("score", 1.0),
                "rep_frame_path": frame_path,
                "rep_crop_path": crop_path,
                "frames": tr.get("frames", [])
            })
        scenes.append({"scene_id": sid, "entities": ents})
    return {"meta": {"source_tracks": cfg["tracks_json"]}, "scenes": scenes}

def _clip_reclass_entities(data: Dict[str, Any], cfg: Dict[str, Any]):

    model, preprocess = clip.load(cfg["clip_model"], device=device)
    model.eval()
    prompts = [f"a photo of a {c}" for c in cfg["clip_classes"]]
    with torch.no_grad():
        toks = clip.tokenize(prompts).to(device)
        text_feats = model.encode_text(toks)
        text_feats /= text_feats.norm(dim=-1, keepdim=True)

    jobs = []
    for si, sc in enumerate(data["scenes"]):
        for ei, ent in enumerate(sc["entities"]):
            cp = ent.get("rep_crop_path")
            if cp and os.path.exists(cp):
                jobs.append((si, ei, cp))
    kept = 0
    min_conf = float(clip_min_conf)

    for i in range(0, len(jobs), cfg["clip_batch"]):
        batch = jobs[i:i+cfg["clip_batch"]]
        imgs = []
        for _, _, p in batch:
            try:
                imgs.append(preprocess(Image.open(p).convert("RGB")))
            except:
                imgs.append(None)

        imgs = [x for x in imgs if x is not None]
        if not imgs:
            continue

        with torch.no_grad():
            ims = torch.stack(imgs).to(device)
            img_feats = model.encode_image(ims)
            img_feats /= img_feats.norm(dim=-1, keepdim=True)
            sims = img_feats @ text_feats.T
            probs = sims.softmax(dim=-1)

        for k, row in enumerate(probs):
            si, ei, _ = batch[k]
            top = int(torch.argmax(row).item())
            conf = float(row[top].item())
            new_label = cfg["clip_classes"][top]

            ent = data["scenes"][si]["entities"][ei]
            old_label = ent.get("label", "object")

            ent.setdefault("meta", {})
            ent["meta"]["clip"] = {
                "top1": new_label,
                "conf": conf,
                "old_label": old_label,
            }
            if conf >= min_conf:
                ent.setdefault("clip_labels", [])
                ent["clip_labels"].append({
                    "name": new_label,
                    "score": conf
                })
                kept += 1


def build_entities(cfg: Dict[str, Any]):
    data = _collect_entities(cfg)
    if use_clip:
        _clip_reclass_entities(data, cfg)

    out_path = Path(cfg["out_entities"]); ensure_dir(out_path.parent)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def get_args():
    parser = argparse.ArgumentParser(
        description="Crop object detections and run CLIP for OD"
    )
    parser.add_argument(
        "tracks_path",
        type=Path,
        help="output of your object tracker",
    )
    parser.add_argument(
        "out_entities",
        type=Path,
        help="Path for the output merged json",
    )
    parser.add_argument(
        "crops_dir",
        type=Path,
        help="Path for the crops",
    )
    return parser.parse_args()

def main():
    args = get_args()
    tracks_path = str(args.tracks_path)
    out_entities = str(args.out_entities)
    crops_dir = str(args.crops_dir)

    with open("/scratch/Fabrice/vision/idea_3/config.yml", "r", encoding="utf-8") as f:
        yml = yaml.safe_load(f)

    cfg = {
        "tracks_json": tracks_path,
        "crops_dir": crops_dir,
        "out_entities": out_entities,
    }
    cfg.update(yml["object_tracking_config"])

    build_entities(cfg)

if __name__ == "__main__":
    main()
