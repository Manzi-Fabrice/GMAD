# ********************************* main.py ****************************
# CLIP-based visual attribute enrichment for detected entities.
# ***********************************************************************
from pathlib import Path
import json, os, hashlib
from typing import List, Dict, Any, Optional, Tuple
import torch
import clip  
from PIL import Image
import numpy as np
import cv2
import random
import yaml
import argparse

top_k = 2
min_score = 0.15
add_geo_size = True
geo_small_thresh = 0.02
geo_large_thresh = 0.20
device = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def _best_frame(frames: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not frames: return {}
    return max(frames, key=lambda f: f.get("det_score", 0.0))

def _safe_crop(img_bgr: np.ndarray, bbox: List[float], pad: int = 2) -> Optional[np.ndarray]:
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)
    if x2 <= x1 or y2 <= y1: return None
    return img_bgr[y1:y2, x1:x2]

def _resolve_image_for_entity(ent: Dict[str, Any], prefer_crop=True) -> Optional[Image.Image]:
    if prefer_crop:
        cp = ent.get("rep_crop_path")
        if cp and Path(cp).exists():
            try:
                return Image.open(cp).convert("RGB")
            except Exception:
                pass
    best = _best_frame(ent.get("frames", []))
    frame_path = best.get("image_path", ent.get("rep_frame_path"))
    bbox = best.get("bbox")
    if frame_path and bbox and Path(frame_path).exists():
        img_bgr = cv2.imread(frame_path)
        if img_bgr is None: return None
        crop_bgr = _safe_crop(img_bgr, bbox, pad=2)
        if crop_bgr is None: return None
        h, w = crop_bgr.shape[:2]
        if min(h, w) < 32:
            scale = max(1, int(32 / max(1, min(h, w))))
            crop_bgr = cv2.resize(crop_bgr, (w*scale, h*scale), interpolation=cv2.INTER_LINEAR)
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(crop_rgb)
    return None

def bbox_rel_area(b, img_w, img_h):
    x1,y1,x2,y2 = b
    a = max(0.0, x2-x1) * max(0.0, y2-y1)
    return a / max(1.0, img_w*img_h)

class ClipAttrTagger:
    def __init__(self, model_name: str, device:str,
                 attributes: List[str], templates: List[str]):
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

        self.attributes = list(dict.fromkeys([a.strip() for a in attributes if a.strip()]))
        self.templates = templates

        texts: List[str] = []
        self.attr2idx: Dict[str, List[int]] = {}
        for attr in self.attributes:
            idxs = []
            for tmpl in self.templates:
                texts.append(tmpl.format(attr=attr))
                idxs.append(len(texts) - 1)
            self.attr2idx[attr] = idxs

        with torch.no_grad():
            toks = clip.tokenize(texts).to(self.device)
            t_feats = self.model.encode_text(toks)
            t_feats = t_feats / t_feats.norm(dim=-1, keepdim=True)
        self.text_feats = t_feats
        self.texts = texts

    @torch.no_grad()
    def rank(self, pil_images: List[Image.Image]) -> List[List[Tuple[str, float]]]:
        if not pil_images: return []
        imgs = torch.stack([self.preprocess(im) for im in pil_images]).to(self.device)
        with torch.no_grad():
            i_feats = self.model.encode_image(imgs)
            i_feats = i_feats / i_feats.norm(dim=-1, keepdim=True)
            sims_all = i_feats @ self.text_feats.T
        sims = sims_all.detach().cpu().numpy()

        ranked: List[List[Tuple[str, float]]] = []
        for row in sims:
            per_attr = []
            for attr, idxs in self.attr2idx.items():
                per_attr.append((attr, float(np.max(row[idxs]))))
            per_attr.sort(key=lambda x: -x[1])
            ranked.append(per_attr)
        return ranked


def _collect_attribute_vocab(cfg: Dict[str, Any]) -> List[str]:
    vocab = []
    presets_cfg = cfg.get("presets", {})
    for name in cfg.get("attribute_presets", []):
        vocab += presets_cfg.get(name, [])
    vocab += cfg.get("extra_attributes", [])
    core = cfg.get("core_attributes", [])
    vocab += core
    seen = set(); out = []
    for a in vocab:
        a = a.strip()
        if not a or a in seen: continue
        seen.add(a); out.append(a)
    return out

def enrich_entities(cfg: Dict[str, Any]):
    set_seed(42)

    ent_path  = Path(cfg["entities_in"])
    out_path  = Path(cfg["entities_out"])
    ensure_dir(out_path.parent)
    attributes = _collect_attribute_vocab(cfg)
    templates  = cfg["templates"]

    B = int(cfg["batch_size"])
    k = int(top_k)
    min_s = float(min_score)
    data = json.load(open(ent_path, "r", encoding="utf-8"))

    tagger = ClipAttrTagger(
        model_name=cfg["clip_model"],
        device=device,
        attributes=attributes,
        templates=templates
    )

    jobs: List[Tuple[int, int, Image.Image]] = []
    missing = 0
    for si, sc in enumerate(data.get("scenes", [])):
        for ei, ent in enumerate(sc.get("entities", [])):
            im = _resolve_image_for_entity(ent, prefer_crop=cfg["prefer_saved_crop"])
            if im is not None:
                jobs.append((si, ei, im))
            else:
                data["scenes"][si]["entities"][ei].setdefault("attrs", [])
                missing += 1

    for i in range(0, len(jobs), B):
        batch = jobs[i:i+B]
        ims = [im for _,_,im in batch]
        ranked_lists = tagger.rank(ims)
        for (si, ei, _), ranked in zip(batch, ranked_lists):
            chosen = ranked[:k]
            if min_s is not None:
                chosen = [(n, s) for (n, s) in chosen if s >= float(min_s)]
            data["scenes"][si]["entities"][ei]["attrs"] = [
                {"name": n, "score": round(float(s), 4), "src": "clip"} for (n, s) in chosen
            ]

    if add_geo_size:
        sm = float(geo_small_thresh)
        lg = float(geo_large_thresh)
        for si, sc in enumerate(data.get("scenes", [])):
            for ei, ent in enumerate(sc.get("entities", [])):
                frames = ent.get("frames", [])
                if not frames: continue
                best = max(frames, key=lambda f: f.get("det_score", 0.0))
                bbox = best.get("bbox"); fpath = best.get("image_path") or ent.get("rep_frame_path")
                if bbox and fpath and Path(fpath).exists():
                    im = cv2.imread(fpath)
                    if im is None: continue
                    rel = bbox_rel_area(bbox, im.shape[1], im.shape[0])
                    tag = "large" if rel > lg else "small" if rel < sm else None
                    if tag:
                        ent.setdefault("attrs", [])
                        ent["attrs"].append({"name": tag, "score": round(float(rel),4), "src": "geo"})

    meta = data.setdefault("meta", {})
    meta.update({
        "clip_model": cfg["clip_model"],
        "device": device,
        "top_k": k,
        "min_score": min_s,
        "templates_sha1": hashlib.sha1(("|".join(templates)).encode()).hexdigest()[:10],
        "attributes_sha1": hashlib.sha1(("|".join(attributes)).encode()).hexdigest()[:10],
        "attribute_count": len(attributes),
        "presets": cfg.get("attribute_presets", []),
        "extra_attributes": cfg.get("extra_attributes", []),
    })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def get_args():
    parser = argparse.ArgumentParser(
        description="Enrich entities.json with CLIP-based visual attributes"
    )
    parser.add_argument(
        "entities_in",
        type=Path,
        help="Path to entities.json (from build_entities.py)",
    )
    parser.add_argument(
        "entities_out",
        type=Path,
        help="Path to output enriched entities_with_attrs.json",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("/scratch/Fabrice/vision/idea_3/config.yml"),
        help="Path to YAML config for attribute presets / CLIP settings",
    )
    return parser.parse_args()

def main():
    args = get_args()
    entities_in = str(args.entities_in)
    entities_out = str(args.entities_out)

    with open(args.config, "r", encoding="utf-8") as f:
        root = yaml.safe_load(f)

    cfg = dict(root["attribute_enrichment"])
    cfg["entities_in"] = entities_in
    cfg["entities_out"] = entities_out

    enrich_entities(cfg)


if __name__ == "__main__":
    main()
