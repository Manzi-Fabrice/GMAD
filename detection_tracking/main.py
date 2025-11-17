# ********************************* main.py ****************************
# Object Detection Pipeline
# ***********************************************************************

from pathlib import Path
import json
import argparse
from typing import List, Dict, Any, Optional
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

MIN_SCORE = 0.0 
OD_MODEL = "microsoft/Florence-2-large"

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32

def _load_manifest(p: str) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def _iter_all_frames(manifest: Dict[str, Any]):
    for sc in manifest["scenes"]:
        for fr in sc["frames"]:
            yield {
                "scene_id": sc["scene_id"],
                "image_path": fr["path"],
                "t": fr.get("t", None)
            }

def _post_to_boxes(parsed: Dict[str, Any], W: int, H: int):
    out = []
    od = parsed.get("<OD>") or parsed.get("od") or {}
    bboxes = od.get("bboxes", [])
    labels = od.get("labels", [])
    scores = od.get("scores", None)
    for i, bb in enumerate(bboxes):
        x1, y1, x2, y2 = [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])]
        x1 = max(0.0, min(x1, W)); y1 = max(0.0, min(y1, H))
        x2 = max(0.0, min(x2, W)); y2 = max(0.0, min(y2, H))
        sc = float(scores[i]) if scores is not None and i < len(scores) else 1.0
        lb = str(labels[i]) if i < len(labels) else "object"
        out.append({"bbox": [x1, y1, x2, y2], "score": sc, "label": lb, "cls_id": -1})
    return out

def run_detect_florence_all(manifest_path: str, out_path: str, allowed_labels: Optional[List[str]] = None, min_score: float = 0.0, dtype: str = "auto"):

    model_id = OD_MODEL
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=torch_dtype
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    manifest = _load_manifest(manifest_path)
    items = list(_iter_all_frames(manifest))
    scenes_out: Dict[int, Dict[str, Any]] = {}

    for meta in items:
        img_path = meta["image_path"]
        im = Image.open(img_path).convert("RGB")
        W, H = im.width, im.height
        prompt = "<OD>"
        inputs = processor(text=prompt, images=im, return_tensors="pt")
        inputs = {
            k: v.to(device, torch_dtype) if v.dtype.is_floating_point else v.to(device)
            for k, v in inputs.items()
        }

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024
            )
        gen_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(
            gen_text, task=prompt, image_size=(W, H)
        )

        boxes = _post_to_boxes(parsed, W, H)
        if allowed_labels is not None:
            allowed = set(allowed_labels)
            boxes = [b for b in boxes if b["label"] in allowed]
        if min_score > 0.0:
            boxes = [b for b in boxes if b.get("score", 1.0) >= min_score]

        entry = {
            "image_path": img_path,
            "t": meta["t"],
            "boxes": boxes
        }
        sid = meta["scene_id"]
        scenes_out.setdefault(sid, {"scene_id": sid, "detections": []})["detections"].append(entry)

    for sc in scenes_out.values():
        sc["detections"].sort(key=lambda d: (d["t"] if d["t"] is not None else 0.0))

    out = {
        "meta": {
            "model": model_id,
            "task": "<OD>",
            "device": device,
            "torch_dtype": str(torch_dtype),
            "source_manifest": str(Path(manifest_path).resolve()),
            "allowed_labels": allowed_labels,
            "min_score": min_score
        },
        "scenes": [scenes_out[k] for k in sorted(scenes_out.keys())]
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

def get_args():
    parser = argparse.ArgumentParser(
        description="Runs object detection model on sampled frames."
    )
    parser.add_argument(
        "scenes_path",
        type=Path,
        help="Path to the json of scenes",
    )
    parser.add_argument(
        "out_path",
        type=Path,
        help="Path for the output detections json",
    )
    return parser.parse_args()

def main():
    args = get_args()
    scenes_path = str(args.scenes_path)
    out_path = str(args.out_path)
    run_detect_florence_all(manifest_path=scenes_path, out_path=out_path, allowed_labels=None, min_score=MIN_SCORE, dtype="auto")

if __name__ == "__main__":
    main()
