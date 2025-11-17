# ********************************* object_tracking.py ****************************
# multi-object tracker built on IoU matching.
# ***********************************************************************
from __future__ import annotations
from pathlib import Path
import json
from typing import List, Dict, Any, Tuple
import numpy as np
import argparse

HIGH_THRESH=0.7
LOW_THRESH=0.20
MATCH_THRESH=0.50
TRACK_BUFFER=10
MIN_FRAMES=1

def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(0.0, (a[2] - a[0])) * max(0.0, (a[3] - a[1]))
    b_area = max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1]))
    union = a_area + b_area - inter
    return float(inter / union) if union > 0 else 0.0


def greedy_match(tracks: List[dict], dets: List[dict], iou_thresh: float, same_class: bool = True):
    if not tracks or not dets:
        return [], list(range(len(tracks))), list(range(len(dets)))

    iou_mat = np.full((len(tracks), len(dets)), -1.0, dtype=np.float32)
    for i, tr in enumerate(tracks):
        tb = np.array(tr["bbox"], dtype=np.float32)
        t_label = max(tr["label_counts"].items(), key=lambda kv: kv[1])[0] if tr["label_counts"] else None
        t_cls = max(tr["cls_counts"].items(), key=lambda kv: kv[1])[0] if tr["cls_counts"] else None
        for j, de in enumerate(dets):
            if same_class:
                if (t_label is not None and de["label"] != t_label) and (t_cls is not None and de["cls_id"] != t_cls):
                    continue
            iou_mat[i, j] = iou_xyxy(tb, np.array(de["bbox"], dtype=np.float32))

    matches = []
    used_t, used_d = set(), set()
    while True:
        i, j = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
        if iou_mat[i, j] < iou_thresh:
            break
        if i in used_t or j in used_d:
            iou_mat[i, j] = -1.0
            continue
        matches.append((i, j))
        used_t.add(i); used_d.add(j)
        iou_mat[i, :] = -1.0
        iou_mat[:, j] = -1.0

    unmatched_tracks = [i for i in range(len(tracks)) if i not in used_t]
    unmatched_dets = [j for j in range(len(dets)) if j not in used_d]
    return matches, unmatched_tracks, unmatched_dets

class ByteTrackLite:

    def __init__(self,high_thresh: float = 0.5,low_thresh: float = 0.1,match_thresh: float = 0.8,track_buffer: int = 30,min_frames: int = 2,):
        assert 0 <= low_thresh <= high_thresh <= 1
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.min_frames = min_frames

        self.tracks: List[dict] = []      
        self.dead_tracks: List[dict] = [] 
        self._next_id = 1

    def _new_id(self) -> str:
        tid = f"t{self._next_id}"
        self._next_id += 1
        return tid

    def _start_track(self, det: dict, tstamp: float, frame_path: str):
        self.tracks.append({
            "id": self._new_id(),
            "label_counts": {det["label"]: 1},
            "cls_counts": {int(det["cls_id"]): 1},
            "score_sum": det["score"],
            "score_n": 1,
            "frames": [{
                "t": tstamp,
                "bbox": det["bbox"],
                "det_score": det["score"],
                "image_path": frame_path,  
            }],
            "bbox": det["bbox"],   
            "lost": 0             
        })

    def _update_track(self, tr: dict, det: dict, tstamp: float, frame_path: str):
        tr["frames"].append({
            "t": tstamp,
            "bbox": det["bbox"],
            "det_score": det["score"],
            "image_path": frame_path,      
        })
        tr["bbox"] = det["bbox"]
        tr["lost"] = 0
        tr["score_sum"] += det["score"]
        tr["score_n"] += 1
        tr["label_counts"][det["label"]] = tr["label_counts"].get(det["label"], 0) + 1
        tr["cls_counts"][int(det["cls_id"])] = tr["cls_counts"].get(int(det["cls_id"]), 0) + 1

    def _mark_lost(self, idx: int):
        self.tracks[idx]["lost"] += 1

    def _purge_lost(self):
        keep = []
        for tr in self.tracks:
            if tr["lost"] > self.track_buffer:
                self.dead_tracks.append(tr)
            else:
                keep.append(tr)
        self.tracks = keep

    def step(self, tstamp: float, detections: List[dict], frame_path: str):

        D_high = [d for d in detections if d["score"] >= self.high_thresh]
        D_low  = [d for d in detections if self.low_thresh <= d["score"] < self.high_thresh]
        matches, un_tr_idx, _ = greedy_match(self.tracks, D_high, self.match_thresh)
        matched_det_indices = set()
        for tr_i, dh_i in matches:
            self._update_track(self.tracks[tr_i], D_high[dh_i], tstamp, frame_path)
            matched_det_indices.add(dh_i)

        unmatched_tracks_stage1 = [self.tracks[i] for i in un_tr_idx]
        matches2, un_tr_idx2, _ = greedy_match(unmatched_tracks_stage1, D_low, self.match_thresh)
        stage1_unmatched_indices = un_tr_idx 

        for local_tr_i, dl_i in matches2:
            global_tr_i = stage1_unmatched_indices[local_tr_i]
            self._update_track(self.tracks[global_tr_i], D_low[dl_i], tstamp, frame_path)

        matched_global_from_stage2 = set(stage1_unmatched_indices[local] for (local, _) in matches2)
        for gi in stage1_unmatched_indices:
            if gi not in matched_global_from_stage2:
                self._mark_lost(gi)

        for j in range(len(D_high)):
            if j not in matched_det_indices:
                self._start_track(D_high[j], tstamp, frame_path)

        self._purge_lost()

    def finalize(self):
        all_tracks = self.dead_tracks + self.tracks
        out = []
        for tr in all_tracks:
            if len(tr["frames"]) < self.min_frames:
                continue
            label = max(tr["label_counts"].items(), key=lambda kv: kv[1])[0]
            cls_id = max(tr["cls_counts"].items(), key=lambda kv: kv[1])[0]
            score = tr["score_sum"] / max(1, tr["score_n"])
            out.append({
                "id": tr["id"],
                "label": label,
                "cls_id": int(cls_id),
                "score": float(score),
                "frames": tr["frames"]
            })
        out.sort(key=lambda t: t["frames"][0]["t"])
        return out


def _load_detections(dets_path: str) -> Dict[str, Any]:
    with open(dets_path, "r", encoding="utf-8") as f:
        return json.load(f)

def _group_frames_by_scene(det_json: Dict[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
    out = {}
    for sc in det_json.get("scenes", []):
        sid = int(sc["scene_id"])
        frames = []
        for d in sc.get("detections", []):
            frames.append({
                "t": d.get("t", None),
                "image_path": d.get("image_path", None),
                "boxes": d.get("boxes", [])
            })
        frames = [f for f in frames if f["t"] is not None and f["image_path"]]
        frames.sort(key=lambda x: x["t"])
        out[sid] = frames
    return out


def associate_by_scene( dets_path: str, out_path: str,high_thresh: float = 0.5,low_thresh: float = 0.2,match_thresh: float = 0.45,track_buffer: int = 30,min_frames: int = 2,):
    det_json = _load_detections(dets_path)
    frames_by_scene = _group_frames_by_scene(det_json)

    scenes_out = []
    for sid, frames in sorted(frames_by_scene.items(), key=lambda kv: kv[0]):
        tracker = ByteTrackLite(
            high_thresh=high_thresh,
            low_thresh=low_thresh,
            match_thresh=match_thresh,
            track_buffer=track_buffer,
            min_frames=min_frames,
        )

        for fr in frames:
            t = float(fr["t"])
            frame_path = fr["image_path"]
            dets = []
            for b in fr["boxes"]:
                dets.append({
                    "bbox": [float(x) for x in b["bbox"]],
                    "score": float(b["score"]),
                    "label": str(b.get("label", "object")),
                    "cls_id": int(b.get("cls_id", -1)),
                })
            tracker.step(t, dets, frame_path)  

        tracks = tracker.finalize()
        scenes_out.append({"scene_id": sid, "tracks": tracks})

    out = {
        "meta": {
            "source_detections": str(Path(dets_path).resolve()),
            "high_thresh": high_thresh,
            "low_thresh": low_thresh,
            "match_thresh": match_thresh,
            "track_buffer": track_buffer,
            "min_frames": min_frames
        },
        "scenes": scenes_out
    }

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

def get_args():
    parser = argparse.ArgumentParser(
        description="Object tracking across similar frames using IO"
    )
    parser.add_argument("detection_path", type=Path, help="Path to the json of objection detection",)
    parser.add_argument("out_path",type=Path, help="Path for the output detections json",)
    return parser.parse_args()

def main():
    args = get_args()
    det_path = str(args.detection_path)
    output_path = str(args.out_path)

    associate_by_scene(dets_path=det_path,out_path= output_path,high_thresh=HIGH_THRESH,low_thresh=LOW_THRESH, match_thresh=MATCH_THRESH, track_buffer=TRACK_BUFFER,min_frames=MIN_FRAMES)

if __name__ == "__main__":
    main()
