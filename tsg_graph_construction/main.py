# ********************************* main.py ****************************
# Build a Temporal Scene Graph (TSG) from entities.json and tracks.json
# ***********************************************************************

from pathlib import Path
import json
import math
from collections import defaultdict, Counter
import argparse

DEFAULT_WIDTH, DEFAULT_HEIGHT = 1920, 1080
BBOX_IS_XYWH = False
TAU_NEAR   = 0.18  
TAU_IOU    = 0.08   
MARGIN     = 0.03  
MIN_SUPPORT_FRAMES = 2   
MERGE_GAP_FRAMES   = 1   
MAX_EVIDENCE_PER_EDGE = 5
MAX_SAMPLE_FRAMES = 3

def qid(scene_id, tid):
    return f"sc{scene_id}_{tid}" if scene_id is not None else str(tid)

def to_xyxy(b):
    if b is None:
        return None
    x1, y1, a, b2 = b[:4]
    if BBOX_IS_XYWH:
        x2, y2 = x1 + a, y1 + b2
    else:
        x2, y2 = a, b2
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return (float(x1), float(y1), float(x2), float(y2))

def centers(b): x1,y1,x2,y2=b; return (0.5*(x1+x2), 0.5*(y1+y2))
def size_wh(b): x1,y1,x2,y2=b; return (x2-x1, y2-y1)
def norm_dist(c1,c2,W,H): return math.hypot(c1[0]-c2[0], c1[1]-c2[1]) / math.hypot(W,H)

def iou(a,b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    ix1,iy1=max(ax1,bx1),max(ay1,by1)
    ix2,iy2=min(ax2,bx2),min(ay2,by2)
    iw,ih=max(0.0,ix2-ix1),max(0.0,iy2-iy1)
    inter=iw*ih
    if inter<=0: return 0.0
    areaA=(ax2-ax1)*(ay2-ay1); areaB=(bx2-bx1)*(by2-by1)
    return float(inter/(areaA+areaB-inter+1e-6))

def left_of(a,b,W):
    cxa,_=centers(a); cxb,_=centers(b)
    wa,_=size_wh(a);  wb,_=size_wh(b)
    ok=(cxa+MARGIN*wa)<(cxb-MARGIN*wb)
    score=max(0.0, min(1.0, (cxb-cxa)/max(1.0, W)))
    return ok,score

def above(a,b,H):
    _,cya=centers(a); _,cyb=centers(b)
    _,ha=size_wh(a);  _,hb=size_wh(b)
    ok=(cya+MARGIN*ha)<(cyb-MARGIN*hb)
    score=max(0.0, min(1.0, (cyb-cya)/max(1.0, H)))
    return ok,score

def near(a,b,W,H):
    d=norm_dist(centers(a),centers(b),W,H)
    ok=d<=TAU_NEAR
    return ok, (1.0 - d/TAU_NEAR) if ok else 0.0


def load_entities_any_shape(p):

    if not Path(p).exists():
        print(f"entities file not found: {p}")
        return None, {}

    data = json.load(open(p, "r", encoding="utf-8"))

    if "entities" in data and isinstance(data["entities"], list):
        ents = data["entities"]
        sid = data.get("scene_id")
        node_by_id = {}
        for e in ents:
            if "id" not in e: 
                continue
            e = dict(e)
            e["id"] = qid(sid, e["id"])
            node_by_id[e["id"]] = e
        return ([sid] if sid is not None else None, node_by_id)

    scenes = data.get("scenes", [])
    if isinstance(scenes, list) and scenes:
        node_by_id = {}
        scene_ids = []
        for sc in scenes:
            sid = sc.get("scene_id")
            scene_ids.append(sid)
            ents = sc.get("entities", [])
            for e in ents:
                if "id" not in e:
                    continue
                e = dict(e)
                e["id"] = qid(sid, e["id"])
                node_by_id[e["id"]] = e
        return (scene_ids, node_by_id)
    return None, {}

def iter_tracks_any_shape(data): 
    if "tracks" in data:
        scenes = [data]
    else:
        scenes = data.get("scenes", [])
    for sc in scenes:
        sid = sc.get("scene_id")
        for tr in sc.get("tracks", []):
            tid = qid(sid, tr.get("id"))
            label = tr.get("label", "obj")
            for i, fr in enumerate(tr.get("frames", [])):
                t = fr.get("t", i)
                yield tid, label, {
                    "t": t,
                    "bbox": fr.get("bbox"),
                    "image_path": fr.get("image_path")
                }

def load_tracks(p):
    data = json.load(open(p, "r", encoding="utf-8"))
    time_index = defaultdict(list)
    count_frames = 0
    seen = set()
    for tid, lab, fr in iter_tracks_any_shape(data):
        seen.add(tid)
        bb = to_xyxy(fr.get("bbox"))
        if bb is None:
            continue
        rec = {"id": tid, "label": lab, "bbox": bb,
               "image_path": fr.get("image_path"), "t": fr.get("t")}
        time_index[rec["t"]].append(rec)
        count_frames += 1
    count_tracks = len(seen)
    print(f"[DBG] parsed {count_tracks} tracks with {count_frames} frames (from {p})")
    return time_index, {"tracks": count_tracks, "frames": count_frames}


def spatial_relations(objs, W, H):

    out = []
    n = len(objs)
    for i in range(n):
        A = objs[i]
        for j in range(n):
            if i == j:
                continue
            B = objs[j]
            a, b = A["bbox"], B["bbox"]
            ok, s = left_of(a, b, W)
            if ok: out.append(("left_of", A["id"], B["id"], s))
            ok, s = above(a, b, H)
            if ok: out.append(("above", A["id"], B["id"], s))
            ok, s = near(a, b, W, H)
            if ok: out.append(("near", A["id"], B["id"], s))
            s = iou(a, b)
            if s > TAU_IOU: out.append(("overlap", A["id"], B["id"], s))
    return out

def merge_edges_across_time(per_frame_edges):

    buckets = defaultdict(list)
    for t, sub, s, d, sc, ev in per_frame_edges:
        buckets[(sub, s, d)].append((t, sc, ev))

    result = {}
    for key, items in buckets.items():
        items.sort(key=lambda x: x[0])
        intervals, ev_keep, scores = [], [], []
        cur_start = cur_end = None
        last_t = None
        for t, sc, ev in items:
            if cur_start is None:
                cur_start = cur_end = t
            else:
                if last_t is not None and (t - last_t) <= MERGE_GAP_FRAMES:
                    cur_end = t
                else:
                    intervals.append({"t0": cur_start, "t1": cur_end})
                    cur_start = cur_end = t
            scores.append(sc)
            ev_keep.append(ev)
            last_t = t
        if cur_start is not None:
            intervals.append({"t0": cur_start, "t1": cur_end})

        support = sum(intv["t1"] - intv["t0"] + 1 for intv in intervals)
        if support >= MIN_SUPPORT_FRAMES:
            avg_score = sum(scores) / max(1, len(scores))
            result[key] = {
                "intervals": intervals,
                "score": float(avg_score),
                "evidence": ev_keep[:min(MAX_EVIDENCE_PER_EDGE, len(ev_keep))]
            }
    return result

def get_args():
    parser = argparse.ArgumentParser(
        description="Build a Temporal Scene Graph (TSG) from entities.json and tracks.json",
    )

    parser.add_argument(
        "--entities",
        type=Path,
        required=True,
        help="Path to entities_with_attrs.json",
    )

    parser.add_argument(
        "--tracks",
        type=Path,
        required=True,
        help="Path to tracks.json",
    )

    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output path for tsg.json",
    )

    return parser.parse_args()

def main():
    args = get_args()
    scene_ids, node_by_id = load_entities_any_shape(args.entities)
    time_index, tstats = load_tracks(args.tracks)
    OUT_TSG_JSON = args.out

    frames = sorted(time_index.keys())
    per_frame_edges = []
    multi_obj_frames = 0
    for idx, t in enumerate(frames):
        objs = time_index[t]
        n = len(objs)
        print(f"  frame {t}: {n} object(s)")
        if n < 2:
            continue
        multi_obj_frames += 1
        rels = spatial_relations(objs, DEFAULT_WIDTH, DEFAULT_HEIGHT)
        print(f"    → found {len(rels)} raw relations")
        if idx < MAX_SAMPLE_FRAMES:
            for r in rels[:5]:
                print(f"      sample: {r}")

        id2bbox = {o["id"]: o["bbox"] for o in objs}
        frame_path = objs[0].get("image_path")
        for (sub, s, d, sc) in rels:
            ev = {
                "t": t,
                "frame": frame_path,
                "bbox_src": id2bbox.get(s),
                "bbox_dst": id2bbox.get(d)
            }
            per_frame_edges.append((t, sub, s, d, sc, ev))

    merged = merge_edges_across_time(per_frame_edges)
    if len(node_by_id) == 0:
        label_by_id = {}
        seen_counts = Counter()
        for _, objs in time_index.items():
            for o in objs:
                tid = o["id"]
                if tid not in label_by_id:
                    label_by_id[tid] = o.get("label", "object")
                seen_counts[tid] += 1
        nodes = []
        for tid in sorted(label_by_id.keys()):
            nodes.append({
                "id": tid,
                "label": label_by_id[tid],
                "attrs": [],
                "meta": {"frames_seen": int(seen_counts[tid])}
            })
    else:
        nodes = []
        for nid, ent in node_by_id.items():
            nodes.append({
                "id": nid,
                "label": ent.get("label", "object"),
                "attrs": ent.get("attrs", []),
                "meta": {
                    "score": ent.get("score", None),
                    "rep_crop_path": ent.get("rep_crop_path", None),
                    "clip": ent.get("clip", None),
                    "clip_labels": ent.get("clip_labels", []),
                    "cls_id": ent.get("cls_id", None)
                }
            })
    label_by_id = {n["id"]: n.get("label", "object") for n in nodes}

    edges = [
        {
            "type": "spatial",
            "subtype": sub,
            "src": s,
            "dst": d,
            "score": data["score"],
            "intervals": data["intervals"],
            "evidence": data["evidence"]
        }
        for (sub, s, d), data in merged.items()
    ]
    rel_hist = Counter(e["subtype"] for e in edges)
    out_deg = Counter(e["src"] for e in edges)
    in_deg  = Counter(e["dst"] for e in edges)
    out = {
        "scene_ids": scene_ids,  
        "nodes": nodes,
        "edges": edges,
        "meta": {
            "defaults": {"W": DEFAULT_WIDTH, "H": DEFAULT_HEIGHT},
            "bbox_format_xywh": bool(BBOX_IS_XYWH),
            "input_stats": {
                "entities_count": len(node_by_id),
                "tracks": tstats["tracks"],
                "frames": tstats["frames"]
            },
            "relation_hist": dict(rel_hist),
            "merge_params": {
                "MIN_SUPPORT_FRAMES": MIN_SUPPORT_FRAMES,
                "MERGE_GAP_FRAMES": MERGE_GAP_FRAMES,
                "MAX_EVIDENCE_PER_EDGE": MAX_EVIDENCE_PER_EDGE
            },
            "spatial_params": {
                "TAU_NEAR": TAU_NEAR,
                "TAU_IOU": TAU_IOU,
                "MARGIN": MARGIN
            }
        }
    }

    Path(OUT_TSG_JSON).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT_TSG_JSON, "w"), indent=2)

if __name__ == "__main__":
    main()
