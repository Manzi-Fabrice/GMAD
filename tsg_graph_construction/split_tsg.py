# ********************************* split_tsg.py ****************************
# Split a global TSG into per-scene TSG files with per-scene metadata
# ***********************************************************************
import json
from pathlib import Path
from collections import Counter
import argparse

def sid_of(qid: str):
    if not isinstance(qid, str) or not qid.startswith("sc"):
        return None
    try:
        return int(qid.split("_")[0][2:])
    except Exception:
        return None

def parse_args():
    parser = argparse.ArgumentParser(
        description="Split a global TSG JSON into per-scene TSG files."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the TSG JSON file (output of main TSG builder).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Output directory for per-scene TSG JSON files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = args.input
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        tsg = json.load(f)

    nodes = tsg.get("nodes", [])
    edges = tsg.get("edges", [])
    meta = tsg.get("meta", {})

    node_sid = {n["id"]: sid_of(n["id"]) for n in nodes}
    nodes_by_scene = {}
    for n in nodes:
        sid = node_sid.get(n["id"])
        nodes_by_scene.setdefault(sid, []).append(n)

    edges_by_scene = {}
    for e in edges:
        sid_src = node_sid.get(e["src"])
        sid_dst = node_sid.get(e["dst"])
        if sid_src is None or sid_dst is None:
            continue
        if sid_src != sid_dst:
            continue
        edges_by_scene.setdefault(sid_src, []).append(e)

    for sid in sorted(nodes_by_scene.keys()):
        scene_nodes = nodes_by_scene.get(sid, [])
        scene_edges = edges_by_scene.get(sid, [])

        rel_hist = Counter(e["subtype"] for e in scene_edges)
        out_deg = Counter(e["src"] for e in scene_edges)
        in_deg = Counter(e["dst"] for e in scene_edges)

        scene_meta = {
            "version": meta.get("version"),
            "defaults": meta.get("defaults", {}),
            "bbox_format_xywh": meta.get("bbox_format_xywh", False),
            "input_stats": {
                "nodes": len(scene_nodes),
                "edges": len(scene_edges),
            },
            "relation_hist": dict(rel_hist),
            "merge_params": meta.get("merge_params", {}),
            "spatial_params": meta.get("spatial_params", {}),
            "degree": {
                "top_out": out_deg.most_common(5),
                "top_in": in_deg.most_common(5),
            },
        }
        out = {
            "scene_id": sid,
            "nodes": scene_nodes,
            "edges": scene_edges,
            "meta": scene_meta,
        }
        out_path = outdir / f"tsg_scene{sid:04d}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
