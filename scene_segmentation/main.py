
# ********************************* main.py ****************************
# Detect scenes, refine intervals, sample frames, write scenes.json
# ***********************************************************************


import json
import argparse
from pathlib import Path
import sys
import cv2 
from scenedetect import open_video
from scenedetect.scene_manager import SceneManager
from scenedetect.detectors import ContentDetector
from scenedetect.frame_timecode import FrameTimecode

THRESHOLD = 38.0          
MIN_SCENE_LEN_S = 0.8     
START_SEC = 1.0          
MAX_T = 58.0              
MERGE_GAP_S = 0.2        
MIN_KEEP_S = 0.6          
TOP_K = 999                 
FRAME_NAME = "keyframe.jpg"       
MANY_FRAMES = True              
SAMPLE_FPS = 2.0   
EXPORT_ONE_FRAME_PER_SCENE = True

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def detect_scenes_pyscenedetect(video_path: str, threshold: float,min_scene_len_s: float,start_sec: float):
    video = open_video(video_path)
    scene_manager = SceneManager()
    fps_video = float(video.frame_rate)
    min_scene_len_frames = max(1, int(round(min_scene_len_s * fps_video)))

    scene_manager.add_detector(ContentDetector(
        threshold=threshold,
        min_scene_len=min_scene_len_frames
    ))
    if start_sec > 0:
        video.seek(FrameTimecode(start_sec, fps=video.frame_rate))
    scene_manager.detect_scenes(video)
    scenes = scene_manager.get_scene_list()
    scenes_sec = [(s.get_seconds(), e.get_seconds()) for (s, e) in scenes]
    return scenes_sec


def clip_range(scenes_sec, t_max: float):
    out = []
    for s, e in scenes_sec:
        if s >= t_max:
            break
        out.append((max(0.0, s), min(e, t_max)))
    return out


def drop_too_short(scenes_sec, min_len: float):
    return [(s, e) for (s, e) in scenes_sec if (e - s) >= min_len]


def merge_small_gaps(scenes_sec, max_gap: float):
    if not scenes_sec:
        return []
    scenes_sec = sorted(scenes_sec, key=lambda se: se[0])
    merged = [scenes_sec[0]]
    for s, e in scenes_sec[1:]:
        prev_s, prev_e = merged[-1]
        if s - prev_e < max_gap:
            merged[-1] = (prev_s, max(prev_e, e))
        else:
            merged.append((s, e))
    return merged


def keep_top_k_longest(scenes_sec, k: int):
    if k <= 0 or k >= len(scenes_sec):
        return scenes_sec
    ranked = sorted(scenes_sec, key=lambda se: (se[1] - se[0]), reverse=True)
    top = ranked[:k]
    return sorted(top, key=lambda se: se[0])


def sample_frames(video_path: str, scenes_sec, out_root: Path, fps: float = 2.0):
    cap = cv2.VideoCapture(video_path)
    frames_root = out_root / "frames"
    ensure_dir(frames_root)
    per_scene = []

    for idx, (s, e) in enumerate(scenes_sec, start=1):
        scene_dir = frames_root / f"scene_{idx:04d}"
        ensure_dir(scene_dir)

        frames = []
        if fps > 0:
            step = 1.0 / fps
            t = s
            while t <= e:
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                frame_path = scene_dir / f"frame_{round(t,3):.3f}.jpg"
                cv2.imwrite(str(frame_path), frame)
                frames.append({"t": round(t, 3), "path": str(frame_path)})
                t += step

        if not frames:
            t_mid = (s + e) / 2.0
            cap.set(cv2.CAP_PROP_POS_MSEC, t_mid * 1000.0)
            ok, frame = cap.read()
            if ok and frame is not None:
                frame_path = scene_dir / "frame_mid.jpg"
                cv2.imwrite(str(frame_path), frame)
                frames.append({"t": round(t_mid, 3), "path": str(frame_path)})

        per_scene.append(frames)

    cap.release()
    return per_scene



def save_manifest(scenes_sec, per_scene, out_dir: Path):
    ensure_dir(out_dir)
    manifest = {
        "scenes": [
            {
                "scene_id": i,
                "t_start": round(s, 3),
                "t_end": round(e, 3),
                "duration": round(e - s, 3),
                "frames": per_scene[i - 1] if per_scene is not None else []
            }
            for i, (s, e) in enumerate(scenes_sec, start=1)
        ]
    }
    with open(out_dir / "scenes.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest

def get_args():
    parser = argparse.ArgumentParser(
        description="Detect scenes in a video and optionally sample frames & write a scenes.json manifest."
    )
    parser.add_argument(
        "video_path",
        type=Path,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "out_dir",
        type=Path,
        help="Output directory where scenes, frames, and manifest will be written.",
    )
    return parser.parse_args()


def main():
    args = get_args()
    video_path = str(args.video_path)
    out_dir = args.out_dir

    scenes_raw = detect_scenes_pyscenedetect(
        video_path=video_path,
        threshold=THRESHOLD,
        min_scene_len_s=MIN_SCENE_LEN_S,
        start_sec=START_SEC
    )
    scenes = clip_range(scenes_raw, t_max=MAX_T)
    scenes = drop_too_short(scenes, min_len=MIN_KEEP_S)
    scenes = merge_small_gaps(scenes, max_gap=MERGE_GAP_S)
    scenes = keep_top_k_longest(scenes, k=TOP_K)

    per_scene = []
    if EXPORT_ONE_FRAME_PER_SCENE:
        per_scene = sample_frames(
            video_path=video_path,
            scenes_sec=scenes,
            out_root=out_dir,
            fps=SAMPLE_FPS,
        )
    save_manifest(scenes, per_scene if EXPORT_ONE_FRAME_PER_SCENE else None, out_dir)


if __name__ == "__main__":
    main()
    