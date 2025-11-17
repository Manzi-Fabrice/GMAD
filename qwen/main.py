# ********************************* main.py ****************************
# Run Qwen Model on the video
# ***********************************************************************
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import argparse
import json
import subprocess
from pathlib import Path
from string import Template
import torch
import yaml
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

QMODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
MAX_SEG_LEN = 15.0
min_pixels = 256 * 28 * 28
max_pixels = 512 * 28 * 28


vl = Qwen2_5_VLForConditionalGeneration.from_pretrained(
   QMODEL,
   torch_dtype=torch.float16,
   device_map="auto",
   attn_implementation="sdpa",
)
schema_prompt_tpl: Template | None = None
vl_proc = AutoProcessor.from_pretrained(
   QMODEL,
   min_pixels=min_pixels,
   max_pixels=max_pixels,
)


def get_video_duration(video_path: str | Path) -> float:
   cmd = [
       "ffprobe",
       "-v", "error",
       "-show_entries", "format=duration",
       "-of", "default=nw=1:nk=1",
       str(video_path),
   ]
   out = subprocess.run(cmd, capture_output=True, text=True, check=True)
   return float(out.stdout.strip())




def get_video_fps(video_path: str | Path) -> float:
   cmd = [
       "ffprobe",
       "-v", "error",
       "-select_streams", "v:0",
       "-show_entries", "stream=avg_frame_rate",
       "-of", "default=nw=1:nk=1",
       str(video_path),
   ]
   out = subprocess.run(cmd, capture_output=True, text=True, check=True)
   s = out.stdout.strip().replace("\r", "")
   if "/" in s:
       num, den = s.split("/")
       num = float(num)
       den = float(den) if float(den) != 0 else 1.0
       return num / den
   return float(s)


def clip_segment_copy(input_path: str | Path, t0: float, t1: float, out_path: str | Path):
   duration = max(0.0, float(t1) - float(t0))
   cmd = [
       "ffmpeg", "-y",
       "-ss", f"{t0:.3f}", "-i", str(input_path),
       "-t", f"{duration:.3f}",
       "-vcodec", "copy",
       "-acodec", "copy",
       str(out_path),
   ]
   subprocess.run(cmd, check=True)


def extract_cues(path: str | Path, fps: float = 1.0, max_tokens: int = 256) -> dict:
   global schema_prompt_tpl
   if schema_prompt_tpl is None:
       raise RuntimeError("schema_prompt_tpl is not initialized. Load it from YAML before calling extract_cues().")


   duration = get_video_duration(path)
   fmt = schema_prompt_tpl.substitute(duration=f"{duration:.2f}", fps=fps)


   msg = [{
       "role": "user",
       "content": [
           {"type": "video", "path": str(path)},
           {
               "type": "text",
               "text": (
                   f"Analyze this video at {fps} fps and output a cue sheet for full temporal coverage.\n{fmt}"
               ),
           },
       ],
   }]


   inputs = vl_proc.apply_chat_template(
       msg,
       video_fps=fps,
       add_generation_prompt=True,
       tokenize=True,
       return_dict=True,
       return_tensors="pt",
   ).to(vl.device)


   torch.cuda.empty_cache()
   with torch.no_grad():
       out_ids = vl.generate(
           **inputs,
           max_new_tokens=max_tokens,
           do_sample=False,
           top_p=1.0,
       )


   gen_ids = out_ids[:, inputs.input_ids.shape[1]:]
   raw = vl_proc.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()


   start = raw.find("{")
   end = raw.rfind("}")
   text = raw[start:end + 1] if start != -1 and end != -1 else "{}"


   def parse_json(s: str) -> dict:
       try:
           return json.loads(s)
       except json.JSONDecodeError:
           s2 = s.replace("\n", " ").replace("\t", " ").replace("\\n", " ")
           try:
               return json.loads(s2)
           except Exception:
               return {}


   cues = parse_json(text)


   try:
       events = cues.get("salient_events", [])
       if events:
           covered = sum(float(e.get("end_s", 0)) - float(e.get("start_s", 0)) for e in events)
           pct = (covered / duration * 100.0) if duration > 0 else 0.0
   except Exception:
       pass


   torch.cuda.empty_cache()
   return cues or {}




def adjust_event_times(cues: dict, offset_s: float) -> dict:
   if not cues:
       return {}
   evs = cues.get("salient_events", [])
   for e in evs:
       try:
           e["start_s"] = float(e.get("start_s", 0)) + offset_s
           e["end_s"] = float(e.get("end_s", 0)) + offset_s
       except Exception:
           pass
   return cues




def merge_cues(cues_list, total_duration: float) -> dict:
   merged = {
       "video_duration_s": total_duration,
       "scene_setting": {
           "environment": "unknown",
           "location_type": "unknown",
           "time_of_day": "unknown",
           "lighting_mood": "unknown",
       },
       "primary_objects": [],
       "key_actions": [],
       "detected_emotions": [],
       "salient_events": [],
   }


   def push_unique(dst, src, key=None, cap=5):
       for x in src:
           if key:
               sig = x.get(key, "")
               if not any(y.get(key, "") == sig for y in dst):
                   dst.append(x)
           else:
               if x not in dst:
                   dst.append(x)
           if len(dst) >= cap:
               break


   for c in cues_list:
       if not c:
           continue
       ss = c.get("scene_setting", {})
       for k in ["environment", "location_type", "time_of_day", "lighting_mood"]:
           if ss.get(k, "unknown") != "unknown" and merged["scene_setting"][k] == "unknown":
               merged["scene_setting"][k] = ss[k]


       push_unique(merged["primary_objects"], c.get("primary_objects", []), key="name", cap=5)
       push_unique(merged["key_actions"], c.get("key_actions", []), cap=5)
       push_unique(merged["detected_emotions"], c.get("detected_emotions", []), cap=5)
       merged["salient_events"].extend(c.get("salient_events", []))


   evs = []
   for e in merged["salient_events"]:
       try:
           s = float(e.get("start_s", 0.0))
           t = float(e.get("end_s", s))
           if t < s:
               t = s
           evs.append({
               "start_s": s,
               "end_s": t,
               "event": str(e.get("event", ""))[:500],
           })
       except Exception:
           pass


   evs.sort(key=lambda x: x["start_s"])


   fixed = []
   last_end = None
   for e in evs:
       if last_end is not None and e["start_s"] < last_end and (last_end - e["start_s"]) <= 1.0:
           e["start_s"] = last_end
       fixed.append(e)
       last_end = max(last_end or e["end_s"], e["end_s"])


   merged["salient_events"] = fixed
   return merged




def save_json(obj: dict, path: str | Path):
   path = Path(path)
   with path.open("w", encoding="utf-8") as f:
       json.dump(obj, f, ensure_ascii=False, indent=2)




def make_timeline(cues_json: Path, timeline_json: Path,
                 pad_in_s: float = 0.20, pad_out_s: float = 0.10):
   cues = json.loads(cues_json.read_text(encoding="utf-8"))
   events = cues.get("salient_events", [])
   if not events:
       tl = [{"t0": 0.50, "t1": 3.0}]
       timeline_json.write_text(
           json.dumps(tl, ensure_ascii=False, indent=2),
           encoding="utf-8",
       )
       return


   tl = []
   for ev in events:
       try:
           s = float(ev.get("start_s", 0.0))
           t = float(ev.get("end_s", s + 1.0))
           t0 = max(0.0, s - pad_in_s)
           t1 = max(t0 + 0.3, t + pad_out_s)
           tl.append({"t0": t0, "t1": t1})
       except Exception:
           tl.append({"t0": pad_in_s, "t1": pad_in_s + 1.0})


   timeline_json.write_text(
       json.dumps(tl, ensure_ascii=False, indent=2),
       encoding="utf-8",
   )




def get_args():
   parser = argparse.ArgumentParser(
       description="Generate Qwen2.5-VL cue sheets and timelines from a video and scene manifest.",
   )
   parser.add_argument(
       "video_path",
       type=Path,
       help="Path to the input video",
   )
   parser.add_argument(
       "scenes_path",
       type=Path,
       help="Path to scenes.json (with 'scenes': [{t_start, t_end}, ...])",
   )
   parser.add_argument(
       "--config",
       type=Path,
       default=Path("/scratch/Fabrice/vision/idea_3/prompts.yaml"),
       help="Path to YAML config for Qwen prompt (default: prompts.yaml)",
   )
   parser.add_argument(
       "--out-cues",
       type=Path,
       default=None,
       help="Output path for merged cues JSON (default: <video_stem>_cues.json)",
   )
   parser.add_argument(
       "--out-timeline",
       type=Path,
       default=None,
       help="Output path for timeline JSON (default: <video_stem>_timeline.json)",
   )
   return parser.parse_args()

def split_long_segment(t0: float, t1: float, max_len: float = 15.0):
    segments = []
    cur = t0
    while cur < t1 - 1e-3:
        nt = min(t1, cur + max_len)
        segments.append((cur, nt))
        cur = nt
    return segments


def main():
   global schema_prompt_tpl


   args = get_args()
   with args.config.open("r", encoding="utf-8") as f:
       cfg = yaml.safe_load(f)
   schema_prompt_tpl = Template(cfg["qwen_prompt"])


   video = args.video_path
   scenes_manifest_path = args.scenes_path
   cues_out = args.out_cues or Path("cues.json")
   timeline_out = args.out_timeline or Path(f"{video.stem}_timeline.json")


   with scenes_manifest_path.open("r", encoding="utf-8") as f:
       scenes_manifest = json.load(f)


   segments = [
       (float(sc["t_start"]), float(sc["t_end"]))
       for sc in scenes_manifest.get("scenes", [])
   ]

   for i, (t0, t1) in enumerate(segments):
    print(f"SEGMENT {i} {t0:.2f}–{t1:.2f} (len={t1 - t0:.2f}s)")


    new_segments = []
    for t0, t1 in segments:
        if (t1 - t0) > MAX_SEG_LEN:
            new_segments.extend(split_long_segment(t0, t1, max_len=MAX_SEG_LEN))
        else:
            new_segments.append((t0, t1))
    segments = new_segments

   cues_all = []
   for idx, (t0, t1) in enumerate(segments):
       seg_path = Path(f"/tmp/seg_scene{idx + 1}_{int(t0)}_{int(t1)}.mp4")
       clip_segment_copy(video, t0, t1, seg_path)


       seg_cues = extract_cues(seg_path, fps=1.0, max_tokens=256)
       seg_cues = adjust_event_times(seg_cues, offset_s=t0)
       if seg_cues:
           seg_cues["video_duration_s"] = get_video_duration(video)
           cues_all.append(seg_cues)


       torch.cuda.empty_cache()


   merged = merge_cues(cues_all, total_duration=get_video_duration(video))
   save_json(merged, cues_out)
   make_timeline(cues_out, timeline_out)




if __name__ == "__main__":
   main()


