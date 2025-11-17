
# ********************************* main.py ****************************
# Fusion Layer: Generate Grounded Audio Descriptions from TSG + Qwen Cues.
# ***********************************************************************
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
import yaml
from openai import OpenAI

client = OpenAI()

def get_args():
    parser = argparse.ArgumentParser(
        description="Generate grounded audio descriptions from TSG + Qwen cues.",
    )
    parser.add_argument(
        "--tsg",
        type=Path,
        required=True,
        help="Path to tsg.json",
    )
    parser.add_argument(
        "--cues",
        type=Path,
        required=True,
        help="Path to Qwen cues JSON (e.g., prince_ali_cues.json)",
    )
    parser.add_argument(
        "--lang",
        type=str,
        required=True,
        help="Target language for narration (e.g., 'English', 'Kinyarwanda')",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path("/scratch/Fabrice/vision/idea_3/prompts.yaml"),
        required=False,
        help="Path to YAML file containing prompt_1 and prompt_2 (default: prompts.yaml)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=False,
        help="Output directory for narration JSON files (default: ./outputs_<lang>)",
    )
    return parser.parse_args()

def _load_tsg(tsg_path: str | Path):
    return json.loads(Path(tsg_path).read_text(encoding="utf-8"))


def _load_cues(cues_path: str | Path):
    return json.loads(Path(cues_path).read_text(encoding="utf-8"))


def _summarize_entities(tsg: dict, max_entities: int = 40):

    nodes = tsg.get("nodes", [])
    GENERIC = {"object", "unknown", "background", "thing", "animal"}
    from collections import defaultdict, Counter
    stats = defaultdict(lambda: {
        "count": 0,
        "attrs": Counter(),
        "od_labels": Counter(),
        "clip_labels": Counter(),
    })
    for n in nodes:
        od_label = n.get("label", "object")
        attrs = n.get("attrs", [])
        meta = n.get("meta", {}) or {}
        clip_meta = meta.get("clip") or {}
        clip_top1 = None
        clip_conf = 0.0

        if isinstance(clip_meta, dict):
            clip_top1 = clip_meta.get("top1")
            try:
                clip_conf = float(clip_meta.get("conf", 0.0))
            except Exception:
                clip_conf = 0.0
        canonical_label = od_label
        if clip_top1 and clip_conf >= 0.60 and clip_top1 not in GENERIC:
            canonical_label = clip_top1

        bucket = stats[canonical_label]
        bucket["count"] += 1
        bucket["od_labels"][od_label] += 1
        clip_labels_list = meta.get("clip_labels", [])
        for cl in clip_labels_list:
            name = cl.get("name") if isinstance(cl, dict) else str(cl)
            if name:
                bucket["clip_labels"][name] += 1
        for a in attrs:
            if isinstance(a, dict):
                name = a.get("name")
            else:
                name = str(a)
            if name:
                bucket["attrs"][name] += 1
    items = []
    for label, info in stats.items():
        count = info["count"]
        attr_counter = info["attrs"]
        od_counter = info["od_labels"]
        clip_counter = info["clip_labels"]

        top_attrs = [name for (name, _) in attr_counter.most_common(6)]
        top_od = [name for (name, _) in od_counter.most_common(3)]
        top_clip = [name for (name, _) in clip_counter.most_common(3)]

        items.append({
            "label": label,
            "count": count,
            "od_labels": top_od,
            "clip_labels": top_clip,
            "top_attrs": top_attrs,
        })
    items.sort(key=lambda x: -x["count"])
    return items[:max_entities]

def _summarize_events(cues: dict, max_events: int = 40):
    return cues.get("salient_events", [])[:max_events]


def build_fusion_prompt(prompt_1: str,tsg_path: str | Path,cues_path: str | Path,target_language: str = "English",max_entities: int = 40,max_events: int = 40,) -> str:
    tsg = _load_tsg(tsg_path)
    cues = _load_cues(cues_path)

    grounded_entities = _summarize_entities(tsg, max_entities=max_entities)
    qwen_events = _summarize_events(cues, max_events=max_events)

    scene_setting = cues.get("scene_setting", {})
    video_duration = cues.get("video_duration_s", None)

    context = {
        "scene_setting": scene_setting,
        "video_duration_s": video_duration,
        "grounded_entities": grounded_entities,
        "qwen_salient_events": qwen_events,
    }
    context_json = json.dumps(context, indent=2, ensure_ascii=False)

    prompt = (
        prompt_1
        .replace("{{target_language}}", target_language)
        .replace("{{context_json}}", context_json)
    )
    print("this is fusion prompt", prompt)
    return prompt

def chat_once(model: str,system: str,user: str,temperature: float = 0.4,max_tokens: int = 1200,) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


def build_grounding_prompt(prompt_2: str,context: dict,llm1_narration,target_language: str = "English",) -> str:
    if isinstance(llm1_narration, str):
        candidate_narration = json.loads(llm1_narration)
    else:
        candidate_narration = llm1_narration

    context_json = json.dumps(context, indent=2, ensure_ascii=False)
    narration_json = json.dumps(candidate_narration, indent=2, ensure_ascii=False)
    prompt = (
        prompt_2
        .replace("{{target_language}}", target_language)
        .replace("{{context_json}}", context_json)
        .replace("{{narration_json}}", narration_json)
    )
    print("this is the grounding prompt", prompt)
    return prompt

def main():
    args = get_args()

    tsg_path = args.tsg
    cues_path = args.cues
    target_language = args.lang
    prompts_path = args.prompts

    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        out_dir = Path(f"./outputs_{target_language}")

    out_dir.mkdir(parents=True, exist_ok=True)

    prompts_data = yaml.safe_load(prompts_path.read_text(encoding="utf-8"))
    prompt_1 = prompts_data["prompt_1"]
    prompt_2 = prompts_data["prompt_2"]
    tsg = _load_tsg(tsg_path)
    cues = _load_cues(cues_path)
    grounded_entities = _summarize_entities(tsg)
    qwen_events = _summarize_events(cues)
    scene_setting = cues.get("scene_setting", {})
    video_duration = cues.get("video_duration_s", None)

    context = {
        "scene_setting": scene_setting,
        "video_duration_s": video_duration,
        "grounded_entities": grounded_entities,
        "qwen_salient_events": qwen_events,
    }
    user_prompt_1 = build_fusion_prompt(
        prompt_1=prompt_1,
        tsg_path=tsg_path,
        cues_path=cues_path,
        target_language=target_language,
    )

    system_prompt = (
        "You are a careful model that strictly follows instructions "
        "and outputs valid JSON when requested."
    )

    print("Running LLM-1 (generation)...")
    result_1_text = chat_once(
        model="gpt-4.1",
        system=system_prompt,
        user=user_prompt_1,
        temperature=0.4,
        max_tokens=3000,
    )

    result_1_json = json.loads(result_1_text)

    out_stage1 = out_dir / "narration_stage1.json"
    out_stage1.write_text(
        json.dumps(result_1_json, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    user_prompt_2 = build_grounding_prompt(
        prompt_2=prompt_2,
        context=context,
        llm1_narration=result_1_json,
        target_language=target_language,
    )
    result_2_text = chat_once(
        model="gpt-4.1",
        system=system_prompt,
        user=user_prompt_2,
        temperature=0.2,
        max_tokens=3000,
    )

    result_2_json = json.loads(result_2_text)
    out_stage2 = out_dir / "narration_stage2_grounded.json"
    out_stage2.write_text(json.dumps(result_2_json, indent=2, ensure_ascii=False),encoding="utf-8",)
    return result_1_json, result_2_json


if __name__ == "__main__":
    main()
