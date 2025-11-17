
# ********************************* main.py ****************************
# Text to Audion using Multilingual Eleven Model
# ***********************************************************************

import json
import os
import time
import random
from pathlib import Path
from typing import List, Dict
from argparse import ArgumentParser
from elevenlabs.client import ElevenLabs
from elevenlabs import save

ELEVEN_MODEL = "eleven_multilingual_v2"

ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
if not ELEVENLABS_API_KEY:
    raise RuntimeError(
        "ELEVENLABS_API_KEY is Missing "
    )
tts = ElevenLabs(api_key=ELEVENLABS_API_KEY)

VOICE_BY_LANG: Dict[str, str] = {
    "Arabic": "JBFqnCBsd6RMkjVDRZzb",
    "Chinese": "JBFqnCBsd6RMkjVDRZzb",
    "English": "JBFqnCBsd6RMkjVDRZzb",
    "German": "JBFqnCBsd6RMkjVDRZzb",
    "Korean": "JBFqnCBsd6RMkjVDRZzb",
    "Spanish": "JBFqnCBsd6RMkjVDRZzb",
    "Swahili": "JBFqnCBsd6RMkjVDRZzb",
}
DEFAULT_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"


def backoff(retry: int, base: float = 0.8, jitter: float = 0.25) -> float:
    return base * (2 ** retry) + random.uniform(0, jitter)


def to_srt_timestamp_from_seconds(seconds: float) -> str:
    ms_total = int(round(seconds * 1000))
    hours, rem = divmod(ms_total, 3600000)
    minutes, rem = divmod(rem, 60000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def write_srt(events: List[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for i, ev in enumerate(events, start=1):
            start_s = float(ev["start_s"])
            end_s = float(ev["end_s"])
            text = ev["description"].strip()

            start_ts = to_srt_timestamp_from_seconds(start_s)
            end_ts = to_srt_timestamp_from_seconds(end_s)

            f.write(f"{i}\n{start_ts} --> {end_ts}\n{text}\n\n")


def synthesize_mp3(text: str, out_path: Path, voice_id: str) -> Path:
    for r in range(5):
        try:
            stream = tts.text_to_speech.convert(
                text=text,
                voice_id=voice_id,
                model_id=ELEVEN_MODEL,
                output_format="mp3_44100_128",
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            save(stream, str(out_path))
            return out_path
        except Exception as e:
            print(f"Error during TTS attempt")
            if r == 4:
                raise
            time.sleep(backoff(r))
    return out_path


def process_language_dir(input_dir: Path, output_root: Path) -> None:
    lang_name = input_dir.name.replace("outputs_", "")
    stage2_path = input_dir / "narration_stage2_grounded.json"

    if not stage2_path.exists():
        print(f"[{lang_name}] narration_stage2_grounded.json not found, skipping.")
        return

    print(f"\n=== Processing language: {lang_name} ===")
    data = json.loads(stage2_path.read_text(encoding="utf-8"))
    events: List[dict] = data

    lang_out_dir = output_root / lang_name
    audio_dir = lang_out_dir / "tts_mp3"
    srt_path = lang_out_dir / "narration_stage2_grounded.srt"
    transcript_path = lang_out_dir / "narration_stage2_grounded.txt"

    voice_id = VOICE_BY_LANG.get(lang_name, DEFAULT_VOICE_ID)

    transcript_lines: List[str] = []
    for i, ev in enumerate(events, start=1):
        desc = ev["description"].strip()
        start_s = float(ev["start_s"])
        start_ms = int(round(start_s * 1000))
        filename = f"{i:03d}_{start_ms:06d}.mp3"
        out_mp3 = audio_dir / filename
        synthesize_mp3(desc, out_mp3, voice_id=voice_id)
        transcript_lines.append(desc)

    write_srt(events, srt_path)
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text("\n".join(transcript_lines), encoding="utf-8")

def parse_args():
    parser = ArgumentParser(
        description="Generate TTS MP3s, SRT, and transcript from narration JSON."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more input directories (e.g. outputs_English).",
    )
    parser.add_argument(
        "-o",
        "--output-root",
        type=str,
        default="output/description",
        help="Root directory where results will be written (default: output/description).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = Path(args.output_root).resolve()

    for input_path in args.inputs:
        lang_dir = Path(input_path).resolve()
        if not lang_dir.is_dir():
            print(f"Skipping {lang_dir}: not a directory.")
            continue
        process_language_dir(lang_dir, output_root)

if __name__ == "__main__":
    main()
