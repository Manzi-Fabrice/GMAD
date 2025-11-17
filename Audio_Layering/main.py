# ********************************* main.py ****************************
# Layer Generated Audio on Top of Video
# ***********************************************************************
import json
import re
import subprocess
import shutil
import tempfile
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple
from argparse import ArgumentParser

from pydub import AudioSegment
from pydub.silence import detect_silence, split_on_silence

TARGET_SR = 48000
TARGET_CH = 2
DEFAULT_FADE_MS = 10
DUCK_DB = -7.0
BG_PRE_GAIN_DB = 0.0
NORMALIZE_PEAK = -1.0

SOLO_TAIL_HOLD_MS = 70
SOLO_TAIL_FADE_MS = 220
SOLO_TAIL_DIP_DB = -2.5
ZERO_X_SCAN_MS = 30

START_AT_S = 0.35

MIN_WINDOW_MS = 300
TRIM_SIL_THRESH = -45
TRIM_SIL_MIN_MS = 120
END_MARGIN_MS = 100

SILENCE_THRESH_DB = -40
MIN_SILENCE_LEN_MS = 100

NEIGHBOR_GAP_THR_MS = 180
CROSSFADE_MS = 120
MAX_ADVANCE_MS = 90

SOLO_TAIL_FLATTEN_MS = 60
SOLO_TAIL_ATTEN_DB = -4.0

PER_EVENT_RE = re.compile(r"^([a-zA-Z\-]+)_(\d{3})_.*\.(mp3|wav|m4a|flac|ogg)$")
SINGLE_FILE_RE = re.compile(r"^([a-zA-Z\-]+)_narration\.(mp3|wav|m4a|flac|ogg)$")


def run(cmd: List[str]):
    subprocess.run(cmd, check=True)


def video_duration_ms(video: Path) -> int:
    r = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(float(r.stdout.strip()) * 1000)


def extract_audio_from_video(video: Path, out_wav: Path):
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video),
            "-vn",
            "-ar",
            str(TARGET_SR),
            "-ac",
            str(TARGET_CH),
            "-acodec",
            "pcm_s16le",
            "-f",
            "wav",
            str(out_wav),
        ]
    )


def load_seg(path: Path) -> AudioSegment:
    seg = AudioSegment.from_file(path)
    if seg.frame_rate != TARGET_SR:
        seg = seg.set_frame_rate(TARGET_SR)
    if seg.channels != TARGET_CH:
        seg = seg.set_channels(TARGET_CH)
    return seg


def trim_trailing_silence(seg: AudioSegment) -> AudioSegment:
    end_sil = detect_silence(
        seg, min_silence_len=TRIM_SIL_MIN_MS, silence_thresh=TRIM_SIL_THRESH
    )
    if end_sil and end_sil[-1][1] >= len(seg) - 1 and end_sil[-1][0] > 0:
        return seg[: end_sil[-1][0]]
    return seg


def normalize_peak(seg: AudioSegment, target_dbfs=-1.0) -> AudioSegment:
    if target_dbfs is None:
        return seg
    change = target_dbfs - seg.max_dBFS
    return seg.apply_gain(change)


def overlay_with_duck(
    bg: AudioSegment, fg: AudioSegment, t0_ms: int, duck_db: float
) -> AudioSegment:
    return bg.overlay(fg, position=int(t0_ms), gain_during_overlay=duck_db)


def infer_windows(timeline_raw: List[dict], vid_len_ms: int) -> List[Tuple[int, int]]:
    shift = int(round(START_AT_S * 1000))
    t0s = [
        max(0, int(round(float(x["t0"]) * 1000)) + shift) for x in timeline_raw
    ]
    t1s = []
    has_t1 = any("t1" in x for x in timeline_raw)
    if has_t1:
        for i, x in enumerate(timeline_raw):
            fallback_t1 = t0s[i + 1] if i + 1 < len(t0s) else vid_len_ms
            t1_val = float(x.get("t1", fallback_t1 / 1000))
            t1 = int(round(t1_val * 1000)) + shift
            t1s.append(min(vid_len_ms, max(t0s[i] + MIN_WINDOW_MS, t1)))
    else:
        for i, t0 in enumerate(t0s):
            nxt = t0s[i + 1] if i + 1 < len(t0s) else vid_len_ms
            t1s.append(min(vid_len_ms, max(t0 + MIN_WINDOW_MS, nxt)))
    if END_MARGIN_MS:
        t1s = [
            max(t0s[i] + MIN_WINDOW_MS, t1 - END_MARGIN_MS)
            for i, t1 in enumerate(t1s)
        ]
    return list(zip(t0s, t1s))


def find_per_event_files(narr_dir: Path) -> Dict[str, List[Path]]:
    per_lang: Dict[str, List[Tuple[int, Path]]] = defaultdict(list)
    for p in narr_dir.iterdir():
        m = PER_EVENT_RE.match(p.name)
        if m:
            lang = m.group(1)
            idx = int(m.group(2))
            per_lang[lang].append((idx, p))
    out: Dict[str, List[Path]] = {}
    for lang, items in per_lang.items():
        items.sort(key=lambda t: t[0])
        out[lang] = [p for _, p in items]
    return out


def find_single_files(narr_dir: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for p in narr_dir.iterdir():
        m = SINGLE_FILE_RE.match(p.name)
        if m:
            out[m.group(1)] = p
    return out


def split_narration_on_silence(
    full_audio: AudioSegment, num_expected: int
) -> List[AudioSegment]:
    chunks = split_on_silence(
        full_audio,
        min_silence_len=MIN_SILENCE_LEN_MS,
        silence_thresh=SILENCE_THRESH_DB,
        keep_silence=50,
    )
    if len(chunks) < num_expected:
        while len(chunks) < num_expected:
            if chunks:
                last_chunk = chunks[-1]
            else:
                last_chunk = AudioSegment.silent(
                    duration=1000, frame_rate=TARGET_SR
                )
            chunks.append(last_chunk)
    elif len(chunks) > num_expected:
        step = max(1, len(chunks) // num_expected)
        merged = []
        for i in range(0, len(chunks), step):
            end_idx = min(i + step, len(chunks))
            chunk_group = chunks[i:end_idx]
            merged_chunk = sum(
                chunk_group, AudioSegment.silent(duration=0, frame_rate=TARGET_SR)
            )
            merged.append(merged_chunk)
        chunks = merged[:num_expected]
        while len(chunks) < num_expected:
            chunks.append(
                AudioSegment.silent(duration=800, frame_rate=TARGET_SR)
            )
    return chunks


def fit_to_window(seg: AudioSegment, t0_ms: int, t1_ms: int) -> AudioSegment:
    win_len = max(MIN_WINDOW_MS, t1_ms - t0_ms)
    s = seg
    if DEFAULT_FADE_MS:
        s = s.fade_in(DEFAULT_FADE_MS).fade_out(DEFAULT_FADE_MS)
    s = trim_trailing_silence(s)
    if len(s) > win_len:
        speed = len(s) / win_len
        if speed <= 1.2:
            s = s.speedup(playback_speed=speed)
        else:
            cutoff = win_len
            s = s[:cutoff].fade_out(
                int(min(DEFAULT_FADE_MS * 2, cutoff * 0.2))
            )
    elif len(s) < win_len:
        s = s + AudioSegment.silent(
            duration=win_len - len(s), frame_rate=TARGET_SR
        )
    return s


def _nearest_zero_cross_from_end(
    seg: AudioSegment, scan_ms: int = ZERO_X_SCAN_MS
) -> int:
    if len(seg) < 2:
        return len(seg)
    scan = min(scan_ms, len(seg))
    mono = seg.set_channels(1)
    samp = mono.get_array_of_samples()
    spms = mono.frame_rate // 1000
    n_scan = int(scan * spms)
    end_idx = len(samp)
    start_idx = max(0, end_idx - n_scan)
    prev = samp[end_idx - 1]
    for i in range(end_idx - 2, start_idx, -1):
        cur = samp[i]
        if (cur <= 0 < prev) or (cur >= 0 > prev):
            ms = int(i / spms)
            return min(ms, len(seg))
        prev = cur
    return len(seg)


def speechy_soft_stop(
    seg: AudioSegment,
    hold_ms: int = SOLO_TAIL_HOLD_MS,
    fade_ms: int = SOLO_TAIL_FADE_MS,
    dip_db: float = SOLO_TAIL_DIP_DB,
) -> AudioSegment:
    total_tail = max(0, hold_ms + fade_ms)
    if len(seg) <= total_tail + DEFAULT_FADE_MS:
        return seg.fade_out(min(fade_ms, max(40, len(seg) // 3)))
    cut_ms = _nearest_zero_cross_from_end(seg, ZERO_X_SCAN_MS)
    if cut_ms < len(seg):
        seg = seg[:cut_ms]
    head_len = len(seg) - total_tail
    head = seg[:head_len]
    hold = (
        seg[head_len : head_len + hold_ms].apply_gain(dip_db)
        if hold_ms > 0
        else AudioSegment.silent(duration=0)
    )
    tail = (
        seg[head_len + hold_ms :]
        if fade_ms > 0
        else AudioSegment.silent(duration=0)
    )
    if fade_ms > 0 and len(tail) > 0:
        tail = tail.fade_out(fade_ms)
    return head + hold + tail


def apply_crossfade_edges(
    prev_seg: AudioSegment, next_seg: AudioSegment, xfade_ms: int = CROSSFADE_MS
) -> Tuple[AudioSegment, AudioSegment]:
    x = max(0, min(xfade_ms, len(prev_seg) // 2, len(next_seg) // 2))
    if x == 0:
        return prev_seg, next_seg
    return prev_seg.fade_out(x), next_seg.fade_in(x)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--video",
        required=True,
        help="Input video file.",
    )
    parser.add_argument(
        "--narration-dir",
        required=True,
        help="Directory with narration audio files.",
    )
    parser.add_argument(
        "--timeline",
        required=True,
        help="JSON file with timeline data.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/description",
        help="Directory for rendered videos.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    video_path = Path(args.video).resolve()
    narr_dir = Path(args.narration_dir).resolve()
    timeline_path = Path(args.timeline).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    timeline_raw = json.loads(timeline_path.read_text(encoding="utf-8"))
    vid_len_ms = video_duration_ms(video_path)
    windows = infer_windows(timeline_raw, vid_len_ms)
    num_windows = len(windows)

    per_event = find_per_event_files(narr_dir)
    singles = find_single_files(narr_dir)
    if not per_event and not singles:
        return

    base = video_path.stem
    langs = set(per_event.keys()) | set(singles.keys())

    for lang in sorted(langs):
        tmpdir = Path(tempfile.mkdtemp(prefix=f"ad_{lang}_"))
        try:
            bg_wav = tmpdir / "bg.wav"
            extract_audio_from_video(video_path, bg_wav)
            bg = AudioSegment.from_wav(bg_wav).set_frame_rate(TARGET_SR).set_channels(
                TARGET_CH
            )
            if BG_PRE_GAIN_DB:
                bg = bg.apply_gain(BG_PRE_GAIN_DB)

            if lang in per_event and per_event[lang]:
                files = per_event[lang]
                narr_segs: List[AudioSegment] = []
                for i in range(num_windows):
                    if i < len(files):
                        seg = load_seg(files[i])
                    else:
                        seg = AudioSegment.silent(
                            duration=600, frame_rate=TARGET_SR
                        )
                    narr_segs.append(seg)
            else:
                single = singles.get(lang)
                if not single:
                    shutil.rmtree(tmpdir, ignore_errors=True)
                    continue
                full_narr = load_seg(single)
                full_narr = trim_trailing_silence(full_narr)
                narr_segs = split_narration_on_silence(
                    full_narr, num_windows
                )

            fit_segs: List[AudioSegment] = []
            for (t0_ms, t1_ms), seg in zip(windows, narr_segs):
                fit_segs.append(fit_to_window(seg, t0_ms, t1_ms))

            starts = [t0 for (t0, _t1) in windows]

            for i in range(len(fit_segs) - 1):
                cur_start = starts[i]
                cur_end = cur_start + len(fit_segs[i])
                nxt_start = starts[i + 1]
                gap = nxt_start - cur_end

                if gap <= NEIGHBOR_GAP_THR_MS:
                    fit_segs[i], fit_segs[i + 1] = apply_crossfade_edges(
                        fit_segs[i], fit_segs[i + 1], CROSSFADE_MS
                    )
                    desired_nxt = cur_end - CROSSFADE_MS
                    max_pull = min(MAX_ADVANCE_MS, nxt_start)
                    new_nxt = max(nxt_start - max_pull, desired_nxt)
                    starts[i + 1] = new_nxt
                else:
                    if (
                        len(fit_segs[i])
                        >= SOLO_TAIL_FLATTEN_MS + DEFAULT_FADE_MS
                    ):
                        fit_segs[i] = speechy_soft_stop(
                            fit_segs[i],
                            hold_ms=SOLO_TAIL_HOLD_MS,
                            fade_ms=SOLO_TAIL_FADE_MS,
                            dip_db=SOLO_TAIL_DIP_DB,
                        )

            if fit_segs and len(fit_segs[-1]) >= SOLO_TAIL_FLATTEN_MS + DEFAULT_FADE_MS:
                fit_segs[-1] = speechy_soft_stop(
                    fit_segs[-1],
                    hold_ms=SOLO_TAIL_HOLD_MS,
                    fade_ms=SOLO_TAIL_FADE_MS,
                    dip_db=SOLO_TAIL_DIP_DB,
                )

            mixed = bg
            for seg_fit, t0_ms in zip(fit_segs, starts):
                mixed = overlay_with_duck(mixed, seg_fit, int(t0_ms), DUCK_DB)

            if NORMALIZE_PEAK is not None:
                mixed = normalize_peak(mixed, NORMALIZE_PEAK)

            mix_wav = tmpdir / f"{lang}_mix.wav"
            mixed.export(
                mix_wav,
                format="wav",
                parameters=["-ar", str(TARGET_SR), "-ac", str(TARGET_CH)],
            )

            out_video = output_dir / f"{base}_{lang}.mp4"
            run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(video_path),
                    "-i",
                    str(mix_wav),
                    "-map",
                    "0:v",
                    "-map",
                    "1:a",
                    "-vcodec",
                    "copy",
                    "-acodec",
                    "aac",
                    "-ab",
                    "192k",
                    "-strict",
                    "-2",
                    str(out_video),
                ]
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
