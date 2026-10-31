#!/usr/bin/env python3
import concurrent.futures
import json
import pathlib
import subprocess
import time
import urllib.parse
import urllib.request

OUT = pathlib.Path("benchmark")
OUT.mkdir(exist_ok=True)


def fetch_bytes(request, attempts=4):
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except (OSError, TimeoutError):
            if attempt + 1 == attempts:
                raise
            time.sleep(2 ** attempt)


def fetch_row(offset):
    url = (
        "https://datasets-server.huggingface.co/rows?"
        + urllib.parse.urlencode(
            {
                "dataset": "google/fleurs",
                "config": "de_de",
                "split": "validation",
                "offset": offset,
                "length": 1,
            }
        )
    )
    request = urllib.request.Request(
        url, headers={"User-Agent": "AnySpeechRecognizer-Helvetia-Probe/1.0"}
    )
    return json.loads(fetch_bytes(request))["rows"][0]


with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    selected = list(executor.map(fetch_row, range(10)))
references = []
for index, entry in enumerate(selected):
    row = entry["row"]
    audio_field = row["audio"]
    audio_url = audio_field[0]["src"] if isinstance(audio_field, list) else audio_field["src"]
    target = OUT / f"clip_{index:02d}.wav"
    audio_request = urllib.request.Request(
        audio_url, headers={"User-Agent": "AnySpeechRecognizer-Helvetia-Probe/1.0"}
    )
    target.write_bytes(fetch_bytes(audio_request))
    references.append(
        {
            "index": index,
            "row_idx": entry["row_idx"],
            "speaker_id": row.get("speaker_id"),
            "transcription": row["transcription"],
            "raw_transcription": row["raw_transcription"],
            "path": str(target),
        }
    )

(OUT / "references.json").write_text(
    json.dumps(references, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)

# Use two known-different FLEURS speakers for a deterministic diarization probe.
first = selected[0]
second_index = next(
    (
        i
        for i, candidate in enumerate(selected[1:], start=1)
        if candidate["row"].get("speaker_id") != first["row"].get("speaker_id")
    ),
    1,
)
source_paths = [OUT / "clip_00.wav", OUT / f"clip_{second_index:02d}.wav"]

def duration(path):
    return float(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            text=True,
        ).strip()
    )


durations = [duration(path) for path in source_paths]
silence_seconds = 1.0
subprocess.run(
    [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source_paths[0]),
        "-f",
        "lavfi",
        "-t",
        str(silence_seconds),
        "-i",
        "anullsrc=r=16000:cl=mono",
        "-i",
        str(source_paths[1]),
        "-filter_complex",
        "[0:a]aformat=sample_fmts=s16:sample_rates=16000:channel_layouts=mono[a0];"
        "[1:a]aformat=sample_fmts=s16:sample_rates=16000:channel_layouts=mono[a1];"
        "[2:a]aformat=sample_fmts=s16:sample_rates=16000:channel_layouts=mono[a2];"
        "[a0][a1][a2]concat=n=3:v=0:a=1[out]",
        "-map",
        "[out]",
        "-c:a",
        "pcm_s16le",
        str(OUT / "diarization.wav"),
    ],
    check=True,
)
start_second = durations[0] + silence_seconds
rttm = (
    f"SPEAKER diarization 1 0.000 {durations[0]:.6f} <NA> <NA> speaker0 <NA> <NA>\n"
    f"SPEAKER diarization 1 {start_second:.6f} {durations[1]:.6f} <NA> <NA> speaker1 <NA> <NA>\n"
)
(OUT / "diarization.rttm").write_text(rttm, encoding="utf-8")
(OUT / "dataset.json").write_text(
    json.dumps(
        {
            "dataset": "google/fleurs",
            "config": "de_de",
            "split": "validation",
            "row_indices": [r["row_idx"] for r in selected],
            "diarization_clip_indices": [0, second_index],
            "silence_seconds": silence_seconds,
        },
        ensure_ascii=False,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
print(json.dumps({"clips": len(selected), "diarization_second_clip": second_index}))
