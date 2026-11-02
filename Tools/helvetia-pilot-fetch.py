#!/usr/bin/env python3
import array
import concurrent.futures
import json
import math
import pathlib
import random
import subprocess
import time
import urllib.parse
import urllib.request
import wave

OUT = pathlib.Path("pilot-benchmark")
AUDIO = OUT / "audio"
REFS = OUT / "references"
AUDIO.mkdir(parents=True, exist_ok=True)
REFS.mkdir(parents=True, exist_ok=True)
COUNT = 50
NOISE_COUNT = 20


def fetch_bytes(request, attempts=4):
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except (OSError, TimeoutError):
            if attempt + 1 == attempts:
                raise
            time.sleep(2**attempt)


def fetch_row(offset):
    url = "https://datasets-server.huggingface.co/rows?" + urllib.parse.urlencode(
        {
            "dataset": "google/fleurs",
            "config": "de_de",
            "split": "validation",
            "offset": offset,
            "length": 1,
        }
    )
    request = urllib.request.Request(url, headers={"User-Agent": "Helvetia-AnySpeech-Pilot/1.0"})
    return json.loads(fetch_bytes(request))["rows"][0]


def convert_audio(data, target):
    raw = target.with_suffix(".source.wav")
    raw.write_bytes(data)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(raw),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(target),
        ],
        check=True,
    )
    raw.unlink()


def read_samples(path):
    with wave.open(str(path), "rb") as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 16000):
            raise RuntimeError(f"Unexpected PCM format: {path}")
        samples = array.array("h")
        samples.frombytes(wav.readframes(wav.getnframes()))
        return samples


def write_samples(path, samples):
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(samples.tobytes())


def duration(path):
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / wav.getframerate()


def add_white_noise(source, target, snr_db, seed):
    signal = read_samples(source)
    signal_rms = math.sqrt(sum(sample * sample for sample in signal) / max(1, len(signal)))
    rng = random.Random(seed)
    noise = [rng.gauss(0.0, 1.0) for _ in signal]
    noise_rms = math.sqrt(sum(value * value for value in noise) / max(1, len(noise)))
    scale = signal_rms / (10 ** (snr_db / 20) * noise_rms)
    mixed = array.array("h", (max(-32768, min(32767, round(sample + value * scale))) for sample, value in zip(signal, noise)))
    write_samples(target, mixed)


with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    selected = list(executor.map(fetch_row, range(COUNT)))

references = []
items = []
for index, entry in enumerate(selected):
    row = entry["row"]
    audio_field = row["audio"]
    audio_url = audio_field[0]["src"] if isinstance(audio_field, list) else audio_field["src"]
    request = urllib.request.Request(audio_url, headers={"User-Agent": "Helvetia-AnySpeech-Pilot/1.0"})
    clean = AUDIO / f"clean_{index:03d}.wav"
    convert_audio(fetch_bytes(request), clean)
    reference = {
        "row_idx": entry["row_idx"],
        "speaker_id": row.get("speaker_id"),
        "transcription": row["transcription"],
        "raw_transcription": row["raw_transcription"],
    }
    references.append(reference)
    items.append({"scenario": "clean", "index": index, "path": str(clean), "transcription": row["transcription"]})
    if index < NOISE_COUNT:
        for snr in (10, 5):
            noisy = AUDIO / f"noise{snr}_{index:03d}.wav"
            add_white_noise(clean, noisy, snr_db=snr, seed=20261031 + index * 100 + snr)
            items.append({"scenario": f"noise{snr}", "index": index, "path": str(noisy), "transcription": row["transcription"]})

# Ten-minute-class long-form input: concatenate all clips with 250 ms gaps.
long_samples = array.array("h")
silence_025 = array.array("h", [0]) * 4000
long_text = []
for index, reference in enumerate(references):
    if index:
        long_samples.extend(silence_025)
    long_samples.extend(read_samples(AUDIO / f"clean_{index:03d}.wav"))
    long_text.append(reference["transcription"])
long_path = AUDIO / "long_000.wav"
write_samples(long_path, long_samples)
items.append({"scenario": "long", "index": 0, "path": str(long_path), "transcription": " ".join(long_text)})

# Pick four distinct speakers for sequential and overlapping diarization fixtures.
unique = []
seen = set()
for index, reference in enumerate(references):
    speaker = reference.get("speaker_id") or f"row-{index}"
    if speaker not in seen:
        unique.append(index)
        seen.add(speaker)
    if len(unique) == 4:
        break
if len(unique) < 4:
    unique = [0, 1, 2, 3]

seq_samples = array.array("h")
seq_rttm = []
seq_cursor = 0.0
silence_05 = array.array("h", [0]) * 8000
for speaker_number, clip_index in enumerate(unique):
    samples = read_samples(AUDIO / f"clean_{clip_index:03d}.wav")
    if speaker_number:
        seq_samples.extend(silence_05)
        seq_cursor += 0.5
    clip_duration = len(samples) / 16000
    seq_rttm.append(
        f"SPEAKER diarization-sequential 1 {seq_cursor:.6f} {clip_duration:.6f} <NA> <NA> speaker{speaker_number} <NA> <NA>"
    )
    seq_samples.extend(samples)
    seq_cursor += clip_duration
write_samples(AUDIO / "diarization-sequential.wav", seq_samples)
(REFS / "diarization-sequential.rttm").write_text("\n".join(seq_rttm) + "\n", encoding="utf-8")

first, second = unique[:2]
first_samples = read_samples(AUDIO / f"clean_{first:03d}.wav")
second_samples = read_samples(AUDIO / f"clean_{second:03d}.wav")
overlap_start = 2.0
overlap_offset = int(overlap_start * 16000)
length = max(len(first_samples), overlap_offset + len(second_samples))
overlap_samples = array.array("h", [0]) * length
for pos, sample in enumerate(first_samples):
    overlap_samples[pos] = sample
for pos, sample in enumerate(second_samples, start=overlap_offset):
    overlap_samples[pos] = max(-32768, min(32767, overlap_samples[pos] + sample))
write_samples(AUDIO / "diarization-overlap.wav", overlap_samples)
overlap_rttm = (
    f"SPEAKER diarization-overlap 1 0.000000 {len(first_samples)/16000:.6f} <NA> <NA> speaker0 <NA> <NA>\n"
    f"SPEAKER diarization-overlap 1 {overlap_start:.6f} {len(second_samples)/16000:.6f} <NA> <NA> speaker1 <NA> <NA>\n"
)
(REFS / "diarization-overlap.rttm").write_text(overlap_rttm, encoding="utf-8")

(OUT / "references.json").write_text(json.dumps(references, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(OUT / "items.json").write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(OUT / "dataset.json").write_text(
    json.dumps(
        {
            "dataset": "google/fleurs",
            "config": "de_de",
            "split": "validation",
            "row_indices": list(range(COUNT)),
            "clean_clips": COUNT,
            "noise_clips_per_snr": NOISE_COUNT,
            "noise_type": "deterministic white Gaussian",
            "noise_snr_db": [10, 5],
            "long_seconds": len(long_samples) / 16000,
            "diarization_clip_indices": unique,
        },
        ensure_ascii=False,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
print(json.dumps({"items": len(items), "long_seconds": len(long_samples) / 16000, "diarization_clips": unique}))
