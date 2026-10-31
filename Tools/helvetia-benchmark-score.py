#!/usr/bin/env python3
import json
import pathlib
import sys
import unicodedata

ROOT = pathlib.Path("benchmark")
RESULTS = pathlib.Path("results")
references = json.loads((ROOT / "references.json").read_text(encoding="utf-8"))


def normalize(text):
    text = unicodedata.normalize("NFKC", text).lower()
    text = "".join(" " if unicodedata.category(ch)[0] in {"P", "S"} else ch for ch in text)
    return text.split()


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for row, expected in enumerate(reference, start=1):
        current = [row]
        for column, actual in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (expected != actual),
                )
            )
        previous = current
    return previous[-1]


def read_ndjson(path):
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def score_backend(name):
    clips = []
    total_edits = 0
    total_words = 0
    total_elapsed = 0.0
    for reference in references:
        path = RESULTS / f"{name}_{reference['index']:02d}.ndjson"
        if not path.exists():
            continue
        records = read_ndjson(path)
        hypothesis = "".join(
            record.get("text", "")
            for record in records
            if record.get("type") == "result" and record.get("isFinal") is True
        )
        ref_words = normalize(reference["transcription"])
        hyp_words = normalize(hypothesis)
        edits = edit_distance(ref_words, hyp_words)
        metadata = next((r for r in records if r.get("type") == "metadata"), {})
        evaluation = next((r for r in records if r.get("type") == "evaluation"), {})
        elapsed = float(evaluation.get("elapsedSeconds", 0.0))
        total_edits += edits
        total_words += len(ref_words)
        total_elapsed += elapsed
        clips.append(
            {
                "index": reference["index"],
                "reference": " ".join(ref_words),
                "hypothesis": " ".join(hyp_words),
                "edits": edits,
                "reference_words": len(ref_words),
                "wer": edits / max(1, len(ref_words)),
                "elapsed_seconds": elapsed,
                "hardware": metadata.get("hardware"),
            }
        )
    if not clips:
        return None
    return {
        "clips": clips,
        "clip_count": len(clips),
        "total_edits": total_edits,
        "reference_words": total_words,
        "wer": total_edits / max(1, total_words),
        "elapsed_seconds": total_elapsed,
    }

summary = {
    "dataset": json.loads((ROOT / "dataset.json").read_text(encoding="utf-8")),
    "parakeet": score_backend("parakeet"),
    "apple": score_backend("apple"),
}
diarization = RESULTS / "diarization.ndjson"
if diarization.exists():
    records = read_ndjson(diarization)
    summary["diarization"] = next(
        (record for record in records if record.get("type") == "evaluation"), None
    )
if len(sys.argv) > 1:
    summary["apple_prepare_error"] = pathlib.Path(sys.argv[1]).read_text(
        encoding="utf-8", errors="replace"
    )[-4000:]
pathlib.Path("summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps({key: value and value.get("wer") if isinstance(value, dict) else value for key, value in summary.items() if key in {"parakeet", "apple"}}))
