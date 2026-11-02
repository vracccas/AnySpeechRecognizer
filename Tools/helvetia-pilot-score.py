#!/usr/bin/env python3
import json
import pathlib
import unicodedata

ROOT = pathlib.Path("pilot-benchmark")
RESULTS = pathlib.Path("pilot-results")
items = json.loads((ROOT / "items.json").read_text(encoding="utf-8"))


def normalize(text):
    text = unicodedata.normalize("NFKC", text).lower()
    text = "".join(" " if unicodedata.category(ch)[0] in {"P", "S"} else ch for ch in text)
    return text.split()


def distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for row, expected in enumerate(reference, start=1):
        current = [row]
        for column, actual in enumerate(hypothesis, start=1):
            current.append(min(current[-1] + 1, previous[column] + 1, previous[column - 1] + (expected != actual)))
        previous = current
    return previous[-1]


def read_ndjson(path):
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return records


def score(backend):
    scenarios = {}
    for item in items:
        path = RESULTS / f"{backend}_{item['scenario']}_{item['index']:03d}.ndjson"
        if not path.exists():
            continue
        records = read_ndjson(path)
        hypothesis = "".join(
            record.get("text", "")
            for record in records
            if record.get("type") == "result" and record.get("isFinal") is True
        )
        reference_words = normalize(item["transcription"])
        hypothesis_words = normalize(hypothesis)
        edits = distance(reference_words, hypothesis_words)
        evaluation = next((r for r in records if r.get("type") == "evaluation"), {})
        bucket = scenarios.setdefault(
            item["scenario"],
            {"items": [], "total_edits": 0, "reference_words": 0, "elapsed_seconds": 0.0, "peak_resident_bytes": 0},
        )
        bucket["items"].append(
            {
                "index": item["index"],
                "edits": edits,
                "reference_words": len(reference_words),
                "wer": edits / max(1, len(reference_words)),
                "hypothesis": " ".join(hypothesis_words),
            }
        )
        bucket["total_edits"] += edits
        bucket["reference_words"] += len(reference_words)
        bucket["elapsed_seconds"] += float(evaluation.get("elapsedSeconds", 0.0))
        bucket["peak_resident_bytes"] = max(bucket["peak_resident_bytes"], int(evaluation.get("peakResidentBytes", 0)))
    for bucket in scenarios.values():
        bucket["item_count"] = len(bucket["items"])
        bucket["wer"] = bucket["total_edits"] / max(1, bucket["reference_words"])
    return scenarios or None


def diarization(name):
    path = RESULTS / f"diarization-{name}.ndjson"
    if not path.exists():
        return None
    return next((r for r in read_ndjson(path) if r.get("type") == "evaluation"), None)

summary = {
    "dataset": json.loads((ROOT / "dataset.json").read_text(encoding="utf-8")),
    "v3": score("v3"),
    "v3live": score("v3live"),
    "ultra": score("ultra"),
    "diarization_sequential": diarization("sequential"),
    "diarization_overlap": diarization("overlap"),
}
pathlib.Path("pilot-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(
    json.dumps(
        {
            backend: {scenario: round(values["wer"], 6) for scenario, values in scenarios.items()}
            for backend, scenarios in (("v3", summary["v3"]), ("v3live", summary["v3live"]), ("ultra", summary["ultra"]))
            if scenarios
        }
    )
)
