from pathlib import Path
import re
import importlib.util
import numpy as np

PROJECT_ROOT = Path(r"D:\AI_FINAL_PROJECT")
THERMAL_ROOT = PROJECT_ROOT / "ThermalImagesDataset" / "ThermalImagesDataset-main"
ANNOTATIONS_FILE = THERMAL_ROOT / "Sources" / "lstm" / "annotations.py"
SEQUENCES_ROOT = THERMAL_ROOT / "Sources" / "lstm" / "data" / "sequences"
OUTPUT_FILE = PROJECT_ROOT / "outputs" / "thermal_label_audit.txt"
FILENAME_PATTERN = re.compile(r"^(?P<recording>.+)_face_\d+_(?P<start>\d+)-(?P<end>\d+)\.txt$")


def load_annotations(path):
    if not path.exists():
        raise FileNotFoundError(f"Could not find annotations file:\n{path}")
    spec = importlib.util.spec_from_file_location("thermal_annotations", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load annotations.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    candidates = [
        value for value in vars(module).values()
        if isinstance(value, dict) and value
        and any(isinstance(v, dict) and ("yawns" in v or "offset" in v) for v in value.values())
    ]
    if not candidates:
        raise RuntimeError("Could not identify the annotations dictionary in annotations.py.")
    return candidates[0]


def get_yawn_intervals(annotation):
    clean = []
    for item in annotation.get("yawns", []) or []:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            try:
                clean.append((int(item[0]), int(item[1])))
            except (TypeError, ValueError):
                pass
    return clean


def frame_labels(frame_numbers, intervals, inclusive_end):
    labels = np.zeros(len(frame_numbers), dtype=np.int64)
    for i, frame_no in enumerate(frame_numbers):
        for start, end in intervals:
            is_positive = (start <= frame_no <= end) if inclusive_end else (start <= frame_no < end)
            if is_positive:
                labels[i] = 1
                break
    return labels


def main():
    print("Thermal label audit starting...")
    if not SEQUENCES_ROOT.exists():
        raise FileNotFoundError(f"Could not find sequences folder:\n{SEQUENCES_ROOT}")
    annotations = load_annotations(ANNOTATIONS_FILE)
    print(f"Annotation recordings loaded: {len(annotations)}")

    records, unmatched, malformed = [], [], []
    for folder_name in ("normal", "yawning"):
        folder = SEQUENCES_ROOT / folder_name
        if not folder.exists():
            print(f"WARNING: missing folder: {folder}")
            continue
        for file_path in folder.glob("*.txt"):
            match = FILENAME_PATTERN.match(file_path.name)
            if not match:
                malformed.append(str(file_path))
                continue
            recording = match.group("recording")
            start = int(match.group("start"))
            annotation = annotations.get(recording)
            if annotation is None:
                unmatched.append((file_path.name, recording))
                continue
            try:
                data = np.loadtxt(file_path, delimiter=";", dtype=np.float64, ndmin=2)
            except Exception as exc:
                records.append({"file": file_path.name, "recording": recording, "status": f"READ_ERROR: {exc}"})
                continue
            if data.ndim != 2 or data.shape[1] < 4 or len(data) == 0:
                records.append({"file": file_path.name, "recording": recording, "status": f"BAD_SHAPE: {data.shape}"})
                continue
            observed = data[:, 3]
            if not np.all(np.isin(observed, [0, 1])):
                records.append({"file": file_path.name, "recording": recording, "status": "NON_BINARY_LABELS"})
                continue

            intervals = get_yawn_intervals(annotation)
            offset = int(annotation.get("offset", 0) or 0)
            candidates = []
            for offset_mode, effective_offset in (
                ("filename indices are absolute", 0),
                ("filename indices need annotation offset", offset),
            ):
                frame_numbers = np.arange(start, start + len(observed)) + effective_offset
                for endpoint_mode, inclusive in (("inclusive", True), ("exclusive", False)):
                    expected = frame_labels(frame_numbers, intervals, inclusive)
                    mismatch_count = int(np.sum(observed != expected))
                    candidates.append((mismatch_count, endpoint_mode, offset_mode, expected))
            mismatches, endpoint_mode, offset_mode, expected = min(candidates, key=lambda x: x[0])
            agreement = 100.0 * (len(observed) - mismatches) / len(observed)
            records.append({
                "file": file_path.name, "recording": recording, "folder": folder_name,
                "rows": len(observed), "observed_positive": int(np.sum(observed)),
                "expected_positive_best": int(np.sum(expected)), "mismatches_best": mismatches,
                "agreement_best_pct": agreement, "endpoint_mode_best": endpoint_mode,
                "offset_mode_best": offset_mode, "offset": offset, "status": "OK" if mismatches == 0 else "MISMATCH",
            })

    valid = [r for r in records if "mismatches_best" in r]
    if not valid:
        raise RuntimeError("No sequence files could be compared. Check annotations and filenames.")
    total_frames = sum(r["rows"] for r in valid)
    total_mismatches = sum(r["mismatches_best"] for r in valid)
    overall_agreement = 100.0 * (total_frames - total_mismatches) / total_frames
    exact_files = sum(r["mismatches_best"] == 0 for r in valid)

    lines = [
        "THERMAL SEQUENCE LABEL AUDIT", "=" * 72,
        f"Sequence files compared: {len(valid)}",
        f"Files with exact match under best tested convention: {exact_files}",
        f"Total frame rows compared: {total_frames}",
        f"Total mismatched frame labels under best tested convention: {total_mismatches}",
        f"Overall frame-level agreement under best tested convention: {overall_agreement:.4f}%",
        f"Files with recording name not found in annotations: {len(unmatched)}",
        f"Files with unrecognized filename format: {len(malformed)}", "",
        "IMPORTANT: The script chooses the best of inclusive/exclusive interval endpoints",
        "and two offset interpretations per file. Review this with the original indexing",
        "rules; best-match agreement alone does not prove the selected convention is correct.",
        "", "PER-RECORDING SUMMARY", "-" * 72,
    ]
    by_recording = {}
    for r in valid:
        agg = by_recording.setdefault(r["recording"], {"files": 0, "frames": 0, "mismatches": 0})
        agg["files"] += 1
        agg["frames"] += r["rows"]
        agg["mismatches"] += r["mismatches_best"]
    for recording, agg in sorted(by_recording.items()):
        pct = 100.0 * (agg["frames"] - agg["mismatches"]) / agg["frames"]
        lines.append(f"{recording}: files={agg['files']}, frames={agg['frames']}, mismatches={agg['mismatches']}, agreement={pct:.2f}%")

    lines += ["", "FILES WITH MISMATCHES", "-" * 72]
    mismatched = sorted((r for r in valid if r["mismatches_best"] > 0),
                        key=lambda x: x["mismatches_best"], reverse=True)
    for r in mismatched[:200]:
        lines.append(
            f"{r['file']} | mismatch={r['mismatches_best']}/{r['rows']} "
            f"({r['agreement_best_pct']:.2f}% agreement) | endpoint={r['endpoint_mode_best']} "
            f"| offset={r['offset_mode_best']}"
        )
    if len(mismatched) > 200:
        lines.append(f"... {len(mismatched) - 200} more mismatched files omitted")

    lines += ["", "UNMATCHED RECORDING NAMES (first 100)", "-" * 72]
    lines.extend(f"{name} | parsed recording={recording}" for name, recording in unmatched[:100])
    if not unmatched:
        lines.append("None")
    lines += ["", "UNRECOGNIZED FILENAMES (first 50)", "-" * 72]
    lines.extend(malformed[:50] or ["None"])

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print("\nAUDIT COMPLETE")
    print(f"Files compared: {len(valid)}")
    print(f"Exact files under best tested convention: {exact_files}/{len(valid)}")
    print(f"Overall frame-level agreement: {overall_agreement:.4f}%")
    print(f"Unmatched recordings: {len(unmatched)}")
    print(f"Report saved to: {OUTPUT_FILE}")
    print("Open the report and share its summary plus any per-recording lines with low agreement.")


if __name__ == "__main__":
    main()
