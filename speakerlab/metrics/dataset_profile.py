import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import yaml


SegmentRecord = Tuple[float, float, str]

SUPPORTED_ANNOTATION_EXTS = {".rttm", ".textgrid"}
COMMON_AUDIO_EXTS = [".wav", ".flac", ".mp3", ".m4a", ".ogg", ".sph"]


def load_dataset_config(config_path: Path) -> Dict[str, List[Dict[str, str]]]:
    with config_path.open("r", encoding="utf-8") as file_obj:
        config = yaml.load(file_obj, Loader=yaml.FullLoader)
    assert isinstance(config, dict), "dataset.yaml must be a dict keyed by dataset name."
    return config


def read_wav_list(wav_list_path: Path) -> List[str]:
    with wav_list_path.open("r", encoding="utf-8") as file_obj:
        rows = [line.strip() for line in file_obj if line.strip()]
    assert rows, f"Empty wav list: {wav_list_path}"
    return rows


def normalize_dataset_item(dataset_name: str, dataset_items: List[Dict[str, str]]) -> Dict[str, str]:
    assert isinstance(dataset_items, list), f"Dataset config for {dataset_name} must be a list."
    merged_item: Dict[str, str] = {}
    for item in dataset_items:
        assert isinstance(item, dict) and len(item) == 1, f"Invalid dataset entry for {dataset_name}: {item}"
        merged_item.update(item)
    required_keys = {"wav_dir", "wav_list", "ref_rttms"}
    missing_keys = required_keys.difference(merged_item.keys())
    assert not missing_keys, f"Missing keys for {dataset_name}: {sorted(missing_keys)}"
    return merged_item


def resolve_audio_path(raw_entry: str, wav_dir: Path) -> Tuple[str, Path]:
    parts = raw_entry.split()
    assert parts, "wav_list contains an empty row."

    if len(parts) == 1:
        utt_id = Path(parts[0]).stem
        path_token = parts[0]
    else:
        utt_id = parts[0]
        path_token = parts[-1]

    base_path = Path(path_token)
    if not base_path.is_absolute():
        base_path = wav_dir / path_token
    if base_path.exists():
        return utt_id, base_path.resolve()

    assert base_path.suffix == "", f"Audio file not found: {base_path}"

    for extension in COMMON_AUDIO_EXTS:
        candidate = base_path.with_suffix(extension)
        if candidate.exists():
            return utt_id, candidate.resolve()

    for extension in COMMON_AUDIO_EXTS:
        candidate = (wav_dir / utt_id).with_suffix(extension)
        if candidate.exists():
            return utt_id, candidate.resolve()

    raise FileNotFoundError(f"Audio file not found for row: {raw_entry}")


def build_annotation_index(ref_dir: Path) -> Dict[str, Path]:
    assert ref_dir.exists(), f"Annotation directory does not exist: {ref_dir}"
    files = [path for path in ref_dir.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_ANNOTATION_EXTS]
    assert files, f"No RTTM/TextGrid files found under: {ref_dir}"

    index: Dict[str, Path] = {}
    for path in sorted(files):
        index[path.stem.lower()] = path
    return index


def resolve_annotation_path(annotation_index: Dict[str, Path], candidates: Sequence[str]) -> Path | None:
    for candidate in candidates:
        key = candidate.lower()
        if key in annotation_index:
            return annotation_index[key]
    return None


def parse_rttm(annotation_path: Path) -> List[SegmentRecord]:
    segments: List[SegmentRecord] = []
    with annotation_path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            row = line.strip()
            if not row or row.startswith("#"):
                continue
            parts = row.split()
            assert len(parts) >= 8, f"Malformed RTTM line in {annotation_path}: {row}"
            assert parts[0] == "SPEAKER", f"Unsupported RTTM row in {annotation_path}: {row}"
            start_time = float(parts[3])
            duration = float(parts[4])
            end_time = start_time + duration
            speaker = parts[7]
            assert end_time >= start_time, f"Invalid RTTM segment in {annotation_path}: {row}"
            if duration > 0:
                segments.append((start_time, end_time, speaker))
    assert segments, f"No valid RTTM segments found in {annotation_path}"
    return sorted(segments, key=lambda item: (item[0], item[1], item[2]))


def _extract_quoted_value(line: str) -> str:
    first_quote = line.find('"')
    last_quote = line.rfind('"')
    assert first_quote >= 0 and last_quote > first_quote, f"Expected quoted value: {line}"
    return line[first_quote + 1:last_quote]


def parse_textgrid(annotation_path: Path) -> List[SegmentRecord]:
    lines = annotation_path.read_text(encoding="utf-8").splitlines()
    segments: List[SegmentRecord] = []
    current_tier_class = ""
    current_tier_name = ""
    index = 0

    while index < len(lines):
        row = lines[index].strip()
        if row.startswith("class ="):
            current_tier_class = _extract_quoted_value(row)
        elif row.startswith("name ="):
            current_tier_name = _extract_quoted_value(row)
        elif row.startswith("intervals ["):
            assert index + 3 < len(lines), f"Incomplete TextGrid interval in {annotation_path}"
            xmin_row = lines[index + 1].strip()
            xmax_row = lines[index + 2].strip()
            text_row = lines[index + 3].strip()
            assert xmin_row.startswith("xmin ="), f"Malformed TextGrid xmin in {annotation_path}: {xmin_row}"
            assert xmax_row.startswith("xmax ="), f"Malformed TextGrid xmax in {annotation_path}: {xmax_row}"
            assert text_row.startswith("text ="), f"Malformed TextGrid text in {annotation_path}: {text_row}"

            if current_tier_class == "IntervalTier":
                start_time = float(xmin_row.split("=", 1)[1].strip())
                end_time = float(xmax_row.split("=", 1)[1].strip())
                text_value = _extract_quoted_value(text_row)
                speaker = current_tier_name if current_tier_name else text_value
                if text_value and end_time > start_time:
                    segments.append((start_time, end_time, speaker))
            index += 3
        index += 1

    assert segments, f"No valid TextGrid segments found in {annotation_path}"
    return sorted(segments, key=lambda item: (item[0], item[1], item[2]))


def parse_annotation(annotation_path: Path) -> List[SegmentRecord]:
    suffix = annotation_path.suffix.lower()
    if suffix == ".rttm":
        return parse_rttm(annotation_path)
    if suffix == ".textgrid":
        return parse_textgrid(annotation_path)
    raise ValueError(f"Unsupported annotation file: {annotation_path}")


def get_audio_duration(audio_path: Path) -> float:
    info = sf.info(str(audio_path))
    assert info.samplerate > 0, f"Invalid samplerate in {audio_path}"
    return float(info.frames) / float(info.samplerate)


def merge_intervals(intervals: Iterable[Tuple[float, float]]) -> List[Tuple[float, float]]:
    sorted_intervals = sorted(intervals, key=lambda item: (item[0], item[1]))
    if not sorted_intervals:
        return []

    merged: List[List[float]] = [[sorted_intervals[0][0], sorted_intervals[0][1]]]
    for start_time, end_time in sorted_intervals[1:]:
        last_interval = merged[-1]
        if start_time <= last_interval[1]:
            last_interval[1] = max(last_interval[1], end_time)
        else:
            merged.append([start_time, end_time])
    return [(item[0], item[1]) for item in merged]


def compute_timeline_statistics(segments: List[SegmentRecord]) -> Dict[str, object]:
    events: List[Tuple[float, int, str]] = []
    speaker_duration = Counter()
    speaker_segment_count = Counter()
    speaker_segment_durations: Dict[str, List[float]] = defaultdict(list)
    base_intervals: List[Tuple[float, float]] = []

    for start_time, end_time, speaker in segments:
        duration = end_time - start_time
        speaker_duration[speaker] += duration
        speaker_segment_count[speaker] += 1
        speaker_segment_durations[speaker].append(duration)
        base_intervals.append((start_time, end_time))
        events.append((start_time, 1, speaker))
        events.append((end_time, -1, speaker))

    events.sort(key=lambda item: (item[0], item[1]))

    active_counts = Counter()
    active_speakers = 0
    prev_time = events[0][0]
    speech_duration = 0.0
    overlap_duration = 0.0
    single_speaker_duration = 0.0
    concurrency_duration = Counter()
    speech_regions = 0
    overlap_regions = 0
    longest_overlap = 0.0
    current_overlap = 0.0
    current_speech = 0.0
    longest_speech = 0.0
    max_concurrency = 0

    for timestamp, delta, speaker in events:
        interval_duration = timestamp - prev_time
        if interval_duration > 0:
            concurrency = active_speakers
            concurrency_duration[concurrency] += interval_duration
            max_concurrency = max(max_concurrency, concurrency)
            if concurrency > 0:
                speech_duration += interval_duration
                current_speech += interval_duration
            if concurrency == 0 and current_speech > 0:
                speech_regions += 1
                longest_speech = max(longest_speech, current_speech)
                current_speech = 0.0
            if concurrency == 1:
                single_speaker_duration += interval_duration
            if concurrency > 1:
                overlap_duration += interval_duration
                current_overlap += interval_duration
            if concurrency <= 1 and current_overlap > 0:
                overlap_regions += 1
                longest_overlap = max(longest_overlap, current_overlap)
                current_overlap = 0.0
        active_counts[speaker] += delta
        if active_counts[speaker] == 0:
            del active_counts[speaker]
        active_speakers = len(active_counts)
        prev_time = timestamp

    if current_speech > 0:
        speech_regions += 1
        longest_speech = max(longest_speech, current_speech)
    if current_overlap > 0:
        overlap_regions += 1
        longest_overlap = max(longest_overlap, current_overlap)

    merged_intervals = merge_intervals(base_intervals)
    return {
        "speech_duration": speech_duration,
        "overlap_duration": overlap_duration,
        "single_speaker_duration": single_speaker_duration,
        "max_concurrency": max_concurrency,
        "speech_regions": speech_regions,
        "overlap_regions": overlap_regions,
        "longest_overlap": longest_overlap,
        "longest_speech": longest_speech,
        "merged_intervals": merged_intervals,
        "speaker_duration": dict(speaker_duration),
        "speaker_segment_count": dict(speaker_segment_count),
        "speaker_segment_durations": dict(speaker_segment_durations),
        "concurrency_duration": dict(concurrency_duration),
    }


def compute_longest_silence(merged_intervals: List[Tuple[float, float]], audio_duration: float) -> float:
    if not merged_intervals:
        return audio_duration

    longest_silence = max(0.0, merged_intervals[0][0])
    prev_end = merged_intervals[0][1]
    for start_time, end_time in merged_intervals[1:]:
        longest_silence = max(longest_silence, start_time - prev_end)
        prev_end = end_time
    longest_silence = max(longest_silence, audio_duration - prev_end)
    return longest_silence


def compute_entropy(values: Sequence[float]) -> float:
    positive_values = [value for value in values if value > 0]
    if not positive_values:
        return 0.0
    total = sum(positive_values)
    probabilities = [value / total for value in positive_values]
    entropy = -sum(probability * math.log(probability) for probability in probabilities)
    if len(probabilities) == 1:
        return 1.0
    return entropy / math.log(len(probabilities))


def summarise_segments(file_id: str, audio_path: Path, annotation_path: Path, segments: List[SegmentRecord]) -> Tuple[Dict[str, float], List[Dict[str, float]], Dict[int, float]]:
    audio_duration = get_audio_duration(audio_path)
    annotation_end = max(end_time for _, end_time, _ in segments)
    timeline_stats = compute_timeline_statistics(segments)
    speaker_duration = timeline_stats["speaker_duration"]
    speaker_segment_count = timeline_stats["speaker_segment_count"]
    speaker_segment_durations = timeline_stats["speaker_segment_durations"]
    merged_intervals = timeline_stats["merged_intervals"]
    segment_lengths = [end_time - start_time for start_time, end_time, _ in segments]
    speaker_total_lengths = list(speaker_duration.values())
    speaker_total_segments = list(speaker_segment_count.values())

    speech_duration = float(timeline_stats["speech_duration"])
    overlap_duration = float(timeline_stats["overlap_duration"])
    single_speaker_duration = float(timeline_stats["single_speaker_duration"])
    silence_duration = max(0.0, audio_duration - speech_duration)
    total_segment_duration = float(sum(segment_lengths))
    dominant_speaker_duration = max(speaker_total_lengths)
    longest_silence = compute_longest_silence(merged_intervals, audio_duration)
    longest_segment = max(segment_lengths)
    shortest_segment = min(segment_lengths)
    avg_speaker_duration = float(np.mean(speaker_total_lengths))
    median_speaker_duration = float(np.median(speaker_total_lengths))
    avg_speaker_segment_count = float(np.mean(speaker_total_segments))
    avg_active_speakers = total_segment_duration / speech_duration if speech_duration > 0 else 0.0

    file_stats = {
        "file_id": file_id,
        "audio_path": str(audio_path),
        "annotation_path": str(annotation_path),
        "annotation_type": annotation_path.suffix.lower().lstrip("."),
        "audio_duration": audio_duration,
        "annotation_end": annotation_end,
        "annotation_coverage_ratio": annotation_end / audio_duration if audio_duration > 0 else 0.0,
        "num_segments": float(len(segments)),
        "num_speakers": float(len(speaker_duration)),
        "segment_rate_per_min": len(segments) / (audio_duration / 60.0) if audio_duration > 0 else 0.0,
        "speech_duration": speech_duration,
        "speech_ratio": speech_duration / audio_duration if audio_duration > 0 else 0.0,
        "overlap_duration": overlap_duration,
        "overlap_ratio": overlap_duration / audio_duration if audio_duration > 0 else 0.0,
        "overlap_in_speech_ratio": overlap_duration / speech_duration if speech_duration > 0 else 0.0,
        "single_speaker_duration": single_speaker_duration,
        "single_speaker_ratio": single_speaker_duration / audio_duration if audio_duration > 0 else 0.0,
        "silence_duration": silence_duration,
        "silence_ratio": silence_duration / audio_duration if audio_duration > 0 else 0.0,
        "speech_region_count": float(timeline_stats["speech_regions"]),
        "overlap_region_count": float(timeline_stats["overlap_regions"]),
        "max_concurrency": float(timeline_stats["max_concurrency"]),
        "avg_active_speakers_during_speech": avg_active_speakers,
        "avg_segment_duration": float(np.mean(segment_lengths)),
        "median_segment_duration": float(np.median(segment_lengths)),
        "max_segment_duration": longest_segment,
        "min_segment_duration": shortest_segment,
        "longest_speech_region": float(timeline_stats["longest_speech"]),
        "longest_overlap_region": float(timeline_stats["longest_overlap"]),
        "longest_silence": longest_silence,
        "total_segment_duration": total_segment_duration,
        "avg_speaker_duration": avg_speaker_duration,
        "median_speaker_duration": median_speaker_duration,
        "dominant_speaker_duration": dominant_speaker_duration,
        "dominant_speaker_ratio": dominant_speaker_duration / total_segment_duration if total_segment_duration > 0 else 0.0,
        "speaker_duration_entropy": compute_entropy(speaker_total_lengths),
        "avg_segments_per_speaker": avg_speaker_segment_count,
    }

    speaker_rows: List[Dict[str, float]] = []
    for speaker, total_duration in sorted(speaker_duration.items(), key=lambda item: (-item[1], item[0])):
        durations = speaker_segment_durations[speaker]
        speaker_rows.append({
            "file_id": file_id,
            "speaker": speaker,
            "speaker_total_duration": total_duration,
            "speaker_duration_ratio": total_duration / total_segment_duration if total_segment_duration > 0 else 0.0,
            "speaker_segment_count": float(speaker_segment_count[speaker]),
            "speaker_avg_segment_duration": float(np.mean(durations)),
            "speaker_median_segment_duration": float(np.median(durations)),
            "speaker_max_segment_duration": max(durations),
            "speaker_min_segment_duration": min(durations),
        })

    concurrency_duration = timeline_stats["concurrency_duration"]
    return file_stats, speaker_rows, {int(key): float(value) for key, value in concurrency_duration.items()}


def weighted_average(values: Sequence[float], weights: Sequence[float]) -> float:
    assert len(values) == len(weights), "Values and weights must have the same length."
    weight_sum = float(sum(weights))
    if weight_sum == 0:
        return 0.0
    return float(sum(value * weight for value, weight in zip(values, weights))) / weight_sum


def build_dataset_summary(dataset_name: str, file_rows: List[Dict[str, float]], concurrency_duration: Dict[int, float]) -> Dict[str, float]:
    audio_durations = [row["audio_duration"] for row in file_rows]
    speech_durations = [row["speech_duration"] for row in file_rows]
    total_audio_duration = float(sum(audio_durations))
    total_speech_duration = float(sum(speech_durations))
    total_overlap_duration = float(sum(row["overlap_duration"] for row in file_rows))
    total_silence_duration = float(sum(row["silence_duration"] for row in file_rows))
    total_files = len(file_rows)

    summary = {
        "dataset": dataset_name,
        "num_files": float(total_files),
        "total_audio_hours": total_audio_duration / 3600.0,
        "total_speech_hours": total_speech_duration / 3600.0,
        "total_overlap_hours": total_overlap_duration / 3600.0,
        "total_silence_hours": total_silence_duration / 3600.0,
        "avg_audio_duration": float(np.mean(audio_durations)),
        "median_audio_duration": float(np.median(audio_durations)),
        "p90_audio_duration": float(np.percentile(audio_durations, 90)),
        "avg_num_speakers": float(np.mean([row["num_speakers"] for row in file_rows])),
        "median_num_speakers": float(np.median([row["num_speakers"] for row in file_rows])),
        "avg_num_segments": float(np.mean([row["num_segments"] for row in file_rows])),
        "median_num_segments": float(np.median([row["num_segments"] for row in file_rows])),
        "weighted_speech_ratio": weighted_average([row["speech_ratio"] for row in file_rows], audio_durations),
        "weighted_overlap_ratio": weighted_average([row["overlap_ratio"] for row in file_rows], audio_durations),
        "weighted_overlap_in_speech_ratio": weighted_average(
            [row["overlap_in_speech_ratio"] for row in file_rows],
            speech_durations,
        ),
        "weighted_silence_ratio": weighted_average([row["silence_ratio"] for row in file_rows], audio_durations),
        "weighted_dominant_speaker_ratio": weighted_average(
            [row["dominant_speaker_ratio"] for row in file_rows],
            [row["total_segment_duration"] for row in file_rows],
        ),
        "avg_segment_duration": float(np.mean([row["avg_segment_duration"] for row in file_rows])),
        "avg_active_speakers_during_speech": weighted_average(
            [row["avg_active_speakers_during_speech"] for row in file_rows],
            speech_durations,
        ),
        "max_concurrency_observed": float(max(row["max_concurrency"] for row in file_rows)),
        "concurrency_duration_ge2_hours": sum(duration for count, duration in concurrency_duration.items() if count >= 2) / 3600.0,
        "concurrency_duration_ge3_hours": sum(duration for count, duration in concurrency_duration.items() if count >= 3) / 3600.0,
    }
    return summary


def write_csv(rows: List[Dict[str, float]], output_path: Path) -> None:
    assert rows, f"No rows to write: {output_path}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_file_level_distributions(file_rows: List[Dict[str, float]], output_path: Path, ignore_overlap: bool = False) -> None:
    metrics = [
        ("audio_duration", "Audio Duration (s)"),
        ("num_speakers", "Speaker Count"),
        ("num_segments", "Segment Count"),
        ("speech_ratio", "Speech Ratio"),
        ("overlap_ratio", "Overlap Ratio"),
        ("overlap_in_speech_ratio", "Overlap / Speech"),
        ("avg_segment_duration", "Avg Segment Duration (s)"),
        ("dominant_speaker_ratio", "Dominant Speaker Ratio"),
        ("longest_silence", "Longest Silence (s)"),
    ]
    if ignore_overlap:
        metrics = [m for m in metrics if "overlap" not in m[0] and "num_segments" not in m[0]]

    num_metrics = len(metrics)
    cols = 3
    rows = math.ceil(num_metrics / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
    axes_flat = np.array(axes).flatten()

    for idx, (key, title) in enumerate(metrics):
        axis = axes_flat[idx]
        values = [row[key] for row in file_rows]
        axis.hist(values, bins=20, edgecolor="black")
        axis.set_title(f"{title}\nmean={np.mean(values):.4f}, median={np.median(values):.4f}")
        axis.set_xlabel(title)
        axis.set_ylabel("File Count")
        axis.grid(True, linestyle=":", alpha=0.3)

    for idx in range(num_metrics, len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_relationships(file_rows: List[Dict[str, float]], output_path: Path, ignore_overlap: bool = False) -> None:
    pairs = [
        ("audio_duration", "num_speakers", "Audio Duration (s)", "Speaker Count"),
        ("audio_duration", "overlap_ratio", "Audio Duration (s)", "Overlap Ratio"),
        ("num_speakers", "overlap_ratio", "Speaker Count", "Overlap Ratio"),
        ("num_segments", "speech_ratio", "Segment Count", "Speech Ratio"),
    ]
    if ignore_overlap:
        pairs = [p for p in pairs if "overlap" not in p[1]]

    num_pairs = len(pairs)
    cols = 2
    rows = math.ceil(num_pairs / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
    axes_flat = np.array(axes).flatten()

    for idx, (x_key, y_key, x_label, y_label) in enumerate(pairs):
        axis = axes_flat[idx]
        x_values = [row[x_key] for row in file_rows]
        y_values = [row[y_key] for row in file_rows]
        axis.scatter(x_values, y_values, alpha=0.7)
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.grid(True, linestyle=":", alpha=0.3)

    for idx in range(num_pairs, len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_speaker_level_distributions(speaker_rows: List[Dict[str, float]], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = [
        ("speaker_total_duration", "Speaker Total Duration (s)"),
        ("speaker_segment_count", "Speaker Segment Count"),
        ("speaker_avg_segment_duration", "Speaker Avg Segment Duration (s)"),
        ("speaker_duration_ratio", "Speaker Duration Ratio"),
    ]
    for axis, (key, title) in zip(axes.flatten(), metrics):
        values = [row[key] for row in speaker_rows]
        axis.hist(values, bins=20, edgecolor="black")
        axis.set_title(f"{title}\nmean={np.mean(values):.4f}, median={np.median(values):.4f}")
        axis.set_xlabel(title)
        axis.set_ylabel("Speaker Count")
        axis.grid(True, linestyle=":", alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_concurrency_distribution(concurrency_duration: Dict[int, float], output_path: Path) -> None:
    sorted_items = sorted(concurrency_duration.items(), key=lambda item: item[0])
    counts = [item[0] for item in sorted_items]
    hours = [item[1] / 3600.0 for item in sorted_items]
    fig, axis = plt.subplots(figsize=(10, 6))
    axis.bar(counts, hours, width=0.8)
    axis.set_xlabel("Active Speaker Count")
    axis.set_ylabel("Duration (hours)")
    axis.set_title("Concurrency Duration Distribution")
    axis.grid(True, axis="y", linestyle=":", alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_cross_dataset_summary(dataset_rows: List[Dict[str, float]], output_path: Path, ignore_overlap: bool = False) -> None:
    dataset_names = [row["dataset"] for row in dataset_rows]
    metrics = [
        ("total_audio_hours", "Total Audio Duration (hours)"),
        ("avg_num_speakers", "Avg Speaker Count"),
        ("weighted_speech_ratio", "Weighted Speech Ratio"),
        ("weighted_overlap_ratio", "Weighted Overlap Ratio"),
    ]
    if ignore_overlap:
        metrics = [m for m in metrics if "overlap" not in m[0]]

    num_metrics = len(metrics)
    cols = 2
    if ignore_overlap:
        cols = 3
    rows = math.ceil(num_metrics / cols)
    if rows == 0:
        return
    fig, axes = plt.subplots(rows, cols, figsize=(16, 5 * rows))
    axes_flat = np.array(axes).flatten()

    for idx, (key, title) in enumerate(metrics):
        axis = axes_flat[idx]
        values = [row[key] for row in dataset_rows]
        axis.bar(dataset_names, values)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=20)
        axis.grid(True, axis="y", linestyle=":", alpha=0.3)

    for idx in range(num_metrics, len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def profile_dataset(dataset_name: str, dataset_items: List[Dict[str, str]], base_output_dir: Path, ignore_overlap: bool = False) -> Dict[str, float]:
    assert dataset_items, f"Dataset config is empty: {dataset_name}"
    dataset_item = normalize_dataset_item(dataset_name, dataset_items)
    wav_dir = Path(dataset_item["wav_dir"])
    wav_list_path = Path(dataset_item["wav_list"])
    ref_dir = Path(dataset_item["ref_rttms"])

    rows = read_wav_list(wav_list_path)
    annotation_index = build_annotation_index(ref_dir)
    file_rows: List[Dict[str, float]] = []
    speaker_rows: List[Dict[str, float]] = []
    concurrency_duration = Counter()

    for raw_entry in rows:
        file_id, audio_path = resolve_audio_path(raw_entry, wav_dir)
        annotation_path = resolve_annotation_path(
            annotation_index,
            [file_id, audio_path.stem, audio_path.name],
        )
        if annotation_path is None:
            print(f"Skip {audio_path}: annotation not found.")
            continue
        segments = parse_annotation(annotation_path)
        file_stats, per_speaker_rows, file_concurrency = summarise_segments(file_id, audio_path, annotation_path, segments)
        file_stats["dataset"] = dataset_name
        file_rows.append(file_stats)
        for row in per_speaker_rows:
            row["dataset"] = dataset_name
        speaker_rows.extend(per_speaker_rows)
        concurrency_duration.update(file_concurrency)

    assert file_rows, f"No matched audio/annotation pairs found for dataset: {dataset_name}"

    dataset_out_dir = base_output_dir / dataset_name
    write_csv(file_rows, dataset_out_dir / "file_statistics.csv")
    write_csv(speaker_rows, dataset_out_dir / "speaker_statistics.csv")
    plot_file_level_distributions(file_rows, dataset_out_dir / "file_distributions.png", ignore_overlap)
    plot_relationships(file_rows, dataset_out_dir / "relationship_plots.png", ignore_overlap)
    plot_speaker_level_distributions(speaker_rows, dataset_out_dir / "speaker_distributions.png")
    
    if not ignore_overlap:
        plot_concurrency_distribution(dict(concurrency_duration), dataset_out_dir / "concurrency_distribution.png")

    dataset_summary = build_dataset_summary(dataset_name, file_rows, dict(concurrency_duration))
    write_csv([dataset_summary], dataset_out_dir / "dataset_summary.csv")
    return dataset_summary


def main() -> None:
    script_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Profile diarization datasets defined in dataset.yaml.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(script_root / "config" / "dataset.yaml"),
        help="Path to dataset.yaml.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(script_root / "results" / "dataset_profile"),
        help="Directory used to save CSV summaries and matplotlib figures.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional dataset names to profile. If omitted, all datasets in dataset.yaml are processed.",
    )
    parser.add_argument(
        "--ignore_overlap",
        action="store_true",
        help="Whether to ignore overlap statistics in the output plots and omit them.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    output_dir = Path(args.out_dir)
    dataset_config = load_dataset_config(config_path)

    selected_datasets = set(args.datasets) if args.datasets else set(dataset_config.keys())
    missing_datasets = [name for name in selected_datasets if name not in dataset_config]
    assert not missing_datasets, f"Datasets not found in config: {missing_datasets}"

    dataset_summaries: List[Dict[str, float]] = []
    for dataset_name, dataset_items in dataset_config.items():
        if dataset_name not in selected_datasets:
            continue
        dataset_summaries.append(profile_dataset(dataset_name, dataset_items, output_dir, args.ignore_overlap))

    write_csv(dataset_summaries, output_dir / "all_datasets_summary.csv")
    if len(dataset_summaries) > 1:
        plot_cross_dataset_summary(dataset_summaries, output_dir / "all_datasets_summary.png", args.ignore_overlap)

    print(f"Profile finished. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()