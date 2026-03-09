import csv
import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate


def parse_rttm_file(rttm_path: Path) -> Annotation:
	"""Parse a single RTTM file into a pyannote.core.Annotation.

	Expected RTTM format lines (SPEAKER entries):
	SPEAKER <file-id> <channel> <onset> <duration> <ortho> <stype> <name> <confidence>

	Only <file-id>, <onset>, <duration>, and <name> are used to build annotations.
	"""
	ann = Annotation()
	with rttm_path.open("r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line or line.startswith("#"):
				continue
			parts = line.split()
			if len(parts) < 9:
				# Skip malformed lines
				continue
			kind = parts[0]
			if kind != "SPEAKER":
				# Only process diarization speaker lines
				continue
			# rttm fields
			# 0: SPEAKER
			# 1: file-id
			# 2: channel
			# 3: onset
			# 4: duration
			# 5: ortho
			# 6: stype
			# 7: name (speaker label)
			# 8: confidence
			try:
				onset = float(parts[3])
				duration = float(parts[4])
			except ValueError:
				continue
			label = parts[7]
			segment = Segment(onset, onset + duration)
			ann[segment] = label
	return ann


def _normalize_name(name: str) -> str:
	"""Normalize RTTM filename by removing optional `_MS###` suffix before extension.

	Example:
	  R8002_M8002_MS802.rttm -> R8002_M8002
	  R8002_M8002.rttm       -> R8002_M8002
	"""
	stem = name
	if stem.endswith(".rttm"):
		stem = stem[:-5]
	# remove trailing _MS<digits> if present
	if "_MS" in stem:
		prefix, suffix = stem.rsplit("_MS", 1)
		if suffix.isdigit():
			stem = prefix
	return stem


def collect_rttm_pairs(ref_dir: Path, hyp_dir: Path) -> List[Tuple[Path, Path]]:
	"""Match RTTM files across directories ignoring `_MS###` suffix differences."""
	ref_files: Dict[str, Path] = {}
	for p in ref_dir.glob("*.rttm"):
		ref_files[_normalize_name(p.name)] = p
	pairs: List[Tuple[Path, Path]] = []
	for h in hyp_dir.glob("*.rttm"):
		key = _normalize_name(h.name)
		if key in ref_files:
			pairs.append((ref_files[key], h))
	return sorted(pairs, key=lambda x: _normalize_name(x[0].name))


def compute_metrics(ref_dir: Path, hyp_dir: Path):
	metric = DiarizationErrorRate()
	filenames: List[str] = []
	ders: List[float] = []
	confusions: List[float] = []
	false_alarms: List[float] = []
	missed_detections: List[float] = []
	speech_total: List[float] = []

	pairs = collect_rttm_pairs(ref_dir, hyp_dir)
	if not pairs:
		raise FileNotFoundError("No matching RTTM files found between reference and hypothesis directories.")

	for ref_path, hyp_path in pairs:
		reference = parse_rttm_file(ref_path)
		hypothesis = parse_rttm_file(hyp_path)

		filenames.append(_normalize_name(ref_path.name))
		der = metric(reference, hypothesis)
		ders.append(der)

		components = metric.compute_components(reference, hypothesis)
		# components keys expected: 'confusion', 'false alarm', 'missed detection'
		st = float(components.get("total", 0.0) or 0.0)
		confusion_time = float(components.get("confusion", 0.0) or 0.0)
		fa_time = float(components.get("false alarm", 0.0) or 0.0)
		miss_time = float(components.get("missed detection", 0.0) or 0.0)

		confusions.append((confusion_time / st) if st > 0 else 0.0)
		false_alarms.append((fa_time / st) if st > 0 else 0.0)
		missed_detections.append((miss_time / st) if st > 0 else 0.0)
		speech_total.append(st)
		

	return {
		"filename": filenames,
		"der": ders,
		"confusion": confusions,
		"false_alarm": false_alarms,
		"missed_detection": missed_detections,
		"speech_total": speech_total,
	}


def save_csv(values: Dict[str, List[float]], out_path: Path):
	"""Save per-file metrics to a CSV file, sorted by DER descending."""
	filenames = values.get("filename", [])
	ders = values.get("der", [])
	confusions = values.get("confusion", [])
	false_alarms = values.get("false_alarm", [])
	missed_detections = values.get("missed_detection", [])

	rows = list(zip(filenames, ders, confusions, false_alarms, missed_detections))
	rows.sort(key=lambda r: r[1], reverse=True)

	out_path.parent.mkdir(parents=True, exist_ok=True)
	with out_path.open("w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(["filename", "der", "confusion", "false_alarm", "missed_detection"])
		for row in rows:
			writer.writerow([row[0], f"{row[1]:.4f}", f"{row[2]:.4f}", f"{row[3]:.4f}", f"{row[4]:.4f}"])


def plot_histograms(values: Dict[str, List[float]], out_path: Path):
	"""Plot 2x2 histograms: DER, confusion, false alarm, missed detection.

	Title of each subplot shows the average value.
	"""
	fig, axes = plt.subplots(2, 2, figsize=(10, 8))

	keys = ["der", "confusion", "false_alarm", "missed_detection"]
	titles = {
		"der": "DER",
		"confusion": "Confusion",
		"false_alarm": "False Alarm",
		"missed_detection": "Missed Detection",
	}

	weights = values.get("speech_total", [])

	def weighted_average(v: List[float], w: List[float]) -> float:
		if not v or not w:
			return 0.0
		sw = sum(w)
		if sw == 0:
			return 0.0
		return sum(vi * wi for vi, wi in zip(v, w)) / sw

	for ax, key in zip(axes.flatten(), keys):
		vals = values.get(key, [])
		avg = weighted_average(vals, weights)
		ax.hist(vals, bins=20, edgecolor="black")
		ax.set_xlabel("{} value".format(titles[key]))
		ax.set_ylabel("File count")
		ax.set_title(f"Average {titles[key]}: {avg:.4f}")
		ax.grid(True, linestyle=":", alpha=0.3)

	fig.tight_layout()
	out_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_path)
	plt.close(fig)


def main():
	parser = argparse.ArgumentParser(description="Compute diarization error rate (DER) histograms from RTTM directories.")
	parser.add_argument("--ref_dir", type=str, required=True, help="Directory with reference RTTM files.")
	parser.add_argument("--hyp_dir", type=str, required=True, help="Directory with hypothesis/predicted RTTM files.")
	parser.add_argument(
		"--out_dir", type=str, default=".", help="Output directory for results (der_hist.png and der_results.csv)."
	)
	args = parser.parse_args()

	ref_dir = Path(args.ref_dir)
	hyp_dir = Path(args.hyp_dir)
	out_dir = Path(args.out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	metrics = compute_metrics(ref_dir, hyp_dir)
	plot_histograms(metrics, out_dir / "der_hist.png")
	save_csv(metrics, out_dir / "der_results.csv")

	# Print weighted averages (by speech total) for quick CLI feedback
	weights = metrics.get("speech_total", [])
	sw = sum(weights) if weights else 0.0
	def wavg(arr: List[float]) -> float:
		if not arr or sw == 0:
			return 0.0
		return sum(a * w for a, w in zip(arr, weights)) / sw

	print(f"Weighted average der: {wavg(metrics['der']):.4f}")
	print(f"Weighted average confusion: {wavg(metrics['confusion']):.4f}")
	print(f"Weighted average false_alarm: {wavg(metrics['false_alarm']):.4f}")
	print(f"Weighted average missed_detection: {wavg(metrics['missed_detection']):.4f}")


if __name__ == "__main__":
	main()
