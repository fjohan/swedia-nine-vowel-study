"""Token extraction and PCA engine, bundled from the original nine-vowel workflow."""
from __future__ import annotations
import argparse
import csv
import math
import warnings
from collections import defaultdict
from pathlib import Path
import numpy as np
import parselmouth
from acoustic_helpers import locate_recording, read_wav, spectral_vector, RESOURCE_ALIASES
PERCENTAGES=(20,50,80)

def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def finite_median(values) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if len(array) else math.nan


def bark(hz: float) -> float:
    return 26.81 / (1.0 + 1960.0 / hz) - 0.53 if math.isfinite(hz) and hz > 0 else math.nan


def interpolation(times, values, query) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(times) & np.isfinite(values)
    if np.count_nonzero(valid) < 2:
        return np.full(len(query), np.nan)
    return np.interp(query, times[valid], values[valid])


def time_windows(start: float, end: float) -> dict[int, np.ndarray]:
    duration = end - start
    return {
        percentage: np.linspace(
            start + (percentage / 100.0 - 0.05) * duration,
            start + (percentage / 100.0 + 0.05) * duration,
            11,
        )
        for percentage in PERCENTAGES
    }


def fieldnames(n_bands: int) -> list[str]:
    metadata = [
        "token_id", "recording", "village", "speaker", "word",
        "lexical_status", "target_label", "ipa", "surface_seg_label",
        "vowel_start_s", "vowel_end_s", "duration_s", "geo_x", "geo_y",
        "fasttrack_ceiling_hz", "fasttrack_candidate", "fasttrack_boundary",
        "fasttrack_status", "fasttrack_error",
    ]
    measurements = [
        field
        for percentage in PERCENTAGES
        for field in (
            f"fasttrack_f1_{percentage}", f"fasttrack_f2_{percentage}",
            f"fasttrack_bark_f1_{percentage}", f"fasttrack_bark_f2_{percentage}",
        )
    ]
    spectra = [
        f"_spectrum_{percentage}_band_{band:02d}"
        for percentage in PERCENTAGES for band in range(1, n_bands + 1)
    ]
    return metadata + measurements + ["fasttrack_VL_hz", "fasttrack_VL_bark"] + spectra


def extract(args: argparse.Namespace, tokens: list[dict[str, str]],
            coordinates, filters: np.ndarray, work_file: Path) -> list[dict[str, str]]:
    try:
        from fasttrackpy import CandidateTracks
    except ImportError as error:
        raise SystemExit(
            "FastTrackPy is unavailable. Run this script with fasttrackpy/bin/python."
        ) from error

    fields = fieldnames(args.pca_bands)
    completed = set()
    if work_file.exists() and work_file.stat().st_size:
        completed = {row["token_id"] for row in read_csv(work_file)}
    new_file = not work_file.exists() or work_file.stat().st_size == 0
    grouped = defaultdict(list)
    for row in tokens:
        grouped[row["recording"]].append(row)
    recordings = sorted(grouped)
    if args.limit_recordings:
        recordings = recordings[:args.limit_recordings]
    remaining_limit = args.limit_tokens

    with work_file.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if new_file:
            writer.writeheader()
        for recording_index, recording in enumerate(recordings, 1):
            pending = [row for row in grouped[recording] if row["token_id"] not in completed]
            if remaining_limit:
                pending = pending[:remaining_limit]
            if not pending:
                continue
            wav_path = locate_recording(recording, args.media_dirs)
            sound = parselmouth.Sound(str(wav_path))
            rate, samples = read_wav(wav_path, args.pca_sample_rate)
            for source in pending:
                start, end = float(source["vowel_start_s"]), float(source["vowel_end_s"])
                windows = time_windows(start, end)
                village = source["village"]
                resource_key = RESOURCE_ALIASES.get(village, village)
                geo_x, geo_y, _ = coordinates[resource_key]
                row = {key: source.get(key, "") for key in fields}
                row.update({"geo_x": geo_x, "geo_y": geo_y})

                # Median Bark-spectrum vector within each local temporal window.
                for percentage, times in windows.items():
                    spectra = np.asarray([
                        spectral_vector(
                            samples, rate, float(time), args.pca_window_ms,
                            args.pca_n_fft, filters,
                        )[0]
                        for time in times
                    ])
                    spectrum = np.median(spectra, axis=0)
                    for band, value in enumerate(spectrum, 1):
                        row[f"_spectrum_{percentage}_band_{band:02d}"] = float(value)

                try:
                    clip = sound.extract_part(
                        max(sound.xmin, start - 0.04),
                        min(sound.xmax, end + 0.04), preserve_times=True,
                    )
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        candidates = CandidateTracks(
                            sound=clip,
                            min_max_formant=args.fasttrack_min_ceiling,
                            max_max_formant=args.fasttrack_max_ceiling,
                            nstep=args.fasttrack_steps, n_formants=4,
                            window_length=args.window_length,
                            time_step=args.time_step,
                            pre_emphasis_from=args.pre_emphasis,
                        )
                    for warning in caught:
                        print(f"  WARNING {source['token_id']}: {warning.message}", flush=True)
                    winner = candidates.winner
                    row.update({
                        "fasttrack_ceiling_hz": float(winner.maximum_formant),
                        "fasttrack_candidate": int(candidates.winner_idx) + 1,
                        "fasttrack_boundary": int(winner.maximum_formant in
                                                   (args.fasttrack_min_ceiling,
                                                    args.fasttrack_max_ceiling)),
                        "fasttrack_status": "ok", "fasttrack_error": "",
                    })
                    values = {}
                    for percentage, times in windows.items():
                        for formant in (1, 2):
                            values[formant, percentage] = finite_median(interpolation(
                                winner.time_domain,
                                winner.smoothed_formants[formant - 1], times))
                            hz = values[formant, percentage]
                            row[f"fasttrack_f{formant}_{percentage}"] = hz
                            row[f"fasttrack_bark_f{formant}_{percentage}"] = bark(hz)
                    row["fasttrack_VL_hz"] = math.hypot(
                        values[1, 80] - values[1, 20],
                        values[2, 80] - values[2, 20])
                    row["fasttrack_VL_bark"] = math.hypot(
                        bark(values[1, 80]) - bark(values[1, 20]),
                        bark(values[2, 80]) - bark(values[2, 20]))
                except Exception as error:
                    row.update({
                        "fasttrack_status": "error",
                        "fasttrack_error": f"{type(error).__name__}: {error}",
                    })
                    print(f"  ERROR {source['token_id']}: {row['fasttrack_error']}", flush=True)
                writer.writerow(row)
                handle.flush()
                if remaining_limit:
                    remaining_limit -= 1
                    if remaining_limit == 0:
                        break
            print(f"[{recording_index}/{len(recordings)}] {recording}", flush=True)
            if args.limit_tokens and remaining_limit == 0:
                break
    return read_csv(work_file)


def fit_and_project(args: argparse.Namespace, rows: list[dict[str, str]],
                    output_file: Path) -> dict:
    spectrum_fields = {
        percentage: [f"_spectrum_{percentage}_band_{band:02d}"
                     for band in range(1, args.pca_bands + 1)]
        for percentage in PERCENTAGES
    }
    balanced = defaultdict(list)
    for row in rows:
        for percentage in PERCENTAGES:
            vector = np.asarray([float(row[field]) for field in spectrum_fields[percentage]])
            key = (row["village"], row["speaker"], row["target_label"], percentage)
            balanced[key].append(vector)
    matrix = np.asarray([
        np.mean(vectors, axis=0) for _, vectors in sorted(balanced.items())
    ])
    pca_mean = matrix.mean(axis=0)
    _, singular, components = np.linalg.svd(matrix - pca_mean, full_matrices=False)
    for component in components:
        if component[np.argmax(np.abs(component))] < 0:
            component *= -1
    variance = singular ** 2 / (len(matrix) - 1)
    explained = variance / variance.sum()

    final_fields = [field for field in fieldnames(args.pca_bands)
                    if not field.startswith("_spectrum_")]
    for percentage in PERCENTAGES:
        final_fields.extend(
            f"pca_pc{component}_{percentage}"
            for component in range(1, args.pca_components + 1))
    final_fields.extend(("pca_VL_pc1_pc2",))
    with output_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=final_fields)
        writer.writeheader()
        for source in rows:
            row = {field: source.get(field, "") for field in final_fields}
            scores = {}
            for percentage in PERCENTAGES:
                vector = np.asarray([float(source[field])
                                     for field in spectrum_fields[percentage]])
                score = (vector - pca_mean) @ components[:args.pca_components].T
                scores[percentage] = score
                for component, value in enumerate(score, 1):
                    row[f"pca_pc{component}_{percentage}"] = float(value)
            row["pca_VL_pc1_pc2"] = float(np.linalg.norm(
                scores[80][:2] - scores[20][:2]))
            writer.writerow(row)

    np.savez(
        args.output_dir / "spectral_pca_model.npz",
        mean=pca_mean, components=components,
        explained_variance_ratio=explained,
        percentages=np.asarray(PERCENTAGES),
    )
    with (args.output_dir / "spectral_pca_loadings.csv").open(
            "w", encoding="utf-8", newline="") as handle:
        fields = ["component", "explained_variance_ratio"] + [
            f"bark_band_{band:02d}" for band in range(1, args.pca_bands + 1)]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(args.pca_components):
            writer.writerow({
                "component": index + 1,
                "explained_variance_ratio": float(explained[index]),
                **{f"bark_band_{band:02d}": float(value)
                   for band, value in enumerate(components[index], 1)},
            })
    return {
        "balanced_pca_cells": len(matrix),
        "explained_variance_ratio": explained[:args.pca_components].tolist(),
    }
