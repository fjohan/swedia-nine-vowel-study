"""Bundled acoustic/inventory helpers, copied without numerical changes from the original workflows."""
from __future__ import annotations
import math
import re
import shlex
import wave
from pathlib import Path
import numpy as np
from scipy.signal import get_window, resample_poly

VOWEL_INITIALS = set("aeiouyAEIOUYäöåÄÖÅ29<")


BASE_TARGETS = [
    {"word": "sot", "target": "u:", "ipa": "u:", "seg": "u:"},
    {"word": "låt", "target": "o:", "ipa": "o:", "seg": "o:", "group": "regular"},
    {"word": "lat", "target": "A:", "ipa": "ɑ:", "seg": "A:"},
    {"word": "nät", "target": "ä:", "ipa": "æ:", "seg": "ä:"},
    {"word": "leta", "target": "e:", "ipa": "e:", "seg": "e:"},
    {"word": "typ", "target": "y:", "ipa": "y:", "seg": "y:"},
    {"word": "lus", "target": "U:", "ipa": "ʉ̟:", "seg": "U:"},
    {"word": "söt", "target": "ö:", "ipa": "ø:", "seg": "ö:"},
]


EXTRA_COUNTS = [
    {"word": "låt", "target": "o2:", "ipa": "o:", "seg": "o:", "group": "second", "header": "second_låt_o"},
    {"word": "lås", "target": "o_alt:", "ipa": "o:", "seg": "o:", "group": "alternative", "header": "lås_o"},
    {"word": "blöt", "target": "ö_alt:", "ipa": "ø:", "seg": "ö:", "group": "alternative", "header": "blöt_ø"},
    {"word": "gles", "target": "e_alt:", "ipa": "e:", "seg": "e:", "group": "alternative", "header": "gles_e"},
    {"word": "lös", "target": "ö2_alt:", "ipa": "ø:", "seg": "ö:", "group": "alternative", "header": "lös_ø"},
    {"word": "läs", "target": "ä_alt:", "ipa": "æ:", "seg": "ä:", "group": "alternative", "header": "läs_æ"},
    {"word": "rot", "target": "u_alt:", "ipa": "u:", "seg": "u:", "group": "alternative", "header": "rot_u"},
    {"word": "fräs", "target": "ä2_alt:", "ipa": "æ:", "seg": "ä:", "group": "alternative", "header": "fräs_æ"},
    {"word": "gråt", "target": "o3_alt:", "ipa": "o:", "seg": "o:", "group": "alternative", "header": "gråt_o"},
    {"word": "båt", "target": "o4_alt:", "ipa": "o:", "seg": "o:", "group": "alternative", "header": "båt_o"},
]


ALTERNATIVE_TARGET_MAP = {
    "u:": ["u_alt:"],
    "o:": ["o_alt:", "o3_alt:", "o4_alt:"],
    "ä:": ["ä_alt:", "ä2_alt:"],
    "e:": ["e_alt:"],
    "ö:": ["ö_alt:", "ö2_alt:"],
}


def normalized_word(word: str) -> str:
    return word.strip().casefold()


def looks_vowel_like(label: str) -> bool:
    return bool(label) and (label[0] in VOWEL_INITIALS or label.startswith("\\}"))


def parse_textgrid_tier(path: Path, tier_name: str) -> list[tuple[float, float, str]]:
    """Read intervals from one tier of a long-text Praat TextGrid."""
    text = path.read_text(encoding="utf-8")
    tier_blocks = re.split(r"(?m)^    item \[\d+\]:\s*$", text)[1:]
    for block in tier_blocks:
        name = re.search(r'(?m)^        name = "(.*)"\s*$', block)
        if not name or name.group(1) != tier_name:
            continue
        intervals = []
        for match in re.finditer(
            r'(?ms)^        intervals \[\d+\]:\s*\n'
            r'            xmin = ([^\n]+)\n'
            r'            xmax = ([^\n]+)\n'
            r'            text = "((?:[^"\\]|\\.)*)"',
            block,
        ):
            label = match.group(3).replace('""', '"')
            if label:
                intervals.append((float(match.group(1)), float(match.group(2)), label))
        return intervals
    raise ValueError(f"Tier {tier_name!r} not found in {path}")


def second_lat_window(words: list[tuple[float, float, str]]) -> tuple[float, float] | None:
    """Return a likely time window for the later length-contrast låt group.

    In these lists the second låt group generally occurs after vägg and before
    lott.  If that frame is absent, return None and count all låt as regular.
    """
    normalized = [(start, end, normalized_word(word)) for start, end, word in words]
    vagg_end = max((end for _, end, word in normalized if word == "vägg"), default=None)
    if vagg_end is None:
        return None
    lott_start = min((start for start, _, word in normalized if word == "lott" and start > vagg_end), default=None)
    if lott_start is None:
        return None
    return vagg_end, lott_start


RECORDING_RE = re.compile(r"^(?P<place>.+)_(?P<kind>[a-z]+)_(?P<suffix>\d+(?:_\d+)*)$")


def recording_parts(path: Path) -> tuple[str, str, tuple[int, ...]] | None:
    match = RECORDING_RE.match(path.stem)
    if not match:
        return None
    return match["place"], match["kind"], tuple(int(part) for part in match["suffix"].split("_"))


RESOURCE_ALIASES = {
    "n_rorum": "nrorum", "pitea": "pite", "s_finnskoga": "sodrafinnskoga",
    "s_mellosa": "stmellosa", "st_anna": "stanna", "v_vingaker": "vingaker",
    "ref": "reference",
}


def read_coordinates(path: Path) -> dict[str, tuple[int, int, str]]:
    coordinates = {}
    for raw in path.read_bytes().decode("latin-1").splitlines():
        line = raw.strip()
        if not line.startswith("{") or not line.endswith("}"):
            continue
        fields = shlex.split(line[1:-1])
        if len(fields) >= 6:
            coordinates[fields[0]] = (int(fields[4]), int(fields[5]), fields[2])
    return coordinates


VOWEL_IPA = {target["target"]: target["ipa"] for target in BASE_TARGETS}


def lexical_targets() -> dict[str, tuple[str, str]]:
    """Map inventory base and approved alternative words to base categories."""
    result = {target["word"]: (target["target"], "base") for target in BASE_TARGETS}
    alternative_to_base = {
        alternative: base
        for base, alternatives in ALTERNATIVE_TARGET_MAP.items()
        for alternative in alternatives
    }
    for target in EXTRA_COUNTS:
        base = alternative_to_base.get(target["target"])
        if base:
            result[target["word"]] = (base, "alternative")
    return result


def read_wav(path: Path, target_rate: int = 16000) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        raw = handle.readframes(handle.getnframes())
    if width != 2:
        raise ValueError(f"Expected 16-bit PCM audio in {path}, found {width * 8}-bit")
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    if rate != target_rate:
        divisor = math.gcd(rate, target_rate)
        samples = resample_poly(samples, target_rate // divisor, rate // divisor)
        rate = target_rate
    return rate, samples


def hz_to_bark(hz: np.ndarray | float) -> np.ndarray:
    hz = np.asarray(hz, dtype=float)
    return 26.81 / (1.0 + 1960.0 / np.maximum(hz, 1e-12)) - 0.53


def bark_filterbank(freqs: np.ndarray, n_bands: int, max_bark: float) -> tuple[np.ndarray, np.ndarray]:
    """Return equally spaced triangular filters on the Traunmüller Bark scale."""
    edges = np.linspace(0.0, max_bark, n_bands + 2)
    centers = edges[1:-1]
    bark_freqs = hz_to_bark(freqs)
    filters = np.zeros((n_bands, len(freqs)))
    for index, (left, center, right) in enumerate(zip(edges[:-2], centers, edges[2:])):
        filters[index] = np.maximum(
            0.0,
            np.minimum((bark_freqs - left) / (center - left), (right - bark_freqs) / (right - center)),
        )
    return centers, filters


def spectral_vector(
    samples: np.ndarray,
    rate: int,
    center_s: float,
    window_ms: float,
    n_fft: int,
    filters: np.ndarray,
) -> tuple[np.ndarray, float]:
    length = round(rate * window_ms / 1000.0)
    center = round(center_s * rate)
    start = center - length // 2
    frame = np.zeros(length)
    source_start, source_end = max(0, start), min(len(samples), start + length)
    if source_end > source_start:
        frame[source_start - start:source_end - start] = samples[source_start:source_end]
    rms_dbfs = 20.0 * np.log10(max(np.sqrt(np.mean(frame ** 2)), 1e-12))
    frame *= get_window("hann", length, fftbins=True)
    power = np.abs(np.fft.rfft(frame, n=n_fft)) ** 2
    energies = (filters @ power) / np.maximum(filters.sum(axis=1), 1e-12)
    db = 10.0 * np.log10(np.maximum(energies, 1e-20))
    # Remove overall level while retaining spectral shape; 80 is a readable reference level.
    total_db = 10.0 * np.log10(np.maximum(energies.sum(), 1e-20))
    return db + (80.0 - total_db), float(rms_dbfs)


def find_vowel_segment(word_start: float, word_end: float, segments: list[tuple[float, float, str]]):
    candidates = [
        segment for segment in segments
        if word_start <= (segment[0] + segment[1]) / 2 <= word_end and looks_vowel_like(segment[2])
    ]
    return candidates[0] if len(candidates) == 1 else None


def locate_recording(stem: str, media_dirs: list[Path]) -> Path:
    matches = [directory / f"{stem}.wav" for directory in media_dirs if (directory / f"{stem}.wav").exists()]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one WAV for {stem}, found: {matches}")
    return matches[0]
