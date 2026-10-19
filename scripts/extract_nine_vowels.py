#!/usr/bin/env python3
"""Extract annotated nine-vowel tokens; independently refit spectral PCA."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import sys
import wave

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault('MPLCONFIGDIR', '/tmp/swedia-nine-vowels')
import numpy as np
import extraction_engine as engine
from acoustic_helpers import (lexical_targets, VOWEL_IPA, find_vowel_segment,
    locate_recording, bark_filterbank, parse_textgrid_tier, second_lat_window,
    normalized_word, recording_parts, read_coordinates, RESOURCE_ALIASES)


def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def inventory(args, coordinates):
    tokens, exclusions = [], []
    mapping = lexical_targets()
    mapping.update({word: ('i:', 'base' if word == 'dis' else 'alternative') for word in ('dis', 'ris', 'gris')})
    for grid in sorted((args.corpus_root/'TextGrids_all_wos').glob('*.TextGrid')):
        parts = recording_parts(grid)
        if not parts or parts[0] in ('ref', 'reference', 'referens'):
            continue
        village, kind, suffix = parts
        key = RESOURCE_ALIASES.get(village, village)
        if key not in coordinates:
            exclusions.append(dict(recording=grid.stem, word='', reason='no_resource_coordinates'))
            continue
        try:
            wav = locate_recording(grid.stem, args.media_dirs)
            with wave.open(str(wav)) as audio:
                duration = audio.getnframes()/audio.getframerate()
            words = parse_textgrid_tier(grid, 'ord')
            segments = parse_textgrid_tier(grid, 'seg')
        except Exception as error:
            exclusions.append(dict(recording=grid.stem, word='', reason=str(error)))
            continue
        second = second_lat_window(words)
        for index, (start_word, end_word, word) in enumerate(words, 1):
            normalized = normalized_word(word)
            if normalized not in mapping:
                continue
            target, status = mapping[normalized]
            reason = None
            if normalized == 'låt' and second and second[0] <= (start_word+end_word)/2 <= second[1]:
                reason = 'second_length_contrast_lat_group'
            segment = find_vowel_segment(start_word, end_word, segments)
            if segment is None:
                reason = reason or 'not_exactly_one_vowel_segment'
            elif target == 'i:' and segment[2] not in ('i:', 'I:'):
                reason = reason or 'i_target_surface_not_i_or_I_long:' + segment[2]
            elif not 0 <= segment[0] < segment[1] <= duration:
                reason = reason or 'invalid_audio_interval'
            if reason:
                exclusions.append(dict(recording=grid.stem, word=word, reason=reason))
                continue
            start, end, surface = segment
            # Nonempty word interval index identifies repetitions and split recordings.
            token_id = f'{grid.stem}_word{index:04d}'
            tokens.append(dict(token_id=token_id, word_id=token_id, recording=grid.stem,
                village=village, speaker=f'{kind}_{suffix[0]}', word=word,
                lexical_status=status, target_label=target, ipa='i:' if target=='i:' else VOWEL_IPA[target],
                surface_seg_label=surface, vowel_start_s=start, vowel_end_s=end, duration_s=end-start,
                word_start_s=start_word, word_end_s=end_word,
                wav_file=str(wav.relative_to(args.corpus_root)), textgrid_file=str(grid.relative_to(args.corpus_root)),
                geo_x=coordinates[key][0], geo_y=coordinates[key][1]))
    return tokens, exclusions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus-root', type=Path, required=True,
        help='External directory containing Media/, sounds/ and TextGrids_all_wos/. No raw corpus is bundled.')
    parser.add_argument('--resource',type=Path,default=ROOT/'data/village_resource.txt')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'data')
    parser.add_argument('--inventory-only', action='store_true')
    parser.add_argument('--reuse-cache',type=Path,help='Optional spectral-feature CSV from the original workflow under identical settings; PCA is always refitted.')
    parser.add_argument('--limit-recordings', type=int, default=0, help='Pilot only; new PCA uses only selected recordings.')
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.corpus_root=args.corpus_root.resolve()
    args.media_dirs = [args.corpus_root/'Media', args.corpus_root/'sounds']
    if not (args.corpus_root/'TextGrids_all_wos').is_dir():
        parser.error('--corpus-root must contain TextGrids_all_wos/')
    # Identical extraction settings to the existing curated token workflow.
    defaults = dict(fasttrack_min_ceiling=4000., fasttrack_max_ceiling=7000., fasttrack_steps=20,
        window_length=.025, time_step=.002, pre_emphasis=50., pca_sample_rate=16000,
        pca_window_ms=25., pca_n_fft=1024, pca_bands=20, pca_max_bark=21., pca_components=3,
        limit_tokens=0)
    for key, value in defaults.items():setattr(args, key, value)
    coordinates = read_coordinates(args.resource)
    tokens, exclusions = inventory(args, coordinates)
    if args.limit_recordings:
        selected = sorted({row['recording'] for row in tokens})[:args.limit_recordings]
        tokens = [row for row in tokens if row['recording'] in selected]
    if not tokens:raise ValueError('No eligible tokens')
    write_csv(args.output_dir/'token_inventory.csv', tokens, list(tokens[0]))
    write_csv(args.output_dir/'exclusions.csv', exclusions, ['recording', 'word', 'reason'])
    print(f'Inventory: {len(tokens)} tokens, {len({r["recording"] for r in tokens})} recordings, '
          f'{len({r["village"] for r in tokens})} villages; vowels {dict(Counter(r["ipa"] for r in tokens))}', flush=True)
    if args.inventory_only:return
    work = args.output_dir/'token_spectral_features.csv'
    old_cache = args.reuse_cache
    identity = lambda r: (r['recording'], r['word'], round(float(r['vowel_start_s']), 6), round(float(r['vowel_end_s']), 6))
    token_map = {row['token_id']: row for row in tokens}
    if work.exists():
        cached = engine.read_csv(work)
        if len({r['token_id'] for r in cached}) != len(cached):
            raise ValueError('Duplicate cached token IDs; use a separate output directory.')
        for row in cached:
            expected = token_map.get(row['token_id'])
            if expected is None or identity(row) != identity(expected) or row['ipa'] != expected['ipa']:
                raise ValueError('Existing cache differs from inventory; use a separate output directory.')
    # Reuse only exact recording/word/boundary matches; PCA scores are never reused.
    seeded = 0
    if not work.exists() and old_cache is not None:
        if not old_cache.is_file():raise FileNotFoundError(old_cache)
        lookup = {identity(r): r for r in engine.read_csv(old_cache)}
        fields = engine.fieldnames(args.pca_bands)
        reused = []
        for token in tokens:
            cached = lookup.get(identity(token))
            if cached is not None and cached['fasttrack_status']=='ok':
                row = dict(cached);row.update(token);reused.append(row)
        write_csv(work, reused, fields)
        seeded = len(reused)
        (args.output_dir/'cache_provenance.json').write_text(json.dumps(dict(source=str(old_cache),
            reused_tokens=seeded, reused='FastTrack measurements and spectra only; never PCA scores'),indent=2)+'\n')
        print(f'Reused {seeded} exact-match acoustic measurements; all PCA scores will be newly fitted.', flush=True)
    frequencies = np.fft.rfftfreq(args.pca_n_fft, 1/args.pca_sample_rate)
    _, filters = bark_filterbank(frequencies, args.pca_bands, args.pca_max_bark)
    measured = engine.extract(args, tokens, coordinates, filters, work)
    if set(row['token_id'] for row in measured) != set(token_map):
        raise ValueError('Resumed cache does not match this inventory; use a separate output directory.')
    # Balance PCA by speaker × canonical vowel × timepoint, pooling lexical alternatives.
    pca_rows = [dict(row, target_label=row['ipa']) for row in measured]
    intermediate = args.output_dir/'token_trajectories_internal.csv'
    pca = engine.fit_and_project(args, pca_rows, intermediate)
    final = []
    fields = ['village','speaker','vowel','word','word_id','geo_x','geo_y']
    fields += [field for t in (20,50,80) for field in (
        f'fasttrack_f1_{t}',f'fasttrack_f2_{t}',f'fasttrack_bark_f1_{t}',f'fasttrack_bark_f2_{t}',f'pca1_{t}',f'pca2_{t}')]
    fields += ['recording','wav_file','textgrid_file','word_start_s','word_end_s',
        'vowel_start_s','vowel_end_s','duration_s','surface_seg_label','lexical_status',
        'fasttrack_status','fasttrack_error','fasttrack_ceiling_hz','fasttrack_boundary']
    for row in engine.read_csv(intermediate):
        token = token_map[row['token_id']]
        output = dict(row);output.update(token);output['vowel']=row['ipa'].replace(':','ː')
        for t in (20,50,80):
            for k in (1,2):output[f'pca{k}_{t}']=row[f'pca_pc{k}_{t}']
        for key in fields:
            try:
                if not math.isfinite(float(output.get(key,''))):output[key]=''
            except (TypeError,ValueError):pass
        final.append(output)
    final.sort(key=lambda row:(row['village'],row['speaker'],row['recording'],float(row['word_start_s'])))
    write_csv(args.output_dir/'nine_vowel_token_trajectories.csv', final, fields)
    settings = dict(defaults, input_annotations='TextGrids_all_wos', regions='All; no Gotland/Finland exclusion',
        software={package:version(package) for package in ('fasttrackpy','praat-parselmouth','numpy','scipy')},
        i_words=['dis','ris','gris'], i_surface_labels=['i:','I:'],
        other_vowels='Existing base and approved alternatives; exactly one vowel segment; surface variation allowed',
        time_windows='11 equally spaced measurements within ±5% of vowel duration around 20/50/80%; median',
        bark='26.81/(1+1960/Hz)-0.53, applied to the local median Hz value',
        pca='New SVD PCA of level-normalized 20-band Bark log spectra; mean per speaker/vowel/timepoint cell; centre bands, no band SD scaling',
        pca_display='Original unrotated scores; signs fixed by largest absolute loading being positive; not comparable numerically to previous PCA models',
        cached_measurements_seeded=(json.loads((args.output_dir/'cache_provenance.json').read_text())['reused_tokens']
            if (args.output_dir/'cache_provenance.json').exists() and args.reuse_cache is not None else seeded),
        n_tokens=len(final), n_villages=len({r['village'] for r in final}),
        n_speakers=len({(r['village'],r['speaker']) for r in final}),
        fasttrack_status=dict(Counter(r['fasttrack_status'] for r in final)), **pca)
    (args.output_dir/'settings.json').write_text(json.dumps(settings,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(f'Wrote {len(final)} rows to {args.output_dir/"nine_vowel_token_trajectories.csv"}',flush=True)


if __name__=='__main__':main()
