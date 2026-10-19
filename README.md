# SweDia nine-vowel acoustics

Standalone snapshot of the /iː/-inclusive workflow. Existing
recordings, annotations and older analyses are **not** part of this directory.
No acoustic extraction or clustering was rerun to prepare this snapshot.

## Browse the results

Start with [index.html](index.html), or serve this directory locally:

```sh
python3 -m http.server 8000
```

Then open `http://localhost:8000/`. Everything is static: no PHP, database,
CDN, build service or connection to the original repository is required.
The same directory can be the root of a GitHub repository and deployed with
GitHub Pages. The HTML/SVG/PNG assets and CSV downloads are already included.

- [Village vowel spaces](vowel_spaces/index.html): one village per row,
  PCA left and FastTrack Bark right, with a sticky timepoint selector and search.
- [Four clustering reports](Reports/index.html):
  1. [Whole nine-vowel configurations](Reports/Nine_vowel_clustering/index.html).
  2. [Independent-vowel profiles, no geography](Reports/Nine_vowel_profiles/index.html).
  3. [Gabriel-constrained vowel clusters only](Reports/Nine_vowel_profiles_Gabriel_inputs/index.html).
  4. [Gabriel constraints at both stages](Reports/Nine_vowel_profiles_Gabriel/index.html).
- [Token measurement CSV](data/nine_vowel_token_trajectories.csv).

## Contents

```text
data/           Token measurements, inventory/exclusions, PCA model/loadings,
                extraction settings and village-resource coordinates
scripts/        Extraction, ellipse plots, viewers, clustering and report templates
vowel_spaces/   Ready-to-view side-by-side viewer and individual village SVGs
Reports/        Four reports, figures, input anchors, assignments and stability data
```

There are **13,558 tokens, 273 village–speaker identities and 91 villages**.
Gotland and Finland are included. The clustering reports use the same
**83 villages** with complete nine-vowel anchors in both methods at all timepoints.

## Measurements

The nine vowels are /uː oː ɑː æː eː yː ʉ̟ː øː iː/. The eight original vowels
use the established base words and approved lexical alternatives. The later
length-contrast *låt* group remains excluded. /iː/ uses *dis*, *ris* and *gris*,
accepting `i:` and `I:` surface labels. For the other vowels, surface variation
does not change lexical vowel identity. Each eligible token has exactly one
vowel segment assigned to its annotated word and valid audio boundaries.

The CSV retains individual word occurrences, **not speaker averages**. It contains
village, local speaker ID, canonical IPA vowel, word, unique word ID, geography,
FastTrack Hz, FastTrack Bark and unrotated PCA1/PCA2 at 20%, 50% and 80%, plus
WAV/TextGrid paths, word/vowel boundaries and tracking metadata. Times are seconds
within the original WAV. Paths are relative to the external corpus root.
Use `(village, speaker)` as a speaker key; `word_id` identifies one occurrence.

FastTrack tests 20 ceilings from 4000–7000 Hz, with four formants, 25 ms windows,
2 ms steps and 50 Hz pre-emphasis. Each temporal measurement is the median of
11 interpolated values spanning ±5% of vowel duration. Bark transforms the
median Hz value using `26.81 / (1 + 1960 / Hz) - 0.53`. Boundary ceilings are
flagged rather than rejected; acoustic correctness was not manually guaranteed.

The new PCA was fitted across **all nine vowels**, using 16 kHz audio,
25 ms Hann windows, 1024-point FFTs and 20 Bark-band spectra up to 21 Bark.
Overall spectral level is removed; local spectral vectors are median-aggregated.
Training uses one mean spectrum per speaker × canonical vowel × temporal-point
cell, pooling lexical alternatives. Bands are centred but not SD-standardized.
PCA scores are unrounded and unrotated. Signs are fixed by the largest absolute
loading being positive. They are not interchangeable with earlier eight-vowel
PCA scores. The NPZ retains the fitted model and the loading CSV documents it.

The snapshot originally reused 10,827 matching eight-vowel acoustic measurements
under identical settings, **never old PCA scores**. `data/cache_provenance.json`
records that historical source; it is not a dependency of this package. Large
spectral caches and the redundant internal PC3 table are omitted. Extraction
can regenerate them from the external corpus.

## Reproduce from the supplied CSV

Python 3.12 is recommended if you also want FastTrack extraction. Plotting and
clustering require only NumPy, SciPy and Matplotlib; scikit-learn is not needed.

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

Run commands from this directory. Script defaults are based on their own paths,
not the location of the original repository.

```sh
# All six village ellipse plot sets plus the SVG/HTML viewer:
python scripts/build_vowel_spaces.py

# Whole-vowel-space Ward clustering and its report:
python scripts/cluster_village_vowel_spaces.py

# Independent vowel profiles and all three geographic variants:
python scripts/cluster_vowel_profiles.py
python scripts/cluster_vowel_profiles.py --geography vowels
python scripts/cluster_vowel_profiles.py --geography both
```

These commands regenerate their output directories. Use the clustering scripts'
`--output` option for separate experiments. Whole-space clustering defaults to
300 speaker bootstraps; profile analyses use 200 and 20 tie-order repeats.
Both accept `--input`, `--seed` and `--bootstrap`. Profile clustering also accepts
`--clusters` (default 2, 3, 4) and `--tie-repetitions`. `--render-only` on the
profile script refreshes its HTML from saved `profiles.json` without reclustering.

For one ellipse set:

```sh
python scripts/plot_vowel_spaces.py --method fasttrack --bark --timepoint 50 \
  --group-by village --include-incomplete --plot \
  --output-dir vowel_spaces/fasttrack_bark_t50
python scripts/build_vowel_space_viewer.py
```

The bundled ellipse script includes /iː/ and omits midpoints by default. It
first calculates token medians within speakers, then village medians across
speakers. All available vowel anchors enter the covariance ellipse; /iː/ is
unpaired, while the six original arrows remain. The ellipse is drawn at two
covariance SDs, not as a confidence interval. Individual SVGs hold axes fixed
over time within village and method.

## Re-extract from recordings and annotations

Raw corpus files are deliberately excluded. Supply a directory containing
`Media/`, `sounds/` and `TextGrids_all_wos/`. The bundled Latin-1
`data/village_resource.txt` supplies geographic metadata; `--resource` can
select another compatible resource file.

```sh
python -m pip install -r requirements-extraction.txt
python scripts/extract_nine_vowels.py --corpus-root /path/to/corpus \
  --output-dir regenerated-data
```

Use Python 3.12 for the pinned FastTrackPy/Parselmouth extraction dependencies.
`--inventory-only` audits eligible annotations without acoustics. Extraction
resumes from its output directory's spectral-feature cache. Optional
`--reuse-cache /path/to/token_fasttrack_spectral_features.csv` seeds exact
recording/word/boundary matches under the same settings; a new nine-vowel PCA
is still fitted. No old repository or cache is required for a fresh run.
`--limit-recordings N` is a pilot option; always give pilots a separate output
directory. The scripts import the small helper modules bundled in `scripts/`,
not `MoreScripts` or a different checkout.

## Clustering interpretation

Whole-configuration Ward clusters retain position, size and orientation, with
one pooled SD per acoustic axis. Independent-vowel profiles compare 50% position,
full trajectories, and early/late movement differences. Literal exact rows are
compared at 9/9; relaxed groups must agree on the same named cells across every
member. Greedy groups are nested from 9 through 3, not connected-component chains
of partly matching villages. Their sensitivity to tie order is reported.

Gabriel variants use a coordinate-only graph with no distance cutoff. Ward
merges only adjacent groups; the `both` variant also restricts profile merges.
Here exact rows split into geographically connected pieces, so distinct rows
and connected groups can have different counts. Speaker bootstraps rebuild the
graph on retained villages. Geographic coherence is therefore partly imposed,
not independent evidence of dialect boundaries. No water/travel barrier is modelled.

Whole speakers are resampled, keeping their vowels/timepoints together.
Incomplete resampled villages are omitted for that replicate; there is no
imputation. Reported stability is conditional on coverage. Tokens and the PCA
model are not resampled. Exact-row return can be low across nine cells; high
partition agreement among mostly singletons does not establish shared profiles.
The reports are exploratory and do not estimate a true number of dialect regions.

## Snapshot checks and publication

```sh
python scripts/validate_bundle.py
```

The validation checks local report links, dynamic plot assets, nine-vowel table
identity/coverage, and the SHA-256 snapshot manifest. `MANIFEST.sha256` describes
the packaged files, not a new analysis run. Rebuilding results will intentionally
change those hashes; use `--skip-hashes` to check a rebuilt package's links/data.

No Git repository was initialized and nothing was committed or uploaded during
packaging. Check data-sharing rights before publishing. No new licence has been
assigned here; the external corpus has its own permissions. External reference
reports linked in the methods are citations only, not runtime dependencies.
