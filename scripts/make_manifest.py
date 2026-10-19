#!/usr/bin/env python3
"""Record a SHA-256 manifest of the distributable snapshot (no analyses run)."""
from pathlib import Path
import hashlib

ROOT=Path(__file__).resolve().parent.parent
EXCLUDED={'data/token_spectral_features.csv','data/token_trajectories_internal.csv'}


def main():
    rows=[]
    for path in sorted(ROOT.rglob('*')):
        relative=path.relative_to(ROOT)
        if not path.is_file() or path.name=='MANIFEST.sha256':continue
        if any(p in ('.git','__pycache__','Media','sounds','TextGrids_all_wos') or p.startswith('.venv') for p in relative.parts):continue
        if relative.as_posix() in EXCLUDED:continue
        with path.open('rb') as handle:digest=hashlib.file_digest(handle,'sha256').hexdigest()
        rows.append(digest+'  '+relative.as_posix())
    (ROOT/'MANIFEST.sha256').write_text('\n'.join(rows)+'\n',encoding='utf-8')
    print(f'Manifest: {len(rows)} files')


if __name__=='__main__':main()
