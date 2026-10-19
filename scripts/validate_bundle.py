#!/usr/bin/env python3
"""Validate packaged data/assets/links using only the Python standard library."""
import argparse
import csv
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import urlsplit,unquote
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parent.parent


class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        for key,value in attrs:
            if key in ('src','href') and value:self.links.append(value)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-hashes',action='store_true')
    args=parser.parse_args();errors=[];html_count=0
    for path in [ROOT/'index.html',*(ROOT/'Reports').rglob('*.html'),*(ROOT/'vowel_spaces').glob('*.html')]:
        html_count+=1;links=Links();links.feed(path.read_text(encoding='utf-8'))
        for link in links.links:
            parsed=urlsplit(link)
            if parsed.scheme or parsed.netloc or not parsed.path:continue
            target=(path.parent/unquote(parsed.path)).resolve()
            if not target.is_relative_to(ROOT):errors.append(f'Link leaves bundle: {path.relative_to(ROOT)} → {link}')
            elif not target.exists():errors.append(f'Missing link: {path.relative_to(ROOT)} → {link}')
    table=ROOT/'data/nine_vowel_token_trajectories.csv'
    with table.open(encoding='utf-8',newline='') as handle:rows=list(csv.DictReader(handle))
    if len(rows)!=13558 or len({r['word_id'] for r in rows})!=13558:errors.append('Token count/IDs differ from snapshot')
    if len({r['village'] for r in rows})!=91:errors.append('Expected 91 villages')
    if len({(r['village'],r['speaker']) for r in rows})!=273:errors.append('Expected 273 speakers')
    if len({r['vowel'] for r in rows})!=9:errors.append('Expected nine vowels')
    for r in rows:
        if r['vowel']=='iː' and (r['word'].casefold() not in ['dis','ris','gris'] or r['surface_seg_label'] not in ['i:','I:']):errors.append('Unexpected /iː/ eligibility: '+r['word_id'])
    svg_count=0
    for method in ('pca','fasttrack_bark'):
        for time in (20,50,80):
            folder=ROOT/'vowel_spaces'/f'{method}_t{time}'
            with (folder/'vowel_positions.csv').open(encoding='utf-8',newline='') as handle:
                villages=sorted({r['village'] for r in csv.DictReader(handle)})
            for v in villages:
                image=ROOT/'vowel_spaces/villages'/method/f'{v}_t{time}.svg'
                try:ET.parse(image);svg_count+=1
                except (OSError,ET.ParseError) as error:errors.append(str(error))
            for page in range(1,5):
                if not (folder/f'midpoint_vowel_spaces_{page:02d}.png').exists():errors.append('Missing paginated figure')
    whole=ROOT/'Reports/Nine_vowel_clustering'
    for method in ('pca','fasttrack_bark'):
        for time in ('50','20_50_80'):
            directory=whole/f'{method}_{time}'
            for k in range(2,7):
                for kind in ('map','spaces','coassignment'):
                    if not (directory/f'{kind}_k{k}.png').exists():errors.append('Missing cluster figure')
    for name in ('Nine_vowel_profiles','Nine_vowel_profiles_Gabriel_inputs','Nine_vowel_profiles_Gabriel'):
        directory=ROOT/'Reports'/name
        payload=json.loads((directory/'profiles.json').read_text(encoding='utf-8'))
        if len(payload['analyses'])!=18:errors.append('Missing profile analyses: '+name)
        for a in payload['analyses']:
            for file in ('vowel_clusters.png','village_profiles.csv','groups.csv','vowel_diagnostics.csv','axis_scaling.csv','bootstrap_profile_groups.npz'):
                if not (directory/a['slug']/file).exists():errors.append('Missing profile asset: '+a['slug']+'/'+file)
    manifest_count=0
    if not args.skip_hashes:
        for line in (ROOT/'MANIFEST.sha256').read_text(encoding='utf-8').splitlines():
            expected,relative=line.split('  ',1);path=ROOT/relative
            if not path.is_file():errors.append('Missing manifest file: '+relative);continue
            with path.open('rb') as handle:actual=hashlib.file_digest(handle,'sha256').hexdigest()
            if actual!=expected:errors.append('Changed snapshot file: '+relative)
            manifest_count+=1
    if errors:raise SystemExit('\n'.join(errors))
    print(f'PASS: {html_count} HTML pages, {svg_count} village SVGs, four reports and all dynamic plot assets; {len(rows)} token rows; {manifest_count} snapshot hashes checked.')


if __name__=='__main__':main()
