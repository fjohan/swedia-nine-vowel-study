#!/usr/bin/env python3
"""Build side-by-side individual-village SVGs and the sticky-toolbar viewer."""
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
from plot_vowel_spaces import draw_group, ellipse_outline, plt, np, VOWELS


def main():
    output = ROOT/'vowel_spaces'
    datasets = {}
    for method in ('pca', 'fasttrack_bark'):
        for t in (20, 50, 80):
            directory = output/f'{method}_t{t}'
            with (directory/'vowel_positions.csv').open(encoding='utf-8', newline='') as handle:
                rows = list(csv.DictReader(handle))
            for row in rows:
                for k in ('display_x', 'display_y'):row[k] = float(row[k])
            with (directory/'ellipse_angles.csv').open(encoding='utf-8', newline='') as handle:
                angles = list(csv.DictReader(handle))
            for row in angles:
                for k in ('centroid_x','centroid_y','major_axis_sd','minor_axis_sd','ellipse_angle_deg',
                          'ellipse_angle_from_horizontal_signed_deg','axis_ratio'):
                    row[k] = float(row[k])
                row['orientation_reliable'] = int(row['orientation_reliable'])
            datasets[method,t] = (rows, {r['village']:r for r in angles})
    villages = sorted({r['village'] for rows, angles in datasets.values() for r in rows})
    for method in ('pca','fasttrack_bark'):
        directory = output/'villages'/method
        directory.mkdir(parents=True, exist_ok=True)
        for village in villages:
            # Hold scales fixed over time within each village and acoustic method.
            extents = []
            for t in (20,50,80):
                rows, angles = datasets[method,t]
                extents.extend((r['display_x'],r['display_y']) for r in rows if r['village']==village)
                x,y = ellipse_outline(angles[village])
                extents.extend(zip(x,y))
            points = np.asarray(extents)
            lo,hi = points.min(0),points.max(0)
            margin = np.maximum((hi-lo)*.12,.2)
            for t in (20,50,80):
                rows, angles = datasets[method,t]
                values = [r for r in rows if r['village']==village]
                fig, ax = plt.subplots(figsize=(5.8,4.4), constrained_layout=True)
                draw_group(ax,values,{(village,'','vowels'):angles[village]},
                           'pca' if method=='pca' else 'fasttrack','village',VOWELS+('iː',),False)
                ax.set_xlim(lo[0]-margin[0],hi[0]+margin[0])
                ax.set_ylim(lo[1]-margin[1],hi[1]+margin[1])
                ax.set_aspect('equal',adjustable='box')
                ax.set_title(f'{"PCA" if method=="pca" else "FastTrack Bark"} · {t}% · {len(values)}/9 vowels',fontsize=11)
                ax.set_xlabel('−PC1 (score units)' if method=='pca' else '−F2 (Bark)')
                ax.set_ylabel('−PC2 (score units)' if method=='pca' else '−F1 (Bark)')
                fig.savefig(directory/f'{village}_t{t}.svg')
                plt.close(fig)
        print(f'{method}: {len(villages)*3} village plots',flush=True)
    template = Path(__file__).with_name('vowel_space_viewer.html').read_text(encoding='utf-8')
    (output/'index.html').write_text(template.replace('__VILLAGES__',json.dumps(villages,ensure_ascii=False)),encoding='utf-8')
    print(f'Wrote {output/"index.html"}',flush=True)


if __name__=='__main__':main()
