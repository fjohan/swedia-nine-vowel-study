#!/usr/bin/env python3
"""Rebuild all six nine-vowel village plot sets and their HTML/SVG viewer."""
import sys
from pathlib import Path
import plot_vowel_spaces
import build_vowel_space_viewer

ROOT=Path(__file__).resolve().parent.parent


def main():
    for method in ('pca','fasttrack'):
        for time in (20,50,80):
            name='fasttrack_bark' if method=='fasttrack' else 'pca'
            sys.argv=[__file__,'--input',str(ROOT/'data/nine_vowel_token_trajectories.csv'),
                '--method',method,'--timepoint',str(time),'--include-i','--no-midpoints',
                '--include-incomplete','--group-by','village','--plot',
                '--output-dir',str(ROOT/'vowel_spaces'/f'{name}_t{time}')]
            if method=='fasttrack':sys.argv.append('--bark')
            plot_vowel_spaces.main()
    build_vowel_space_viewer.main()


if __name__=='__main__':main()
