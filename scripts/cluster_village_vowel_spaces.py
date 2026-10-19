#!/usr/bin/env python3
"""Exploratory Ward clustering of complete nine-vowel village configurations."""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import warnings
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault('MPLCONFIGDIR','/tmp/swedia-nine-vowel-clusters')
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import linkage, cut_tree, dendrogram, leaves_list
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linear_sum_assignment
from plot_vowel_spaces import VOWELS as BASE_VOWELS, PAIRS, COLORS

VOWELS=BASE_VOWELS+('iː',)
TIMES=(20,50,80)
KS=tuple(range(2,7))
PALETTE=list(plt.get_cmap('tab10').colors[:6])


def write_csv(path,rows,fields=None):
    with path.open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]) if rows else fields);writer.writeheader();writer.writerows(rows)


def nanmedian(a,axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        return np.nanmedian(a,axis=axis)


def adjusted_rand(a,b):
    _,a=np.unique(a,return_inverse=True);_,b=np.unique(b,return_inverse=True)
    table=np.zeros((a.max()+1,b.max()+1),dtype=int)
    np.add.at(table,(a,b),1)
    choose=lambda n:n*(n-1)/2
    observed=choose(table).sum();total=choose(len(a))
    expected=choose(table.sum(0)).sum()*choose(table.sum(1)).sum()/total
    maximum=(choose(table.sum(0)).sum()+choose(table.sum(1)).sum())/2
    return float((observed-expected)/(maximum-expected)) if maximum>expected else 1.


def silhouette(X,labels):
    distances=squareform(pdist(X));values=[]
    for i,label in enumerate(labels):
        same=np.flatnonzero(labels==label)
        if len(same)==1:values.append(0.);continue
        a=distances[i,same].sum()/(len(same)-1)
        b=min(distances[i,labels==other].mean() for other in np.unique(labels) if other!=label)
        values.append((b-a)/max(a,b) if max(a,b)>0 else 0.)
    return float(np.mean(values))


def load_speakers(path):
    grouped=defaultdict(list);geo={}
    with path.open(encoding='utf-8',newline='') as handle:
        for row in csv.DictReader(handle):
            if row['vowel'] not in VOWELS:continue
            grouped[row['village'],row['speaker'],row['vowel']].append(row)
            xy=(float(row['geo_x']),float(row['geo_y']))
            if row['village'] in geo and geo[row['village']]!=xy:raise ValueError('Inconsistent village coordinates')
            geo[row['village']]=xy
    speakers=defaultdict(set)
    for village,speaker,vowel in grouped:speakers[village].add(speaker)
    result={method:{} for method in ('pca','fasttrack_bark')}
    for method in result:
        for village,names in sorted(speakers.items()):
            tensor=np.full((len(names),3,9,2),np.nan)
            for s,speaker in enumerate(sorted(names)):
                for v,vowel in enumerate(VOWELS):
                    rows=grouped.get((village,speaker,vowel),[])
                    for t,time in enumerate(TIMES):
                        columns=(f'pca1_{time}',f'pca2_{time}') if method=='pca' else (f'fasttrack_bark_f2_{time}',f'fasttrack_bark_f1_{time}')
                        for axis,col in enumerate(columns):
                            numbers=[]
                            for row in rows:
                                try:value=float(row[col])
                                except (ValueError,TypeError):continue
                                if math.isfinite(value):numbers.append(-value)
                            if numbers:tensor[s,t,v,axis]=np.median(numbers)
            result[method][village]=tensor
    return result,geo


def mean_space_plots(path,anchors,labels,villages,times,method,k):
    ncolumns=len(times) if len(times)>1 else min(3,k)
    nrows=k if len(times)>1 else math.ceil(k/ncolumns)
    fig,axes=plt.subplots(nrows,ncolumns,figsize=(4.5*ncolumns,3.8*nrows),squeeze=False,constrained_layout=True)
    lo=anchors.min(axis=(0,1,2));hi=anchors.max(axis=(0,1,2));pad=np.maximum((hi-lo)*.08,.2)
    for c in range(k):
        for t,time in enumerate(times):
            ax=axes[c,t] if len(times)>1 else axes.flat[c]
            subset=anchors[labels==c,t];means=subset.mean(0)
            ax.scatter(subset[...,0],subset[...,1],s=6,color='#74818a',alpha=.12)
            for v,vowel in enumerate(VOWELS):
                ax.scatter(*means[v],s=30,color=COLORS[vowel],zorder=4)
                ax.annotate('/'+vowel+'/',means[v],xytext=(4,3),textcoords='offset points',fontsize=8)
            for first,second in PAIRS:
                a,b=means[VOWELS.index(first)],means[VOWELS.index(second)]
                ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='#555',alpha=.6,lw=1,shrinkA=5,shrinkB=5))
            ax.set_title(f'Cluster {c+1} · {sum(labels==c)} villages · {time}%',color=PALETTE[c])
            ax.set_xlim(lo[0]-pad[0],hi[0]+pad[0]);ax.set_ylim(lo[1]-pad[1],hi[1]+pad[1]);ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.15)
            ax.set_xlabel('−PC1' if method=='pca' else '−F2 (Bark)');ax.set_ylabel('−PC2' if method=='pca' else '−F1 (Bark)')
    if len(times)==1:
        for ax in list(axes.flat)[k:]:ax.set_visible(False)
    fig.suptitle('Cluster-average village anchors (equal village weights); grey points = individual village anchors')
    fig.savefig(path,dpi=150);plt.close(fig)


def analyse(method,times,all_speakers,villages,geo,args):
    slug=method+('_50' if len(times)==1 else '_20_50_80');directory=args.output/slug;directory.mkdir(parents=True,exist_ok=True)
    indices=[TIMES.index(t) for t in times]
    speakers=[all_speakers[v][:,indices] for v in villages]
    anchors=np.array([nanmedian(a) for a in speakers])
    axis_mean=anchors.mean(axis=(0,1,2));axis_sd=anchors.std(axis=(0,1,2),ddof=1)
    if np.any(axis_sd<=0):raise ValueError('Zero-variance acoustic axis')
    standardize=lambda a:((a-axis_mean)/axis_sd).reshape(len(a),-1)/np.sqrt(len(times))
    X=standardize(anchors);tree=linkage(X,method='ward');labels=cut_tree(tree,n_clusters=KS)
    n=len(villages);rng=np.random.default_rng(args.seed)
    retained=[];cooccurrence=np.zeros((n,n));coassigned=np.zeros((len(KS),n,n))
    aris=[[] for _ in KS];agreements=np.zeros((len(KS),n));coverage=np.zeros(n)
    for repetition in range(args.bootstrap):
        boot=np.array([nanmedian(a[rng.integers(0,len(a),len(a))]) for a in speakers])
        valid=np.isfinite(boot).all(axis=(1,2,3));ids=np.flatnonzero(valid);retained.append(len(ids))
        if len(ids)<=max(KS):continue
        bootlabels=cut_tree(linkage(standardize(boot[valid]),method='ward'),n_clusters=KS)
        ix=np.ix_(ids,ids);cooccurrence[ix]+=1;coverage[ids]+=1
        for j,k in enumerate(KS):
            base=labels[valid,j];sample=bootlabels[:,j]
            aris[j].append(adjusted_rand(base,sample))
            coassigned[j][ix]+=(sample[:,None]==sample[None,:])
            contingency=np.zeros((k,k),int);np.add.at(contingency,(base,sample),1)
            a,b=linear_sum_assignment(-contingency);mapping=dict(zip(b,a))
            aligned=np.array([mapping[v] for v in sample]);agreements[j,ids]+=(aligned==base)
    fig,ax=plt.subplots(figsize=(16,6),constrained_layout=True)
    dendrogram(tree,labels=villages,leaf_rotation=90,leaf_font_size=6,color_threshold=0,above_threshold_color='#52636d',ax=ax)
    ax.set(title=slug+': Ward dendrogram',ylabel='Ward linkage height (scaled configuration distance)')
    fig.savefig(directory/'dendrogram.png',dpi=160);plt.close(fig)
    order=leaves_list(tree);summaries=[];assignments=[];means=[]
    xy=np.array([geo[v] for v in villages])
    for j,k in enumerate(KS):
        lab=labels[:,j];scores=np.asarray(aris[j]);agreement=np.divide(agreements[j],coverage,out=np.full(n,np.nan),where=coverage>0)
        probability=np.divide(coassigned[j],cooccurrence,out=np.full((n,n),np.nan),where=cooccurrence>0)
        within=(lab[:,None]==lab[None,:])&~np.eye(n,dtype=bool)
        between=lab[:,None]!=lab[None,:]
        summaries.append(dict(analysis=slug,method=method,timepoints=' '.join(map(str,times)),k=k,n_villages=n,
            silhouette=silhouette(X,lab),bootstrap_ari_median=float(np.median(scores)),
            bootstrap_ari_p025=float(np.quantile(scores,.025)),bootstrap_ari_p975=float(np.quantile(scores,.975)),
            within_cluster_coassignment=float(np.nanmean(probability[within])),between_cluster_coassignment=float(np.nanmean(probability[between])),
            bootstrap_villages_median=float(np.median(retained)),bootstrap_villages_min=min(retained),bootstrap_villages_max=max(retained)))
        for i,village in enumerate(villages):
            assignments.append(dict(village=village,geo_x=geo[village][0],geo_y=geo[village][1],k=k,cluster=int(lab[i]+1),
                bootstrap_membership_agreement=float(agreement[i]),bootstrap_present=int(coverage[i]),bootstrap_repetitions=args.bootstrap))
        fig,ax=plt.subplots(figsize=(8,10),constrained_layout=True)
        for c in range(k):
            mask=lab==c
            ax.scatter(xy[mask,0],xy[mask,1],s=45,c=[PALETTE[c]],label=f'Cluster {c+1} (n={sum(mask)})',edgecolors='white',linewidths=.5)
        for i,village in enumerate(villages):ax.annotate(village,xy[i],xytext=(3,3),textcoords='offset points',fontsize=5.5)
        ax.invert_yaxis();ax.set_aspect('equal');ax.set(xlabel='geo_x',ylabel='geo_y (larger = south)',title=f'{slug} · k={k}\nGeography was not used in clustering');ax.legend(fontsize=8)
        fig.savefig(directory/f'map_k{k}.png',dpi=160);plt.close(fig)
        mean_space_plots(directory/f'spaces_k{k}.png',anchors,lab,villages,times,method,k)
        fig,ax=plt.subplots(figsize=(11,10),constrained_layout=True)
        image=ax.imshow(probability[np.ix_(order,order)],vmin=0,vmax=1,cmap='viridis')
        ax.set_xticks(range(n),[villages[i] for i in order],rotation=90,fontsize=5)
        ax.set_yticks(range(n),[villages[i] for i in order],fontsize=5)
        ax.set_title(f'{slug} · k={k}: bootstrap co-clustering\nConditional on both villages having complete resampled inventories')
        fig.colorbar(image,ax=ax,shrink=.7,label='Fraction of jointly observed bootstrap replicates')
        fig.savefig(directory/f'coassignment_k{k}.png',dpi=150);plt.close(fig)
        for c in range(k):
            for t,time in enumerate(times):
                for v,vowel in enumerate(VOWELS):
                    x,y=anchors[lab==c,t,v].mean(0)
                    means.append(dict(k=k,cluster=c+1,n_villages=int(sum(lab==c)),timepoint=time,vowel=vowel,display_x=x,display_y=y))
        np.savez_compressed(directory/f'coassignment_k{k}.npz',villages=np.array(villages),probability=probability,joint_replicates=cooccurrence)
    features=[]
    for i,village in enumerate(villages):
        row=dict(village=village,geo_x=geo[village][0],geo_y=geo[village][1])
        for t,time in enumerate(times):
            for v,vowel in enumerate(VOWELS):
                for a,axis in enumerate(('x','y')):row[f'{vowel}_{time}_{axis}']=float(anchors[i,t,v,a])
        features.append(row)
    write_csv(directory/'village_features.csv',features);write_csv(directory/'cluster_assignments.csv',assignments);write_csv(directory/'cluster_average_positions.csv',means)
    (directory/'scaling.json').write_text(json.dumps(dict(axis_mean=axis_mean.tolist(),axis_sd=axis_sd.tolist(),time_divisor=float(np.sqrt(len(times))),axis_order='display X, display Y'),indent=2))
    np.savez_compressed(directory/'ward_model.npz',villages=np.array(villages),features=X,linkage=tree,labels=labels,k=np.array(KS))
    print(f'{slug}: {n} villages; median retained in bootstrap {np.median(retained):g}; silhouettes '+', '.join(f'k={r["k"]}: {r["silhouette"]:.3f}' for r in summaries),flush=True)
    return summaries


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'data/nine_vowel_token_trajectories.csv')
    parser.add_argument('--output',type=Path,default=ROOT/'Reports/Nine_vowel_clustering')
    parser.add_argument('--bootstrap',type=int,default=300)
    parser.add_argument('--seed',type=int,default=20261017)
    args=parser.parse_args()
    if args.bootstrap<20:parser.error('Use at least 20 bootstrap repetitions')
    args.output.mkdir(parents=True,exist_ok=True)
    speakers,geo=load_speakers(args.input)
    villages=[];excluded=[]
    for village in sorted(geo):
        missing={}
        for method,by_village in speakers.items():
            median=nanmedian(by_village[village])
            absent=[v for i,v in enumerate(VOWELS) if not np.isfinite(median[:,i,:]).all()]
            if absent:missing[method]=' '.join(absent)
        if missing:excluded.append(dict(village=village,missing=json.dumps(missing,ensure_ascii=False)))
        else:villages.append(village)
    if len(villages)<=max(KS):raise ValueError('Too few complete villages')
    write_csv(args.output/'excluded_villages.csv',excluded,['village','missing'])
    summaries=[]
    for method in ('pca','fasttrack_bark'):
        for times in ((50,),TIMES):summaries.extend(analyse(method,times,speakers[method],villages,geo,args))
    write_csv(args.output/'summary.csv',summaries)
    comparisons=[]
    models={}
    for method in ('pca','fasttrack_bark'):
        for suffix in ('50','20_50_80'):
            slug=method+'_'+suffix;models[slug]=np.load(args.output/slug/'ward_model.npz')['labels']
    names=list(models)
    for i,a in enumerate(names):
        for b in names[i+1:]:
            for j,k in enumerate(KS):comparisons.append(dict(analysis_a=a,analysis_b=b,k=k,adjusted_rand=adjusted_rand(models[a][:,j],models[b][:,j])))
    write_csv(args.output/'between_analysis_agreement.csv',comparisons)
    settings=dict(input=str(args.input),n_villages=len(villages),n_excluded=len(excluded),bootstrap=args.bootstrap,seed=args.seed,
        clustering='Ward hierarchical clustering; Euclidean configuration distances; k=2..6; geography excluded',
        aggregation='Coordinate-wise token medians within speaker/vowel/timepoint, then medians across speakers',
        scaling='One SD per acoustic axis pooled across included villages, vowels and selected timepoints; fixed for bootstrap. Divide concatenated features by sqrt(number of timepoints). No Procrustes alignment or per-village normalization.',
        bootstrap_method='Resample whole speakers within each village; retain all vowel/timepoint measurements together. Villages missing any resampled anchor omitted only in that replicate. No imputation. Coassignment denominators count joint presence; ARI evaluated on retained villages.',
        cluster_averages='Arithmetic mean of village medians, equal village weights; displayed in original acoustic units',
        caution='Descriptive exploratory groups, not established dialect regions. Static and all-timepoint fits are overlapping, not independent evidence. Bootstrap does not resample tokens or refit spectral PCA.')
    (args.output/'settings.json').write_text(json.dumps(settings,indent=2)+'\n')
    template=Path(__file__).with_name('clustering_viewer.html').read_text()
    payload=dict(summaries=summaries,comparisons=comparisons,settings=settings)
    (args.output/'index.html').write_text(template.replace('__DATA__',json.dumps(payload)),encoding='utf-8')
    print(f'Wrote {args.output}',flush=True)


if __name__=='__main__':main()
