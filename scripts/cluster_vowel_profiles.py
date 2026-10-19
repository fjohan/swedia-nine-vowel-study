#!/usr/bin/env python3
"""Independent vowel clusters and nested common-core village profiles."""
from __future__ import annotations
import argparse
from collections import defaultdict
import heapq
import json
from pathlib import Path
import numpy as np
from scipy.cluster.hierarchy import linkage,cut_tree
from scipy.optimize import linear_sum_assignment
from gabriel_constrained_clustering import gabriel_network, constrained_ward_cuts
from cluster_village_vowel_spaces import (ROOT,VOWELS,load_speakers,nanmedian,
    adjusted_rand,silhouette,write_csv,plt,COLORS)

THRESHOLDS=tuple(range(9,2,-1))
MODES=('position50','trajectory','movement')


def features(anchors,mode):
    if mode=='position50':return anchors[:,[1],:]
    if mode=='movement':return np.diff(anchors,axis=1)
    return anchors


def profile_hierarchy(cells,tie_rng=None,adjacency=None):
    """Greedy nested groups; every member agrees on >=m SAME named cells."""
    exact=defaultdict(list)
    for i,row in enumerate(cells):exact[tuple(map(int,row))].append(i)
    initial=[]
    for core,members in sorted(exact.items(),key=lambda x:x[1][0]):
        if adjacency is None:initial.append((tuple(members),core));continue
        # Exact-row classes can be disconnected: retain their connected pieces.
        unseen=set(members)
        while unseen:
            start=min(unseen);component={start};unseen.remove(start);stack=[start]
            while stack:
                node=stack.pop();reached={v for v in unseen if adjacency[node,v]}
                unseen-=reached;component|=reached;stack.extend(reached)
            initial.append((tuple(sorted(component)),core))
    active=dict(enumerate(initial))
    heap=[];next_id=len(active)
    def push(a,b):
        members_a,core_a=active[a];members_b,core_b=active[b]
        boundary=int(np.count_nonzero(adjacency[np.ix_(members_a,members_b)])) if adjacency is not None else 0
        if adjacency is not None and boundary==0:return
        core=tuple(x if x==y else -1 for x,y in zip(core_a,core_b))
        shared=sum(x>=0 for x in core)
        # Stronger cores first, then larger union. No geographic tie-breaking.
        tie=float(tie_rng.random()) if tie_rng is not None else min(members_a+members_b)
        heapq.heappush(heap,(-shared,-boundary,-len(members_a)-len(members_b),tie,a,b,core))
    keys=list(active)
    for i,a in enumerate(keys):
        for b in keys[i+1:]:push(a,b)
    result={}
    for threshold in THRESHOLDS:
        if threshold<9:
            while heap:
                item=heapq.heappop(heap);negshared,_,_,_,a,b,core=item
                if a not in active or b not in active:continue
                if -negshared<threshold:heapq.heappush(heap,item);break
                members=tuple(sorted(active[a][0]+active[b][0]))
                del active[a],active[b]
                new=next_id;next_id+=1;active[new]=(members,core)
                for other in list(active):
                    if other!=new:push(min(other,new),max(other,new))
        groups=sorted(active.values(),key=lambda x:(-len(x[0]),x[0]))
        labels=np.empty(len(cells),int)
        for i,(members,core) in enumerate(groups):labels[list(members)]=i
        result[threshold]=dict(labels=labels,groups=groups)
    return result


def partitions_for_vowels(anchors,mode,ks,scales=None,adjacency=None):
    labels=np.empty((len(ks),len(anchors),9),int);diagnostics=[];fitted_scales=[]
    for v in range(9):
        f=features(anchors[:,:,v,:],mode)
        mean,sd=(f.mean(axis=(0,1)),f.std(axis=(0,1),ddof=1)) if scales is None else scales[v]
        sd=np.maximum(sd,1e-10)
        X=((f-mean)/sd).reshape(len(f),-1)/np.sqrt(f.shape[1])
        if adjacency is None:
            tree=linkage(X,method='ward');cuts=cut_tree(tree,n_clusters=ks)
        else:cuts=constrained_ward_cuts(X,adjacency,ks)
        labels[:,:,v]=cuts.T;fitted_scales.append((mean,sd))
        if scales is None:
            for j,k in enumerate(ks):diagnostics.append(dict(vowel=VOWELS[v],k=k,silhouette=silhouette(X,cuts[:,j]),
                cluster_sizes=' '.join(str(sum(cuts[:,j]==c)) for c in range(k))))
    return labels,fitted_scales,diagnostics


def plot_vowels(path,anchors,cells,k,method,mode):
    fig,axes=plt.subplots(3,3,figsize=(13,11),constrained_layout=True)
    colors=plt.get_cmap('tab10').colors
    for v,(vowel,ax) in enumerate(zip(VOWELS,axes.flat)):
        for c in range(k):
            a=anchors[cells[:,v]==c,:,v,:];mean=a.mean(0)
            t=[1] if mode=='position50' else [0,1,2]
            ax.scatter(a[:,t,0],a[:,t,1],s=8,alpha=.12,color=colors[c])
            ax.plot(mean[t,0],mean[t,1],'o-',color=colors[c],label=f'C{c+1} (n={len(a)})',lw=2,ms=5)
            for i in t:ax.annotate(str((20,50,80)[i]),mean[i],xytext=(4,3),textcoords='offset points',fontsize=7,color=colors[c])
        ax.set_title('/'+vowel+'/');ax.grid(alpha=.15);ax.set_aspect('equal',adjustable='datalim');ax.legend(fontsize=6)
        ax.set_xlabel('−PC1' if method=='pca' else '−F2 (Bark)');ax.set_ylabel('−PC2' if method=='pca' else '−F1 (Bark)')
    fig.suptitle(f'{method} · {mode} · {k} clusters per vowel\nMeans shown in original acoustic space; movement clustering uses differences, not these absolute positions')
    fig.savefig(path,dpi=150);plt.close(fig)


def analyse(method,mode,speakers,villages,geo,args):
    source=[speakers[v] for v in villages];anchors=np.array([nanmedian(a) for a in source])
    graph=gabriel_network([geo[v] for v in villages]) if args.geography!='none' else None
    profile_graph=graph if args.geography=='both' else None
    labels,scales,diagnostics=partitions_for_vowels(anchors,mode,args.clusters,adjacency=graph)
    originals=[profile_hierarchy(c,adjacency=profile_graph) for c in labels];n=len(villages);reps=args.bootstrap
    rng=np.random.default_rng(args.seed)
    row_return=np.zeros((len(args.clusters),n));cell_return=np.zeros((len(args.clusters),n,9));coverage=np.zeros(n)
    aris=defaultdict(list);vowel_aris=defaultdict(list);jac=defaultdict(lambda:np.zeros(n));pair_sum=defaultdict(lambda:np.zeros((n,n)))
    joint=np.zeros((n,n));retained=[]
    for b in range(reps):
        boot=np.array([nanmedian(a[rng.integers(0,len(a),len(a))]) for a in source])
        valid=np.isfinite(boot).all(axis=(1,2,3));ids=np.flatnonzero(valid);retained.append(len(ids))
        if len(ids)<=max(args.clusters):continue
        coverage[ids]+=1;joint[np.ix_(ids,ids)]+=1
        bootgraph=gabriel_network([geo[villages[i]] for i in ids]) if graph is not None else None
        bootlabels,_,_=partitions_for_vowels(boot[valid],mode,args.clusters,scales,bootgraph)
        for j,k in enumerate(args.clusters):
            aligned=np.empty_like(bootlabels[j])
            for v in range(9):
                base=labels[j,valid,v];sample=bootlabels[j,:,v]
                vowel_aris[j,v].append(adjusted_rand(base,sample))
                table=np.zeros((k,k),int);np.add.at(table,(base,sample),1)
                a,c=linear_sum_assignment(-table);mapping=dict(zip(c,a));aligned[:,v]=[mapping[x] for x in sample]
            matches=aligned==labels[j,valid]
            cell_return[j,ids]+=matches;row_return[j,ids]+=matches.all(1)
            hierarchy=profile_hierarchy(bootlabels[j],adjacency=bootgraph if args.geography=='both' else None)
            for threshold in THRESHOLDS:
                base=originals[j][threshold]['labels'][valid];sample=hierarchy[threshold]['labels']
                aris[j,threshold].append(adjusted_rand(base,sample))
                pair_sum[j,threshold][np.ix_(ids,ids)]+=(sample[:,None]==sample[None,:])
                for i,index in enumerate(ids):
                    old=base==base[i];new=sample==sample[i]
                    jac[j,threshold][index]+=np.count_nonzero(old&new)/np.count_nonzero(old|new)
    tie_scores=defaultdict(list)
    tie_rng=np.random.default_rng(args.seed+1)
    for j,k in enumerate(args.clusters):
        for _ in range(args.tie_repetitions):
            alt=profile_hierarchy(labels[j],tie_rng,profile_graph)
            for t in THRESHOLDS:tie_scores[j,t].append(adjusted_rand(originals[j][t]['labels'],alt[t]['labels']))
    configurations=[];summaries=[]
    for j,k in enumerate(args.clusters):
        slug=f'{method}_{mode}_k{k}';directory=args.output/slug;directory.mkdir(exist_ok=True)
        plot_vowels(directory/'vowel_clusters.png',anchors,labels[j],k,method,mode)
        row_stability=np.divide(row_return[j],coverage,out=np.full(n,np.nan),where=coverage>0)
        cell_stability=np.divide(cell_return[j],coverage[:,None],out=np.full((n,9),np.nan),where=coverage[:,None]>0)
        rows=[];group_rows=[];threshold_payload={}
        for t in THRESHOLDS:
            groups=originals[j][t]['groups'];lab=originals[j][t]['labels'];scores=np.array(aris[j,t])
            jaccard=np.divide(jac[j,t],coverage,out=np.full(n,np.nan),where=coverage>0)
            co=np.divide(pair_sum[j,t],joint,out=np.full((n,n),np.nan),where=joint>0)
            payload_groups=[]
            for g,(members,core) in enumerate(groups):
                ix=np.array(members);intra=co[np.ix_(ix,ix)];mask=~np.eye(len(ix),dtype=bool)
                pair_retained=float(np.nanmean(intra[mask])) if len(ix)>1 else None
                item=dict(group=g+1,n_villages=len(members),members=[villages[i] for i in members],
                    shared_vowels=[VOWELS[v] for v,c in enumerate(core) if c>=0],
                    core=[int(c+1) if c>=0 else None for c in core],
                    mean_bootstrap_jaccard=float(np.nanmean(jaccard[ix])),pair_retained=pair_retained)
                payload_groups.append(item)
                group_rows.append(dict(threshold=t,group=g+1,n_villages=len(members),members=' '.join(item['members']),
                    common_core=' '.join(f'{VOWELS[v]}={c+1}' for v,c in enumerate(core) if c>=0),
                    n_shared_vowels=len(item['shared_vowels']),mean_bootstrap_jaccard=item['mean_bootstrap_jaccard'],
                    pair_retained=pair_retained if pair_retained is not None else ''))
            summary=dict(analysis=slug,method=method,mode=mode,k=k,threshold=t,n_groups=len(groups),
                n_singletons=sum(len(m)==1 for m,c in groups),largest_group=max(len(m) for m,c in groups),
                distinct_exact_rows=len({tuple(row) for row in labels[j]}),
                bootstrap_ari_median=float(np.median(scores)),bootstrap_ari_p025=float(np.quantile(scores,.025)),bootstrap_ari_p975=float(np.quantile(scores,.975)),
                village_jaccard_median=float(np.nanmedian(jaccard)),exact_row_return_median=float(np.nanmedian(row_stability)),
                tie_order_ari_median=float(np.median(tie_scores[j,t])),tie_order_ari_min=float(np.min(tie_scores[j,t])),
                bootstrap_retained_median=float(np.median(retained)))
            summaries.append(summary)
            threshold_payload[str(t)]=dict(labels=(lab+1).tolist(),groups=payload_groups,summary=summary,jaccard=jaccard.tolist())
            for i,village in enumerate(villages):
                rows.append(dict(village=village,geo_x=geo[village][0],geo_y=geo[village][1],threshold=t,group=int(lab[i]+1),
                    **{v:int(labels[j,i,vv]+1) for vv,v in enumerate(VOWELS)},
                    exact_row_return=float(row_stability[i]),bootstrap_jaccard=float(jaccard[i]),bootstrap_present=int(coverage[i]),bootstrap_repetitions=reps))
        diag=[dict(d,bootstrap_ari_median=float(np.median(vowel_aris[j,VOWELS.index(d['vowel'])]))) for d in diagnostics if d['k']==k]
        write_csv(directory/'village_profiles.csv',rows);write_csv(directory/'groups.csv',group_rows);write_csv(directory/'vowel_diagnostics.csv',diag)
        np.savez_compressed(directory/'bootstrap_profile_groups.npz',villages=np.array(villages),thresholds=np.array(THRESHOLDS),joint_replicates=joint,
            coassignment=np.array([np.divide(pair_sum[j,t],joint,out=np.full((n,n),np.nan),where=joint>0) for t in THRESHOLDS]))
        write_csv(directory/'axis_scaling.csv',[dict(vowel=v,mean_x=float(s[0][0]),mean_y=float(s[0][1]),sd_x=float(s[1][0]),sd_y=float(s[1][1])) for v,s in zip(VOWELS,scales)])
        configurations.append(dict(slug=slug,method=method,mode=mode,k=k,cells=(labels[j]+1).tolist(),thresholds=threshold_payload,
            row_return=row_stability.tolist(),cell_return=cell_stability.tolist(),coverage=coverage.astype(int).tolist(),vowel_diagnostics=diag))
        exact=threshold_payload['9']['summary']
        print(f'{slug}: exact {exact["n_groups"]} profiles ({exact["n_singletons"]} singleton); ≥3: {threshold_payload["3"]["summary"]["n_groups"]} groups; exact-row return {exact["exact_row_return_median"]:.1%}',flush=True)
    return configurations,summaries


def save_report(output,payload):
    settings=payload['settings'];geography=settings['geography']
    settings['profile_matching']=('Gabriel-connected components of exact rows at 9; adjacent-only nested common-core merges at 8..3.' if geography=='both' else 'Exact rows at 9; greedy nested common-core groups at 8..3.')+' Identical labels on the SAME named vowels across ALL members; not pairwise-only thresholding.'
    settings['merge_priority']=('More shared named cells, then more boundary edges, then larger combined group, then fixed alphabetical village index and internal IDs.' if geography=='both' else 'More shared named cells, then larger combined group, then fixed alphabetical village index and internal IDs.')+' Randomized tie-order audit changes only the final tie criterion.'
    (output/'settings.json').write_text(json.dumps(settings,ensure_ascii=False,indent=2)+'\n')
    (output/'profiles.json').write_text(json.dumps(payload,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    template=Path(__file__).with_name('vowel_profile_viewer.html').read_text(encoding='utf-8')
    statement={'none':'No geographic constraint is applied at either stage.',
        'vowels':'The individual-vowel clusters are Gabriel-constrained, but exact and relaxed profile groups are not geographically constrained.',
        'both':'Both individual-vowel clusters and profile merges are Gabriel-constrained. At 9/9, groups are connected pieces of exact rows, not necessarily whole row-equality classes.'}[geography]
    template=template.replace('No geographic constraint is applied at either stage.',statement)
    if geography=='both':
        template=template.replace('At 9/9, a profile is exactly one distinct row.','At 9/9, identical rows are split into their Gabriel-connected components; one row can therefore form several connected groups.')
        template=template.replace('Exact profiles are literal row equality, unaffected by any within-column relabelling.','Exact categorical rows are invariant to within-column relabelling. This connected version divides each such row into its connected components before relaxation.')
        template=template.replace('then deterministic alphabetical/internal ordering.','then deterministic alphabetical/internal ordering. For geographically constrained profile merges, more shared boundary edges take priority before combined group size.')
    if geography!='none':
        template=template.replace('<title>Nine-vowel village profiles</title>','<title>Nine-vowel profiles: Gabriel '+('both stages' if geography=='both' else 'vowel inputs')+'</title>')
    (output/'index.html').write_text(template.replace('__DATA__',json.dumps(payload,ensure_ascii=False,allow_nan=False)),encoding='utf-8')
    print(f'Wrote {output}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'data/nine_vowel_token_trajectories.csv')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--geography',choices=('none','vowels','both'),default='none',
        help='Gabriel constraints: none, individual vowel Ward clusters only, or both vowel clusters and profile hierarchy.')
    parser.add_argument('--render-only',action='store_true',help='Rebuild HTML and method metadata from existing profiles.json without reclustering.')
    parser.add_argument('--clusters',nargs='+',type=int,default=[2,3,4])
    parser.add_argument('--bootstrap',type=int,default=200)
    parser.add_argument('--tie-repetitions',type=int,default=20)
    parser.add_argument('--seed',type=int,default=20261018)
    args=parser.parse_args()
    if args.bootstrap<20 or args.tie_repetitions<1 or min(args.clusters)<2:parser.error('Use >=20 bootstrap replicates, >=1 tie repeat, and >=2 clusters')
    args.clusters=sorted(set(args.clusters))
    if args.output is None:
        suffix={'none':'','vowels':'_Gabriel_inputs','both':'_Gabriel'}[args.geography]
        args.output=ROOT/'Reports'/('Nine_vowel_profiles'+suffix)
    args.output.mkdir(parents=True,exist_ok=True)
    if args.render_only:
        save_report(args.output,json.loads((args.output/'profiles.json').read_text(encoding='utf-8')))
        return
    speakers,geo=load_speakers(args.input)
    villages=[v for v in sorted(geo) if all(np.isfinite(nanmedian(speakers[m][v])).all() for m in speakers)]
    if len(villages)<=max(args.clusters):raise ValueError('Too few complete villages')
    excluded=[dict(village=v,reason='Missing at least one vowel/timepoint/coordinate') for v in sorted(set(geo)-set(villages))]
    write_csv(args.output/'excluded_villages.csv',excluded,['village','reason'])
    results=[];summaries=[]
    for method in ('pca','fasttrack_bark'):
        for mode in MODES:
            r,s=analyse(method,mode,speakers[method],villages,geo,args);results.extend(r);summaries.extend(s)
    settings=dict(input=str(args.input),n_villages=len(villages),bootstrap=args.bootstrap,seed=args.seed,tie_repetitions=args.tie_repetitions,
        clusters_per_vowel=args.clusters,geography=args.geography,
        graph='Plain Gabriel graph, no distance cutoff; empty closed diameter disk with radius-squared-relative tolerance 1e-10. Rebuilt on retained villages in each bootstrap. No acoustic information, silent graph completion or travel/water constraints.',
        ward='Adjacency-constrained Ward minimizes SSE increase among graph-linked groups; unit village weights; cuts follow merge sequence rather than sorting potentially nonmonotone heights.' if args.geography!='none' else 'Unconstrained Ward',
        aggregation='Token medians per speaker/vowel/timepoint, then medians across speakers',
        modes=dict(position50='2 coordinates at 50%',trajectory='6 coordinates at 20/50/80%',movement='4 differences: 50−20 and 80−50, each with 2 acoustic axes'),
        vowel_clustering='Independent Ward trees for each vowel/method/mode; one pooled SD per axis within vowel; divide concatenation by sqrt(number of temporal vectors); no Procrustes alignment',
        profile_matching=('Gabriel-connected components of exact rows at 9; adjacent-only nested common-core merges at 8..3.' if args.geography=='both' else 'Exact rows at 9; greedy nested common-core groups at 8..3.')+' Identical labels on the SAME named vowels across ALL members; not pairwise-only thresholding.',
        merge_priority=('More shared named cells, then more boundary edges, then larger combined group, then fixed alphabetical village index and internal IDs.' if args.geography=='both' else 'More shared named cells, then larger combined group, then fixed alphabetical village index and internal IDs.')+' Randomized tie-order audit changes only the final tie criterion.',
        bootstrap_method='Whole speakers resampled jointly across vowels/timepoints. Incomplete resampled villages omitted for that replicate. Per-vowel clusters and profile hierarchy both refitted. Axis scales and spectral PCA fixed. Label-matched exact-row return; label-invariant group ARI/Jaccard/pair retention conditional on presence.',
        caution='Categorical compression and prescribed k can create or split profiles; shared-core partitions are greedy and tie-sensitive, not unique optimal regions. Exact-row return is especially stringent over nine cells.')
    (args.output/'settings.json').write_text(json.dumps(settings,ensure_ascii=False,indent=2)+'\n')
    write_csv(args.output/'summary.csv',summaries)
    payload=dict(villages=villages,geo=[geo[v] for v in villages],vowels=VOWELS,analyses=results,settings=settings)
    if args.geography!='none':
        graph=gabriel_network([geo[v] for v in villages]);edges=[dict(village_a=villages[i],village_b=villages[j],geo_distance=float(np.linalg.norm(np.array(geo[villages[i]])-geo[villages[j]]))) for i,j in zip(*np.triu(graph,1).nonzero())]
        payload['gabriel_edges']=[[int(i),int(j)] for i,j in zip(*np.triu(graph,1).nonzero())]
        settings['n_gabriel_edges']=len(edges);write_csv(args.output/'gabriel_edges.csv',edges)
        settings['profile_constraint']='Connected pieces of exact rows at 9, then adjacent-only common-core merges; priority shared cells, boundary-edge count, combined size, fixed order' if args.geography=='both' else 'Literal exact rows and unconstrained relaxed profile merges; profiles need not be connected'
        (args.output/'settings.json').write_text(json.dumps(settings,ensure_ascii=False,indent=2)+'\n')
    save_report(args.output,payload)


if __name__=='__main__':main()
