"""Plain geographic Gabriel adjacency and adjacency-constrained Ward merges."""
from functools import lru_cache
import heapq
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


@lru_cache(maxsize=512)
def _gabriel(coordinates):
    X=np.asarray(coordinates,float);n=len(X)
    if len(np.unique(X,axis=0))!=n:raise ValueError('Gabriel graph requires distinct coordinates')
    middle=(X[:,None,:]+X[None,:,:])/2
    radius2=np.sum((X[:,None,:]-X[None,:,:])**2,axis=-1)/4
    distance2=np.sum((middle[:,:,None,:]-X[None,None,:,:])**2,axis=-1)
    occupied=distance2<=radius2[:,:,None]+1e-10*np.maximum(1,radius2[:,:,None])
    i,j=np.indices((n,n));occupied[i,j,i]=False;occupied[i,j,j]=False
    adjacency=~occupied.any(-1);np.fill_diagonal(adjacency,False)
    if connected_components(csr_matrix(adjacency),directed=False,return_labels=False)!=1:
        raise ValueError('Gabriel graph disconnected; no silent completion edges are added')
    adjacency.setflags(write=False)
    return adjacency


def gabriel_network(coordinates):
    """Empty CLOSED diameter disk; tolerance relative to squared radius; no cutoff."""
    return _gabriel(tuple(tuple(map(float,xy)) for xy in coordinates))


def constrained_ward_cuts(X,adjacency,ks):
    """Ward SSE-increase criterion, with only graph-adjacent groups eligible.

    Cuts follow merge sequence, not height order (constrained heights can invert).
    Graph edges are contracted after every merge; disconnected groups never merge.
    """
    X=np.asarray(X,float);n=len(X);ks=list(ks)
    if not 1<=min(ks)<=max(ks)<=n:raise ValueError('Invalid requested cluster count')
    if adjacency.shape!=(n,n) or not np.array_equal(adjacency,adjacency.T):raise ValueError('Invalid adjacency')
    means=np.zeros((2*n-1,X.shape[1]));means[:n]=X
    sizes=np.ones(2*n-1,int);active={i:(i,) for i in range(n)}
    neighbours={i:set(np.flatnonzero(adjacency[i])) for i in range(n)}
    heap=[];result={}
    def push(a,b):
        a,b=sorted((a,b));delta=means[a]-means[b]
        cost=float(sizes[a]*sizes[b]/(sizes[a]+sizes[b])*np.dot(delta,delta))
        heapq.heappush(heap,(cost,a,b))
    def save():
        k=len(active)
        if k in ks:
            labels=np.empty(n,int)
            for c,members in enumerate(sorted(active.values(),key=min)):labels[list(members)]=c
            result[k]=labels
    for i,j in zip(*np.triu(adjacency,1).nonzero()):push(int(i),int(j))
    save()
    for new in range(n,2*n-1):
        while heap:
            _,a,b=heapq.heappop(heap)
            if a in active and b in active:break
        else:raise ValueError('No adjacent merge available')
        members=active.pop(a)+active.pop(b)
        sizes[new]=sizes[a]+sizes[b]
        means[new]=(sizes[a]*means[a]+sizes[b]*means[b])/sizes[new]
        adjacent=(neighbours.pop(a)|neighbours.pop(b))-{a,b}
        neighbours[new]=adjacent;active[new]=members
        for other in adjacent:
            neighbours[other].difference_update((a,b));neighbours[other].add(new);push(other,new)
        save()
        if len(active)<=min(ks):break
    return np.column_stack([result[k] for k in ks])
