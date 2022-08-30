import numpy as np
import pandas as pd
import networkx as nx
import statistics as s
from itertools import product
import csv
from math import floor
import matplotlib.pyplot as plt
import numpy as np
from NNetwork import NNetwork as nn
#import utils.NNetwork as nn
from sklearn import svm
from sklearn import metrics, model_selection
from tqdm import trange
from sklearn.cluster import KMeans
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA ### Use truncated SVD / online PCA later for better computational efficiency
import warnings
from scipy.stats import bernoulli
import random

warnings.filterwarnings("ignore")

def FCA_nx(G, s, k, iteration):
    """Implements the Firefly Cellular Automata model (as part of REU2022@UW-Madison)
       Author: Bella Wu

    Args:
        G (NetworkX Graph): Input graph to the model
        s (array): Current state
        k (int): k-color FCA
        iteration (int): number of iterations

    Returns:
        ret: states at each iteration
        synchronize: whether the system synchronizes at the final iteration
    """
    b = (k-1)//2 # Blinking color
    ret = s
    s_next = np.zeros(G.number_of_nodes())
    for h in range(iteration):
        if h != 0:
            s = s_next # Update to the newest state
            ret = np.vstack((ret, s_next))
        s_next = np.zeros(G.number_of_nodes())
        for i in range(G.number_of_nodes()):
            flag = False # True if inhibited by the blinking neighbor
            if s[i] > b:
                for j in range(G.number_of_nodes()):
                    if s[j] == b and list(G.nodes)[j] in list(G.adj[list(G.nodes)[i]]):
                        flag = True
                if flag:
                    s_next[i] = s[i]
                else:
                    s_next[i] = (s[i]+1)%k
            else:
                s_next[i] = (s[i]+1)%k

    synchronize = False
    if len(np.unique(ret[-1])) == 1 and iteration != 1:
        synchronize = True

    return ret, synchronize

def FCA_nn(G, s, k, p1=1, p2=1, iteration=10):
    """
    Implements the Firefly Cellular Automata model (as part of REU2022@UW-Madison)
       Author: Bella Wu

    Args:
        G: nnetwork
        s: current state (array)
        k: total number of FCA color
        p1: the probability of each vertex's neighbors presenting in the network
        p2: the probability of each vertex's blinking neighbors presenting in the network
        (if both p1 and p2 are set to 1, no stochasticity is given to the network)
        iteration: number of iteration
    """

    b = (k-1)//2 #blinking color
    ret = s #storing the output value
    #s_next = np.zeros(G.number_nodes)
    nodes = G.vertices
    trajectory = [np.asarray(s)]

    for h in range(iteration):
        if h != 0:
            s = s_next #update to the newest state
            #ret = np.vstack((ret, s_next))
            #trajectory.append(s_next)

        s_next = np.zeros(G.number_nodes)

        for i in range(G.number_nodes):
            #randomly select the current node's neighbors
            neighblist = G.neighbors(nodes[i]) #storing the neighbor list of the current node
            neighb_len = sum(bernoulli.rvs(p1, size=len(neighblist))) #flip a coin: randomly generate a number smaller than total number of nodes
            neighblist = random.sample(neighblist, neighb_len) #get the subset of neighborlist

            flag = False #true if inhibited by the blinking neighbor

            if s[i]>b:
                neigh_b = 0 #storing the number of blinking neighbors
                for j in range(G.number_nodes):
                    if s[j] == b and nodes[j] in neighblist:
                        neigh_b += 1

                #randomly select the current node's blinking neighbors
                if sum(bernoulli.rvs(p2, size=neigh_b)) > 0:
                    flag = True

                if flag:
                    s_next[i] = s[i]
                else:
                    s_next[i] = (s[i]+1)%k
            else:
                s_next[i] = (s[i]+1)%k

        trajectory.append(s_next)
    return trajectory

#generate all possible different connected networks with n nodes
#https://matplotlib.org/matplotblog/posts/draw-all-graphs-of-n-nodes/
def make_graphs(n=2, i=None, j=None):
    """Make a graph recursively, by either including, or skipping each edge.

    Edges are given in lexicographical order by construction."""
    out = []
    if i is None: # First call

        out  = [[(0,1)]+r for r in make_graphs(n=n, i=0, j=1)]
    elif j<n-1:
        out += [[(i,j+1)]+r for r in make_graphs(n=n, i=i, j=j+1)]
        out += [          r for r in make_graphs(n=n, i=i, j=j+1)]
    elif i<n-1:
        out = make_graphs(n=n, i=i+1, j=i+1)
    else:
        out = [[]]
    return out

def perm(n, s=None):
    """All permutations of n elements."""
    if s is None: return perm(n, tuple(range(n)))
    if not s: return [[]]
    return [[i]+p for i in s for p in perm(n, tuple([k for k in s if k!=i]))]

def permute(g, n):
    """Create a set of all possible isomorphic codes for a graph,

    as nice hashable tuples. All edges are i<j, and sorted lexicographically."""
    ps = perm(n)
    out = set([])
    for p in ps:
        out.add(tuple(sorted([(p[i],p[j]) if p[i]<p[j]
                              else (p[j],p[i]) for i,j in g])))
    return list(out)

def connected(g):
    """Check if the graph is fully connected, with Union-Find."""
    nodes = set([i for e in g for i in e])
    roots = {node: node for node in nodes}

    def _root(node, depth=0):
        if node==roots[node]: return (node, depth)
        else: return _root(roots[node], depth+1)

    for i,j in g:
        ri,di = _root(i)
        rj,dj = _root(j)
        if ri==rj: continue
        if di<=dj: roots[ri] = rj
        else:      roots[rj] = ri
    return len(set([_root(node)[0] for node in nodes]))==1

def filter(gs, target_nv):
    """Filter all improper graphs: those with not enough nodes,

    those not fully connected, and those isomorphic to previously considered."""
    mem = set({})
    gs2 = []
    for g in gs:
        nv = len(set([i for e in g for i in e]))
        if nv != target_nv:
            continue
        if not connected(g):
            continue
        if tuple(g) not in mem:
            gs2.append(g)
            mem |= set(permute(g, target_nv))
    return gs2

#compute width to check the half-circle concentration
#from L2PSync repo
def width_compute(coloring, kappa):
    differences = [np.max(coloring) - np.min(coloring)]
    for j in range(1,kappa+1):
        shifted = (np.array(coloring) + j) % kappa
        differences.append(np.max(shifted) - np.min(shifted))
    return np.min(differences)


# FCA_iter: total iteration for the FCA model, used for label
# baseline_iter: the iteration for baseline model, usually less than FCA_iter
# num_edges, num_nodes, min_degree, max_degree, diameter, quartile_1, quartile_2, quartile_3, states, y, baseline_width
"""
def FCA_datagen(num_nodes, kappa, FCA_iter, baseline_iter, file_name):
    # generate all possible permutation of color lists
    color_list = list(product(range(0, kappa), repeat=num_nodes))
    # list of all possible different networks with n nodes
    gs = make_graphs(num_nodes)
    gs = filter(gs, num_nodes)

    # generate the toy dataset
    file = open(file_name, 'w+', newline='')

    graph_list = []  # storing the order of graph in the generated dataset

    header = ["num_edges", "num_nodes", "min_degree", "max_degree", "diameter", "quartile_1",
              "quartile_2", "quartile_3", "y", "baseline_width"]
    for i in range(baseline_iter):
        for j in range(num_nodes):
            header.append("s" + str(i + 1) + "_" + str(j + 1))

    with file:
        write = csv.writer(file)

        write.writerow(header)
        for col in color_list:
            for i in gs:
                G = nx.Graph()
                G.add_edges_from(i)
                graph_list.append(i)

                num_edges = G.number_of_edges()
                min_degree = min(list(G.degree), key=lambda x: x[1])[1]
                max_degree = max(list(G.degree), key=lambda x: x[1])[1]
                diameter = nx.diameter(G)
                quartile_1 = s.quantiles(col, n=4)[0]
                quartile_2 = s.quantiles(col, n=4)[1]
                quartile_3 = s.quantiles(col, n=4)[2]

                sample = [num_edges, num_nodes, min_degree, max_degree, diameter,
                          quartile_1, quartile_2, quartile_3]
                states, label = FCA(G, col, kappa, FCA_iter)

                width = width_compute(states[FCA_iter - 1], kappa)
                y = False
                if (width < floor(kappa / 2)):  # half circle concentration
                    y = True
                sample.append(y)

                baseline_width = width_compute(states[baseline_iter - 1], kappa)
                baseline = False
                if (baseline_width < floor(kappa / 2)):  # half circle concentration
                    baseline = True
                sample.append(baseline)

                for j in range(baseline_iter):
                    sample = sample + list(states[j])

                write.writerow(sample)
    return graph_list
"""


### from network_sampling_ex
def coding(X, W, H0,
          r=None,
          a1=0, #L1 regularizer
          a2=0, #L2 regularizer
          sub_iter=[5],
          stopping_grad_ratio=0.0001,
          nonnegativity=True,
          subsample_ratio=1):
    """
    Find \hat{H} = argmin_H ( || X - WH||_{F}^2 + a1*|H| + a2*|H|_{F}^{2} ) within radius r from H0
    Use row-wise projected gradient descent
    """
    H1 = H0.copy()
    i = 0
    dist = 1
    idx = np.arange(X.shape[1])
    if subsample_ratio>1:  # subsample columns of X and solve reduced problem (like in SGD)
        idx = np.random.randint(X.shape[1], size=X.shape[1]//subsample_ratio)
    A = W.T @ W ## Needed for gradient computation
    grad = W.T @ (W @ H0 - X)
    while (i < np.random.choice(sub_iter)):
        step_size = (1 / (((i + 1) ** (1)) * (np.trace(A) + 1)))
        H1 -= step_size * grad
        if nonnegativity:
            H1 = np.maximum(H1, 0)  # nonnegativity constraint
        i = i + 1
    return H1


def ALS(X,
        n_components = 10, # number of columns in the dictionary matrix W
        n_iter=100,
        a0 = 0, # L1 regularizer for H
        a1 = 0, # L1 regularizer for W
        a12 = 0, # L2 regularizer for W
        H_nonnegativity=True,
        W_nonnegativity=True,
        compute_recons_error=False,
        subsample_ratio = 10):

        '''
        Given data matrix X, use alternating least squares to find factors W,H so that
                                || X - WH ||_{F}^2 + a0*|H|_{1} + a1*|W|_{1} + a12 * |W|_{F}^{2}
        is minimized (at least locally)
        '''

        d, n = X.shape
        r = n_components

        #normalization = np.linalg.norm(X.reshape(-1,1),1)/np.product(X.shape) # avg entry of X
        #print('!!! avg entry of X', normalization)
        #X = X/normalization

        # Initialize factors
        W = np.random.rand(d,r)
        H = np.random.rand(r,n)
        # H = H * np.linalg.norm(X) / np.linalg.norm(H)
        for i in trange(n_iter):
            #H = coding_within_radius(X, W.copy(), H.copy(), a1=a0, nonnegativity=H_nonnegativity, subsample_ratio=subsample_ratio)
            #W = coding_within_radius(X.T, H.copy().T, W.copy().T, a1=a1, a2=a12, nonnegativity=W_nonnegativity, subsample_ratio=subsample_ratio).T
            H = coding(X, W.copy(), H.copy(), a1=a0, nonnegativity=H_nonnegativity, subsample_ratio=subsample_ratio)
            W = coding(X.T, H.copy().T, W.copy().T, a1=a1, a2=a12, nonnegativity=W_nonnegativity, subsample_ratio=subsample_ratio).T
            W /= np.linalg.norm(W)
            if compute_recons_error and (i % 10 == 0) :
                print('iteration %i, reconstruction error %f' % (i, np.linalg.norm(X-W@H)**2))
        return W, H


def display_dictionary(W, save_name=None, score=None, grid_shape=None, figsize=[10,10]):
    k = int(np.sqrt(W.shape[0]))
    rows = int(np.sqrt(W.shape[1]))
    cols = int(np.sqrt(W.shape[1]))
    if grid_shape is not None:
        rows = grid_shape[0]
        cols = grid_shape[1]

    figsize0=figsize
    if (score is None) and (grid_shape is not None):
        figsize0=(cols, rows)
    if (score is not None) and (grid_shape is not None):
        figsize0=(cols, rows+0.2)

    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=figsize0,
                            subplot_kw={'xticks': [], 'yticks': []})


    for ax, i in zip(axs.flat, range(100)):
        if score is not None:
            idx = np.argsort(score)
            idx = np.flip(idx)

            ax.imshow(W.T[idx[i]].reshape(k, k), cmap="viridis", interpolation='nearest')
            ax.set_xlabel('%1.2f' % score[i], fontsize=13)  # get the largest first
            ax.xaxis.set_label_coords(0.5, -0.05)
        else:
            ax.imshow(W.T[i].reshape(k, k), cmap="viridis", interpolation='nearest')
            if score is not None:
                ax.set_xlabel('%1.2f' % score[i], fontsize=13)  # get the largest first
                ax.xaxis.set_label_coords(0.5, -0.05)

    plt.tight_layout()
    # plt.suptitle('Dictionary learned from patches of size %d' % k, fontsize=16)
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

    if save_name is not None:
        plt.savefig( save_name, bbox_inches='tight')
    plt.show()
