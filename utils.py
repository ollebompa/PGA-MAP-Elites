'''
Copyright (c) 2020 Scott Fujimoto 
ReplayBuffer Based on Twin Delayed Deep Deterministic Policy Gradients (TD3)
Implementation by Scott Fujimoto https://github.com/sfujim/TD3 Paper: https://arxiv.org/abs/1802.09477
Modified by Olle Nilsson: olle.nilsson19@imperial.ac.uk
'''

'''
Copyright 2019, INRIA
CVT Uility functions based on pymap_elites framework https://github.com/resibots/pymap_elites/blob/master/map_elites/
pymap_elites main contributor(s):
    Jean-Baptiste Mouret, jean-baptiste.mouret@inria.fr
    Eloise Dalin , eloise.dalin@inria.fr
    Pierre Desreumaux , pierre.desreumaux@inria.fr
Modified by Olle Nilsson: olle.nilsson19@imperial.ac.uk
'''



import numpy as np
import torch
from pathlib import Path
from sklearn.cluster import KMeans
from itertools import count
from sklearn.neighbors import KDTree
import pickle



class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6), load=False):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.additions = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def add(self, transitions):
        l = len(transitions[0])
        idx = np.arange(self.ptr, self.ptr + l) % self.max_size
        self.state[idx] = transitions[0]
        self.action[idx] = transitions[1]
        self.next_state[idx] = transitions[2]
        self.reward[idx] = transitions[3]
        self.not_done[idx] = 1. - transitions[4]

        self.ptr = (self.ptr + l) % self.max_size
        self.size = min(self.size + l, self.max_size)
        self.additions += 1


    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
        torch.FloatTensor(self.state[ind]).to(self.device),
        torch.FloatTensor(self.action[ind]).to(self.device),
        torch.FloatTensor(self.next_state[ind]).to(self.device),
        torch.FloatTensor(self.reward[ind]).to(self.device),
        torch.FloatTensor(self.not_done[ind]).to(self.device)
        )


    def sample_state(self, batch_size, steps):
        states = []
        for _ in range(steps):
            ind = np.random.randint(0, self.size, size=batch_size)
            states.append(torch.FloatTensor(self.state[ind]).to(self.device))

        return states


    def save(self, filename):
        with open(f"{filename}", 'wb') as replay_buffer_file:
            pickle.dump(self, replay_buffer_file)

    
    def load(self, filename):
        with open(f"{filename}", 'rb') as replay_buffer_file:
            replay_buffer = pickle.load(replay_buffer_file)
        return replay_buffer
        

class Individual:
    _ids = count(0)
    def __init__(self, x, desc, fitness, centroid=None):
        x.id = next(self._ids)
        Individual.current_id = x.id
        self.x = x
        self.desc = desc
        self.fitness = fitness
        self.centroid = centroid
        self.novelty = None


def add_to_archive(s, centroid, archive, kdt, main=True):
    niche_index = kdt.query([centroid], k=1)[1][0][0]
    niche = kdt.data[niche_index]
    n = make_hashable(niche)
    if main:
        s.centroid = n
    if n in archive:
        if s.fitness > archive[n].fitness:
            if main:
                s.x.novel = False
                s.x.delta_f = s.fitness - archive[n].fitness
            archive[n] = s
            return 1
        return 0
    else:
        archive[n] = s
        if main:
            s.x.novel = True
            s.x.delta_f = None
        return 1


def __centroids_filename(k, dim): 
    return 'CVT/centroids_' + str(k) + '_' + str(dim) + '.dat'


def write_centroids(centroids):
    k = centroids.shape[0]
    dim = centroids.shape[1]
    filename = __centroids_filename(k, dim)
    with open(filename, 'w') as f:
        for p in centroids:
            for item in p:
                f.write(str(item) + ' ') 
            f.write('\n')


def cvt(k, dim, samples, cvt_use_cache=True):
    # check if we have cached values
    fname = __centroids_filename(k, dim)
    if cvt_use_cache:
        if Path(fname).is_file():
            print("WARNING: using cached CVT:", fname)
            if dim == 1:
                if k == 1:
                    return np.expand_dims(np.expand_dims(np.loadtxt(fname), axis=0), axis=1)
                return np.expand_dims(np.loadtxt(fname), axis=1)
            else:
                if k == 1:
                    return np.expand_dims(np.loadtxt(fname), axis=0)
                return np.loadtxt(fname)
    # otherwise, compute cvt
    print("Computing CVT (this can take a while...):", fname)
    x = np.random.rand(samples, dim) 
    k_means = KMeans(init='k-means++', n_clusters=k,
                     n_init=1, max_iter=1000000, n_jobs=1, verbose=1, tol=1e-8) #Full is the proper Expectation Maximization algorithm
    k_means.fit(x)
    write_centroids(k_means.cluster_centers_)
    return k_means.cluster_centers_



def make_hashable(array):
    return tuple(map(float, array))

# format: fitness, centroid, desc, 
# fitness, centroid, desc and are vectors
def save_archive(archive, gen, archive_name, save_path, save_models=False):
    def write_array(a, f):
        for i in a:
            f.write(str(i) + ' ')
    filename = f"{save_path}/archive_{archive_name}_" + str(gen) + '.dat'
    with open(filename, 'w') as f:
        for k in archive.values():
            f.write(str(k.fitness) + ' ')
            write_array(k.centroid, f)
            write_array(k.desc, f)
            f.write(str(k.x.id) + ' ')
            f.write("\n") 
            if save_models:
                k.x.save(f"{save_path}/models/{archive_name}_actor_" + str(k.x.id))
       