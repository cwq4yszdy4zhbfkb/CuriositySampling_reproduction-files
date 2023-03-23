import os
import numpy as np
import pyemma
import pathlib

# calculates fraction of configurations explored in the cluster-discretized configurational landscape
def cluster_fraction(featurized_trajectory, clustering_alg):
    trajectory_all_ind = set()
    all_clusters_index = set(list(np.concatenate(clustering_alg.transform(clustering_alg.clustercenters))))
    fractions = []
    if isinstance(featurized_trajectory, list):
        traj_len = featurized_trajectory[0].shape[0]
    for i in range(traj_len):
        features = np.concatenate([f[i] for f in featurized_trajectory], axis=0)
        current_ind = set(np.squeeze(clustering_alg.transform(features)))
        trajectory_all_ind = current_ind | trajectory_all_ind
        frac = len(trajectory_all_ind)/len(all_clusters_index)
        fractions.append(frac)
    return fractions

# create features from files, turn off cossin if you use them for ploting/clustering (it's better to cluster in 2D)
def featurize(pdb, files, cossin=True):
    torsions_feat = pyemma.coordinates.featurizer(pdb)
    # periodic off for implicit solvent as RfaH-CTD
    torsions_feat.add_backbone_torsions(cossin=cossin, periodic=False)
    torsions_data = pyemma.coordinates.load(files, features=torsions_feat, stride=1)
    return torsions_data

# dmin=0.25 for RfaH-CTD, dmin=0.05 for WLALL5
cluster = pyemma.coordinates.cluster_regspace(tica_output, dmin=0.25, max_centers=5000, stride=1, fixed_seed=1)
dtrajs_concatenated = np.concatenate(cluster.dtrajs)

directory_bd = "/home/USER/PROJECT/trajectories"
dcd_files = [str(path.absolute()) for path in list(pathlib.Path(directory_bd).glob('**/*_traj.dcd'))]
dcd_files.sort(key=os.path.getmtime)
pdb_files = '/home/USER/PROJECT/trajectories/step5_input.pdb'

features = featurize(pdb_files, dcd_files, cossin=True)
features_bd = tica.transform(features)
features_bd = [f.reshape(-1, 100, f.shape[-1]) for f in features_bd]
fractions = cluster_fraction(features_bd, cluster)
