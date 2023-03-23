import pyemma
from pyemma.util.contexts import settings
import mdtraj as md
import pandas as pd
import numpy as np
import os
import tqdm
import sys
from glob import glob
import shutil
from copy import copy, deepcopy

import logging
urllib3_logger = logging.getLogger('pyemma')
urllib3_logger.setLevel(logging.CRITICAL)

# based on https://github.com/choderalab/msm-mfpt/blob/master/vamp_scoring/automatic_eigenvalue/villin/villin_score_msmlag10ns_splittica_alleigen.py
# modified by cwq4yszdy4zhbfkb 
def score_vamp(featurized_trajs, tica_lag, tica_dim, microstate_number, msm_lag, number_of_splits=2, repeats=15, neig=51, stride=1, nonrev=True):
    """
    
    
    returns: list (of size number of splits) of arrays with shape ((neig - 2), 4), where the 4 are elements (test vamp2 km, test vamp2 cm, train vamp2 km, train vamp2 cm)
    neig can differ if the number of microstates is too low.
    """
    validation_fraction=1/number_of_splits
    number_of_splits=number_of_splits*repeats
    scores = []
    # stride trajectories
    featurized_trajs = [f[::stride] for f in featurized_trajs]
    with pyemma.util.contexts.settings(show_progress_bars=False):
        for n in range(number_of_splits):
            ntot = int(len(featurized_trajs))
            nval = int(len(featurized_trajs) * validation_fraction)
            ival = np.random.choice(len(featurized_trajs), size=nval, replace=False)
            # 0 - train, 1 - test
            split = [np.array([i for i in range(ntot) if i not in ival]), np.array([i for i in range(ntot) if i in ival])]
            train_data = [f for i, f in enumerate(featurized_trajs) if i in split[0]]
            test_data = [f for i, f in enumerate(featurized_trajs) if i in split[1]]

            if tica_dim == '95p':
                tica_kinetic = pyemma.coordinates.tica(train_data, lag=tica_lag, var_cutoff=1, kinetic_map=True)
                dim_kinetic = np.argwhere(np.cumsum(tica_kinetic.eigenvalues**2)/np.sum(tica_kinetic.eigenvalues**2) > 0.95)[0,0] + 1
                Y_kinetic_train = tica_kinetic.transform(train_data)
                Y_kinetic_test = tica_kinetic.transform(test_data)
                
                tica_commute = pyemma.coordinates.tica(train_data, lag=tica_lag, var_cutoff=1, kinetic_map=False, commute_map=True)
                dim_commute = np.argwhere(np.cumsum(tica_commute.timescales)/np.sum(tica_commute.timescales) > 0.95)[0,0] + 1
                Y_commute_train = tica_commute.transform(train_data)
                Y_commute_test = tica_commute.transform(test_data)

                if nonrev:
                    tica_nrev = pyemma.coordinates.tica(train_data, lag=tica_lag, var_cutoff=1, kinetic_map=False, commute_map=False, reversible=False)
                    dim_nrev = np.argwhere(np.cumsum(tica_nrev.timescales)/np.sum(tica_commute.timescales) > 0.95)[0,0] + 1       
                    Y_nrev_train = tica_commute.transform(train_data)
                    Y_nrev_test = tica_commute.transform(test_data)

            else:
                tica_kinetic = pyemma.coordinates.tica(train_data, lag=tica_lag, dim=tica_dim, kinetic_map=True)
                Y_kinetic_train = tica_kinetic.get_output()
                Y_kinetic_test = tica_kinetic.transform(test_data)

                
                tica_commute = pyemma.coordinates.tica(train_data, lag=tica_lag, dim=tica_dim, kinetic_map=False, commute_map=True)
                Y_commute_train = tica_commute.get_output()
                Y_commute_test = tica_commute.transform(test_data)

                if nonrev:
                    tica_nrev = pyemma.coordinates.tica(train_data, lag=tica_lag, dim=tica_dim, kinetic_map=False, commute_map=False, reversible=False)
                    Y_nrev_train = tica_commute.get_output()
                    Y_nrev_train = [t.real for t in Y_nrev_train]
                    Y_nrev_test = tica_commute.transform(test_data)
                    Y_nrev_test = [t.real for t in Y_nrev_test]

            rep_j = 0
            conv_try = 5
            try:
                while rep_j <= conv_try:
                    try:
                        kmeans_kinetic = pyemma.coordinates.cluster_kmeans(Y_kinetic_train, k=microstate_number, max_iter=500, stride=10)
                        dtrajs_kinetic_train = kmeans_kinetic.dtrajs
                        dtrajs_kinetic_test = kmeans_kinetic.transform(Y_kinetic_test)
                        dtrajs_kinetic_test = [np.concatenate(traj) for traj in dtrajs_kinetic_test]
        
                        kmeans_commute = pyemma.coordinates.cluster_kmeans(Y_commute_train, k=microstate_number, max_iter=500, stride=10)
                        dtrajs_commute_train = kmeans_commute.dtrajs
                        dtrajs_commute_test = kmeans_commute.transform(Y_commute_test)
                        dtrajs_commute_test = [np.concatenate(traj) for traj in dtrajs_commute_test]
                        if nonrev:
                            kmeans_nrev = pyemma.coordinates.cluster_kmeans(Y_nrev_train, k=microstate_number, max_iter=500, stride=10)
                            dtrajs_nrev_train = kmeans_nrev.dtrajs
                            dtrajs_nrev_test = kmeans_nrev.transform(Y_nrev_test)
                            dtrajs_nrev_test = [np.concatenate(traj) for traj in dtrajs_nrev_test]
                    
                        # we're doing only VAMP-2 and automatically selecting the max. number of eigenvalues later
                        # for now have to enumerate as many k's as possible - let's go every 1 up to min number of clusters in any model
                        msm_kinetic = pyemma.msm.estimate_markov_model(dtrajs_kinetic_train, msm_lag, score_method='VAMP2')
                        msm_commute = pyemma.msm.estimate_markov_model(dtrajs_commute_train, msm_lag, score_method='VAMP2')

                        if nonrev:
                            msm_nrev = pyemma.msm.estimate_markov_model(dtrajs_nrev_train, msm_lag, score_method='VAMP2')
                        scores_eig = []
                        if tica_dim == '95p':
                            for score_k in np.arange(2, neig):
                                score_kinetic = msm_kinetic.score(dtrajs_kinetic_test, score_method='VAMP2', score_k=score_k)
                                score_kinetic_train = msm_kinetic.score(dtrajs_kinetic_train, score_method='VAMP2', score_k=score_k)
                                score_commute = msm_commute.score(dtrajs_commute_test, score_method='VAMP2', score_k=score_k)
                                score_commute_train = msm_commute.score(dtrajs_commute_train, score_method='VAMP2', score_k=score_k)
                                if nonrev:
                                    score_nrev = msm_nrev.score(dtrajs_nrev_test, score_method='VAMP2', score_k=score_k)
                                    score_nrev_train = msm_nrev.score(dtrajs_nrev_train, score_method='VAMP2', score_k=score_k)
                                    scores_eig.append([(score_kinetic, score_commute, score_nrev, score_kinetic_train, score_commute_train, score_nrev_train), (dim_kinetic, dim_commute, dim_nrev)])
                                else:
                                    scores_eig.append([(score_kinetic, score_commute, score_kinetic_train, score_commute_train), (dim_kinetic, dim_commute)])
                
                        else:
                            for score_k in np.arange(2, neig):
                                score_kinetic = msm_kinetic.score(dtrajs_kinetic_test, score_method='VAMP2', score_k=score_k)
                                score_kinetic_train = msm_kinetic.score(dtrajs_kinetic_train, score_method='VAMP2', score_k=score_k)
                                score_commute = msm_commute.score(dtrajs_commute_test, score_method='VAMP2', score_k=score_k)
                                score_commute_train = msm_commute.score(dtrajs_commute_train, score_method='VAMP2', score_k=score_k)
                                if nonrev:
                                    score_nrev = msm_commute.score(dtrajs_nrev_test, score_method='VAMP2', score_k=score_k)
                                    score_nrev_train = msm_commute.score(dtrajs_nrev_train, score_method='VAMP2', score_k=score_k)

                                    scores_eig.append((score_kinetic, score_commute, score_nrev, score_kinetic_train, score_commute_train, score_nrev_train))
                                else:
                                    scores_eig.append((score_kinetic, score_commute, score_kinetic_train, score_commute_train))
                        assert(len(scores_eig) > 0)
                        scores_eig_array = np.array(scores_eig)
                        
                        scores.append(scores_eig_array)
                        rep_j = 1e6
                        break          
                    except Exception as e:
                        print("Error that occured: {}".format(str(e)))
                        if str(e) != "Stationary distribution contains entries smaller than 1e-15 during iteration":
                            raise Exception("Other than MSM convergence issue raised")
                        if rep_j > conv_try:
                            raise Exception("MSM model failed too many times to converge")
                        else:
                            rep_j +=1
            except KeyboardInterrupt:
                print('Interrupted')
                return None
        
    return np.array(scores)


def scores_to_df(scores_dict, nonrev=True):
    size = np.array(list(scores_dict.values())).size
    df_dict = {'Hyperparameters': {}, 'VAMP-2 score': {}, 'Score type': {}, 'Repetition': {}, 'Score % of dim.': {}, 'Number of eigenvalues': {}}
    if nonrev:
        sc_type_array = ['rev-km', 'rev-cm', 'nonrev', 'rev-km', 'rev-cm', 'nonrev']
        score_type_array = ['Test scores', 'Test scores', 'Test scores', 'Training scores', 'Training scores', 'Training scores']
    else:
        sc_type_array = ['km', 'cm', 'km', 'cm']
        score_type_array = ['Test scores', 'Test scores', 'Training scores', 'Training scores']
    _ind = 0
    for i, (key, value) in enumerate(scores_dict.items()):
        for j, rep in enumerate(value):
            for k, eigen in enumerate(rep):
                for l, scaling in enumerate(eigen):
                    sc_type = sc_type_array[l]
                    # key: needs to be string #dims, lagtime, #clusters
                    df_dict['Hyperparameters'][_ind] = "("+sc_type+" ,"+key+")"
                    # assue we start from 2
                    df_dict['Number of eigenvalues'][_ind] = k + 2
                    df_dict['Score % of dim.'][_ind] = (scaling-1)/(k + 2)
                    df_dict['VAMP-2 score'][_ind] = scaling
                    df_dict['Score type'][_ind] = score_type_array[l]
                    df_dict['Repetition'][_ind] = j+1
                    _ind += 1
    # -1 because it end with +=1 at the end
    #assert size == _ind
    return pd.DataFrame(df_dict)

# Modified script from MDTRAJ that returns contact indices
from __future__ import print_function, division
from mdtraj.utils import ensure_type
from mdtraj.utils.six import string_types
from mdtraj.utils.six.moves import xrange
from mdtraj.core import element
import mdtraj as md
import itertools
__all__ = ['compute_contacts', 'squareform']

##############################################################################
# Code
##############################################################################

def compute_contacts(traj, contacts='all', scheme='closest-heavy', ignore_nonprotein=True, periodic=True,
                     soft_min=False, soft_min_beta=20):
    """Compute the distance between pairs of residues in a trajectory.
    Parameters
    ----------
    traj : md.Trajectory
        An mdtraj trajectory. It must contain topology information.
    contacts : array-like, ndim=2 or 'all'
        An array containing pairs of indices (0-indexed) of residues to
        compute the contacts between, or 'all'. The string 'all' will
        select all pairs of residues separated by two or more residues
        (i.e. the i to i+1 and i to i+2 pairs will be excluded).
    scheme : {'ca', 'closest', 'closest-heavy', 'sidechain', 'sidechain-heavy'}
        scheme to determine the distance between two residues:
            'ca' : distance between two residues is given by the distance
                between their alpha carbons
            'closest' : distance is the closest distance between any
                two atoms in the residues
            'closest-heavy' : distance is the closest distance between
                any two non-hydrogen atoms in the residues
            'sidechain' : distance is the closest distance between any
                two atoms in residue sidechains
            'sidechain-heavy' : distance is the closest distance between
                any two non-hydrogen atoms in residue sidechains
    ignore_nonprotein : bool
        When using `contact==all`, don't compute contacts between
        "residues" which are not protein (i.e. do not contain an alpha
        carbon).
    periodic : bool, default=True
        If periodic is True and the trajectory contains unitcell information,
        we will compute distances under the minimum image convention.
    soft_min : bool, default=False
        If soft_min is true, we will use a diffrentiable version of
        the scheme. The exact expression used
         is d = \frac{\beta}{log\sum_i{exp(\frac{\beta}{d_i}})} where
         beta is user parameter which defaults to 20nm. The expression
         we use is copied from the plumed mindist calculator.
         http://plumed.github.io/doc-v2.0/user-doc/html/mindist.html
    soft_min_beta : float, default=20nm
        The value of beta to use for the soft_min distance option.
        Very large values might cause small contact distances to go to 0.
    Returns
    -------
    distances : np.ndarray, shape=(n_frames, n_pairs), dtype=np.float32
        Distances for each residue-residue contact in each frame
        of the trajectory
    residue_pairs : np.ndarray, shape=(n_pairs, 2), dtype=int
        Each row of this return value gives the indices of the residues
        involved in the contact. This argument mirrors the `contacts` input
        parameter. When `all` is specified as input, this return value
        gives the actual residue pairs resolved from `all`. Furthermore,
        when scheme=='ca', any contact pair supplied as input corresponding
        to a residue without an alpha carbon (e.g. HOH) is ignored from the
        input contacts list, meanings that the indexing of the
        output `distances` may not match up with the indexing of the input
        `contacts`. But the indexing of `distances` *will* match up with
        the indexing of `residue_pairs`
    Examples
    --------
    >>> # To compute the contact distance between residue 0 and 10 and
    >>> # residues 0 and 11
    >>> md.compute_contacts(t, [[0, 10], [0, 11]])
    >>> # the itertools library can be useful to generate the arrays of indices
    >>> group_1 = [0, 1, 2]
    >>> group_2 = [10, 11]
    >>> pairs = list(itertools.product(group_1, group_2))
    >>> print(pairs)
    [(0, 10), (0, 11), (1, 10), (1, 11), (2, 10), (2, 11)]
    >>> md.compute_contacts(t, pairs)
    See Also
    --------
    mdtraj.geometry.squareform : turn the result from this function
        into a square "contact map"
    Topology.residue : Get residues from the topology by index
    """
    if traj.topology is None:
        raise ValueError('contact calculation requires a topology')

    if isinstance(contacts, string_types):
        if contacts.lower() != 'all':
            raise ValueError('(%s) is not a valid contacts specifier' % contacts.lower())

        residue_pairs = []
        for i in xrange(traj.n_residues):
            residue_i = traj.topology.residue(i)
            if ignore_nonprotein and not any(a for a in residue_i.atoms if a.name.lower() == 'ca'):
                continue
            for j in xrange(i+3, traj.n_residues):
                residue_j = traj.topology.residue(j)
                if ignore_nonprotein and not any(a for a in residue_j.atoms if a.name.lower() == 'ca'):
                    continue
                if residue_i.chain == residue_j.chain:
                    residue_pairs.append((i, j))

        residue_pairs = np.array(residue_pairs)
        if len(residue_pairs) == 0:
            raise ValueError('No acceptable residue pairs found')

    else:
        residue_pairs = ensure_type(np.asarray(contacts), dtype=int, ndim=2, name='contacts',
                               shape=(None, 2), warn_on_cast=False)
        if not np.all((residue_pairs >= 0) * (residue_pairs < traj.n_residues)):
            raise ValueError('contacts requests a residue that is not in the permitted range')

    # now the bulk of the function. This will calculate atom distances and then
    # re-work them in the required scheme to get residue distances
    scheme = scheme.lower()
    if scheme not in ['ca', 'closest', 'closest-heavy', 'sidechain', 'sidechain-heavy']:
        raise ValueError('scheme must be one of [ca, closest, closest-heavy, sidechain, sidechain-heavy]')

    if scheme == 'ca':
        if soft_min:
            import warnings
            warnings.warn("The soft_min=True option with scheme=ca gives"
                          "the same results as soft_min=False")
        filtered_residue_pairs = []
        atom_pairs = []

        for r0, r1 in residue_pairs:
            ca_atoms_0 = [a.index for a in traj.top.residue(r0).atoms if a.name.lower() == 'ca']
            ca_atoms_1 = [a.index for a in traj.top.residue(r1).atoms if a.name.lower() == 'ca']
            if len(ca_atoms_0) == 1 and len(ca_atoms_1) == 1:
                atom_pairs.append((ca_atoms_0[0], ca_atoms_1[0]))
                filtered_residue_pairs.append((r0, r1))
            elif len(ca_atoms_0) == 0 or len(ca_atoms_1) == 0:
                # residue does not contain a CA atom, skip it
                if contacts != 'all':
                    # if the user manually asked for this residue, and didn't use "all"
                    import warnings
                    warnings.warn('Ignoring contacts pair %d-%d. No alpha carbon.' % (r0, r1))
            else:
                raise ValueError('More than 1 alpha carbon detected in residue %d or %d' % (r0, r1))

        residue_pairs = np.array(filtered_residue_pairs)
        distances = md.compute_distances(traj, atom_pairs, periodic=periodic)


    elif scheme in ['closest', 'closest-heavy', 'sidechain', 'sidechain-heavy']:
        if scheme == 'closest':
            residue_membership = [[atom.index for atom in residue.atoms]
                                  for residue in traj.topology.residues]
        elif scheme == 'closest-heavy':
            # then remove the hydrogens from the above list
            residue_membership = [[atom.index for atom in residue.atoms if not (atom.element == element.hydrogen)]
                                  for residue in traj.topology.residues]
        elif scheme == 'sidechain':
            residue_membership = [[atom.index for atom in residue.atoms if atom.is_sidechain]
                                  for residue in traj.topology.residues]
        elif scheme == 'sidechain-heavy':
            # then remove the hydrogens from the above list
            if 'GLY' in [residue.name for residue in traj.topology.residues]:
              import warnings
              warnings.warn('selected topology includes at least one glycine residue, which has no heavy atoms in its sidechain. The distances involving glycine residues '
                            'will be computed using the sidechain hydrogen instead.')
            residue_membership = [[atom.index for atom in residue.atoms if atom.is_sidechain and not (atom.element == element.hydrogen)] if not residue.name == 'GLY'
                                  else [atom.index for atom in residue.atoms if atom.is_sidechain]
                                  for residue in traj.topology.residues]

        residue_lens = [len(ainds) for ainds in residue_membership]

        atom_pairs = []
        n_atom_pairs_per_residue_pair = []
        for pair in residue_pairs:
            atom_pairs.extend(list(itertools.product(residue_membership[pair[0]], residue_membership[pair[1]])))
            n_atom_pairs_per_residue_pair.append(residue_lens[pair[0]] * residue_lens[pair[1]])

        atom_distances = md.compute_distances(traj, atom_pairs, periodic=periodic)

        # now squash the results based on residue membership
        n_residue_pairs = len(residue_pairs)
        distances = np.zeros((len(traj), n_residue_pairs), dtype=np.float32)
        n_atom_pairs_per_residue_pair = np.asarray(n_atom_pairs_per_residue_pair)
        indexes = []
        atom_pairs = np.array(atom_pairs)
        for i in xrange(n_residue_pairs):
            index = int(np.sum(n_atom_pairs_per_residue_pair[:i]))
            n = n_atom_pairs_per_residue_pair[i]
            if not soft_min:
                distances[:, i] = atom_distances[:, index : index + n].min(axis=1)
                # returns indices in the space of index:index+n, which are atoms of residue
                g = atom_distances[:, index : index + n].argmin(axis=1)
                # returns indices in space of all residue atoms
                r = np.arange(index, index + n + 1)[g]
                # returns pair of trajectory
                atom_ind = atom_pairs[r]
            else:
                distances[:, i] = soft_min_beta / \
                                  np.log(np.sum(np.exp(soft_min_beta/
                                                       atom_distances[:, index : index + n]), axis=1))
            indexes.append(atom_ind)
    else:
        raise ValueError('This is not supposed to happen!')

    return distances, residue_pairs, np.transpose(np.array(indexes), [1, 0, - 1])




# load files 
main_directory = "/home/YOUR_USERNAME/PROJECT/curiosity_chignolin_inwater_per_file_340K_dist_100ns_6dim/"

pdb_file = main_directory+"step5_input.pdb"
traj_files = glob(main_directory + "**/*_traj.dcd")
traj_files.sort(key=lambda x: sum([ord(c) for c in x.split("/")[-2]]))
walkers=4
one_len = len(traj_files)//walkers


# remove last, unfinished files, so that all trajectories have equal length 

del traj_files[-1]
del traj_files[-1]
del traj_files[-1]
del traj_files[-1]

# compute contact indices
md_traj = md.load(traj_files[0], top=pdb_file)
atom_slice = md_traj.topology.select("protein")
md_traj_slice = md_traj.atom_slice(atom_slice)
dist, pairs, ind = compute_contacts(md_traj_slice, scheme="sidechain-heavy")
# first frame as ind reference
ind = ind[0]


# featurise trajectory
feat = pyemma.coordinates.featurizer(pdb_file)
feat.add_distances(ind, periodic=True)
feat_data = pyemma.coordinates.load(traj_files, features=feat)


# Start hyperparameter search for MSM
lags = [10, 50, 100, 200]
dims = [2, 4, 5, 6, 7, 8, 9, 10]
clust = [50, 75, 100, 150, 200]

scores_dict = {}
size = len(lags) * len(dims) * len(clust)
pbar = tqdm.tqdm(total=size)
for lag in lags:
    for dim in dims:
        for k in clust:
            # msm lag 10 ns
            a = score_vamp(feat_data, lag, dim, k, 100)
            # lag scaled to ns
            scores_dict[str(dim)+", "+str(lag//10)+", "+str(k)] = a
            pbar.update(1)


pbar.close()
df = scores_to_df(scores_dict)
df.to_csv("CHIGNOLIN_from_folded_340K.csv")
print(df)
