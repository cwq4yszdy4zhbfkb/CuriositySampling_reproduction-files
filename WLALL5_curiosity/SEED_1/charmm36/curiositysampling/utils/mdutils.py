import numpy as np


def strip_offsets(atom_names):
    """Convert a list of atom + offset strings into lists of atoms.
    Source: https://github.com/mdtraj/mdtraj/blob/c0fa948f9bdc72c9d198c26ffacf8100a0b05638/mdtraj/geometry/dihedral.py#L269
    Arguments:
        atom_names A list of names of the atoms, whose offset prexifs you want to strip
    Notes
    -----
    For example, ["-C", "N", "CA", "C"] will be parsed as
    ["C","N","CA","C"]
    Returns:
        A list of atom names without offsets.
    """
    atoms = []
    for atom in atom_names:
        if atom[0] == "-":
            atoms.append(atom[1:])
        elif atom[0] == "+":
            atoms.append(atom[1:])
        else:
            atoms.append(atom)
    return atoms


def construct_atom_dict(topology):
    """Create dictionary to lookup indices by atom name, residue_id, and chain
    index.
    Source: https://github.com/mdtraj/mdtraj/blob/c0fa948f9bdc72c9d198c26ffacf8100a0b05638/mdtraj/geometry/dihedral.py#L269
    Arguments:
        topology : An OpenMM topology object
    Returns:
        atom_dict : Tree of nested dictionaries such that
        `atom_dict[chain_index][residue_index][atom_name] = atom_index`
    """
    atom_dict = {}
    for chain in topology.chains():
        residue_dict = {}
        for residue in chain.residues():
            local_dict = {}
            for atom in residue.atoms():
                local_dict[atom.name] = atom.index
            residue_dict[residue.index] = local_dict
        atom_dict[chain.index] = residue_dict

    return atom_dict


def atom_sequence(top, atom_names, residue_offsets=None):
    """Find sequences of atom indices corresponding to desired atoms.
    This method can be used to find sets of atoms corresponding to specific
    dihedral angles (like phi or psi). It looks for the given pattern of atoms
    in each residue of a given chain. See the example for details.
    Source: https://github.com/mdtraj/mdtraj/blob/c0fa948f9bdc72c9d198c26ffacf8100a0b05638/mdtraj/geometry/dihedral.py
    Arguments:
    top : an OpenMM topology
        Topology for which you want dihedrals.
    atom_names : A numpy array with atom names used for calculating dihedrals
    residue_offsets :
        A numpy numpy array of integer offsets for each atom. These are used to refer
        to atoms forward or backward in the chain relative to the current
        residue
    Example:
    Here we calculate the phi torsion angles by specifying the correct
    atom names and the residue_id offsets (e.g. forward or backward in
    chain) for each atom.
    >>> traj = mdtraj.load("native.pdb")
    >>> atom_names = ["C" ,"N" , "CA", "C"]
    >>> residue_offsets = [-1, 0, 0, 0]
    >>> found_residue_ids, indices = _atom_sequence(traj, atom_names, residue_offsets)
    """

    atom_names = strip_offsets(atom_names)

    atom_dict = construct_atom_dict(top)

    atom_indices = []
    found_residue_ids = []
    atoms_and_offsets = list(zip(atom_names, residue_offsets))
    for chain in top.chains():
        cid = chain.index
        for residue in chain.residues():
            rid = residue.index
            # Check that desired residue_IDs are in dict
            if all([rid + offset in atom_dict[cid] for offset in residue_offsets]):
                # Check that we find all atom names in dict
                if all(
                    [
                        atom in atom_dict[cid][rid + offset]
                        for atom, offset in atoms_and_offsets
                    ]
                ):
                    # Lookup desired atom indices and and add to list.
                    atom_indices.append(
                        [
                            atom_dict[cid][rid + offset][atom]
                            for atom, offset in atoms_and_offsets
                        ]
                    )
                    found_residue_ids.append(rid)

    atom_indices = np.array(atom_indices)
    found_residue_ids = np.array(found_residue_ids)

    if len(atom_indices) == 0:
        atom_indices = np.empty(shape=(0, 4), dtype=np.int)

    return found_residue_ids, atom_indices
