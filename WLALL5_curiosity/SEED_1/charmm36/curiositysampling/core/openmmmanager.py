from curiositysampling.core import SingleBiasedSim
from curiositysampling.utils import SharedBiases
from curiositysampling.utils import TrajReporter

# from sklearn.preprocessing import OneHotEncoder
import numpy as np
import simtk.openmm as omm
import ray
from uuid import uuid4
from itertools import combinations
from itertools import chain


@ray.remote
class OpenMMManager:
    """OpenMMManager controls MD simulations , so that it allows for running,
    parallerization (using Ray framework) for several walkers, sharing bias (if `metadynamics=True`).
    The OMM prepares files (pdb, force field), creates  directories, and instantiates new ray actors.
    Furthermore, it is responsible for sharing bias between the same cvs, so that, walkers won't start
    for each CV/CVs from the start.
    Arguments:
        max_num_of_instances:  maximum number of Ray actor instances to be run. # generally TODO
        num_of_cores_per_instance: number of cores used be OpenMM instances. # generally TODO
        positions: initial positions for MD simulation.
        system: OpenMM compatibile system configuration for a given molecular system.
        topology: OpenMM compatibile topology for a given molecular system.
        integrator: OpenMM compatibile integrator.
        reporter_stride: How many integration steps have to be passed in order to
                         save an microstate as a observation. The observation is further
                         used as a training example.
        temperature: Temperature for given molecular system
        steps: Number of integration steps, before an observation is returned.
        file_per_cycle: If true, separate trajectory files are saved every cycle.
        saving_frequency: After every integration steps save trajectory in every cycle.
        warmup_cycles: number of cycles before trajectory is saved to file/files.
                       Generally, it's should at least of buffer's size.
        warmup_steps: number of integration steps while performing warmup cycles.
        warmup_reporter_stride: The same as reporter_stride, but for warmup period.
        regular_md: perform regular md simulation, without setting new positions and
                    resetting thermostat.
        cuda: If to use CUDA gpus or CPUs.
        metadynamics: if perform well-tempered metadynamics simulation.

        Those options below only work if metadynamics=True:
        bias_share_object: object that stores bias from metadynamics simulations, across
                           all walkers.
        variables: OpenMM compatibile variables for metadynamics simulation.
        biasFactor: well-tempered metadynamics bias factor
        height: well-tempered metadynamics initial heigh bias factor
        frequency: how often (in number of integration steps) deposit a gaussian
        use_dihedral: If to use dihedral as features for a given selection, if false, distances are used.
        add_chi_angles: If to add chi1, chi2, chi3 to the features of the dihedral angles. Requires to `use_dihedral` to be True.
        selection: Selection used for feature calculations. Default is "protein". The selection is of MDTraj/VMD style.
        positions_only: Use positions as feature calculations. Overrides all other feature options.
    """

    def __init__(
        self,
        max_num_of_instances=2,
        num_of_cores_per_instance=1,
        variables=None,
        positions=None,
        system=None,
        integrator=None,
        topology=None,
        saving_frequency=100,
        steps=1000,
        frequency=500,
        biasFactor=3,
        height=1,
        temperature=300,
        reporter_stride=100,
        regular_md=False,
        metadynamics=False,
        cuda=False,
        file_per_cycle=False,
        warmup_cycles=0,
        warmup_steps=1000,
        warmup_reporter_stride=10,
        use_dihedral=True,
        add_chi_angles=True,
        positions_only=False,
        selection="protein",
    ):
        self.cuda = cuda
        self.file_per_cycle = file_per_cycle
        self.warmup_cycles = warmup_cycles
        self.warmup_steps = warmup_steps
        self.warmup_reporter_stride = warmup_reporter_stride
        # MD simulation prep
        self.positions = positions
        self.system = system
        self.integrator = integrator
        self.topology = topology
        # MD simulation parameters
        self.steps = steps
        self.reporter_stride = reporter_stride
        self.regular_md = regular_md
        self.temperature = temperature
        self.saving_frequency = saving_frequency
        # MD bias parameters
        self.metadynamics = metadynamics
        if self.metadynamics:
            self.frequency = frequency
            self.biasFactor = biasFactor
            self.height = height
            # At the moment it is
            self.variables = variables
            self.num_of_cvs = len(variables)
            self.bias_share_object = SharedBiases(variables=variables)
        else:
            self.frequency = None
            self.biasFactor = None
            self.height = None
            self.variables = None
            self.num_of_cvs = None
            self.bias_share_object = None
        # The 3.5A is based on the
        # https://www.ncbi.nlm.nih.gov/books/NBK21581/
        self.distance_upper_bound = 3.5 * self.topology.getNumResidues()
        self.num_of_nodes = len(
            self.get_numbers_of_atoms(
                topology=self.topology, selection=["CA", "N", "O", "C"]
            )
        )
        # temporarly num of nodes equals num of edges
        self.num_of_edges = self.num_of_nodes
        # Bias_share_object initialization
        # TOBEDONE
        # Technical
        self.instances = {}
        # self.one_hot_encoder = OneHotEncoder(sparse=False, dtype=np.float32)
        # self.one_hot_encoder.fit(np.arange(start=0, stop=self.num_of_disjoint_cvs).reshape(-1, 1))
        self.dihedral = use_dihedral
        self.add_chi_angles = add_chi_angles
        self.positions_only = positions_only
        self.selection = selection

    def get_numbers_of_atoms(self, topology=None, selection=["N", "CA", "O", "CB"]):
        """Returns atom ids in the order from smallest to biggest, based on the
        passed selection.
        Arguments:
            toplogy: OpenMM topology object
            selection: List of atoms names, those should be considered.
        Returns:
            Sorted list of atom ids for a given selection
        """
        list_of_atom_id = []
        for atom in topology.atoms():
            if atom.name in selection:
                list_of_atom_id.append(int(atom.id))
        return list_of_atom_id

    def get_variables(self):
        """Outputs list of tuples of cvs as Bias variable objects.
        Arguments:
            None
        Returns:
            List of BiasVariables.
        """
        return self.variables

    def get_initial_positions(self):
        return self.positions

    def get_instance(self, sim_id=None):
        return self.instances[sim_id]["instance"]

    def create_new_instance(self):
        """Creates new ray actor that runs openmm biased simulation.
        All necessary files (force field, pdb, etc) and directories
        are prepared during the step.
        The method assigns new id to every ray actor, so that it is
        possible to run several instances at once through the class.
        Args:
            None
        Returns:
            instance id as a string
        """
        new_id = str(uuid4())
        if self.cuda:
            RemoteClass = ray.remote(num_gpus=1)(SingleBiasedSim)
        else:
            RemoteClass = ray.remote(SingleBiasedSim)
        new_instance = RemoteClass.remote(
            sim_id=new_id,
            positions=self.positions,
            system=self.system,
            topology=self.topology,
            integrator=self.integrator,
            bias_share_object=self.bias_share_object,
            variables=self.variables,
            reporter_stride=self.reporter_stride,
            temperature=self.temperature,
            frequency=self.frequency,
            biasFactor=self.biasFactor,
            height=self.height,
            steps=self.steps,
            metadynamics=self.metadynamics,
            cuda=self.cuda,
            file_per_cycle=self.file_per_cycle,
            warmup_cycles=self.warmup_cycles,
            warmup_steps=self.warmup_steps,
            warmup_reporter_stride=self.warmup_reporter_stride,
            regular_md=self.regular_md,
            saving_frequency=self.saving_frequency,
            selection=self.selection,
        )
        self.instances[new_id] = {
            "instance": new_instance,
            "locked": False,
            "running_id": None,
        }

        return new_id

    def step(self, sim_id=None, action=None):
        """Performs specified in `__init__` number of steps of MD simulation.
        The method is non-blocking and results are obtained by calling the
        method several times with the same sim_id.
        Arguments:
            sim_id: simulation id, that distingushes this instance from other.
            action: An Quantity object in units of nm, that contains
                    molecular system positions (including water, ions etc.).
        """
        instance = self.instances[sim_id]["instance"]
        if self.instances[sim_id]["locked"] == False:
            self.instances[sim_id]["locked"] = True
            running_id = instance.run.remote(
                action=action,
                dihedral=self.dihedral,
                add_chi_angles=self.add_chi_angles,
                positions_only=self.positions_only,
            )
            self.instances[sim_id]["running_id"] = running_id
        else:
            running_id = self.instances[sim_id]["running_id"]

        # We save running id for later, to be checked by `results` method
        ready, not_ready = ray.wait([running_id], timeout=0)

        if len(ready) == 0:
            return None
        else:
            nodes, edges, free_energy_matrix, md_time = ray.get(ready[0])
            self.instances[sim_id]["locked"] = False
            return nodes, edges, free_energy_matrix, md_time
