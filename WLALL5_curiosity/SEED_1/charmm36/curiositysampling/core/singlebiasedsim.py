from curiositysampling.utils import SettableBiasVariable
from curiositysampling.utils import SharedMetadynamics
from curiositysampling.utils import TrajReporter
from curiositysampling.utils import DCDReporterMultiFile
from curiositysampling.utils import atom_sequence
import ray
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
import numpy as np
import os
import copy
from simtk.openmm.app import DCDFile
from time import time
import mdtraj as md
from itertools import combinations


class SingleBiasedSim(SharedMetadynamics):
    """Class that implements Molecular Dynamics simulation, that can be restarted from cetain configuration,
    that is provided externally. Furthermore, the class allows for parallerization for several walkers using
    Ray framework. Additionally, metadynamics can be performed.
    Arguments:
        sim_id: simulation id, that distingushes this instance from other. Usually an uuid.
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
        saving_frequency: after how many integration steps, energies, positions are saved
        warmup_cycles: number of cycles before trajectory is saved to file/files.
                       Generally, it's should at least of buffer's size.
        warmup_steps: number of integration steps while performing warmup cycles.
        warmup_reporter_stride: The same as reporter_Stride, but during warmup
        regular_md: perform regular md simulation, without setting new positions and
                    resetting thermostat.
        cuda: If set to true, use CUDA and GPUs, if set to False, you CPU platform
        metadynamics: if perform well-tempered metadynamics simulation.

        Those options below only work if metadynamics=True:
        bias_share_object: object that stores bias from metadynamics simulations, across
                           all walkers.
        variables: OpenMM compatibile variables for metadynamics simulation.
        biasFactor: well-tempered metadynamics bias factor.
        height: well-tempered metadynamics initial heigh bias factor.
        frequency: how often (in number of integration steps) deposit a gaussian.
        selection: Selection used for feature calculations. The selection is of MDtraj/VMD.
    """

    def __init__(
        self,
        sim_id=None,
        positions=None,
        system=None,
        topology=None,
        integrator=None,
        bias_share_object=None,
        variables=None,
        reporter_stride=100,
        temperature=None,
        metadynamics=False,
        biasFactor=3,
        height=1,
        frequency=1000,
        steps=1000,
        cuda=True,
        saving_frequency=100,
        file_per_cycle=False,
        warmup_cycles=0,
        warmup_steps=1000,
        warmup_reporter_stride=10,
        regular_md=False,
        selection="protein",
    ):

        self.sim_id = sim_id
        self.system = system
        self.topology = topology
        self.topology_mdtraj = md.Topology.from_openmm(self.topology)
        self.integrator = integrator
        self.positions = positions
        self.temperature = temperature
        self.file_per_cycle = file_per_cycle
        self.warmup_cycles = warmup_cycles
        self.warmup_steps = warmup_steps
        self.warmup_cycles_to_go = self.warmup_cycles
        self.warmup_reporter_stride = warmup_reporter_stride
        self.reporter_stride = reporter_stride
        self.regular_md = regular_md
        self.metadynamics = metadynamics
        self.selection = selection
        self.distance_ind = None
        if self.metadynamics:
            self.bias_share_object = bias_share_object
            self.biasFactor = biasFactor
            self.height = height
            self.frequency = frequency
            # bias variables
            # structure of the dictionary: action tuple: tuple of atom tuples, e.g.
            # for two CVs (0, 1): ((0, 1, 2, 3), (1, 2, 3, 4))
            self.forces = []
            self.variables = []
            for var in variables:
                f = copy.deepcopy(var.force)
                v = SettableBiasVariable(
                    f,
                    var.minValue,
                    var.maxValue,
                    var.biasWidth,
                    var.periodic,
                    var.gridWidth,
                )
                self.forces.append(f)
                self.variables.append(v)

        self.steps = steps
        os.environ["OPENMM_CPU_THREADS"] = "6"
        # Metadynamics parameters

        # Simulation
        if cuda:
            platform = Platform.getPlatformByName("CUDA")
        else:
            platform = Platform.getPlatformByName("CPU")
        self.simulation = Simulation(
            self.topology, self.system, self.integrator, platform
        )
        # Initiate metadynamics:
        if self.metadynamics:
            super().__init__(
                self.system,
                self.variables,
                self.temperature,
                self.biasFactor,
                self.height,
                self.frequency,
                self.bias_share_object,
            )

        self.simulation.context.setPositions(self.positions)
        self.simulation.minimizeEnergy()
        # Create reporters
        # Create folder for results:
        self.cur_dir = os.getcwd() + "/" + self.sim_id
        if not os.path.exists(self.cur_dir):
            os.makedirs(self.cur_dir)
        # reporter
        if self.warmup_cycles > 0:
            self.trajectory_reporter = TrajReporter(
                report_interval=self.warmup_reporter_stride
            )
        else:
            self.trajectory_reporter = TrajReporter(
                report_interval=self.reporter_stride
            )

        self.simulation.reporters.append(self.trajectory_reporter)
        if self.file_per_cycle:
            self.dcdreporter = DCDReporterMultiFile(
                self.cur_dir + "/traj" + ".dcd", saving_frequency
            )
        else:
            self.dcdreporter = DCDReporter(
                self.cur_dir + "/traj" + ".dcd", saving_frequency
            )
        self.simulation.reporters.append(
            StateDataReporter(
                self.cur_dir + "/scalars" + ".csv",
                saving_frequency,
                time=True,
                potentialEnergy=True,
                totalEnergy=True,
                temperature=True,
            )
        )
        self.out_dcd_file = open(self.cur_dir + "/actionpostions" + ".dcd", "wb")
        self._dcd = DCDFile(
            self.out_dcd_file,
            self.topology,
            self.integrator.getStepSize(),
            self.simulation.currentStep,
            1,
            False,
        )
        # other
        self.run_once = True

    def get_initial_positions(self):
        """Return initial positions for MD simulation"""
        return self.positions

    def run(
        self,
        action,
        dihedral=True,
        add_chi_angles=False,
        save_action_to_file=True,
        positions_only=False,
    ):
        """Performs one MD simulation step (or well-tempered metadynamics if metadynamics=True).
        Arguments:
            action: An Quantity object in units of nm, that contains
                    molecular system positions (including water, ions etc.).
            dihedral: If use cos(dihedral_angle) of protein's backbone dihedral angles instead
                      of distance matrixces as a feature vector.
            save_action_to_file: If save every action to a file, in order to investigate what
                                 molecular configurations were chosen by the algorithm.
            add_chi_angles=: If to add chi angles to features. `dihedral` must be True then.
            positions_only: Use positions as features, overrides all other options.

        Returns:
            Returns a tuple of three variables, free energy at the end of simulation,
            trajectory of all structures saved every `reporter_stride` and trajectory_obs
            with distance_matrices or dihedrals, saved every `reporter_stride`.
        """
        # measure time for purporse of adaptive sleep time
        start_time = time()
        position = action
        # save action to a file
        if save_action_to_file:
            self._dcd.writeModel(
                position,
                periodicBoxVectors=self.simulation.context.getState().getPeriodicBoxVectors(),
            )
        if not self.regular_md:
            self.set_position(position)
        # perform one step of metadynamics or MD step
        if self.warmup_cycles_to_go > 0:
            steps = self.warmup_steps
            self.warmup_cycles_to_go -= 1
            print("Performing warmup cycle")
        else:
            steps = self.steps
            if self.file_per_cycle:
                self.dcdreporter.nextCycle(self.simulation)

            # start saving after warmup is set
            if self.run_once:
                self.trajectory_reporter._reportInterval = self.reporter_stride
                self.simulation.reporters.append(self.dcdreporter)
                # set a new file if file per cycle set
                self.dcdreporter.describeNextReport(self.simulation)
                print("Trajectory is going to be saved from now")
                self.run_once = False
        if self.metadynamics:
            self.step(self.simulation, steps)
        else:
            valid = False
            for i in range(10):
                try:
                    self.simulation.step(steps)
                    valid = True
                    break
                except OpenMMException:
                    print(
                        "WArning, particle coord is nan, restarting from last position. Restart times {}".format(
                            i
                        )
                    )
                    self.set_position(position)
            if not valid:
                raise OpenMMException(
                    "Tried 10 times to restart simulation, Particle is still NaN"
                )

        # get saved trajectory positions
        trajectory = self.trajectory_reporter.get_trajectory()
        # get nparray object out of the Quantity object
        # trajectory = [positions._value for positions in trajectory]
        # get positions of backbone atoms:
        if positions_only:
            sel = self.topology_mdtraj.select(self.selection)
            traj = md.Trajectory(trajectory, self.topology_mdtraj)
            sub_traj = traj.atom_slice(sel)
            traj_feat = sub_traj.xyz.reshape(-1, 3 * sub_traj.xyz.shape[1])
            trajectory_obs = (traj_feat, traj_feat)
        elif not dihedral:
            distance_matrix = np.float32(
                self.get_distance_matrix(
                    trajectory=trajectory, selection=self.selection
                )
            )
            trajectory_obs = (
                distance_matrix,
                distance_matrix,
            )
        else:
            sel = self.topology_mdtraj.select(self.selection)
            traj = md.Trajectory(trajectory, self.topology_mdtraj)
            sub_traj = traj.atom_slice(sel)
            phi = md.compute_phi(sub_traj)[1]
            psi = md.compute_psi(sub_traj)[1]
            if add_chi_angles:
                # pad with zeros to fit dimensions
                chi1_like = np.zeros_like(phi)
                chi1 = md.compute_chi1(sub_traj)[1]
                chi1_like[: chi1.shape[0], : chi1.shape[1]] = chi1
                chi1_like = chi1

                chi2_like = np.zeros_like(phi)
                chi2 = md.compute_chi2(sub_traj)[1]
                chi2_like[: chi2.shape[0], : chi2.shape[1]] = chi2
                chi2 = chi2_like

                chi3_like = np.zeros_like(phi)
                chi3 = md.compute_chi3(sub_traj)[1]
                chi3_like[: chi3.shape[0], : chi3.shape[1]] = chi3
                chi3 = chi3_like

                chi4_like = np.zeros_like(phi)
                chi4 = md.compute_chi4(sub_traj)[1]
                chi4_like[: chi4.shape[0], : chi4.shape[1]] = chi4
                chi4 = chi4_like
                # if all are not zeros
                # often chi4 are all zeros
                if np.any(chi4):
                    angles = np.concatenate([phi, psi, chi1, chi2, chi3, chi4], axis=-1)
                else:
                    angles = np.concatenate([phi, psi, chi1, chi2, chi3], axis=-1)

            else:
                angles = np.concatenate([phi, psi], axis=-1)
            dihedral_matrix = np.concatenate([np.sin(angles), np.cos(angles)], axis=-1)
            trajectory_obs = (dihedral_matrix, dihedral_matrix)

        self.trajectory_reporter.flush_trajectory()
        # get free energy tensor out of simulation
        if self.metadynamics:
            free_energy = np.array(self.getFreeEnergy())
        else:
            free_energy = None
        # measure time for purporse of adaptive sleep time
        md_time = time() - start_time
        return free_energy, trajectory, trajectory_obs, md_time

    def set_position(self, positions):
        """Set molecular configuration, that is going to be sampled during the next step.
        Arguments:
            positions: molecular configuration of type Quantity
        """
        self.simulation.context.setPositions(positions)
        self.simulation.context.setVelocitiesToTemperature(self.temperature)

    def get_distance_matrix(self, trajectory=None, selection=None):
        """Returns distance matrix, of coodinates for a  given
        selection (passed through selection argument).
        Dimension of the matrix depends on the system
        and selection.
        Arguments:
            trajectory: a list of Quantity objects from the OpenMM simulation.
            selection: a list of strings with atom-type names.
        """

        trajectory = md.Trajectory(trajectory, self.topology_mdtraj)
        if self.distance_ind == None:
            sel = self.topology_mdtraj.select(self.selection)
            ind = list(combinations(sel, 2))
            self.distance_ind = ind

        distances = md.compute_distances(trajectory, self.distance_ind)

        return distances
