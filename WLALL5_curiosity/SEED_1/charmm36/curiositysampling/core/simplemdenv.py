import numpy as np
import ray
import time
from copy import deepcopy


class SimpleMDEnv:
    """An Biased Molecular Dynamics Environment, that allows for interaction between biased MD simulation
    and machine learning algorithms. Since MD simulations are very slow in their nature, the env is designed to so that,
    it allows for multiple workers to operate.

    Arguments:
        config: A dictionary that, contains are necessary parameters, at the moment the only one parameter is OpenMMManager object.
    """

    def __init__(self, config):
        # MD variables
        self.openmmmanager = config["openmmmanager"]
        self.sim_id = ray.get(self.openmmmanager.create_new_instance.remote())
        # RND object
        # Observations
        # every discrete space is number of action types,
        # self.action_space = gym.spaces.Discrete(len(self.all_available_actions))
        # RL variables
        self.results_sleep = 0.001
        # Init
        time.sleep(self.results_sleep)

    def reset(self):
        """Resets state of the environment. Initial positions are loaded and
        one step is performed.
        Returns: Initial observation
        """
        self.cur_pos = 0
        # nodes, edges = ray.get(self.openmmmanager.reset.remote(sim_id=self.sim_id))
        action = ray.get(self.openmmmanager.get_initial_positions.remote())
        obs, reward = self.step(action=action)
        return obs

    def step(self, action):
        """Performs one RL step, during which fallowing things happen:
        1. Pass action (openmm's system positions) to the md simulation.
        2. Perform md simulation
        3. Return observation

        Arguments:
            action: positions as a Quantity object from simtk framework, in the
                    nm units.
        Returns:
            Observation dict with `trajectory` key and `dist_matrix` key. The first
            contains Quantity moleculary structures, where the second contains
            feature tensors.
        """
        state = action
        free_energy_matrix, trajectory, trajectory_obs = self._md_simulation(state)
        if any(
            (
                free_energy_matrix == "Wait",
                trajectory == "Wait",
                trajectory_obs == "Wait",
            )
        ):
            return "Wait", "Wait"
        self.current_trajectory = trajectory
        self.current_free_energy = free_energy_matrix
        # trajectory_nodes are of shape (traj_size, num_nodes)
        # trajectory_edges are of shape (traj_size, num_edges, num_edges)
        trajectory_nodes, trajectory_edges = trajectory_obs
        # consider few modes, reward during backpropagation, after and before
        # implement buffer later
        # buffer_md_obs = self.openmmmanager.state_buffer.
        # dictionary of rewards, with indicies corresponding to order
        # to implement, generally adds to backpropagate buffer to backpropagate in future
        # The obs is still a dictionary
        obs = self._next_observation(
            trajectory_nodes, trajectory_edges, self.current_trajectory
        )
        #
        return obs, np.zeros(trajectory_nodes.shape[0])

    def _next_observation(self, nodes, edges, current_trajectory):
        """Calculates next observation ready to be used by `step` method.
        Arguments:
            nodes: node (atom types) of the observation
            edges: feature matrix, that corresponds to the nodes and forms a graph
            current_trajectory: Quantity objetcs, from the OpenMM simulation
        """
        # We ommit nodes at the moment
        return {"dist_matrix": edges, "trajectory": current_trajectory}

    def _md_simulation(self, action):
        """Perform molecular dynamics simulation in the context of
        openmm manager object. After the simulation, state of
        the system is returned as a free energy matrix, trajectory
        (positions as Quantity objects) and features.
        If `metadynamics` was set to false, free energy is
        None.
        Arguments:
            action: An simtk's Quantity object with molecular system positions
        """
        results = None
        while True:
            results = ray.get(
                self.openmmmanager.step.remote(sim_id=self.sim_id, action=action)
            )
            if results is not None:
                break
            else:
                # by 10, because we check 10 times
                # per simulation time (thus it slows
                # performance max by 1/10)
                # otherwise we wait double of it
                return "Wait", "Wait", "Wait"
        # fix issue of memory leak
        # https://github.com/ray-project/ray/issues/9504
        results_copy = deepcopy(results)
        del results
        free_energy_matrix, trajectory, trajectory_obs, md_time = results_copy
        return free_energy_matrix, trajectory, trajectory_obs

    def close(self):
        """The method closes openmmmanager and his
        subsequent agents.
        """
        pass
        # Status TODO
        # self.openmmmanager.close()
        # ray.kill(slf.openmmmanager)

    def get_current_freeenergy(self):
        """Returns last free energy matrix."""
        return self.current_free_energy
