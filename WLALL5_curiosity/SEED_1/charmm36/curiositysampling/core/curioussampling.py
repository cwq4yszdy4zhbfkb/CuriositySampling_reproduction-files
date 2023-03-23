import tensorflow as tf
import simtk.openmm as omm
import numpy as np
import ray
import random
import copy
import sys
from curiositysampling.core import SimpleMDEnv
from curiositysampling.core import RndTrain
from time import time, sleep


class CuriousSampling:
    """The class defines object, that allows running molecular curiosity sampling with openmm framework.
    Example:
        ```python
        from curiositysampling.core import OpenMMManager
        from curiositysampling.core import CuriousSampling
        import ray
        from simtk.openmm.app import *
        from simtk.openmm import *
        from simtk.unit import *
        from openmmtools.testsystems import AlanineDipeptideImplicit


        ray.init()
        testsystem = AlanineDipeptideExplicit(hydrogenMass=4*amu)
        system = testsystem.system
        integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 4*femtoseconds)
        topology = testsystem.topology
        positions = testsystem.positions
        omm = OpenMMManager.remote(positions=testsystem.positions, system=system, topology=testsystem.topology,
                                   integrator=integrator, steps=2000, reporter_stride=10)
        config_env = {'openmmmanager': omm}
        config_rnd = {'model': 'dense_units': [layer_size]*layers,
                               'dense_units_ae_enc': [layer_size]*layers,
                               'dense_units_ae_dec': [layer_size]*layers,
                               'dense_activ': 'fullsort',
                               'dense_layernorm': False,
                               'dense_out': 1,
                               'dense_out_activ': 'linear',
                               'curiosity_activ': 'tanh',
                               'initializer': 'glorot_uniform',
                               'spectral': True,
                               'ae_spectral_only': True},
                      'autoencoder': True,
                      'autoencoder_lagtime': 450,
                      'minibatch_size': 200,
                      'clip_by_global_norm': False,
                      'num_of_train_updates': iterations,
                      'num_of_ae_train_updates': 2,
                      'learning_rate_cur': 0.0001,
                      'learning_rate_ae': 0.0001,
                      'obs_stand': False,
                      'reward_stand': False,
                      'ae_train_buffer_size': 50000,
                      'train_buffer_size': 50000,
                      'optimizer': 'sgd',
                      'optimizer_ae': 'nsgd',
                      'target_network_update_freq': 20
                     }

         csm = CuriousSampling(rnd_config=config_rnd, env_config=config_env,
                               number_of_parralel_envs=1, use_buffer=True)
         # define arrays to report output values:
         intrinsic_reward_reporter = []
         action_reporter = []
         state_mean_var_reporter = []
         reward_mean_var_reporter = []

         # run for 20 cycles
         csm.run(20, action_reporter=action_reporter,
         max_reward_reporter=intrinsic_reward_reporter,
         state_mean_var_reporter=state_mean_var_reporter,
         reward_mean_var_reporter=reward_mean_var_reporter)
         ```

         Arguments:
             rnd_config: A dictionary which defines RND model, number of iterations,
                         minibatch size and other parameters, see RNDtrain documentation
                         for more details.
             env_config: A dictionary which defines objects (like OpenMMManager) and
                         parameters associated with molecular dynamics. At the moment
                         the only its key is `omm`, which should be set to OpenMMManager
                         object.

             number_of_parralel_envs: How many envs should be sampled in parallel
             use_buffer: If buffer with past top-performing configurations, should
                         be used.
             random_actions: Pickpup actions from uniform distribution. Used for
                             testing purporse.
             latent_save_frequency: After how many train examples, latent
                                    space is saved (by default 0, not saved).
                                    The latent space can be obtained by
                                    `get_saved_latent()` method.
             latent_save_action: If to save action's latent space into array.
                                 Used for testing purpose.
    """

    def __init__(
        self,
        rnd_config=None,
        env_config=None,
        number_of_parralel_envs=1,
        use_buffer=True,
        buffer_size=150,
        use_metadynamics=False,
        random_actions=False,
        latent_save_frequency=0,
        latent_space_action=False,
    ):

        # env control vars
        self.results_sleep = 0.001
        self.results_sleep_beta = 0.9

        # prepare env instances
        self.env_instances = []
        for i in range(number_of_parralel_envs):
            self.env_instances.append(SimpleMDEnv(env_config))
        initial_observations = []

        start_time_md = time()

        temp_obs = []
        restr = []
        while True:
            # does not go further, envs starts simulation from beginning if it's called twice
            # if obs != "Wait", then env have to be removed from checking
            for i, env in enumerate(self.env_instances):
                if i not in restr:
                    obs = env.reset()
                    if obs != "Wait":
                        temp_obs.append(obs)
                        restr.append(i)
            sleep(self.results_sleep / 1000)
            if len(temp_obs) == len(self.env_instances):
                break
        time_md = time() - start_time_md

        self.results_sleep = self.results_sleep * self.results_sleep_beta + time_md * (
            1 - self.results_sleep_beta
        )

        for obs in temp_obs:
            initial_observations.append(obs)
        self.actions = []
        self.action_mode = "top"
        for init_obs in initial_observations:
            action = random.choice(init_obs["trajectory"])
            self.actions.append(action)
        initial_observation = initial_observations[0]
        self.rnd_train_instance = RndTrain(config=rnd_config)
        self.rnd_train_instance.initialise(initial_observation)
        del initial_observation
        del initial_observations
        # buffer
        if buffer_size > 0:
            self.use_buffer = use_buffer
        else:
            self.use_buffer = False
        self.buffer_size = buffer_size
        self.buffer = []
        self.previously_chosen_obs = None
        # other
        self.random_actions = random_actions
        self.autoencoder = rnd_config["autoencoder"]

        # latent space save
        self.latent_save_frequency = latent_save_frequency
        self.latent_space_counter = 0
        self.latent_space_array = []
        self.latent_space_action = latent_space_action
        self.latent_space_action_array = []

    def get_sim_ids(self):
        """Returns all simulations ids, added to this point in time."""
        sim_ids = []
        for env_instance in self.env_instances:
            sim_id = env_instance.sim_id
            sim_ids.append(sim_id)
        return sim_ids

    def run(
        self,
        cycles,
        action_reporter=None,
        max_reward_reporter=None,
        state_mean_var_reporter=None,
        reward_mean_var_reporter=None,
        previously_chosen=False,
    ):
        """Runs curious sampling for parameters set in the curious sampling
        objects.
        Arguments:
            cycles: Number of cycles, where one cycle is one MD simulation
                    of number of steps length, and training the RND network
                    with samples drawn from the MD sampler.
            action_reporter: Structures chosen with the highest reward. Be careful while storing
                             them, they are Nx3 matrices, where N is number of all atoms.
                             Therefore, they can easily lead to Out of memory situation.
            max_reward_reporter: Maximum reward associated with each env (may change to
                                 the maximum reward at all in the future).
            state_mean_var_reporter: Mean and Variance used to standarize observation,
                                     reported in each cycle.
            reward_mean_var_reporter: Mean and Variance used to standarize reward,
                                      in each cycle.
            previously_chosen: Add previously chosen action as next observation.
                               It should prevent choosing the same action several
                               times, as it's added to the training batch once again.
                               Nevertheless it may not work well with lagged AE.

        """
        for i in range(cycles):
            # sampling
            current_cycle_obs = []
            print("Cycle {0} out of {1} cycles".format(i + 1, cycles))
            print("Current buffer size {}".format(len(self.buffer)))
            start_time_total = time()

            start_time_md = time()
            temp_obs = []
            restr = []
            while True:
                for i, (env, action) in enumerate(
                    zip(self.env_instances, self.actions)
                ):
                    # remove first dimension from action
                    # at the moment step is blocking
                    # it should be changed in future to do MD in parallel
                    if i not in restr:
                        obs, _ = env.step(action)
                        if obs != "Wait":
                            temp_obs.append(obs)
                            restr.append(i)
                sleep(self.results_sleep / 1000)
                if len(temp_obs) == len(self.env_instances):
                    break
            time_md = time() - start_time_md
            self.results_sleep = (
                self.results_sleep * self.results_sleep_beta
                + time_md * (1 - self.results_sleep_beta)
            )
            for obs in temp_obs:
                current_cycle_obs.append(obs)
                # concatenate with prev top obs
                if self.previously_chosen_obs is not None:
                    con_obs_train = self._concatenate_obs(
                        [obs] + self.previously_chosen_obs
                    )
                    # add observations to train buffer
                    self._add_to_train(con_obs_train, shuffle=False, previous_obs=True)
                else:
                    # add observations to train buffer
                    self._add_to_train(obs, shuffle=False, previous_obs=False)
            # one train step through all the examples in the rnd model
            start_time_train = time()
            _, ae_loss_list, vamp2_score = self._train()
            time_train = time() - start_time_train
            if self.autoencoder:
                print("Autoencoder loss is {}".format(np.mean(np.hstack(ae_loss_list))))
                if len(vamp2_score) > 0:
                    print("VAMP2 score is {}".format(np.mean(np.hstack(vamp2_score))))
            # here we start predicting actions
            # we concatenate
            if self.use_buffer:
                # concatenate wwith buffer
                if len(self.buffer) == 0:
                    con_obs = self._concatenate_obs(current_cycle_obs)
                else:
                    # add the one which was chosen previously
                    con_obs = self._concatenate_obs(current_cycle_obs + self.buffer)
                # get top structures, including those from buffer
                number_of_actions = min(
                    len(self.env_instances) + self.buffer_size,
                    con_obs["dist_matrix"].shape[0],
                )
                (
                    self.actions,
                    top_obss,
                    rewards,
                    buffer_reward_unsorted,
                ) = self._get_new_actions(con_obs, number_of_actions)

                if self.action_mode == "ranked":
                    copy_actions = [
                        action.reshape(action.shape[1:])
                        for action in self.actions[: len(self.env_instances)]
                    ]
                elif self.action_mode == "top":
                    copy_actions = [
                        self.actions[0].reshape(self.actions[0].shape[1:])
                    ] * len(self.env_instances)

                self.actions = copy_actions

                if len(self.buffer) < self.buffer_size:
                    self.buffer = top_obss[len(self.env_instances) :]
                else:
                    # exchange random structures from buffer
                    # generally from k fittest individuals we chose n randomly (a genetic algorithm)
                    # top_structure = self._tournament_selection(con_obs, k=10)[0]
                    # version with just top structure
                    print("max reward: {}".format(np.max(rewards)))
                    # top_structure = top_obss[0]
                    # top structure as a top from the simulation only, index 1 is top observation
                    top_structure = self._get_new_actions(
                        self._concatenate_obs(current_cycle_obs), 1
                    )[1][0]
                    # version with random top structure
                    # top_structure = random.choice(top_obss)
                    worst_in_the_buffer = np.argmin(buffer_reward_unsorted)
                    # second_worst_in_the_buffer = np.argsort(buffer_reward_unsorted)[::-1][0]
                    true_reward = self._get_new_actions(
                        self.buffer[worst_in_the_buffer], 1
                    )[-2][0]
                    memorized_reward = buffer_reward_unsorted[worst_in_the_buffer]
                    if not np.isclose(
                        true_reward,
                        memorized_reward,
                        atol=1.0e-1,
                        rtol=1.0e-2,
                    ):

                        raise Exception(
                            "The rewards are not close to themselves, the calculated {0} and buffered {1}".format(
                                true_reward, memorized_reward
                            )
                            + " It may also signify numerical instabilities in the ANN."
                            + "Lower Learning rate or use L1/L2 reg.",
                        )
                    self.buffer[worst_in_the_buffer] = top_structure
                    # self.buffer[second_worst_in_the_buffer] = top_obss[1]

            else:
                con_obs = self._concatenate_obs(current_cycle_obs)
                self.actions, top_obss, rewards, _ = self._get_new_actions(
                    con_obs, len(self.env_instances)
                )
                if self.action_mode == "ranked":
                    copy_actions = [
                        action.reshape(action.shape[1:])
                        for action in self.actions[: len(self.env_instances)]
                    ]
                elif self.action_mode == "top":
                    copy_actions = [
                        self.actions[0].reshape(self.actions[0].shape[1:])
                    ] * len(self.env_instances)

                self.actions = copy_actions
            # store it, to add it to the train buffer
            if previously_chosen:
                self.previously_chosen_obs = top_obss[: len(self.env_instances)]
            else:
                self.previously_chosen_obs = None
            # calculate latent space and add to array
            if self.latent_save_frequency > 0:
                for dist_matrix in obs["dist_matrix"]:
                    if self.latent_space_counter >= self.latent_save_frequency:
                        dist_matrix_tensor = tf.convert_to_tensor(
                            dist_matrix[np.newaxis]
                        )
                        latent = self.rnd_train_instance.target_model_copy(
                            dist_matrix_tensor, training=False
                        )
                        if self.rnd_train_instance.reversible_vampnet:
                            latent = tf.matmul(
                                latent
                                - self.rnd_train_instance.means_copy.read_value(),
                                self.rnd_train_instance.eigenvectors_copy.read_value(),
                            ) / (
                                self.rnd_train_instance.norms_copy.read_value()
                                + tf.keras.backend.epsilon()
                            )
                        self.latent_space_array.append(latent.numpy())
                        self.latent_space_counter = 0
                    self.latent_space_counter += 1
            if self.latent_space_action:
                dist_matrix_tensor = tf.convert_to_tensor(top_obss[0]["dist_matrix"])
                latent = self.rnd_train_instance.target_model_copy(
                    dist_matrix_tensor, training=False
                )
                if self.rnd_train_instance.reversible_vampnet:
                    latent = tf.matmul(
                        latent - self.rnd_train_instance.means_copy.read_value(),
                        self.rnd_train_instance.eigenvectors_copy.read_value(),
                    ) / (
                        self.rnd_train_instance.norms_copy.read_value()
                        + tf.keras.backend.epsilon()
                    )

                self.latent_space_action_array.append(latent)
            # allow for real time printing
            sys.stdout.flush()
            # only top rewards for chosen actions for every env instance
            rewards = rewards[: len(self.env_instances)]
            if action_reporter is not None:
                action_reporter.append(self.actions)
            if max_reward_reporter is not None:
                max_reward_reporter.append(rewards)
            if state_mean_var_reporter is not None:
                state_mean_var_reporter.append(
                    self.rnd_train_instance.get_state_mean_variance()
                )
            if reward_mean_var_reporter is not None:
                reward_mean_var_reporter.append(
                    self.rnd_train_instance.get_reward_mean_variance()
                )
            time_total = time() - start_time_total
            # print times
            print(
                (
                    "Total time: {0:.2f} s"
                    + "   MD time: {1:.2f} s"
                    + "   ML train time {2:.2f} s"
                ).format(time_total, time_md, time_train)
            )

    def _concatenate_obs(self, obs_list):
        """Concatenates observations from a list
        into a single observation - dictionary
        with two ndarrays.
        Arguments:
            obs_list: list of dictionaries with trajectory and feature matrix (called dist_matrix).
        Returns:
            dictionary with trajectory (key: trajectory) and feature matrix (key: dist_matrix).
        """
        obs_base = copy.copy(obs_list[0])
        if len(obs_list) > 1:
            for obs in obs_list[1:]:
                obs_base["trajectory"] = np.concatenate(
                    [obs_base["trajectory"], obs["trajectory"]]
                )
                obs_base["dist_matrix"] = np.concatenate(
                    [obs_base["dist_matrix"], obs["dist_matrix"]]
                )
            if (
                obs_base["trajectory"].shape[0] < 2
                or obs_base["dist_matrix"].shape[0] < 2
            ):
                raise Exception("The shape should be at least 2 after contatenate")

        return obs_base

    def _train(self):
        """Trains RND network for all stored examples."""
        return self.rnd_train_instance.train()

    def _add_to_train(self, obs, shuffle=False, previous_obs=False):
        """Adds observations to the buffer from MD simulations.
        Arguments:
            obs: dictionary with observations
            shuffle: shuffles input against first dimension
            previous_obs: should be set true, if previous obs
                          is used.
        """
        self.rnd_train_instance.add_to_buffer(
            obs, shuffle=shuffle, previous_obs=previous_obs
        )

    def _get_new_actions(self, obs, n):
        """Calculates `n` observations with the highest rewards.
        Number of the input observations have to be higher than n.
        Arguments:
            obs: dictionary with observations.
            n: number of returned observations with highest rewards.
        Returns:
            A tuple of four, that contains sorted `n` actions from highest to lowest,
            `n` top observations sorted from highest to lowest, `n` rewards sorted
            from highest to lowest and unsorted buffer's rewards (used for debug).
        """
        (
            actions,
            dist_matrixs,
            rewards,
            reward_unsorted,
        ) = self.rnd_train_instance.predict_action(
            obs, n, random_actions=self.random_actions
        )
        top_obss = [
            {"trajectory": action, "dist_matrix": dist_matrix}
            for action, dist_matrix in zip(actions, dist_matrixs)
        ]
        # ensure that input shapes are the same as output shapes
        if not reward_unsorted.shape[0] == obs["dist_matrix"].shape[0]:
            msg = "objects shapes are not equal, {0} != {1}".format(
                reward_unsorted.shape[0], obs["dist_matrix"].shape[0]
            )
            raise ValueError(msg)
        # always the indices with numbers higher than shape of obs are buffer's
        buffer_reward_unsorted = reward_unsorted[
            reward_unsorted.shape[0] - len(self.buffer) :
        ]  # len of buffer, since it's size changes at the beginning
        if n > 1 and len(self.buffer) > 0:
            if not buffer_reward_unsorted.shape[0] == len(self.buffer):
                msg = "objects shapes are not equal, {0} != {1}".format(
                    buffer_reward_unsorted.shape[0], len(self.buffer)
                )
                raise ValueError(msg)
        return actions, top_obss, rewards, buffer_reward_unsorted

    def get_free_energy(self):
        """Returns free energy surface, if metadynamics is turned on.
        Otherwise it returns None
        """
        return self.env_instances[0].current_free_energy

    def _tournament_selection(self, obs, k=10):
        """Performs tournament selection algorithm on the observations.
        Arguments:
            obs: dictionary with observations.
            k: number of randomly selected observations

        """
        obs_indices = np.arange(obs["dist_matrix"].shape[0])
        k_obs_ind = random.choices(obs_indices, k=k)
        k_obs = {
            "trajectory": obs["trajectory"][k_obs_ind],
            "dist_matrix": obs["dist_matrix"][k_obs_ind],
        }
        (
            actions,
            dist_matrixs,
            rewards,
            reward_unsorted,
        ) = self.rnd_train_instance.predict_action(k_obs, 1)
        top_obss = [
            {"trajectory": action, "dist_matrix": dist_matrix}
            for action, dist_matrix in zip(actions, dist_matrixs)
        ]
        return top_obss

    def get_saved_latent(self):
        """Returns saved latent space array"""
        return self.latent_space_array

    def get_saved_latent_action(self):
        """Returns saved latent space array"""
        return self.latent_space_action_array
