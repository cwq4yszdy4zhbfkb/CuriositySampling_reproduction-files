import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_addons as tfa
import numpy as np
import simtk.openmm as omm
import ray
import random
import collections
from copy import deepcopy
from curiositysampling.models import RND, RND_decoder
import scipy
from time import time


class RunningMeanStd:
    """
    The class allows for calculating mean and average with Welford's online algorithm.
    The mean and variance is calculated across axis=0, so the resulting mean and variance
    can also be a tensor.
    source: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    source: https://github.com/jcwleo/random-network-distillation-pytorch/blob/master/utils.py
    adapted to work with TF2

    Arguments:
        epsilon: a float number to make mean and variance calculations more stable.
        shape: shape of the tensor, whose mean and variance is calculated.
    """

    def __init__(self, epsilon=1e-4, shape=None):
        self.mean = tf.Variable(
            tf.zeros(shape=shape), name="mean", dtype=tf.float32, trainable=False
        )
        self.var = tf.Variable(
            tf.ones(shape=shape), name="var", dtype=tf.float32, trainable=False
        )
        self.count = tf.Variable(
            epsilon, name="count", dtype=tf.float32, trainable=False
        )

    # @tf.function(experimental_relax_shapes=True)
    def update(self, x):
        """Updates mean and variance with batch tensor x
        Arguments:
            x: batch tensor of shape defined in the init
        """
        batch_mean = tf.reduce_mean(x, axis=0, keepdims=True)
        batch_var = tf.math.reduce_variance(x, axis=0, keepdims=True)
        batch_count = tf.cast(tf.shape(x)[0], dtype=tf.float32)
        self.update_from_moments(batch_mean, batch_var, batch_count)

        return self.mean.read_value(), self.var.read_value()

    # @tf.function(experimental_relax_shapes=True)
    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean.read_value()
        tot_count = self.count.read_value() + batch_count
        new_mean = self.mean.read_value() + delta * batch_count / tot_count
        m_a = self.var.read_value() * (self.count.read_value())
        m_b = batch_var * (batch_count)
        M2 = (
            m_a
            + m_b
            + tf.square(delta)
            * self.count.read_value()
            * batch_count
            / (self.count.read_value() + batch_count)
        )
        new_var = M2 / (self.count.read_value() + batch_count)

        new_count = batch_count + self.count.read_value()

        self.mean.assign(new_mean, read_value=False)
        self.var.assign(new_var, read_value=False)
        self.count.assign(new_count, read_value=False)

    def get_mean_and_var(self):
        """Returns means and variance tensors"""
        return self.mean.read_value(), self.var.read_value()


class RndTrain:
    """The class's object allows for training a RND network with batches of
    molecular dynamics simulation's features (distance matrix, cos of dihedral
    angles). Model's parameters (number of layers, activations, normalization
    layers) are passed through `config` dictionary.
    Arguments:
        config: dictionary with fallowing parameters:
            model: defines model used in the RND, it has fallowing parameters:
                dense_units: a list with number of neurons for Curiosity part (list of uints)
                dense_units_ae_dec: a list with number of neurons for Autoencoder decoder part (list of uints)
                dense_units_ae_enc: a list with number of neurons for Autoencoder encoder part (list of uints)
                dense_activ: a string with Keras's activation function or deel-lip activation
                             (if spectral=True)
                dense_layernorm: whether to use layer normalization or not (bool)
                dense_batch: whether to use batch renormalization or not (bool)
                dense_out: number of outputs (uint)
                dense_out_activ: a string with Keras's activation function (str)
                curiosity_activ: activation functions for curiosity part of the
                                 algorithm (str)
                spectral: Use 1-Lipschitz comp. layers (True or False)
                ae_spectral_only: Use 1-Lipschitz comp. layers only for AE (bool)
                orthonormal: Whetever to constraint output's transformation (W of last layer) of the dim. red. to be orthogonal (bool)
                l1_reg: whether to L1 regularize encoder/node of AE/mTAE/VAMP/SRV (ufloat)
                l2_reg: whether to L2 regularize encoder/node of AE/mTAE/VAMP/SRB (ufloat)
            autoencoder: if to use autoencoder (bool).
            autoencoder_lagtime: lagtime for time-lagged autoencoder (uint,
                                 max is number of steps/stride)
            vampnet: whetever to use Vampnet of Noe for the dim. red. part (bool)
            reversible_vampnet: whetever to use SRV of Ferguson to perform dim. red part (bool)
            mae: whetever to use whitened Time-Lagged Autoencoder (wTAE) from Ferguson as dim. red. (bool)
            minibatch_size: minibatch size while training (uint)
            num_of_train_updates: number of optimization iterations (uint)
            num_of_ae_train_updates: number of optimization iterations
                                     for autoencoder (uint)
            learning_rate_cur: Learning rate for the curiosity part (ufloat)
            learning_rate_ae: learning rate for autoencoder part (ufloat)
            clip_by_global_norm: If to clip update gradient by global norm (ufloat).
            obs_stand: if standarize observation by mean and variance (bool).
                         Sometimes it gives better performance.
            reward_stand: If standarize reward with its (bool)
                          standard deviation. Sometimes it's
                          easier to analyze reward with it
            optimizer: Optimizer (curiosity) to choose. You can choose between
                       nsgd (Nesterov SGD), sgd (SGD), msgd (mSGD),
                       amsgrad (AMSgrad), rmsprop (RMSprop)
                       adadelta (adadelta).
            optimizer_ae: Optimizer (AE) to choose. You can choose between
                       nsgd (Nesterov SGD), sgd (SGD), msgd (mSGD),
                       amsgrad (AMSgrad), rmsprop (RMSprop)
                       adadelta (adadelta).

            ae_train_buffer_size: Size of buffer for training example
                                  for autoencoder part (uint).
            train_buffer_size: Size of buffer for training example
                               for curiosity part (uint).
            target_network_update_freq: Every how many cycles, the autoencoder
                                        part should be updated and used
                                        to judge curiosity. If it's 1 it's updated
                                        everytime but curiosity part of the algorithm
                                        may not catch up and output trash (uint).
            hard_momentum: If to you use predefined momentum istead of adaptive momentum.
                           The momentum coef. are 0.9 both for curiosity and AE.
            vamp2_metric: If to calculate VAMP-2 score for outputs. It may take
                          a quite portion of computational time.
        An example of such config is below:
         ```config_rnd = {'model': 'dense_units': [4, 4],
                                   'dense_units_ae_enc': [4, 4],
                                   'dense_units_ae_dec': [4, 4],
                                   'dense_activ': 'fullsort',
                                   'dense_layernorm': False,
                                   'dense_batchnorm': False,
                                   'dense_out': 1,
                                   'dense_out_activ': 'linear',
                                   'curiosity_activ': 'tanh',
                                   'initializer': 'glorot_uniform',
                                   'spectral': False,
                                   'ae_spectral_only': False,
                                   'orthonormal': False,
                                   'l1_reg': 0.0,
                                   'l2_reg': 0.0001,
                                   'unit_constraint': False},
                          'autoencoder': True,
                          'autoencoder_lagtime': 450,
                          'vampnet': False,
                          'reversible_vampnet': False,
                          'mae': False,
                          'minibatch_size': 200,
                          'clip_by_global_norm': False,
                          'num_of_train_updates': 1,
                          'num_of_ae_train_updates': 2,
                          'learning_rate_cur': 0.0001,
                          'learning_rate_ae': 0.0001,
                          'obs_stand': False,
                          'reward_stand': False,
                          'ae_train_buffer_size': 50000,
                          'train_buffer_size': 50000,
                          'optimizer_ae': 'nsgd',
                          'optimizer': 'nsgd',
                          'target_network_update_freq': 20,
                          'hard_momentum': True,
                          'vamp2_metric': True,
                          'classic_loss': False
                          }```
    """

    def __init__(self, config=None, output_size=2):
        self.config = config
        self.minibatch_size = config["minibatch_size"]
        self.minibatch_size_cur = config["minibatch_size_cur"]
        self.clip_by_global_norm = float(config["clip_by_global_norm"])
        self.num_of_train_updates = config["num_of_train_updates"]
        self.num_of_ae_train_updates = config["num_of_ae_train_updates"]
        self.learning_rate_cur = float(config["learning_rate_cur"])
        self.learning_rate_ae = float(config["learning_rate_ae"])
        self.ae_lagtime = config["autoencoder_lagtime"]
        self.ae_train_buffer_size = config["ae_train_buffer_size"]
        self.train_buffer_size = config["train_buffer_size"]
        self.obs_stand = config["obs_stand"]
        self.reward_stand = config["reward_stand"]
        self.chosen_optimizer = config["optimizer"]
        self.chosen_ae_optimizer = config["optimizer_ae"]
        self.target_network_update_freq = config["target_network_update_freq"]
        self.hard_momentum = config["hard_momentum"]
        self.vamp2_metric = config["vamp2_metric"]
        self.slowp_vector = tf.reshape(
            tf.convert_to_tensor(config["slowp_vector"], dtype=tf.float32), (1, -1)
        )
        self.classic_loss = config["classic_loss"]
        self.shrinkage = config["shrinkage"]
        self.whiten = config["whiten"]
        self.logtrans = config["logtrans"]
        self.one_simulation_size = None
        # start from -1 to avoid problems with first_run method
        self.cycle = 0
        self.test_ae_buffer_iter = 10
        if self.hard_momentum:
            self.momentum_curiosity = 0.9
            self.momentum_ae = 0.99
        else:
            self.momentum_curiosity = 0.9
            self.momentum_ae = 0.0
        # Initialisation of RND model
        self.autoencoder = config["autoencoder"]
        self.vampnet = config["vampnet"]
        self.mae = config["mae"]
        self.reversible_vampnet = config["reversible_vampnet"]
        self.spectral = config["model"]["spectral"]
        # check if mae and vampnet are not turned on together
        if (self.vampnet or self.reversible_vampnet) and self.mae:
            msg = "vampnet and reversible_vampnet can't true, when mae is true. You have to choose mae true or the other."
            raise Exception(msg)
        org_conf = deepcopy(config)
        conf_copy = deepcopy(config["model"])
        del config["model"]["ae_spectral_only"]
        config["model"]["curiosity_activ"] = None
        del conf_copy["dense_units_ae_enc"]
        del conf_copy["dense_units_ae_dec"]
        del conf_copy["orthonormal"]
        if conf_copy["ae_spectral_only"]:
            conf_copy["spectral"] = False
        del conf_copy["ae_spectral_only"]
        del conf_copy["l1_reg"]
        del conf_copy["l2_reg"]
        del conf_copy["unit_constraint"]
        self.predictor_model = RND(**conf_copy)
        if not self.autoencoder:
            del config["model"]["dense_units_ae_enc"]
            del config["model"]["dense_units_ae_dec"]
            self.target_model = RND(**config["model"], target=True)
            self.ae_lagtime = 0
            self.ae_train_buffer_size = None
        if not self.vampnet or not self.mae:
            conf_copy = deepcopy(org_conf["model"])
            conf_copy["dense_units"] = conf_copy["dense_units_ae_enc"]
            del conf_copy["dense_units_ae_enc"]
            del conf_copy["dense_units_ae_dec"]
            del conf_copy["ae_spectral_only"]
            self.target_model = RND(**conf_copy, target=False)
            self.target_model_copy = RND(**conf_copy, target=False)
            self.target_model_count = 0
        else:
            conf_copy = deepcopy(org_conf["model"])
            conf_copy["dense_units"] = conf_copy["dense_units_ae_enc"]
            del conf_copy["dense_units_ae_enc"]
            del conf_copy["dense_units_ae_dec"]
            del conf_copy["ae_spectral_only"]
            self.target_model = RND(**conf_copy, target=False)
            self.target_model_copy = RND(**conf_copy, target=False)
            self.target_model_count = 0
        # Here we set optimizer
        if self.chosen_optimizer == "rmsprop":
            self.optimizer = tf.keras.optimizers.RMSprop(
                learning_rate=self.learning_rate_cur, centered=True
            )
        elif self.chosen_optimizer == "nsgd":
            self.optimizer = tf.keras.optimizers.SGD(
                learning_rate=self.learning_rate_cur,
                nesterov=True,
                momentum=self.momentum_curiosity,
            )
        elif self.chosen_optimizer == "msgd":
            self.optimizer = tf.keras.optimizers.SGD(
                learning_rate=self.learning_rate_cur,
                nesterov=False,
                momentum=self.momentum_curiosity,
            )
        if self.chosen_optimizer == "sgd":
            self.optimizer = tf.keras.optimizers.SGD(
                learning_rate=self.learning_rate_cur
            )
        if self.chosen_optimizer == "amsgrad":
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate_cur, amsgrad=True
            )
        if self.chosen_optimizer == "nadam":
            self.optimizer = tf.keras.optimizers.Nadam(
                learning_rate=self.learning_rate_cur
            )

        if self.chosen_optimizer == "adadelta":
            self.optimizer = tf.keras.optimizers.Adadelta(
                learning_rate=self.learning_rate_cur
            )
        if self.chosen_optimizer == "adamax":
            self.optimizer = tf.keras.optimizers.Adamax(
                learning_rate=self.learning_rate_cur
            )

        if self.chosen_optimizer == "adagrad":
            self.optimizer = tf.keras.optimizers.Adagrad(
                learning_rate=self.learning_rate_cur
            )

        if self.chosen_optimizer == "adamw":
            self.optimizer = tfa.optimizers.AdamW(
                learning_rate=self.learning_rate_cur,
                weight_decay=self.learning_rate_cur / 10,
            )

        if self.chosen_optimizer == "novograd":
            self.optimizer = tfa.optimizers.NovoGrad(
                learning_rate=self.learning_rate_cur,
            )

        if self.chosen_optimizer == "swa":
            self.optimizer = tfa.optimizers.SWA(
                tf.keras.optimizers.SGD(learning_rate=self.learning_rate_cur), 0, 15
            )

        print("Optimzer is {}".format(self.chosen_optimizer))

        # Here we set optimizer for AE
        if self.chosen_ae_optimizer == "rmsprop":
            self.optimizer_ae = tf.keras.optimizers.RMSprop(
                learning_rate=self.learning_rate_ae, centered=True
            )
        elif self.chosen_ae_optimizer == "nsgd":
            self.optimizer_ae = tf.keras.optimizers.SGD(
                learning_rate=self.learning_rate_ae,
                nesterov=True,
                momentum=self.momentum_ae,
            )
        elif self.chosen_ae_optimizer == "msgd":
            self.optimizer_ae = tf.keras.optimizers.SGD(
                learning_rate=self.learning_rate_ae,
                nesterov=False,
                momentum=self.momentum_ae,
            )
        if self.chosen_ae_optimizer == "sgd":
            self.optimizer_ae = tf.keras.optimizers.SGD(
                learning_rate=self.learning_rate_ae
            )
        if self.chosen_ae_optimizer == "amsgrad":
            self.optimizer_ae = tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate_ae, amsgrad=True, beta_1=0.99
            )
        if self.chosen_ae_optimizer == "nadam":
            self.optimizer_ae = tf.keras.optimizers.Nadam(
                learning_rate=self.learning_rate_ae, beta_1=0.99
            )

        if self.chosen_ae_optimizer == "adadelta":
            self.optimizer_ae = tf.keras.optimizers.Adadelta(
                learning_rate=self.learning_rate_ae
            )
        if self.chosen_ae_optimizer == "adamax":
            self.optimizer_ae = tf.keras.optimizers.Adamax(
                learning_rate=self.learning_rate_ae, beta_1=0.99
            )
        if self.chosen_ae_optimizer == "adagrad":
            self.optimizer_ae = tf.keras.optimizers.Adagrad(
                learning_rate=self.learning_rate_ae
            )

        if self.chosen_ae_optimizer == "adamw":
            self.optimizer_ae = tfa.optimizers.AdamW(
                learning_rate=self.learning_rate_ae,
                weight_decay=self.learning_rate_ae / 10,
            )

        if self.chosen_ae_optimizer == "novograd":
            self.optimizer_ae = tfa.optimizers.NovoGrad(
                learning_rate=self.learning_rate_ae,
            )

        if self.chosen_ae_optimizer == "swa":
            self.optimizer_ae = tfa.optimizers.SWA(
                tf.keras.optimizers.SGD(learning_rate=self.learning_rate_ae), 0, 15
            )

        print("AE Optimzer is {}".format(self.chosen_ae_optimizer))
        # statistics objects
        self.epsilon = tf.convert_to_tensor(1e-8)
        self.obs_stat = None
        self.reward_stat = None
        self.logconst = None
        # other
        if self.reversible_vampnet:
            self.means_ = None
            self.cov_0 = None
            self.cov_1 = None
            self.beta = 0.99
            self.norms_ = None
            self.zt0_array = []
            self.ztt_array = []
            self.eigenvectors_copy = None
            self.prev_eigenvalues = None
            self.norms_copy = None
            self.means_copy = None
            self.implied_time_array = []

        # check if vamp or other is true if autoencoder is true
        if not self.autoencoder:
            if any((self.vampnet, self.reversible_vampnet, self.mae)):
                raise ValueError(
                    "To any of the three to work (vampnet, reversible_vampnet or mae), autoencoder have to be set True"
                )
        # check if vamp is true, when reversible_vamp is true
        if self.reversible_vampnet:
            if not self.vampnet:
                raise ValueError(
                    "vampnet have to be true, when reversible_vampnet is true"
                )

    def norm_inputs(self, inp, mean, var, tanh=True, logtrans=False):
        if logtrans:
            if self.logconst is None:
                self.logconst = tf.reduce_max(tf.math.abs(inp)) * 100
            inp = tf.math.log(inp + self.logconst) - tf.math.log(self.logconst)
        if not self.obs_stand:
            return inp

        z_norm = (inp - mean) / tf.sqrt(var + self.epsilon)
        if tanh:
            return 0.5 * (tf.math.tanh(0.01 * z_norm) + 1)
        else:
            return z_norm

    def initialise(self, initial_observation):
        """Initializes RND network, means and variances
        with some initial observation.
        """
        # if vampnet is turned on, then batch_size have to be aligned with sampled
        # data, shifted by lagtime.
        if self.vampnet or self.mae:
            init_size = initial_observation["dist_matrix"].shape[0]
            lagtime_rest = init_size - self.ae_lagtime
            while not (
                (lagtime_rest % self.minibatch_size == 0)
                # use it, when aligment with respect to whole batch size will be needed
                # and (init_size % self.minibatch_size == 0)
            ):
                self.minibatch_size = self.minibatch_size + 1
                if self.minibatch_size > lagtime_rest:
                    raise Exception(
                        "Calculation's can't be done with current minibatch size"
                    )
            # self.minibatch_size = lagtime_rest
            print(
                "Mini-batch size is {}, due to vampnet=True".format(self.minibatch_size)
            )
        if self.train_buffer_size > 0:
            self.train_buffer = collections.deque(
                maxlen=self.train_buffer_size // self.minibatch_size_cur
                + bool(self.train_buffer_size % self.minibatch_size_cur)
            )
        else:
            self.train_buffer = []

        if self.ae_lagtime > 0:
            if self.ae_train_buffer_size > 0:
                self.ae_train_buffer = collections.deque(
                    maxlen=self.ae_train_buffer_size // self.minibatch_size
                    + bool(self.ae_train_buffer_size % self.minibatch_size)
                )
            else:
                self.ae_train_buffer = []
        else:
            self.ae_train_buffer = None
        self.add_to_buffer(initial_observation, shuffle=False)
        self._first_run()

        if self.reversible_vampnet:

            assert self.norms_copy is not None
            assert self.means_copy is not None
            assert self.eigenvectors_copy is not None

            if any(
                (
                    tf.math.reduce_any(tf.math.is_nan(self.eigenvectors_copy)),
                    tf.math.reduce_any(tf.math.is_nan(self.means_copy)),
                    tf.math.reduce_any(tf.math.is_nan(self.norms_copy)),
                )
            ):
                raise StandardError(
                    "Eigenvector, Means or Norms for reversible vampnet are nan"
                )

    def _first_run(self):
        first_time = True

        if not self.obs_stand:
            obs_mean, obs_var = (0.0, 1.0)

        for example in self.train_buffer:
            dist_matrix = tf.convert_to_tensor(example)
            # standarization of state target
            if first_time:
                conf_copy = deepcopy(self.config["model"])
                self.obs_stat = RunningMeanStd(shape=(1, *dist_matrix.shape[1:]))
                self.reward_stat = RunningMeanStd(shape=(1, conf_copy["dense_out"]))
                if self.autoencoder and not (self.vampnet or self.mae):
                    conf_copy["dense_out"] = int(dist_matrix.shape[-1])
                    conf_copy["dense_units"] = conf_copy["dense_units_ae_dec"]
                    del conf_copy["dense_units_ae_enc"]
                    del conf_copy["dense_units_ae_dec"]
                    del conf_copy["curiosity_activ"]
                    del conf_copy["layernorm_out"]
                    del conf_copy["orthonormal"]
                    del conf_copy["l1_reg"]
                    del conf_copy["l2_reg"]
                    self.decoder_model = RND_decoder(**conf_copy)

                first_time = False

            obs_mean, obs_var = self.obs_stat.update(dist_matrix)

        if self.reversible_vampnet:
            obs_mean, obs_var = (0.0, 1.0)
            len_ae_train_buffer = len(self.ae_train_buffer)
            for batch_shift, batch_back in self.ae_train_buffer:
                batch_dist_matrix_back = tf.convert_to_tensor(batch_back)
                batch_dist_matrix_shifted = tf.convert_to_tensor(batch_shift)
                self._calc_basis(
                    batch_dist_matrix_back,
                    batch_dist_matrix_shifted,
                    obs_mean,
                    obs_var,
                    max_batch_size=len_ae_train_buffer,
                )

            if self.eigenvectors_copy is None:
                # Initialize values
                self.eigenvectors_copy = tf.Variable(self.eigenvectors_)
                self.means_copy = tf.Variable(self.means_)
                self.norms_copy = tf.Variable(self.norms_)
            else:
                self.eigenvectors_copy.assign(self.eigenvectors_)
                self.means_copy.assign(self.means_)
                self.norms_copy.assign(self.norms_)

            dist_matrix_standarized = self.norm_inputs(
                dist_matrix, obs_mean, obs_var, logtrans=self.logtrans
            )
            self.obs_mean_copy = obs_mean
            self.obs_var_copy = obs_var
            state_target = self.target_model(dist_matrix_standarized, training=False)
            state_target_copy = self.target_model_copy(
                dist_matrix_standarized, training=False
            )
            if self.ae_lagtime > 0 and self.autoencoder:
                if self.reversible_vampnet:
                    state_target = (
                        tf.matmul(
                            (state_target - self.means_copy.read_value()),
                            self.eigenvectors_copy.read_value(),
                        )
                        / (self.norms_copy.read_value() + self.epsilon)
                    )
            state_predict = self.predictor_model(
                dist_matrix_standarized, training=False
            )

            if self.autoencoder and not (self.vampnet or self.mae):
                state_decoder = self.decoder_model(state_target, training=False)

            regul_loss = tf.reduce_sum(self.predictor_model.losses)
            loss, loss_nm = self.loss_func(
                state_target, state_predict, regul_loss=regul_loss
            )
            # loss standarized
            loss_mean, loss_var = self.reward_stat.update(loss_nm)
            # also with mean, not like in the papaer
            loss_standarized = (loss_nm - loss_mean) / tf.sqrt(loss_var + self.epsilon)

        self.train_buffer.clear()
        if self.ae_lagtime > 0:
            self.ae_train_buffer.clear()

    def train(
        self,
        shuffle_every_batch=True,
        shuffle_inside_batch=True,
        dont_shuffle_autoencoder=True,
        dont_shuffle_curiosity=False,
        gauss_aug=True,
    ):
        """Performs training for the current training buffer (samples observed during current MD simulation)
        with predefined number of optimization steps in the `config`. The buffer is emptied at the end
        of training.
        Arguments:
             shuffle_every_batch: shuffle batches in the list
             shuffle_inside_batch: shuffle inside every batch
             dont_shuffle_autoencoder: dont shuffe inside every batch for autoencoder training and its variants
             dont_shuffle_curiosity: dont shuffe inside every batch for curiosity training loop
        """
        if (
            self.target_model_count >= self.target_network_update_freq
            and self.autoencoder
        ):
            self.target_model_copy.set_weights(self.target_model.get_weights())
            if self.reversible_vampnet:
                self.norms_copy.assign(self.norms_)
                self.means_copy.assign(self.means_)
                self.eigenvectors_copy.assign(self.eigenvectors_)
            if self.obs_stand:
                obs_mean, obs_var = self.obs_stat.get_mean_and_var()
                self.obs_mean_copy = obs_mean
                self.obs_var_copy = obs_var
            print("Target model updated!!!")
            self.target_model_count = 0
        self.target_model_count += 1

        if self.reversible_vampnet:
            first_time_reversible = True
            assert self.norms_copy is not None
            assert self.means_copy is not None
            assert self.eigenvectors_copy is not None
            # calc implied time scale
            implied_time = self.ae_lagtime * (-1 / np.log(self.eigenvalues_))
            self.implied_time_array.append(implied_time)
            # print difference in eigenvalues
            print("Timescales for this epoch: {}".format(implied_time))
            if self.prev_eigenvalues is not None:
                print(
                    "Diff. between epoch eigenvalues: {}".format(
                        self.eigenvalues_ - self.prev_eigenvalues
                    )
                )
            self.prev_eigenvalues = self.eigenvalues_

        # dist_matrix is of dimension (batch_size, distm_dimension, distm_dimension)
        loss_standarized_list = []
        loss_ae_list = []
        vamp2_score_list = []
        # deques are used to calculate change in the training buffer size
        if shuffle_every_batch:
            # we can shuffle, because batches are aligned
            train_buffer_local = list(self.train_buffer)
            random.shuffle(train_buffer_local)
        # update obs statistics
        if self.obs_stand:
            for k, batch_example in enumerate(train_buffer_local):
                batch_dist_matrix = tf.convert_to_tensor(
                    batch_example, dtype=tf.float32
                )
                obs_mean, obs_var = self.obs_stat.update(batch_example)

        if self.autoencoder:
            # separate train updates for ae from curiosity
            if self.ae_lagtime == 0:
                for i in range(self.num_of_ae_train_updates):
                    for k, batch_example in enumerate(train_buffer_local):
                        if not dont_shuffle_autoencoder:
                            if shuffle_inside_batch:
                                random.shuffle(batch_example)
                        batch_dist_matrix = tf.convert_to_tensor(
                            batch_example, dtype=tf.float32
                        )
                        batch_ae_loss, g_norm_ae = self._train_step_ae(
                            batch_dist_matrix,
                            batch_dist_matrix,
                            self.obs_mean_copy,
                            self.obs_var_copy,
                        )
                        loss_ae_list.append(batch_ae_loss.numpy())

                # separate lagged autoencoder training, to use all data in the
                # curiosity part
            if self.ae_lagtime > 0:
                for i in range(self.num_of_ae_train_updates):
                    # we can shuffle, because batches are aligned
                    ae_train_buffer_local = list(self.ae_train_buffer)
                    # it can\'t be used here, due to
                    # timelag
                    # if not self.vampnet:
                    # if shuffle_inside_batch_cur:
                    # np.random.shuffle(ae_train_buffer_local)
                    if shuffle_every_batch:
                        random.shuffle(ae_train_buffer_local)

                    # shift, back = zip(*ae_train_buffer_local)
                    # shift = np.concatenate(shift, axis=0)
                    # back = np.concatenate(back, axis=0)

                    # ind = np.arange(back.shape[0])
                    # np.random.shuffle(ind)
                    # shift = shift[ind]
                    # back = back[ind]

                    # shift = np.split(
                    #    shift,
                    #    range(
                    #        self.minibatch_size,
                    #        shift.shape[0],
                    #        self.minibatch_size,
                    #    ),
                    # )
                    # back = np.split(
                    #    back,
                    #    range(
                    #        self.minibatch_size,
                    #        back.shape[0],
                    #        self.minibatch_size,
                    #    ),
                    # )
                    # ae_train_buffer_local = zip(shift, back)
                    for k, (batch_shift, batch_back) in enumerate(
                        ae_train_buffer_local
                    ):
                        # we shouldn't shuffle withing batch, as it can break up variance calculation
                        if not dont_shuffle_autoencoder:
                            if shuffle_inside_batch:
                                ind = np.arange(batch_shift.shape[0])
                                np.random.shuffle(ind)
                                batch_shift = batch_shift[ind]
                                batch_back = batch_back[ind]
                        batch_dist_matrix_shifted = tf.convert_to_tensor(
                            batch_shift, dtype=tf.float32
                        )
                        batch_dist_matrix_back = tf.convert_to_tensor(
                            batch_back, dtype=tf.float32
                        )
                        if gauss_aug:
                            noise_back = batch_dist_matrix_back * tf.random.normal(
                                shape=batch_dist_matrix_back.shape, stddev=5.0
                            )
                            noise_shift = batch_dist_matrix_shifted * tf.random.normal(
                                shape=batch_dist_matrix_shifted.shape, stddev=5.0
                            )
                            coef = 0.001
                            batch_dist_matrix_back += noise_back * coef
                            batch_dist_matrix_shifted += noise_shift * coef
                        if not (self.vampnet or self.mae):
                            batch_ae_loss, g_norm_ae = self._train_step_ae(
                                batch_dist_matrix_back,
                                batch_dist_matrix_shifted,
                                self.obs_mean_copy,
                                self.obs_var_copy,
                            )
                        elif self.vampnet:
                            if self.vamp2_metric:
                                vamp2_score_list.append(
                                    self.metric_VAMP2(
                                        batch_dist_matrix_back,
                                        batch_dist_matrix_shifted,
                                    )
                                )

                            batch_ae_loss, g_norm_ae = self._train_step_vamp(
                                batch_dist_matrix_back,
                                batch_dist_matrix_shifted,
                                self.obs_mean_copy,
                                self.obs_var_copy,
                            )

                        elif self.mae:
                            batch_ae_loss, g_norm_ae = self._train_step_mae(
                                batch_dist_matrix_back,
                                batch_dist_matrix_shifted,
                                self.obs_mean_copy,
                                self.obs_var_copy,
                            )

                        loss_ae_list.append(batch_ae_loss.numpy())

                if self.reversible_vampnet:
                    for k, (batch_shift, batch_back) in enumerate(self.ae_train_buffer):
                        # transform SRV features through approximated basis vectors
                        self._calc_basis(
                            batch_dist_matrix_back,
                            batch_dist_matrix_shifted,
                            self.obs_mean_copy,
                            self.obs_var_copy,
                            max_batch_size=len(self.ae_train_buffer),
                        )

        if self.ae_lagtime > 0:
            true_train_buff_size_shift = sum(
                [ar[0].shape[0] for ar in ae_train_buffer_local]
            )
            true_train_buff_size_back = sum(
                [ar[1].shape[0] for ar in ae_train_buffer_local]
            )
            print(
                "Lagged AE training buffer current size is: {0} and {1}".format(
                    true_train_buff_size_shift, true_train_buff_size_back
                )
            )
            # remove train buffer local to free memory
            del ae_train_buffer_local
        if not self.hard_momentum:
            # update momentum based on the train examples
            if self.autoencoder and self.ae_lagtime == 0:
                self.momentum_ae = min(
                    1.0 * true_train_buff_size // self.train_buffer_size, 0.99
                )
            elif self.ae_lagtime > 0:
                self.momentum_ae = min(
                    1.0 * true_train_buff_size_back // self.ae_train_buffer_size, 0.99
                )
        # switch for first time in the batch loop
        first_time = True
        # train curiosity
        for i in range(self.num_of_train_updates):

            train_buffer_local = deepcopy(self.train_buffer)
            if not dont_shuffle_curiosity:
                if shuffle_inside_batch:
                    train_buffer_local = np.concatenate(train_buffer_local)
                    np.random.shuffle(train_buffer_local)
                    train_buffer_local = np.split(
                        train_buffer_local,
                        range(
                            self.minibatch_size_cur,
                            train_buffer_local.shape[0],
                            self.minibatch_size_cur,
                        ),
                    )

            # train buffer comes in the order of simulation if shuffle in
            # add_to_buffer method and this one here, are off.
            if shuffle_every_batch:
                train_buffer_local = list(train_buffer_local)
                random.shuffle(train_buffer_local)

            for k, batch_example in enumerate(train_buffer_local):
                # !! With lagtime, the batch matrix is smaller and
                # part of the data is waster!! change it!

                batch_dist_matrix = tf.convert_to_tensor(
                    batch_example, dtype=tf.float32
                )

                if gauss_aug:
                    noise = batch_dist_matrix * tf.random.normal(
                        batch_dist_matrix.shape, stddev=5
                    )
                    coef = 0.001
                    batch_dist_matrix += noise * coef

                # perform one train step
                batch_loss, batch_loss_nm, g_norm = self._train_step(
                    batch_dist_matrix, self.obs_mean_copy, self.obs_var_copy
                )
                if self.reward_stand:
                    if first_time:
                        loss_mean, loss_var = self.reward_stat.update(batch_loss_nm)
                        first_time = False
                    else:
                        loss_mean, loss_var = self.reward_stat.get_mean_and_var()
                else:
                    loss_var = 1.0
                batch_loss_standarized = (batch_loss_nm - loss_mean) / np.sqrt(
                    loss_var + self.epsilon
                )
                loss_standarized_list.append(
                    tf.reduce_mean(batch_loss_standarized, axis=-1)
                )

            true_train_buff_size = sum([ar.shape[0] for ar in train_buffer_local])
            if not self.hard_momentum:
                # update momentum based on the current buffer size
                self.momentum_curiosity = min(
                    1.0 - true_train_buff_size // self.train_buffer_size, 0.99
                )

            # remove train buffer local to free memory
            del train_buffer_local
        print("Main training buffer current size is: {}".format(true_train_buff_size))

        return loss_standarized_list, loss_ae_list, vamp2_score_list

    def metric_VAMP2(self, y_true, y_shifted):
        """Returns the sum of the squared top k eigenvalues of the vamp matrix,
        with k determined by the wrapper parameter k_eig, and the vamp matrix
        defined as:
            V = cov_00 ^ -1/2 * cov_01 * cov_11 ^ -1/2
        Can be used as a metric function in model.fit()

        Arguments:
            y_true: tensorflow tensor.
                parameter not needed for the calculation, added to comply with Keras
                rules for loss fuctions format.

            y_pred: tensorflow tensor with shape [batch_size, 2 * output_size]
                output of the two lobes of the network

        Returns:
            eig_sum_sq: tensorflow float
            sum of the squared k highest eigenvalues in the vamp matrix
        """
        N = y_true.shape[0]
        y_true_standarized = self.norm_inputs(
            y_true, self.obs_mean_copy, self.obs_var_copy, logtrans=self.logtrans
        )
        y_shifted_standarized = self.norm_inputs(
            y_shifted, self.obs_mean_copy, self.obs_var_copy, logtrans=self.logtrans
        )
        zt0 = self.target_model_copy(y_true_standarized)
        ztt = self.target_model_copy(y_shifted_standarized)

        # shape (batch_size, output)
        zt0 = zt0 - tf.reduce_mean(zt0, axis=0, keepdims=True)
        # shape (batch_size, output)
        ztt = ztt - tf.reduce_mean(ztt, axis=0, keepdims=True)
        # Calculate the covariance matrices
        # shape (output, output)
        cov_01 = self.calc_cov(zt0, ztt)
        cov_10 = self.calc_cov(ztt, zt0)
        cov_00 = self.calc_cov(zt0, zt0)
        cov_11 = self.calc_cov(ztt, ztt)
        cov_00_inv = tf.linalg.sqrtm(tf.linalg.inv(cov_00))
        cov_11_inv = tf.linalg.sqrtm(tf.linalg.inv(cov_11))
        vamp_matrix = tf.matmul(tf.matmul(cov_00_inv, cov_01), cov_11_inv)
        # Select the all singular values of the VAMP matrix
        diag = tf.linalg.svd(vamp_matrix, compute_uv=False)
        eig_sum_sq = 1 + tf.reduce_sum(diag ** 2)

        return eig_sum_sq

    def loss_func(self, true, pred, mask=True, regul_loss=None):
        """Loss function to calculate loss, the returned loss is of
        batch dimension shape.
        Arguments:
            true: groundtruth tensor
            pred: predicted tensor
            mask: if to mask values, that are not finite, to prevent nan losses
            regul_loss: pass your regularization loss here
        """
        loss_nm = (true - pred) ** 2
        loss_f = tf.reduce_mean(loss_nm, axis=-1)
        if mask:
            loss_f = tf.clip_by_value(loss_f, 0, 1e16)
            loss_nm = tf.clip_by_value(loss_nm, 0, 1e16)
        if regul_loss is not None:
            loss_f += regul_loss

        return loss_f, loss_nm

    def rao_blackwell_ledoit_wolf(self, cov, N):
        """Rao-Blackwellized Ledoit-Wolf shrinkaged estimator of the covariance
        matrix.
        Arguments:
        ----------
        S : array, shape=(n, n)
        Sample covariance matrix (e.g. estimated with np.cov(X.T))
        n : int
        Number of data points.
        Returns
        .. [1] Chen, Yilun, Ami Wiesel, and Alfred O. Hero III. "Shrinkage
        estimation of high dimensional covariance matrices" ICASSP (2009)
        Based on: https://github.com/msmbuilder/msmbuilder/blob/master/msmbuilder/decomposition/tica.py
        """
        p = len(cov)
        assert cov.shape == (p, p)

        alpha = (N - 2) / (N * (N + 2))
        beta = ((p + 1) * N - 2) / (N * (N + 2))

        trace_cov2 = tf.reduce_sum(cov * cov)
        U = (p * trace_cov2 / tf.linalg.trace(cov) ** 2) - 1
        rho = tf.minimum(alpha + beta / U, 1)

        F = (tf.linalg.trace(cov) / p) * tf.eye(p)
        return (1 - rho) * cov + rho * F, rho

    def calc_cov(self, x, y, rblw=False, use_shrinkage=False, no_normalize=False):
        N = x.shape[0]
        feat = x.shape[1]
        if not no_normalize:
            cov = 1 / (N - 1) * tf.matmul(x, y, transpose_a=True)
        else:
            cov = tf.matmul(x, y, transpose_a=True)
        if rblw and use_shrinkage and self.shrinkage <= 0:
            cov_shrink, shrinkage = self.rao_blackwell_ledoit_wolf(cov, N)
            return cov_shrink
        elif use_shrinkage:
            shrinkage = self.shrinkage
            ident = tf.eye(feat)
            mu = tf.linalg.trace(cov) / feat
            cov_shrink = (1 - shrinkage) * cov + shrinkage * mu * ident
            return cov_shrink
        else:
            return cov

    def _inv(self, x, ret_sqrt=False):
        """Utility function that returns the inverse of a matrix, with the
        option to return the square root of the inverse matrix.
        Original from: https://github.com/markovmodel/deeptime
        Arguments
            x: numpy array with shape [m,m]
               atrix to be inverted

            ret_sqrt: bool, optional, default = False
                if True, the square root of the inverse matrix is returned instead
        Returns:
            x_inv: numpy array with shape [m,m]
                inverse of the original matrix

        """

        # Calculate eigvalues and eigvectors
        eigval_all, eigvec_all = tf.linalg.eigh(x)

        # Filter out eigvalues below threshold and corresponding eigvectors
        eig_th = tf.constant(tf.keras.backend.epsilon(), dtype=tf.float32)
        index_eig = tf.cast(eigval_all > eig_th, tf.int32)
        _, eigval = tf.dynamic_partition(eigval_all, index_eig, 2)
        _, eigvec = tf.dynamic_partition(tf.transpose(eigvec_all), index_eig, 2)

        # Build the diagonal matrix with the filtered eigenvalues or square
        # root of the filtered eigenvalues according to the parameter
        eigval_inv = tf.linalg.diag(1 / eigval)
        eigval_inv_sqrt = tf.linalg.diag(tf.sqrt(1 / eigval))

        cond_sqrt = tf.convert_to_tensor(ret_sqrt)

        diag = tf.cond(cond_sqrt, lambda: eigval_inv_sqrt, lambda: eigval_inv)

        # Rebuild the square root of the inverse matrix
        x_inv = tf.matmul(tf.transpose(eigvec), tf.matmul(diag, eigvec))

        return x_inv

    def loss_func_mae(
        self,
        y_true,
        y_shifted,
        regul_loss=None,
        corr_regul=False,
        orthogonalize=False,
        classic_loss=False,
    ):
        """Loss function to calculate loss, the returned loss is of shape (1, ).
           The loss is calculated based on the expected mean of latent space
           and lagged latent space. So the output have to be arranged
           according MD samples.
           Loss functions based on the paper:
               Chen, Wei, Hythem Sidky, and Andrew L. Ferguson.
               "Capabilities and limitations of time-lagged autoencoders
               for slow mode discovery in dynamical systems." The Journal
               of Chemical Physics 151.6 (2019): 064123.
           Orthogonality constraint added based on the:
               Wang, Wei, et al. "Clustering with orthogonal autoencoder."
               IEEE Access 7 (2019): 62421-62432.
        Arguments:
            y_true: tensorflow tensor, not shifted by tau and truncated in the
                    batch dimension.

            y_shifted: tensorlfow tensor, shifted by tau in the batch dimension and
                    truncated so by size of tau.
            regul_loss: pass your regularization loss here
        """
        # mean-free variables
        z_00 = y_true - tf.reduce_mean(y_true, axis=0)
        z_tt = y_shifted - tf.reduce_mean(y_shifted, axis=0)
        variance = tf.math.reduce_variance(z_00, axis=0, keepdims=True)
        variance_tt = tf.math.reduce_variance(z_tt, axis=0, keepdims=True)
        # defined epsilon, to make the loss function more numerically stable
        # especially that, variance can be close to 0 and z_00/z_tt also
        # Furhtermore, because variance can very small, the division can be
        # very unstable, so the loss is restated so that, it's more numerically stable
        if not classic_loss:
            loss_f = tf.reduce_mean(((z_00 - z_tt) / (tf.sqrt(variance))) ** 2, axis=0)
            loss_f = tf.reduce_sum(loss_f)
        else:
            loss_f = tf.reduce_mean((z_00 - z_tt) ** 2, axis=0, keepdims=True) / (
                variance
            )
            loss_f = tf.reduce_sum(loss_f)

        N = z_00.shape[0]
        k = z_00.shape[1]
        # make outputs dissimilar (not correlated)
        if orthogonalize:
            z_00_norm = z_00 / tf.reshape(tf.norm(z_00, axis=1), (-1, 1))
            orth_00 = tf.matmul(z_00, z_00, transpose_b=True)
            orth_00_norm = tf.matmul(z_00_norm, z_00_norm, transpose_b=True)
            mask_00 = tf.dtypes.cast(orth_00_norm < 0.99, tf.float32)
            z_tt_norm = z_tt / tf.reshape(tf.norm(z_tt, axis=1), (-1, 1))
            orth_11 = tf.matmul(z_tt, z_tt, transpose_b=True)
            orth_11_norm = tf.matmul(z_tt_norm, z_tt_norm, transpose_b=True)
            mask_11 = tf.dtypes.cast(orth_11_norm < 0.99, tf.float32)
            # orth_01 = tf.matmul(z_00_norm, z_tt_norm, transpose_b=True)
            loss_f += tf.norm(orth_00 * mask_00, ord=1) + tf.norm(
                orth_11 * mask_11, ord=1
            )
            # loss_f += tf.norm(orth_01, ord=1) / N
        # turn off shuffling when correlation used!!
        if corr_regul:
            corr_00 = tf.matmul(z_00, z_00, transpose_a=True)
            corr_11 = tf.matmul(z_tt, z_tt, transpose_a=True)
            loss_f += (tf.norm(corr_11, ord=1) + tf.norm(corr_00, ord=1)) / (2 * k)

        if regul_loss is not None:
            loss_f += regul_loss
        return loss_f

    def _koopman_u(self, C):
        """
        Estimate an approximation of the ratio of stationary over empirical distribution from the basis.
        Source: Based on the Pyemma repository
        Arguments:
            C: time-lagged correlation matrix for the whitened and padded data set.
        Returns:
            u: ndarray(M,)
               coefficients of the ratio stationary / empirical dist. from the whitened and expanded basis.
        """
        M = C.shape[0] - 1
        # Compute right and left eigenvectors:
        l, U = tf.linalg.eig(tf.transpose(C))
        idx = tf.argsort(tf.math.abs(l))[::-1]
        l = tf.gather(l, idx, axis=0)
        U = tf.gather(U, idx, axis=1)
        # Extract the eigenvector for eigenvalue one and normalize:
        u = tf.reshape(tf.math.real(U[:, 0]), shape=(-1, 1))
        v = tf.Variable(tf.zeros((M + 1, 1)))
        v[M, 0].assign(1.0)
        u = u / tf.matmul(u, v, transpose_a=True)
        return u

    def _koopman_weight(self, zt0, ztt):
        cov_01 = self.calc_cov(zt0, ztt)
        koop_u = self._koopman_u(cov_01)
        weights = tf.matmul(zt0, koop_u)
        return weights

    def _calc_basis(self, y_true, y_shifted, obs_mean, obs_var, max_batch_size=1):
        """Calculates basis vectors for the SVR method, based on the exponential-average
        approximated covariance matrices.
        Arguments:
             y_true: tensorflow tensor, not shifted by tau and truncated in the
                     batch dimension.

             y_pred: tensorlfow tensor, shifted by tau in the batch dimension and
                     truncated so by size of tau.
             obs_mean: observation mean to normalize input
             obs_var: observation mean to normalize input
             max_batch_size: how much accumalate before calculating the rest of the stats.
         Returns:
             None
        """
        N = y_true.shape[0]
        zt0 = self.target_model(
            self.norm_inputs(y_true, obs_mean, obs_var, logtrans=self.logtrans)
        )
        ztt = self.target_model(
            self.norm_inputs(y_shifted, obs_mean, obs_var, logtrans=self.logtrans)
        )
        # cast to float64 due to errors in cov matrix construction and eigenvalue/vector
        zt0 = tf.cast(zt0, tf.float64)
        ztt = tf.cast(ztt, tf.float64)
        self.ztt_array.append(ztt)
        self.zt0_array.append(zt0)
        if len(self.zt0_array) < max_batch_size:
            return None
        zt0_concat = tf.concat(self.zt0_array, axis=0)
        ztt_concat = tf.concat(self.ztt_array, axis=0)
        x_concat = tf.concat([ztt_concat, zt0_concat], axis=0)
        self.means_ = tf.reduce_mean(x_concat, axis=0)

        M = zt0.shape[1]
        cov_01 = tf.zeros(shape=(M, M), dtype=tf.float64)
        cov_10 = tf.zeros(shape=(M, M), dtype=tf.float64)
        cov_11 = tf.zeros(shape=(M, M), dtype=tf.float64)
        cov_00 = tf.zeros(shape=(M, M), dtype=tf.float64)
        obs = 0
        for zt0, ztt in zip(self.zt0_array, self.ztt_array):
            ztt = ztt - tf.reduce_mean(ztt, axis=0)
            zt0 = zt0 - tf.reduce_mean(zt0, axis=0)
            # weights = self._koopman_weight(zt0, ztt)
            cov_01 += self.calc_cov(zt0, ztt, no_normalize=True)
            cov_10 += self.calc_cov(ztt, zt0, no_normalize=True)
            cov_11 += self.calc_cov(ztt, ztt, no_normalize=True)
            cov_00 += self.calc_cov(zt0, zt0, no_normalize=True)
            obs += N - 1
        cov_01 = cov_01 / obs
        cov_10 = cov_10 / obs
        cov_11 = cov_11 / obs
        cov_00 = cov_00 / obs
        self.cov_0 = 0.5 * (cov_00 + cov_11)
        self.cov_1 = 0.5 * (cov_01 + cov_10)
        assert self.cov_0.shape[0] == zt0.shape[1]
        cov_1_numpy = self.cov_1.numpy()
        cov_0_numpy = self.cov_0.numpy()
        assert cov_1_numpy.dtype == np.float64
        assert cov_0_numpy.dtype == np.float64
        eigvals, eigvecs = scipy.linalg.eigh(cov_1_numpy, b=cov_0_numpy)
        # sorts descending
        idx = np.argsort(np.abs(eigvals))[::-1]
        # remove the slowest process
        self.eigenvectors_ = tf.convert_to_tensor(eigvecs[:, idx], dtype=tf.float32)
        self.eigenvalues_ = tf.convert_to_tensor(eigvals[idx], dtype=tf.float32)
        # transform features trough eigenvectors/basis vectors
        self.means_ = tf.cast(self.means_, tf.float32)
        x_concat = tf.cast(x_concat, tf.float32)
        z = tf.matmul((x_concat - self.means_), self.eigenvectors_)
        self.norms_ = tf.math.sqrt(tf.reduce_mean(z * z, axis=0))
        self.zt0_array = []
        self.ztt_array = []

    def loss_func_vamp(self, y_true, y_pred, reversible=False):
        """Calculates the VAMP-2 score with respect to the network lobes.

        Based on:
            https://github.com/markovmodel/deeptime/blob/master/vampnet/vampnet/vampnet.py
            https://github.com/hsidky/srv/blob/master/hde/hde.py
        Arguments:
            y_true: tensorflow tensor, not shifted by tau and truncated in the
                    batch dimension.

            y_pred: tensorlfow tensor, shifted by tau in the batch dimension and
                    truncated so by size of tau.
        Returns:
            loss_score: tensorflow tensor with shape (1, ).
        """
        N = y_true.shape[0]
        # Remove the mean from the data
        zt0, ztt = (y_true, y_pred)
        tf.cast(zt0, tf.float64)
        tf.cast(ztt, tf.float64)
        tf.cast(N, tf.float64)
        # shape (batch_size, output)
        zt0_mean = tf.reduce_mean(zt0, axis=0, keepdims=True)
        ztt_mean = tf.reduce_mean(ztt, axis=0, keepdims=True)
        # Try to keep the mean about 0
        zt0 = zt0 - zt0_mean
        # shape (batch_size, output)
        ztt = ztt - ztt_mean
        # Calculate the covariance matrices
        # shape (output, output)
        # not using shrinkage should help network learn features
        # despite the noise
        cov_01 = self.calc_cov(zt0, ztt)
        cov_10 = self.calc_cov(ztt, zt0)
        cov_00 = self.calc_cov(zt0, zt0)
        cov_11 = self.calc_cov(ztt, ztt)

        if not reversible:
            # Calculate the inverse of the self-covariance matrices
            cov_00_inv = tf.linalg.sqrtm(tf.linalg.inv(cov_00))
            cov_11_inv = tf.linalg.sqrtm(tf.linalg.inv(cov_11))
            vamp_matrix = tf.matmul(tf.matmul(cov_00_inv, cov_01), cov_11_inv)
            vamp_score = tf.norm(vamp_matrix)
            return -1 - tf.square(vamp_score)
        else:
            cov_0 = 0.5 * (cov_00 + cov_11)
            cov_1 = 0.5 * (cov_01 + cov_10)
            L = tf.linalg.cholesky(cov_0)
            Linv = tf.linalg.inv(L)

            A = tf.matmul(tf.matmul(Linv, cov_1), Linv, transpose_b=True)
            # make sure that all matrices are positive definitive
            # cov_01eigv = tf.linalg.eig(cov_01)[0]
            # cov_10eigv = tf.linalg.eig(cov_10)[0]
            add_loss = -tf.reduce_mean(
                #    tf.math.sign(tf.math.real(cov_01eigv))
                #    + tf.math.sign(tf.math.real(cov_10eigv))
                tf.math.sign(tf.linalg.eigh(cov_11)[0])
                + tf.math.sign(tf.linalg.eigh(cov_00)[0])
            )
            # add_loss += tf.reduce_mean(
            #     tf.abs(tf.math.imag(cov_01eigv)) + tf.abs(tf.math.imag(cov_10eigv))
            # )
            lambdas, eig_v = tf.linalg.eigh(A)
            # zc = tf.concat([zt0, ztt], axis=0)
            # zc -= tf.reduce_mean(zc, axis=0)
            # z = tf.matmul(zc, eig_v)
            # try to keep the norm as unity
            # to make the output have the same norm across epochs
            # thus curiosity won't change the range of outputs during training
            # making the training easier
            # norm = tf.sqrt(tf.reduce_mean(z * z, axis=0))
            # add_loss += tf.reduce_mean((norm - 1) ** 2)
            loss = (
                -1 - tf.reduce_sum(tf.math.sign(lambdas) * tf.abs(lambdas)) + add_loss
            )
            tf.cast(loss, tf.float32)

            return loss

    @tf.function(experimental_relax_shapes=True)
    def _train_step(self, inp_tup, obs_mean, obs_var):
        """One train step as the TF 2 graph.
        Arguments:
            inp_tup: Feature matrix which is an tensor
            obs_mean: observation mean, used for standarization
            obs_var: observation var, used for standarization
        Returns:
            Tuple with loss and global norm
        """

        with tf.GradientTape() as tape:
            # standarization of obs
            inp_tup_standarized = self.norm_inputs(
                inp_tup, obs_mean, obs_var, logtrans=self.logtrans
            )
            if not self.autoencoder:
                state_target = self.target_model(inp_tup_standarized, training=False)
            else:
                state_target = self.target_model_copy(
                    inp_tup_standarized, training=False
                )
            if self.reversible_vampnet and self.ae_lagtime > 0:
                state_target = (
                    tf.matmul(
                        (state_target - self.means_copy.read_value()),
                        self.eigenvectors_copy.read_value(),
                    )
                    / (self.norms_copy.read_value() + self.epsilon)
                )
            state_predict = self.predictor_model(inp_tup_standarized, training=True)
            regul_loss = tf.reduce_sum(self.predictor_model.losses)
            loss, loss_nm = self.loss_func(
                state_target, state_predict, regul_loss=regul_loss
            )
            trainable_var = self.predictor_model.trainable_variables
        grads = tape.gradient((loss), trainable_var)
        global_norm = tf.linalg.global_norm(grads)
        if self.clip_by_global_norm > 0:
            grads, _ = tf.clip_by_global_norm(
                grads, self.clip_by_global_norm, use_norm=global_norm
            )
        self.optimizer.apply_gradients(zip(grads, trainable_var))

        return loss, loss_nm, global_norm

    @tf.function(experimental_relax_shapes=True)
    def _train_step_ae(self, inp_tup, out_tup, obs_mean, obs_var):
        """One train step as the TF 2 graph for autoencoder.
        Arguments:
            inp_tup: Feature matrix which is an tensor
            out_tup: Feature matrix which is tensor,
                     output for the algorithm
            obs_mean: observation mean, used for standarization
            obs_var: observation var, used for standarization
        Returns:
            Tuple with loss and global norm
        """

        with tf.GradientTape() as tape:
            # standarization of obs
            inp_tup_standarized = self.norm_inputs(
                inp_tup, obs_mean, obs_var, logtrans=self.logtrans
            )
            out_tup_standarized = self.norm_inputs(
                out_tup, obs_mean, obs_var, logtrans=self.logtrans
            )
            state_target = self.target_model(inp_tup_standarized, training=True)
            state_decoder = self.decoder_model(state_target, training=True)
            regul_loss = tf.reduce_sum(self.predictor_model.losses)
            loss, _ = self.loss_func(
                state_decoder, out_tup_standarized, regul_loss=regul_loss
            )
            trainable_var = (
                self.target_model.trainable_variables
                + self.decoder_model.trainable_variables
            )
        grads = tape.gradient((loss), trainable_var)
        global_norm = tf.linalg.global_norm(grads)
        if self.clip_by_global_norm > 0:
            grads, _ = tf.clip_by_global_norm(
                grads, self.clip_by_global_norm, use_norm=global_norm
            )
        self.optimizer_ae.apply_gradients(zip(grads, trainable_var))

        return loss, global_norm

    @tf.function(experimental_relax_shapes=True)
    def _train_step_vamp(self, inp_tup, out_tup, obs_mean, obs_var):
        """One train step as the TF 2 graph for autoencoder.
        Arguments:
            inp_tup: Feature matrix which is an tensor
            out_tup: Feature matrix which is tensor,
                     output for the algorithm
            obs_mean: observation mean, used for standarization
            obs_var: observation var, used for standarization
        Returns:
            Tuple with loss and global norm
        """
        with tf.GradientTape() as tape:
            # standarization of obs
            inp_tup_standarized = self.norm_inputs(
                inp_tup, obs_mean, obs_var, logtrans=self.logtrans
            )
            out_tup_standarized = self.norm_inputs(
                out_tup, obs_mean, obs_var, logtrans=self.logtrans
            )
            state_target = self.target_model(inp_tup_standarized, training=True)
            state_decoder = self.target_model(out_tup_standarized, training=True)
            regul_loss = tf.reduce_sum(self.predictor_model.losses)
            loss = self.loss_func_vamp(
                state_decoder, state_target, reversible=self.reversible_vampnet
            )
            loss += regul_loss
            trainable_var = self.target_model.trainable_variables
        grads = tape.gradient((loss), trainable_var)
        global_norm = tf.linalg.global_norm(grads)
        if self.clip_by_global_norm > 0:
            grads, _ = tf.clip_by_global_norm(
                grads, self.clip_by_global_norm, use_norm=global_norm
            )
        self.optimizer_ae.apply_gradients(zip(grads, trainable_var))

        return loss, global_norm

    @tf.function(experimental_relax_shapes=True)
    def _train_step_mae(self, inp_tup, out_tup, obs_mean, obs_var):
        """One train step as the TF 2 graph for autoencoder.
        Arguments:
            inp_tup: Feature matrix which is an tensor
            out_tup: Feature matrix which is tensor,
                     output for the algorithm
            obs_mean: observation mean, used for standarization
            obs_var: observation var, used for standarization
        Returns:
            Tuple with loss and global norm
        """

        with tf.GradientTape() as tape:
            # standarization of obs
            inp_tup_standarized = self.norm_inputs(
                inp_tup, obs_mean, obs_var, logtrans=self.logtrans
            )
            out_tup_standarized = self.norm_inputs(
                out_tup, obs_mean, obs_var, logtrans=self.logtrans
            )
            state_target = self.target_model(inp_tup_standarized, training=True)
            state_decoder = self.target_model(out_tup_standarized, training=True)
            regul_loss = tf.reduce_sum(self.target_model.losses)
            loss = self.loss_func_mae(
                state_decoder,
                state_target,
                regul_loss=regul_loss,
                classic_loss=self.classic_loss,
            )
            trainable_var = self.target_model.trainable_variables
        grads = tape.gradient((loss), trainable_var)
        global_norm = tf.linalg.global_norm(grads)
        if self.clip_by_global_norm > 0:
            grads, _ = tf.clip_by_global_norm(
                grads, self.clip_by_global_norm, use_norm=global_norm
            )
        self.optimizer_ae.apply_gradients(zip(grads, trainable_var))

        return loss, global_norm

    def whiten_data(self, X):
        """Whiten the data before passing further
        arguments:
            X: input data
        returns:
            X_w: Whitened input data
        """
        # Check if first dim. is batch dim.
        if not X.shape[0] > X.shape[1]:
            raise ValueError(
                "You can't use whitening, the batch size is lower than num of features"
            )
        cov = self.calc_cov(X, X)
        eigval, eigvec = tf.linalg.eigh(cov)
        EPS = 1e-5
        D = tf.linalg.diag(1 / tf.math.sqrt(eigval + EPS))
        W = tf.matmul(tf.matmul(eigvec, D), eigvec, transpose_b=True)

        return tf.matmul(X, W)

    def add_to_buffer(self, obs, shuffle=False, previous_obs=False):
        """Adds feature tensors from the observation, to the training buffer.
        Arguments:
            obs: observation sampled from MD simulation, that contains
            `dist_matrix` and `trajectory` keys.
            shuffle: shuffles input with respect to first dimension. It is turned off
                     if lag time is greater than 0.
            previous_obs: if previous obs is added, it should be set True
        Return:
            None
        """
        # TODO
        # move lagtime to training step
        # make sure then, batchs won't be splitted
        # between two different simulations
        # for example by using batch size, that is
        # multiple of simulation size
        if not self.train_buffer_size > 0:
            self.train_buffer.clear()
        dist_matrix_org = obs["dist_matrix"]
        # whiten
        if self.whiten:
            dist_matrix_org = self.whiten_data(dist_matrix_org).numpy()
        self.one_simulation_size = dist_matrix_org.shape[0]
        ae_lagtime = self.ae_lagtime
        if not previous_obs:
            perm = np.arange(dist_matrix_org.shape[0])
        else:
            perm = np.arange(dist_matrix_org.shape[0] - 1)
        if ae_lagtime > 0:
            if not self.ae_train_buffer_size > 0:
                self.ae_train_buffer_size.clear()
            perm_back = perm[:-ae_lagtime]
            perm_front = perm[ae_lagtime:]

            # dist matrix with removed last ae_lagtime entries
            dist_matrix_backward = dist_matrix_org[perm_back]
            # dist matrix shifted by ae_lagtime entries and with those removed
            dist_matrix_shifted = dist_matrix_org[perm_front]
            assert dist_matrix_backward.shape == dist_matrix_shifted.shape
        if shuffle and ae_lagtime == 0:
            np.random.shuffle(perm)
            dist_matrix = dist_matrix_org[perm]
        else:
            dist_matrix = dist_matrix_org
        # resplit buffer maintain proper size
        batch_array = []
        for batch in self.train_buffer:
            batch_array.append(batch)
        batch_array.append(dist_matrix)
        pre_con_shape = batch_array[0].shape
        dist_matrix = np.concatenate(batch_array)
        assert dist_matrix.shape[-1] == pre_con_shape[-1]
        self.train_buffer.clear()
        # end resplit

        ### CURIOSITY BUFFER START
        splited_dist_matrix = np.split(
            dist_matrix,
            range(
                self.minibatch_size_cur, dist_matrix.shape[0], self.minibatch_size_cur
            ),
        )
        for batch_dist_matrix in splited_dist_matrix:
            self.train_buffer.append(batch_dist_matrix)

        ### CURIOSITY BUFFER END

        if ae_lagtime > 0:
            # resplit buffer maintain proper size
            shift_array = []
            back_array = []
            for batch_shift, batch_back in self.ae_train_buffer:
                shift_array.append(batch_shift)
                back_array.append(batch_back)
            shift_array.append(dist_matrix_shifted)
            back_array.append(dist_matrix_backward)
            pre_con_shift_array_shape = shift_array[0].shape
            pre_con_back_array_shape = back_array[0].shape
            shift_array = np.concatenate(shift_array)
            back_array = np.concatenate(back_array)
            assert shift_array.shape[-1] == pre_con_back_array_shape[-1]
            assert back_array.shape[-1] == pre_con_shift_array_shape[-1]
            self.ae_train_buffer.clear()
            # end resplit

            # backward shift
            train_buffer_backward = []
            splited_dist_matrix_backward = np.split(
                back_array,
                range(self.minibatch_size, back_array.shape[0], self.minibatch_size),
            )
            for batch_dist_matrix_backward in splited_dist_matrix_backward:
                train_buffer_backward.append(batch_dist_matrix_backward)

            # forward shift
            train_buffer_shifted = []
            splited_dist_matrix_shifted = np.split(
                shift_array,
                range(self.minibatch_size, shift_array.shape[0], self.minibatch_size),
            )
            for batch_dist_matrix_shifted in splited_dist_matrix_shifted:
                train_buffer_shifted.append(batch_dist_matrix_shifted)

            for batch_shift, batch_back in zip(
                train_buffer_shifted, train_buffer_backward
            ):
                self.ae_train_buffer.append((batch_shift, batch_back))
            # execute only if dequeue is not full yet, otherwise it will lose track and throw error
            if (
                self.ae_lagtime > 0
                and self.cycle
                < self.train_buffer.maxlen
                / (self.one_simulation_size / self.minibatch_size_cur)
                and self.cycle
                < self.ae_train_buffer.maxlen
                / ((self.one_simulation_size - self.ae_lagtime) / self.minibatch_size)
                and self.cycle <= self.test_ae_buffer_iter
                and self.cycle > 0
            ):
                n = self.cycle
                # check if examples are moved by lagtime
                if not previous_obs:
                    train_buf = self.train_buffer[
                        (self.one_simulation_size * n) // self.minibatch_size_cur
                    ][(self.one_simulation_size * n) % self.minibatch_size_cur]
                else:
                    train_buf = self.train_buffer[
                        (self.one_simulation_size * n - bool(n))
                        // self.minibatch_size_cur
                    ][
                        (self.one_simulation_size * n - bool(n))
                        % self.minibatch_size_cur
                    ]
                if not previous_obs:
                    lagtime_rest = self.one_simulation_size - self.ae_lagtime
                else:
                    lagtime_rest = self.one_simulation_size - self.ae_lagtime - 1
                train_ae_buf = self.ae_train_buffer[
                    (lagtime_rest * n) // self.minibatch_size
                ][1][(lagtime_rest * n) % self.minibatch_size]
                # if examples are moved by lagtime, the first example should be the same in the main buffer also
                assert np.sum(train_ae_buf - train_buf) == 0.0
                # shifted version
                if not previous_obs:
                    train_buf_shift = self.train_buffer[
                        (self.one_simulation_size * n + self.ae_lagtime)
                        // self.minibatch_size_cur
                    ][
                        (self.one_simulation_size * n + self.ae_lagtime)
                        % self.minibatch_size_cur
                    ]
                else:
                    train_buf_shift = self.train_buffer[
                        (self.one_simulation_size * n + self.ae_lagtime - bool(n))
                        // self.minibatch_size_cur
                    ][
                        (self.one_simulation_size * n + self.ae_lagtime - bool(n))
                        % self.minibatch_size_cur
                    ]

                train_ae_buf_shift = self.ae_train_buffer[
                    (lagtime_rest * n) // self.minibatch_size
                ][0][(lagtime_rest * n) % self.minibatch_size]
                assert np.sum(train_ae_buf_shift - train_buf_shift) == 0.0
                # update cycle ticker
                self.cycle += 1

    def predict_action(self, obs, n=1, random_actions=False):
        """Predicts `n` structures that are least sampled during all the cycles.
        Arguments:
            obs: observation sampled from MD simulation, that contains
            `dist_matrix` and `trajectory` keys.
            n: number of highest rewards observations to return
            random_actions: Instead of maximum reward, use structures with indices picked from a uniform dist.
        Returns:
            A tuple of four variables is returned, first index are `n` molecular structures sorted from the highest
            reward to lowest. The structures correspond to the `trajectory` key in the observation dict. The second
            are feature tensors, sorted the same way as previous, and also correspond to the `dist_matrix` in the
            obs dict. The third are rewards corresponding to the two previous. The last is array of all rewards,
            unsorted.
        """
        dist_matrix = obs["dist_matrix"]
        splited_dist_matrix = np.split(
            dist_matrix,
            range(self.minibatch_size, dist_matrix.shape[0], self.minibatch_size),
        )
        batch_reward_list = []
        for batch_dist_matrix in splited_dist_matrix:
            batch_dist_matrix = tf.convert_to_tensor(batch_dist_matrix)
            batch_reward = self._calc_reward(batch_dist_matrix)
            if not batch_reward.shape[0] > 0:
                Exception(
                    "Batch reward shape first shape is zero, batch reward: {}".format(
                        batch_reward
                    )
                )
            batch_reward_list.append(batch_reward.numpy())
        batch_reward_array = np.concatenate(batch_reward_list)
        # both have to be equal after spliting
        assert batch_reward_array.shape[0] == dist_matrix.shape[0]
        # get indices of top n
        # see for more here https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
        indices = np.argpartition(batch_reward_array, -n)[-n:]
        # reverse to have descending order
        if not random_actions:
            indices_sorted = indices[np.argsort(batch_reward_array[indices])][::-1]
        else:
            indices_sorted = np.random.randint(0, len(obs["trajectory"]), n)
        actions = [
            obs["trajectory"][ind].reshape((1, *obs["trajectory"][ind].shape))
            for ind in indices_sorted
        ]
        dist_matrixs = [
            obs["dist_matrix"][ind].reshape((1, *obs["dist_matrix"][ind].shape))
            for ind in indices_sorted
        ]
        return (
            actions,
            dist_matrixs,
            batch_reward_array[indices_sorted],
            batch_reward_array,
        )

    def _calc_reward(self, inp_tup):
        """Calculates reward based on the feature tensor provided in the input.
        Arguments:
            inp_tup: Feature tensor
        Returns:
            a tensor of minibatch size.
        """
        obs_mean = self.obs_mean_copy
        obs_var = self.obs_var_copy
        inp_tup_standarized = self.norm_inputs(
            inp_tup, obs_mean, obs_var, logtrans=self.logtrans
        )
        if not self.autoencoder:
            state_target = self.target_model(inp_tup_standarized, training=False)
        else:
            state_target = self.target_model_copy(inp_tup_standarized, training=False)
            if self.reversible_vampnet:
                state_target = (
                    tf.matmul(
                        state_target - self.means_copy.read_value(),
                        self.eigenvectors_copy.read_value(),
                    )
                    / (self.norms_copy.read_value() + self.epsilon)
                )
        state_predict = self.predictor_model(inp_tup_standarized, training=False)
        tf.debugging.check_numerics(
            state_target, "Results are NaN for state target", name=None
        )
        tf.debugging.check_numerics(
            state_predict, "Results are NaN for state predict", name=None
        )

        loss, loss_nm = self.loss_func(state_target, state_predict)
        # loss standarized
        # we substract mean, not like in the paper
        if not self.reward_stand:
            loss_var = 1.0
        else:
            loss_mean, loss_var = self.reward_stat.get_mean_and_var()
        loss_standarized = (loss_nm - loss_mean) / tf.sqrt(loss_var + self.epsilon)
        # weight the rewards with respect to the slow process weights
        loss_standarized = loss_standarized * self.slowp_vector
        return tf.reduce_mean(loss_standarized, axis=-1)

    def get_reward_mean_variance(self):
        """Returns online variance of reward, that is used to
        standarize reward.

        """
        mean, var = self.reward_stat.get_mean_and_var()
        return mean.numpy(), var.numpy()

    def get_state_mean_variance(self):
        """Return mean and variance, that is used to normalize
        state/observation.
        """
        mean, var = self.obs_stat.get_mean_and_var()
        return mean.numpy(), var.numpy()

    def get_implied_time_scales(self):
        """Returns implied time scales if available"""
        if self.reversible_vampnet:
            return self.implied_time_array
        else:
            return None

    def calc_input_latent(self, inp_tup):

        obs_mean = self.obs_mean_copy
        obs_var = self.obs_var_copy
        inp_tup_standarized = self.norm_inputs(
            inp_tup, obs_mean, obs_var, logtrans=self.logtrans
        )
        if not self.autoencoder:
            state_target = self.target_model(inp_tup_standarized, training=False)
        else:
            state_target = self.target_model_copy(inp_tup_standarized, training=False)
            if self.reversible_vampnet:
                state_target = (
                    tf.matmul(
                        state_target - self.means_copy.read_value(),
                        self.eigenvectors_copy.read_value(),
                    )
                    / (self.norms_copy.read_value() + self.epsilon)
                )
        return state_target
