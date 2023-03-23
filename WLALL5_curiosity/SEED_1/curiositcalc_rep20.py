SEED = 1

import random

random.seed(SEED)
import numpy as np

np.random.seed(SEED)


from curiositysampling.ray import OpenMMManager
from curiositysampling.core import CuriousSampling
import ray
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
import time
import shutil
import sys
from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

# only one agent
ray.init(num_gpus=1)


class SimCreator:
    def __init__(self):
        pass

    def getNewInstance(self, dirName=None):
        os.mkdir(dirName)
        self.dirName = dirName
        forcefield = ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
        temperature = 300.00 * kelvin
        friction = 1 / picosecond
        timestep = 0.004 * picoseconds
        psf = CharmmPsfFile("initial_structure.psf")
        psf.box_vectors = ((4.25, 0, 0), (0, 4.25, 0), (0, 0, 4.25))
        pdb = PDBFile("initial_structure.pdb")
        startingPositions = np.array(pdb.positions._value)

        integrator = LangevinMiddleIntegrator(
            300.00 * kelvin,  # Temperature of head bath
            1 / picosecond,  # Friction coefficient
            0.004 * picoseconds,
        )  # Time step

        integrator.setRandomNumberSeed(SEED)

        system = forcefield.createSystem(
            psf.topology,
            nonbondedMethod=PME,
            constraints=HBonds,
            nonbondedCutoff=1.2 * nanometer,
            switchDistance=0.9 * nanometer,
            hydrogenMass=3 * amu,
        )

        system.setDefaultPeriodicBoxVectors(*psf.box_vectors)
        system.addForce(MonteCarloBarostat(1 * bar, 300 * kelvin, 30))
        print(system.getDefaultPeriodicBoxVectors())
        omm = OpenMMManager.remote(
            positions=startingPositions,
            system=system,
            topology=psf.topology,
            integrator=integrator,
            steps=5 * 25000 * 10,
            reporter_stride=5000,
            cuda=True,
            file_per_cycle=True,
            warmup_cycles=0,
            warmup_steps=5 * 25000 * 10,
            warmup_reporter_stride=5000,
            saving_frequency=5000,
            temperature=300 * kelvin,
            use_dihedrals=True,
            use_distances=False,
            selection="protein",
            boosting=True,
            boosting_amd=False,
        )
        self.omm = omm
        config_rnd = {
            "model": {
                "target": {
                    "dense_units": [16, 32, 64],
                    "dense_activ": "mish",
                    "dense_layernorm": False,
                    "dense_batchnorm": False,
                    "input_batchnorm": False,
                    "dense_out": 2,
                    "dense_out_activ": "linear",
                    "layernorm_out": False,
                    "initializer": "lecun_normal",
                    "spectral": False,
                    "orthonormal": False,
                    "l1_reg": 0.0001,
                    "l1_reg_activ": 0.0000,
                    "l2_reg": 0.0001,
                    "l2_reg_activ": 0.0000,
                    "unit_constraint": False,
                    "cnn": True,
                    "strides": [1, 1, 1],
                    "kernel_size": [3, 1, 1],
                    "padding": "valid",
                },
                "predictor": {
                    "dense_units": [16, 32, 64],
                    "dense_activ": "mish",
                    "dense_layernorm": False,
                    "dense_layernorm": False,
                    "dense_batchnorm": False,
                    "input_batchnorm": False,
                    "dense_out": 2,
                    "dense_out_activ": "linear",
                    "layernorm_out": False,
                    "initializer": "lecun_normal",
                    "spectral": False,
                    "orthonormal": False,
                    "l1_reg": 0.0001,
                    "l1_reg_activ": 0.0000,
                    "l2_reg": 0.0001,
                    "l2_reg_activ": 0.0000,
                    "unit_constraint": False,
                    "cnn": True,
                    "strides": [1, 1, 1],
                    "kernel_size": [3, 1, 1],
                    "padding": "valid",
                },
            },
            "vampnet": True,
            "nonrev_srv": False,
            "reversible_vampnet": False,
            "autoencoder": True,
            "mae": False,
            "autoencoder_lagtime": 5,
            "minibatch_size_cur": 200,
            "minibatch_size_ae": 4000,
            "clip_by_global_norm": False,
            "num_of_train_updates": 500,
            "num_of_ae_train_updates": 500,
            "learning_rate_cur": 0.001,
            "learning_rate_ae": 0.01,
            "clr": False,
            "obs_stand": False,
            "reward_stand": False,
            "train_buffer_size": 1000000,
            "optimizer": "adab",
            "optimizer_ae": "adab",
            "target_network_update_freq": 1,
            "hard_momentum": True,
            "vamp2_metric": True,
            "slowp_vector": [1.0, 1.0],
            "classic_loss": False,
            "whiten": False,
            "logtrans": False,
            "shrinkage": 0.0,
            "energy_mode": None,
            "energy_continuous_constant": 25,
            "timescale_mode_target_network_update": False,
            "slowp_kinetic_like_scaling": False,
            "protein_cnn": True,
        }
        config_env = {"openmmmanager": omm}
        csm = CuriousSampling(
            rnd_config=config_rnd,
            env_config=config_env,
            number_of_agents=1,
            random_actions=False,
            buffer_size=20,
            latent_save_frequency=100000,
            latent_space_action=False,
            diskcache=True,
            working_directory=os.getcwd(),
        )
        self.csm = csm
        self.sim_id = 1
        with open("positions.pdb", "w") as f:
            pos = ray.get(omm.get_initial_positions.remote())
            PDBFile.writeHeader(psf.topology, f)
            PDBFile.writeModel(psf.topology, pos, f, 0)
            PDBFile.writeFooter(psf.topology, f)

        return self.csm, self.sim_id

    def __del__(self):
        cwd = os.getcwd()
        shutil.move(
            cwd + "/" + self.sim_id, cwd + "/" + self.dirName + "/" + self.sim_id
        )
        del self.csm
        ray.kill(self.omm)
        del self.omm


csmobj = SimCreator()
intrinsic_reward_reporter = []
state_mean_var_reporter = []
reward_var_reporter = []
num_of_cycles = 401
dirname = "RESULTS"
csm, sim_id = csmobj.getNewInstance(dirName=dirname)
csm.run(
    num_of_cycles,
    action_reporter=None,
    max_reward_reporter=intrinsic_reward_reporter,
    state_mean_var_reporter=state_mean_var_reporter,
    reward_mean_var_reporter=reward_var_reporter,
)

del csmobj
ray.shutdown()
time.sleep(15)
