### SEED ###
SEED = 1

import random

random.seed(SEED)
import numpy as np

np.random.seed(SEED)
### SEED ###


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

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
import pandas as pd

#!rm actionpositions.dcd
ray.init(num_gpus=4)


# Random generator numbers


class SimCreator:
    def __init__(self):
        pass

    def getNewInstance(self, dirName=None):
        os.mkdir(dirName)
        self.dirName = dirName
        pdb = PDBFile("step5_input.pdb")

        forcefield = ForceField("amber14-all.xml", "implicit/gbn2.xml")
        modeller = Modeller(pdb.topology, pdb.positions)
        ### NVT EQ ###

        system = forcefield.createSystem(
            modeller.topology,
            constraints=HBonds,
            nonbondedMethod=NoCutoff,
            hydrogenMass=4 * amu,
        )
        integrator = LangevinMiddleIntegrator(
            300.00 * kelvin,  # Temperature of head bath
            1 / picosecond,  # Friction coefficient
            0.004 * picoseconds,
        )  # Time step
        # SEED
        integrator.setRandomNumberSeed(SEED)
        platform = Platform.getPlatformByName("CUDA")
        properties = {"DeviceIndex": "0", "Precision": "mixed"}
        simulation = Simulation(
            modeller.topology, system, integrator, platform, properties
        )
        simulation.context.setPositions(pdb.positions)
        simulation.minimizeEnergy()
        simulation.step(25000 * 5)  # 0.5 ns of eq
        state = simulation.context.getState(getPositions=True)
        box_vectors = state.getPeriodicBoxVectors(asNumpy=True)
        positions = state.getPositions(asNumpy=True)

        ### NVT EQ ###
        integrator = LangevinMiddleIntegrator(
            300.00 * kelvin,  # Temperature of head bath
            1 / picosecond,  # Friction coefficient
            0.004 * picoseconds,
        )  # Time step
        # SEED
        integrator.setRandomNumberSeed(SEED)
        platform = Platform.getPlatformByName("CUDA")
        properties = {"DeviceIndex": "0", "Precision": "mixed"}
        simulation = Simulation(
            modeller.topology, system, integrator, platform, properties
        )
        simulation.context.setPositions(positions)
        simulation.step(25000 * 5)  # 0.5 ns of eq
        state = simulation.context.getState(getPositions=True)
        box_vectors = state.getPeriodicBoxVectors()
        positions = state.getPositions(asNumpy=True)
        ### equlibration  ###
        print("START CURIOSITY")
        # system without barostat
        system = forcefield.createSystem(
            modeller.topology,
            constraints=HBonds,
            nonbondedMethod=NoCutoff,
            hydrogenMass=3 * amu,
        )
        omm = OpenMMManager.remote(
            positions=positions,
            system=system,
            topology=modeller.topology,
            integrator=integrator,
            steps=5 * 500000,
            reporter_stride=25000,
            cuda=True,
            file_per_cycle=False,
            warmup_cycles=0,
            warmup_steps=5 * 500000,
            warmup_reporter_stride=25000,
            saving_frequency=25000,
            temperature=300 * kelvin,
            use_dihedrals=True,
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
                    "dense_out": 5,
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
                    "strides": [2, 2, 1],
                    "kernel_size": [5, 3, 3],
                    "padding": "valid",
                },
                "predictor": {
                    "dense_units": [16, 32, 64],
                    "dense_activ": "mish",
                    "dense_layernorm": False,
                    "dense_layernorm": False,
                    "dense_batchnorm": False,
                    "input_batchnorm": False,
                    "dense_out": 5,
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
                    "strides": [2, 2, 1],
                    "kernel_size": [5, 3, 3],
                    "padding": "valid",
                },
            },
            "vampnet": True,
            "reversible_vampnet": False,
            "nonrev_srv": False,
            "autoencoder": True,
            "mae": False,
            "autoencoder_lagtime": 5,
            "minibatch_size_ae": 4000,  # should be greater than sample size, i.e. steps/reporter_stride
            "minibatch_size_cur": 200,
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
            "slowp_vector": [1.0, 1.0, 1.0, 1.0, 1.0],
            "classic_loss": False,
            "whiten": False,
            "logtrans": False,
            "shrinkage": 0.0,
            "energy_mode": None,
            "timescale_mode_target_network_update": False,
            "slowp_kinetic_like_scaling": False,
            "protein_cnn": True,
        }
        config_env = {"openmmmanager": omm}
        csm = CuriousSampling(
            rnd_config=config_rnd,
            env_config=config_env,
            number_of_agents=4,
            random_actions=False,
            buffer_size=20,
            latent_save_frequency=1,
            latent_space_action=False,
            working_directory=os.getcwd(),
        )
        self.csm = csm
        self.sim_id = 1
        with open("positions.pdb", "w") as f:
            pos = ray.get(omm.get_initial_positions.remote())
            PDBFile.writeHeader(pdb.topology, f)
            PDBFile.writeModel(pdb.topology, pos, f, 0)
            PDBFile.writeFooter(pdb.topology, f)

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
