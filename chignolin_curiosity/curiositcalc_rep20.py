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
from openmmtools.testsystems import (
    AlanineDipeptideImplicit,
    AlanineDipeptideExplicit,
    SrcExplicit,
    SrcImplicit,
)
import time
import shutil
import sys
from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow as tf
import pandas as pd


ray.init(num_gpus=4)


class SimCreator:
    def __init__(self):
        pass

    def getNewInstance(self, dirName=None):
        os.mkdir(dirName)
        self.dirName = dirName
        psf = CharmmPsfFile("step5_input.psf")
        pdb = PDBFile("step5_input.pdb")

        all_files = []
        for root, dirs, files in os.walk("charmm36"):
            for file in files:
                if file.endswith((".prm", "rtf", "str")):
                    all_files.append("charmm36/" + file)

        params = CharmmParameterSet(*all_files)
        # 64 64 64, below in nm
        psf.box_vectors = ((4.0, 0, 0), (0, 4.0, 0), (0, 0, 4.0))
        # Create an openmm system by calling createSystem on psf

        ### NPT EQ ###
        system = psf.createSystem(
            params,
            nonbondedMethod=PME,
            nonbondedCutoff=1.0 * nanometer,
            switchDistance=0.9 * nanometer,
            constraints=HBonds,
            hydrogenMass=3 * amu,
        )
        system.addForce(MonteCarloBarostat(1 * bar, 340 * kelvin, 30))
        integrator = LangevinMiddleIntegrator(
            340.00 * kelvin,  # Temperature of head bath
            1 / picosecond,  # Friction coefficient
            0.002 * picoseconds,
        )  # Time step
        platform = Platform.getPlatformByName("CUDA")
        properties = {"DeviceIndex": "0, 1", "Precision": "mixed"}
        simulation = Simulation(psf.topology, system, integrator, platform, properties)
        simulation.context.setPositions(pdb.positions)
        simulation.minimizeEnergy()
        print(system.getDefaultPeriodicBoxVectors())
        simulation.step(250009 * 1)  # 1 ns of eq
        # simulation.step(29*1) # 1 ns of eq
        state = simulation.context.getState(getPositions=True)
        box_vectors = state.getPeriodicBoxVectors()
        positions = state.getPositions()

        ### NVT EQ ###

        ### NVT unfold ###
        system = psf.createSystem(
            params,
            nonbondedMethod=PME,
            nonbondedCutoff=1.0 * nanometer,
            switchDistance=0.9 * nanometer,
            constraints=HBonds,
            hydrogenMass=3 * amu,
        )
        system.setDefaultPeriodicBoxVectors(*box_vectors)
        # works only with reinitialize
        system.addForce(MonteCarloBarostat(1 * bar, 340.00 * kelvin, 30))
        # system.addForce(MonteCarloAnisotropicBarostat([1., 1., 1.]*bar, 500*kelvin, True, True, True))
        integrator = LangevinMiddleIntegrator(
            340.00 * kelvin,  # Temperature of head bath
            1 / picosecond,  # Friction coefficient
            0.004 * picoseconds,
        )  # Time step
        platform = Platform.getPlatformByName("CUDA")
        properties = {"DeviceIndex": "0, 1", "Precision": "mixed"}
        simulation = Simulation(psf.topology, system, integrator, platform, properties)
        simulation.context.setPositions(positions)
        print(system.getDefaultPeriodicBoxVectors())
        simulation.step(250000 * 25)  # 25 ns of eq
        # simulation.step(2*25) # 25 ns of eq
        # simulation.step(1) # 25 ns of eq
        state = simulation.context.getState(getPositions=True)
        box_vectors = state.getPeriodicBoxVectors()
        positions = state.getPositions()

        ### NVT unfold  ###

        ### NPT EQ ###
        system = psf.createSystem(
            params,
            nonbondedMethod=PME,
            nonbondedCutoff=1.0 * nanometer,
            switchDistance=0.9 * nanometer,
            constraints=HBonds,
            hydrogenMass=3 * amu,
        )
        system.setDefaultPeriodicBoxVectors(*box_vectors)
        # works only with reinitialize
        system.addForce(MonteCarloBarostat(1 * bar, 340.00 * kelvin, 30))
        # system.addForce(MonteCarloAnisotropicBarostat([1., 1., 1.]*bar, 340*kelvin, True, True, True))
        integrator = LangevinMiddleIntegrator(
            340.00 * kelvin,  # Temperature of head bath
            1 / picosecond,  # Friction coefficient
            0.004 * picoseconds,
        )  # Time step
        platform = Platform.getPlatformByName("CUDA")
        properties = {"DeviceIndex": "0, 1", "Precision": "mixed"}
        simulation = Simulation(psf.topology, system, integrator, platform, properties)
        simulation.context.setPositions(positions)
        print(system.getDefaultPeriodicBoxVectors())
        simulation.step(25000 * 2)  # 0.2 ns of eq
        # simulation.step(2*2) # 0.2 ns of eq
        # simulation.step(1) # 25 ns of eq
        state = simulation.context.getState(getPositions=True)
        box_vectors = state.getPeriodicBoxVectors()
        positions = state.getPositions(asNumpy=True)

        ### NPT EQ  ###

        # system without barostat
        system = psf.createSystem(
            params,
            nonbondedMethod=PME,
            nonbondedCutoff=1.0 * nanometer,
            switchDistance=0.9 * nanometer,
            constraints=HBonds,
            hydrogenMass=3 * amu,
        )
        system.setDefaultPeriodicBoxVectors(*box_vectors)

        integrator = LangevinMiddleIntegrator(
            340.00 * kelvin,  # Temperature
            1 / picosecond,  # Friction coeff
            0.004 * picoseconds,
        )  # Time step

        system.addForce(MonteCarloBarostat(1 * bar, 340 * kelvin, 30))
        # system.addForce(MonteCarloAnisotropicBarostat([1., 1., 1.]*bar, 340*kelvin, True, True, True))
        print(system.getDefaultPeriodicBoxVectors())
        print("START CURIOSITY")
        omm = OpenMMManager.remote(
            positions=positions,
            system=system,
            topology=psf.topology,
            integrator=integrator,
            steps=100 * 250000,
            reporter_stride=25000,
            cuda=True,
            file_per_cycle=True,
            warmup_cycles=0,
            warmup_steps=100 * 250000,
            warmup_reporter_stride=25000,
            saving_frequency=25000,
            temperature=340 * kelvin,
            use_dihedrals=False,
            use_distances=True,
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
                    "dense_out": 6,
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
                    "kernel_size": [3, 3, 3],
                    "padding": "valid",
                },
                "predictor": {
                    "dense_units": [16, 32, 64],
                    "dense_activ": "mish",
                    "dense_layernorm": False,
                    "dense_layernorm": False,
                    "dense_batchnorm": False,
                    "input_batchnorm": False,
                    "dense_out": 6,
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
                    "kernel_size": [3, 3, 3],
                    "padding": "valid",
                },
            },
            "reversible_vampnet": False,
            "vampnet": True,
            "nonrev_srv": False,
            "autoencoder": True,
            "mae": False,
            "autoencoder_lagtime": 10,
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
            "slowp_vector": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "classic_loss": False,
            "whiten": False,
            "logtrans": False,
            "shrinkage": 0.0,
            "energy_mode": None,
            "slowp_kinetic_like_scaling": False,
            "protein_cnn": True,
            "timescale_mode_target_network_update": False,
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
num_of_cycles = 100
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
