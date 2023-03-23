from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from simtk.openmm.app import DCDFile
from openmmtools.testsystems import AlanineDipeptideImplicit, AlanineDipeptideExplicit, SrcExplicit, SrcImplicit
import numpy as np
import random
import time
import shutil
from pprint import pprint
import matplotlib.pyplot as plt 
import seaborn as sns 
import os
import logging

dirName = 'simulation_5us'

# change it 
cuda = True
sim_time = 5000000 # ps, so 5.0 us
#sim_time = 2500 # ps, so 2.5 ns
timestep = 0.004 # ps
steps = int(sim_time/0.004)
saving_frequency = 25000 # every 5000 steps, for timestep 0.004 it's every 20ps
cwd = os.getcwd()

os.mkdir(dirName)
logging.basicConfig(filename=dirName+'/file.log',level=logging.INFO)

if cuda:
    platform = Platform.getPlatformByName('CUDA')
else:
    platform = Platform.getPlatformByName('CPU')

forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
pdb = PDBFile('pentapeptide-impl-solv_autopsf.pdb')
modeller = Modeller(pdb.topology, pdb.positions)
modeller.addHydrogens(forcefield, pH=7.4)
modeller.addSolvent(forcefield, padding=1.2*nanometers) 
system = forcefield.createSystem(modeller.topology, nonbondedMethod=PME, constraints=HBonds, nonbondedCutoff=1.2*nanometer, switchDistance=0.9*nanometer,
                                 hydrogenMass=3*amu)
integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, timestep*picosecond)
topology = modeller.topology
positions = modeller.positions

from simtk.openmm.app import PDBFile, PDBxFile
filehandle = open('positions.pdb', 'w')
PDBFile.writeHeader(topology, filehandle)
PDBFile.writeModel(topology, positions, filehandle, 0)
PDBFile.writeFooter(topology, filehandle)
filehandle.close()
properties = {'Precision': 'mixed'}
simulation = Simulation(topology, system, integrator, platform, properties)
dcdreporter = DCDReporter(dirName+'/traj'+'.dcd', saving_frequency)
simulation.reporters.append(StateDataReporter(dirName+'/scalars'+'.csv', saving_frequency, time=True,
                                              potentialEnergy=True, totalEnergy=True, temperature=True))
simulation.context.setPositions(positions)
simulation.minimizeEnergy()

simulation.context.setVelocitiesToTemperature(600 * kelvin)
integrator.setTemperature(600 * kelvin)

simulation.step(50000*5) # heat it up for 1 ns (instead of 200 ps)


simulation.context.setVelocitiesToTemperature(300 * kelvin)
integrator.setTemperature(300 * kelvin)

simulation.reporters.append(dcdreporter)



logging.info('start simulation, time is {0} ns'.format(sim_time/1000))
logging.info('Using GPU ?: {0}'.format(cuda))
logging.info('timestep {0} ps'.format(timestep))
logging.info('temperature {0} K'.format(300))
start_time = time.time()
simulation.step(steps)
end_time = time.time()
logging.info('Simulation ended, time spent {0} h for {1} ns'.format((-start_time+end_time)/3600, sim_time/1000))
