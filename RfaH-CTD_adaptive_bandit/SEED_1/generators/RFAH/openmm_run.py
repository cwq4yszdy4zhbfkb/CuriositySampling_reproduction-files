import random
#ALL_RND_SEED = 123
#random.seed(ALL_RND_SEED)

import numpy as np

#np.random.seed(ALL_RND_SEED)



from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from simtk.openmm.app import DCDFile
import simtk.openmm
from mdtraj.reporters import XTCReporter
import time
import shutil
import os
import logging

dirName = '.'

forcefield = ForceField('amber14-all.xml', 'implicit/gbn2.xml')

temperature = 300.00*kelvin
friction = 1/picosecond
timestep = 0.004*picoseconds
integrator = LangevinMiddleIntegrator(temperature, friction, timestep)
#integrator.setRandomNumberSeed(ALL_RND_SEED)
if os.path.isfile("input.pdb"):
    pdb = PDBFile("input.pdb")
    startingPositions = pdb.positions
else:
    pdb = PDBFile("../../initial_structure.pdb")

#os.mkdir(dirName)
logging.basicConfig(filename=dirName+'/log.txt',level=logging.INFO)

modeller = Modeller(pdb.topology, pdb.positions)
system = forcefield.createSystem(modeller.topology, constraints=HBonds, nonbondedMethod=NoCutoff, hydrogenMass=4*amu)
# change it 
steps = 500000 * 5
sim_time = steps*0.004
saving_frequency = 25000 # every 10000 steps, for timestep 0.004 it's every 4ps
cwd = os.getcwd()

platform = Platform.getPlatformByName("CUDA")
properties = {'DeviceIndex': '0', 'Precision': 'mixed'}
simulation = Simulation(modeller.topology, system, integrator, platform, properties)
dcdreporter = XTCReporter(dirName+'/output'+'.xtc', saving_frequency)
simulation.reporters.append(dcdreporter)
simulation.reporters.append(StateDataReporter(dirName+'/scalars'+'.csv', saving_frequency, time=True,
                                              potentialEnergy=True, totalEnergy=True, temperature=True,
                                              speed=True))
simulation.context.setPositions(modeller.positions)
simulation.context.setVelocitiesToTemperature(temperature)

with open('structure.pdb', 'w') as f:
    pos = simulation.context.getState(getPositions=True).getPositions()
    #pos = startingPositions
    PDBFile.writeHeader(modeller.topology, f)
    PDBFile.writeModel(modeller.topology, pos, f, 0)
    PDBFile.writeFooter(modeller.topology, f)


logging.info('start simulation, time is {0} ns'.format(sim_time/1000))
logging.info('Using GPU ?: {0}'.format(False))
logging.info('timestep {0} ps'.format(timestep))
logging.info('temperature {0} K'.format(300))
start_time = time.time()
simulation.step(steps)
end_time = time.time()







logging.info('Simulation ended, time spent {0} h for {1} ns'.format((-start_time+end_time)/3600, sim_time/1000))
