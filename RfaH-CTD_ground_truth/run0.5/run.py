from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
import numpy as np
import random
import time
import shutil
import sys







dirName = "simulation_data"
saving_frequency=25000
os.mkdir(dirName)

pdb = PDBFile('step5_input.pdb')
# Create an openmm system by calling createSystem on psf
forcefield = ForceField('amber14-all.xml', 'implicit/gbn2.xml')
modeller = Modeller(pdb.topology, pdb.positions)
### NVT EQ ###

system = forcefield.createSystem(modeller.topology, constraints=HBonds, nonbondedMethod=NoCutoff, hydrogenMass=4*amu)
integrator = LangevinMiddleIntegrator(300.00*kelvin,   # Temperature of head bath
                                      1/picosecond, # Friction coefficient
                                      0.004*picoseconds) # Time step




platform = Platform.getPlatformByName("CUDA")
properties = {'DeviceIndex': '0', 'Precision': 'mixed'}
simulation = Simulation(modeller.topology, system, integrator, platform, properties)
simulation.context.setPositions(pdb.positions)
simulation.minimizeEnergy()
simulation.step(250000*1) # 1 ns of eq
state = simulation.context.getState(getPositions=True)
box_vectors = state.getPeriodicBoxVectors()
positions = state.getPositions()


### NVT EQ ###
### UNFOLDING ###


platform = Platform.getPlatformByName("CUDA")
properties = {'DeviceIndex': '0', 'Precision': 'mixed'}
integrator = LangevinMiddleIntegrator(525.00*kelvin,   # Temperature of head bath
                                      1/picosecond, # Friction coefficient
                                      0.003*picoseconds) # Time step
simulation = Simulation(modeller.topology, system, integrator, platform, properties)
simulation.context.setPositions(pdb.positions)
simulation.step(250000*100) # 75 ns of eq
state = simulation.context.getState(getPositions=True)
box_vectors = state.getPeriodicBoxVectors()
positions = state.getPositions()



### UNFOLDING ###

with open('positions.pdb', 'w') as f:
    pos = positions
    PDBFile.writeHeader(modeller.topology, f)
    PDBFile.writeModel(modeller.topology, pos, f, 0)
    PDBFile.writeFooter(modeller.topology, f)




### PRODUCTION ###
platform = Platform.getPlatformByName("CUDA")
properties = {'DeviceIndex': '0', 'Precision': 'mixed'}
integrator = LangevinMiddleIntegrator(300.00*kelvin,   # Temperature of head bath
                                      1/picosecond, # Friction coefficient
                                      0.004*picoseconds) # Time step
simulation = Simulation(modeller.topology, system, integrator, platform, properties)
simulation.context.setPositions(positions)

dcdreporter = DCDReporter(dirName+'/traj'+'.dcd', saving_frequency)
simulation.reporters.append(dcdreporter)
simulation.reporters.append(StateDataReporter(dirName+'/scalars'+'.csv', saving_frequency, time=True,
                                              potentialEnergy=True, totalEnergy=True, temperature=True, speed=True))
simulation.step(250000 * 30000) # 30 us

