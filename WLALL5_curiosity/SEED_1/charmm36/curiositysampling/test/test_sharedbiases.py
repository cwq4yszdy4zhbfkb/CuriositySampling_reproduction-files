from curiositysampling.utils.shared_bias import SharedBiases
from itertools import combinations
from simtk.openmm import *
import pytest
import numpy as np
from tqdm import tqdm

variabless2D = []
for i in range(10):
    grid1 = np.random.randint(0, 360)
    grid2 = np.random.randint(0, 360)
    phi = openmm.CustomTorsionForce('theta')
    phi.addTorsion(4, 6, 8, 14)
    phi_bias = app.BiasVariable(phi, -np.pi, np.pi, 0.35, True, grid1)

    psi = openmm.CustomTorsionForce('theta')
    psi.addTorsion(6, 8, 14, 16)
    psi_bias = app.BiasVariable(psi, -np.pi, np.pi, 0.35, True, grid2)

    variabless2D.append((phi_bias, psi_bias))


variabless3D = []
for i in range(10):
    grid1 = np.random.randint(0, 360)
    grid2 = np.random.randint(0, 360)
    grid3 = np.random.randint(0, 360)
    phi = openmm.CustomTorsionForce('theta')
    phi.addTorsion(4, 6, 8, 14)
    phi_bias = app.BiasVariable(phi, -np.pi, np.pi, 0.35, True, grid1)

    psi = openmm.CustomTorsionForce('theta')
    psi.addTorsion(6, 8, 14, 16)
    psi_bias = app.BiasVariable(psi, -np.pi, np.pi, 0.35, True, grid2)

    sigma = openmm.CustomTorsionForce('sigma')
    sigma.addTorsion(8, 10, 16, 18)
    sigma_bias = app.BiasVariable(psi, -np.pi, np.pi, 0.35, True, grid3)


    variabless3D.append((phi_bias, psi_bias, sigma_bias))


@pytest.mark.parametrize('var1, var2', list(variabless2D))
def test_sharedbiases2D(var1, var2):
    # init sharedbiases
    sb = SharedBiases(variables=[var1, var2])
    # try add new biases with random actions out of dict
    grid_shape = tuple(reversed([var1.gridWidth, var2.gridWidth]))
    compare_free_energy = np.zeros(grid_shape)
    print('testing {0} cvs with dimensions {1} and {2}'.format(2, var1.gridWidth, 
                                                                    var2.gridWidth))
    for i in tqdm(range(20)):
        grid_shape = grid_shape
        new_bias = np.random.randn(*grid_shape)
        sb.update_bias(bias=new_bias)
        try:
            compare_free_energy += new_bias
        except:
            compare_free_energy = new_bias
    get_total_energy = sb.get_total_bias_for_variables()
    assert np.all(np.isclose(get_total_energy, compare_free_energy)), 'Compare total biases in the shared bias object and here'


@pytest.mark.parametrize('var1, var2, var3', list(variabless3D))
def test_sharedbiases3D(var1, var2, var3):
    # init sharedbiases
    sb = SharedBiases(variables=[var1, var2, var3])
    # try add new biases with random actions out of dict
    grid_shape = tuple(reversed([var1.gridWidth, var2.gridWidth, var3.gridWidth]))
    compare_free_energy = np.zeros(grid_shape)
    print('testing {0} cvs with dimensions {1}, {2} and {3}'.format(3, var1.gridWidth, 
                                                                    var2.gridWidth, 
                                                                    var3.gridWidth))
    for i in tqdm(range(20)):
        grid_shape = grid_shape
        new_bias = np.random.randn(*grid_shape)
        sb.update_bias(bias=new_bias)
        try:
            compare_free_energy += new_bias
        except:
            compare_free_energy = new_bias
    get_total_energy = sb.get_total_bias_for_variables()
    assert np.all(np.isclose(get_total_energy, compare_free_energy)), 'Compare total biases in the shared bias object and here'
