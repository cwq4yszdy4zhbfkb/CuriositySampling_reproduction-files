import simtk.openmm as mm
from simtk.openmm.app import Metadynamics
import numpy as np
import simtk.unit as unit


class SharedMetadynamics(Metadynamics):
    """SharedMetadynamics extends openmm Metadynamics class
    by allowing biased simulations on several CVs with shared
    biases between them.
    """

    def __init__(
        self,
        system,
        variables,
        temperature,
        biasFactor,
        height,
        frequency,
        shared_bias_object,
    ):
        self.system = system
        self.variables = variables
        self.temperature = temperature
        self.biasFactor = biasFactor
        self.height = height
        self.frequency = frequency
        self.shared_bias_object = shared_bias_object
        if len(self.variables) > 3 or len(self.variables) < 1:
            raise ValueError(
                "The number of collective variables should be"
                + " between 1 and 3 (including edge cases)\n"
                + "Now the number of cvs is {}".format(len(self.variables))
            )
        super().__init__(
            self.system,
            self.variables,
            self.temperature,
            self.biasFactor,
            self.height,
            self.frequency,
        )
        # initialize bias object for all cvs
        # update biases if available.

    def _update_forces(self, simulation):
        # update force groups
        freeGroups = set(range(32)) - set(
            force.getForceGroup() for force in self.system.getForces()
        )
        self._force.setForceGroup(max(freeGroups))

        # set function parameters to calculate forces
        if len(self.variables) == 1:
            self._table.setFunctionParameters(self._totalBias.flatten(), *self._limits)
        else:
            self._table.setFunctionParameters(
                *self._widths, self._totalBias.flatten(), *self._limits
            )
        self._force.updateParametersInContext(simulation.context)

    def step(self, simulation, steps):
        # self._update_forces(simulation)
        # perform one step of metadynamics
        stepsToGo = steps
        while stepsToGo > 0:
            nextSteps = stepsToGo
            if simulation.currentStep % self.frequency == 0:
                nextSteps = min(nextSteps, self.frequency)
            else:
                nextSteps = min(
                    nextSteps, self.frequency - simulation.currentStep % self.frequency
                )
            simulation.step(nextSteps)
            if simulation.currentStep % self.frequency == 0:
                # update before, to get total bias and calculate forces
                self._sync_biases()
                position = self._force.getCollectiveVariableValues(simulation.context)
                energy = simulation.context.getState(
                    getEnergy=True, groups={31}
                ).getPotentialEnergy()
                height = self.height * np.exp(
                    -energy / (unit.MOLAR_GAS_CONSTANT_R * self._deltaT)
                )
                self._addGaussian(position, height, simulation.context)
                # update after, to push biases into shared object
                self._sync_biases()
            # The change is due to curiosity modification
            # it is better when simulation is done after the bias
            # because the structure diverges, and algorithm can pickup something
            # different, biaseD
            stepsToGo -= nextSteps

    def _sync_biases(self):
        """Sync bias with shared bias object."""
        # update biases to the object
        # Add only newly added gaussians
        self.shared_bias_object.update_bias(bias=self._selfBias)
        # make the variable zero
        self._selfBias = self._selfBias * 0
        self._totalBias = self._totalBias * 0
        # recompute bias from all processes:
        totalBias = self.shared_bias_object.get_total_bias_for_variables()
        self._totalBias = np.copy(totalBias)


class SettableBiasVariable(mm.app.BiasVariable):
    """Create a Bias Variable as written in the OpenMM documentation,
    but with possibility to update minValue, maxValue, biasWidth
    , periodic and gridWidth. Remember to reinitialize context if set
    different atoms in the force object. Furthermore,
    reinitialize sharedmetadynamics object also.
    """

    def __init__(
        self, force, minValue, maxValue, biasWidth, periodic=False, gridWidth=None
    ):
        super().__init__(
            force, minValue, maxValue, biasWidth, periodic=periodic, gridWidth=gridWidth
        )

    def update_parameters(self, minValue, maxValue, biasWidth, gridWidth=None):
        self.minValue = self._standardize(minValue)
        self.maxValue = self._standardize(maxValue)
        self.biasWidth = self._standardize(biasWidth)
        self.gridWidth = gridWidth
        if gridWidth is None:
            self.gridWidth = int(np.ceil(5 * (maxValue - minValue) / biasWidth))
        else:
            self.gridWidth = gridWidth

        self._scaledVariance = (self.biasWidth / (self.maxValue - self.minValue)) ** 2
