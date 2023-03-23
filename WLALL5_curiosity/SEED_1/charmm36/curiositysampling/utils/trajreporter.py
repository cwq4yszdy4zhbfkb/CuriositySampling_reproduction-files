import simtk.openmm as mm
from simtk.openmm.app.dcdfile import DCDFile
from simtk.unit import nanometer
import numpy as np
import os


class TrajReporter:
    """Stores trajectory every N steps to an object."""

    def __init__(self, report_interval=10, enforcePeriodicBox=None):
        self._reportInterval = report_interval
        self._append = False
        self._enforcePeriodicBox = enforcePeriodicBox
        self.list_to_save = []
        self.first_position = False

    def describeNextReport(self, simulation):
        """Get information about the next report this object will generate.
        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a report for
        Returns
        -------
        tuple
            A six element tuple. The first element is the number of steps
            until the next report. The next four elements specify whether
            that report will require positions, velocities, forces, and
            energies respectively.  The final element specifies whether
            positions should be wrapped to lie in a single periodic box.
        """
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps, True, False, False, False, self._enforcePeriodicBox)

    def report(self, simulation, state):
        positions = state.getPositions(asNumpy=True)
        self.list_to_save.append(positions)

    def get_trajectory(self):
        return self.list_to_save

    def flush_trajectory(self):
        self.list_to_save = []


class DCDReporterMultiFile(object):
    """DCDReporterMultiFile outputs a series of frames from a Simulation to a file.
    To use it, create a DCDReporterMultiFile, then add it to the Simulation's list of reporters.
    The Reporter creates a trajectory file for every cycle of curiosity simulation.
    Arguments:
        file: The filename to write to, the files part named every cycle as
              number_filename.dcd, e.g. 0_niceprotein.dcd, 1_niceprotein.dcd.
              The filename should be given as a full path, e.g. /home/user/niceprotein.dcd
        reportInterval:
            The interval (in time steps) at which to write frames
        append: If True, open an existing DCD file to append to.  If False, create a new file.
        enforcePeriodicBox: Specifies whether particle positions should be translated so the center of every molecule
            lies in the same periodic box.  If None (the default), it will automatically decide whether
            to translate molecules based on whether the system being simulated uses periodic boundary
            conditions.
    """

    def __init__(self, file, reportInterval, append=False, enforcePeriodicBox=None):
        self._reportInterval = reportInterval
        self._append = append
        self._enforcePeriodicBox = enforcePeriodicBox
        if append:
            mode = "r+b"
        else:
            mode = "wb"
        self._mode = mode
        self._dcd = None
        self._path = os.path.dirname(file)
        self._name = os.path.basename(file)
        self._out = None
        self._counter = 0

    def describeNextReport(self, simulation):
        """Get information about the next report this object will generate.
        Arguments:
            simulation : The Simulation to generate a report for
        Returns:
            A six element tuple. The first element is the number of steps
            until the next report. The next four elements specify whether
            that report will require positions, velocities, forces, and
            energies respectively.  The final element specifies whether
            positions should be wrapped to lie in a single periodic box.
        """
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps, True, False, False, False, self._enforcePeriodicBox)

    def nextCycle(self, simulation):
        """Creates a new file, where trajectory is saved.
        The new file has the same name, with a number added at the end.
        """
        # close old file
        if self._out is not None:
            self._out.close()
            del self._dcd
        self._dcd = None
        # create a new name
        self._out = open(
            self._path + "/" + str(self._counter) + "_" + self._name, self._mode
        )
        self._dcd = DCDFile(
            self._out,
            simulation.topology,
            simulation.integrator.getStepSize(),
            simulation.currentStep,
            self._reportInterval,
            self._append,
        )
        self._counter += 1

    def report(self, simulation, state):
        """Generate a report.
        Arguments:
            simulation: The Simulation to generate a report for
            state: The current state of the simulation
        """

        if self._dcd is None:
            self._dcd = DCDFile(
                self._out,
                simulation.topology,
                simulation.integrator.getStepSize(),
                simulation.currentStep,
                self._reportInterval,
                self._append,
            )
        self._dcd.writeModel(
            state.getPositions(), periodicBoxVectors=state.getPeriodicBoxVectors()
        )

    def __del__(self):
        self._out.close()
