import numpy as np
from curiositysampling.utils import SettableBiasVariable


class SharedBiases:
    """The class allows for creating an object which stores biases for different
    simulations and combinations of collective variables. For example we can
    store biases for collective variables CV1, CV2, CV3 (which are vectors),
    and their respective biases from N different dimulations.
    Args:
        # Remove action completly
        all_cvs: a dictionary of action_number: bias variable (SettableBiasVariable)
    """

    def __init__(self, variables=None):
        # Dict of biases initialization
        # (0, 1): (gridWidth, gridWidth)
        # the numbers in the tuple are actions' numbers and
        # in the second tuple - gridwidths
        self.variables = variables
        self.bias_arr = {}
        self.bias_arr = np.zeros(tuple(v.gridWidth for v in reversed(self.variables)))

    def update_bias(self, bias=None):
        if bias.shape == self.bias_arr.shape:
            temp_bias = np.copy(self.bias_arr)
            temp_bias += bias
            self.bias_arr = temp_bias
        else:
            raise ValueError(
                "Shapes dont match, should be {0}, but is {1}".format(
                    self.bias_arr.shape, bias.shape
                )
                + "Check whether it should be reversed"
            )

    def get_total_bias_for_variables(self):
        return self.bias_arr
