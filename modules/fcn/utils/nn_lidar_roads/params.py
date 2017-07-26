"""
File: params.py
Author: Brett Jackson <brett.jackson@here.com>
Version: 3.5.0

Parameter container class used to store parameters for the model

"""

# ==============================================================================
from ..common.params import Params_base


# ==============================================================================
class Params(Params_base):
    # --------------------------------------------------------------------------
    def __init__(self):
        super(Params, self).__init__()

        self.crop_when_scoring = True
        self.cent_when_scoring = False


    # --------------------------------------------------------------------------
    def __str__(self):
        """
        How to display the object when "printed"

        Args:
            Nothing

        Returns:
            Nothing

        """
        the_str = super().__str__()
        return '\n'.join(
            [the_str,
             '\tCropped when scoring: {}'.format(self.crop_when_scoring),
            ])
