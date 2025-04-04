import numpy as np


class Function3D:
    """
    Class representing the 3D function to be optimized.
    f(x, y) = sin(x/8) + cos(y/4) - sin((x·y)/16) + cos(x²/16) + sin(y²/8)
    """

    def __init__(self):
        """Initialize the function class."""
        pass

    def evaluate(self, x, y):
        """
        Evaluate the function at point (x, y).

        Args:
            x (float): x-coordinate
            y (float): y-coordinate

        Returns:
            float: Value of the function at (x, y)
        """
        term1 = np.sin(x / 8)
        term2 = np.cos(y / 4)
        term3 = -np.sin((x * y) / 16)
        term4 = np.cos((x ** 2) / 16)
        term5 = np.sin((y ** 2) / 8)

        return term1 + term2 + term3 + term4 + term5