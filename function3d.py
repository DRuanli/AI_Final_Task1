import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import sympy as sp
from sympy.plotting.plot import plot3d
from typing import Tuple, Any, Callable, Optional, Union


class Function3D:
    """Class to represent and visualize the 3D function"""

    def __init__(self) -> None:
        """Initialize the function parameters"""
        # Define symbolic variables
        self.x_sym, self.y_sym = sp.symbols('x y')

        # Define the symbolic function
        self.f_sym = (sp.sin(self.x_sym/8) + sp.cos(self.y_sym/4) -
                     sp.sin(self.x_sym*self.y_sym/16) +
                     sp.cos(self.x_sym**2/16) +
                     sp.sin(self.y_sym**2/8))

        # Create a lambda function for numerical evaluation
        self.f: Callable[[Union[float, np.ndarray], Union[float, np.ndarray]], Union[float, np.ndarray]] = lambda x, y: (np.sin(x/8) + np.cos(y/4) -
                              np.sin(x*y/16) +
                              np.cos(x**2/16) +
                              np.sin(y**2/8))

    def visualize_sympy(self, x_range: Tuple[float, float] = (-10, 10),
                        y_range: Tuple[float, float] = (-10, 10)) -> Any:
        """Visualize the function using sympy"""
        print("Plotting 3D surface with sympy...")
        plot = plot3d(self.f_sym, (self.x_sym, x_range[0], x_range[1]),
                     (self.y_sym, y_range[0], y_range[1]),
                     title="3D Surface of f(x,y)", xlabel="x", ylabel="y")
        return plot

    def visualize_matplotlib(self, x_range: Tuple[float, float] = (-10, 10),
                            y_range: Tuple[float, float] = (-10, 10),
                            resolution: int = 100) -> Tuple[plt.Figure, plt.Axes, np.ndarray, np.ndarray, np.ndarray]:
        """Visualize the function using matplotlib (more customizable)"""
        print("Plotting 3D surface with matplotlib...")

        # Create a grid of x, y values
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)

        # Evaluate function at all grid points
        Z = self.f(X, Y)

        # Create 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the surface
        surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.8,
                               linewidth=0, antialiased=True)

        # Add labels and colorbar
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('f(X,Y)')
        ax.set_title('3D Surface of f(x,y)')
        fig.colorbar(surf, shrink=0.5, aspect=5)

        return fig, ax, X, Y, Z

    def evaluate(self, x: float, y: float) -> float:
        """Evaluate the function at a specific point"""
        return self.f(x, y)