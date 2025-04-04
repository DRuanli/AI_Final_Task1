import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sympy as sp


class SurfaceVisualizer:
    """
    Class for visualizing the 3D function and the search path.
    """

    def __init__(self, function):
        """
        Initialize the visualizer.

        Args:
            function: The function to visualize
        """
        self.function = function

    def create_sympy_function(self):
        """
        Create a sympy expression for the function.

        Returns:
            sympy.Expr: Sympy expression for the function
        """
        x, y = sp.symbols('x y')
        expr = sp.sin(x / 8) + sp.cos(y / 4) - sp.sin((x * y) / 16) + sp.cos(x ** 2 / 16) + sp.sin(y ** 2 / 8)
        return expr

    def plot_sympy_surface(self, x_range=(-10, 10), y_range=(-10, 10)):
        """
        Plot the 3D surface using sympy as required in the assignment.

        Args:
            x_range (tuple): Range of x values
            y_range (tuple): Range of y values

        Returns:
            The sympy plot object
        """
        x, y = sp.symbols('x y')
        expr = self.create_sympy_function()

        # Create a plot using sympy's plotting module
        plot = sp.plotting.plot3d(
            expr,
            (x, x_range[0], x_range[1]),
            (y, y_range[0], y_range[1]),
            title="3D Surface using Sympy"
        )

        return plot

    def plot_path_matplotlib(self, path):
        """
        Plot the search path using matplotlib for better visualization control.

        Args:
            path (list): List of (x, y) points visited during the search

        Returns:
            matplotlib.figure.Figure: The figure object
        """
        # Create a new figure
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Create a grid for the surface
        x = np.linspace(-10, 10, 100)
        y = np.linspace(-10, 10, 100)
        X, Y = np.meshgrid(x, y)

        # Evaluate function at each point
        Z = np.zeros_like(X)
        for i in range(len(x)):
            for j in range(len(y)):
                Z[i, j] = self.function.evaluate(X[i, j], Y[i, j])

        # Plot the surface
        surface = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)

        # Extract path coordinates
        x_path = [point[0] for point in path]
        y_path = [point[1] for point in path]
        z_path = [self.function.evaluate(x, y) for x, y in path]

        # Plot the path with a red line as specified in the assignment
        ax.plot(x_path, y_path, z_path, 'r-', linewidth=2, label='Search Path')
        ax.scatter(x_path[0], y_path[0], z_path[0], color='green', s=100, label='Start (0,0)')
        ax.scatter(x_path[-1], y_path[-1], z_path[-1], color='red', s=100, label='End')

        # Add labels and legend
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('f(X, Y)')
        ax.set_title('3D Surface with Simulated Annealing Search Path')
        ax.legend()

        return fig