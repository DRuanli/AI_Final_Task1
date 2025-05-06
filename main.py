import numpy as np
import matplotlib.pyplot as plt
from function3d import Function3D
from simulated_annealing import SimulatedAnnealing
from typing import List, Tuple
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from numpy.typing import NDArray

def main() -> None:
    """Main function to run the program"""
    # Create the function object
    function: Function3D = Function3D()

    # Visualize the function using sympy (optional)
    # sympy_plot = function.visualize_sympy()

    # Visualize the function using matplotlib
    fig: Figure
    ax: Axes
    X: NDArray
    Y: NDArray
    Z: NDArray
    fig, ax, X, Y, Z = function.visualize_matplotlib()

    # Create and run the simulated annealing algorithm
    sa: SimulatedAnnealing = SimulatedAnnealing(
        objective_function=function,
        initial_state=(0, 0),  # Start at origin
        max_iterations=15000,  # Increased from 5000
        initial_temp=2.0,  # Increased from 10.0
        alpha=0.85,  # Slower cooling (higher alpha)
        step_size=np.pi / 32  # Step size as specified
    )

    # Run the algorithm
    best_state: Tuple[float, float]
    best_value: float
    path: List[Tuple[float, float]]
    values: List[float]
    best_state, best_value, path, values = sa.run()

    # Visualize the search path
    sa.visualize_path(ax, X, Y, Z)

    # Adjust the view for better visualization
    ax.view_init(elev=30, azim=45)

    # Show the plot
    plt.tight_layout()
    plt.show()

    # Plot the function value over iterations
    plt.figure(figsize=(10, 6))
    plt.plot(values)
    plt.title('Function Value Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('f(x,y)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot the temperature schedule for reference
    iterations = range(len(values))
    temperatures: List[float] = [sa.schedule(i) for i in iterations]

    plt.figure(figsize=(10, 6))
    plt.plot(temperatures)
    plt.title('Temperature Schedule')
    plt.xlabel('Iteration')
    plt.ylabel('Temperature')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()