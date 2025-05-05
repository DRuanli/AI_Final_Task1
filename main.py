import numpy as np
import matplotlib.pyplot as plt
from function3d import Function3D
from simulated_annealing import SimulatedAnnealing

def main():
    """Main function to run the program"""
    # Create the function object
    function = Function3D()

    # Visualize the function using sympy (optional)
    # sympy_plot = function.visualize_sympy()

    # Visualize the function using matplotlib
    fig, ax, X, Y, Z = function.visualize_matplotlib()

    # Create and run the simulated annealing algorithm
    sa = SimulatedAnnealing(
        objective_function=function,
        initial_state=(0, 0),  # Start at origin
        max_iterations=5000,  # Increased from 5000
        initial_temp=10.0,  # Increased from 10.0
        alpha=0.9,  # Slower cooling (higher alpha)
        step_size=np.pi / 32  # Step size as specified
    )

    # Run the algorithm
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
    temperatures = [sa.schedule(i) for i in iterations]

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