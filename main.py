import numpy as np
import matplotlib.pyplot as plt
from function3d import Function3D
from simulated_annealing import SimulatedAnnealing
from visualizer import SurfaceVisualizer


def main():
    """
    Main function to run the Simulated Annealing Search on a 3D surface.
    """
    print("Starting Simulated Annealing Search on 3D Surface...")

    # Create the function
    function = Function3D()

    # Create the Simulated Annealing algorithm
    sa = SimulatedAnnealing(
        function=function,
        initial_state=(0, 0),
        max_iterations=5000,
        initial_temp=100,
        step_size=np.pi / 32
    )

    # Run the algorithm
    best_state, best_value, path = sa.run()

    print(f"Search completed!")
    print(f"Best state found: ({best_state[0]:.4f}, {best_state[1]:.4f})")
    print(f"Value at best state: {best_value:.4f}")
    print(f"Total steps taken: {len(path)}")

    # Create the visualizer
    visualizer = SurfaceVisualizer(function)

    # Plot the surface using sympy (as required in the assignment)
    print("Generating sympy visualization...")
    sympy_plot = visualizer.plot_sympy_surface()

    # Plot the surface and path using matplotlib for better visualization
    print("Generating search path visualization...")
    path_plot = visualizer.plot_path_matplotlib(path)

    print("Displaying visualizations...")
    plt.show()

    print("Process completed successfully!")


if __name__ == "__main__":
    main()