"""
Simulated Annealing Search on 3D Surfaces
Introduction to AI - Final Project Task 1

This program implements a Simulated Annealing Search (SAS) algorithm to find the
maximum value of a given 3D function. It visualizes both the 3D surface and the
search path, using OOP principles for organization.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import sympy as sp
import random
import math
from matplotlib.animation import FuncAnimation
from sympy.plotting.plot import plot3d

class Function3D:
    """Class to represent and visualize the 3D function"""

    def __init__(self):
        """Initialize the function parameters"""
        # Define symbolic variables
        self.x_sym, self.y_sym = sp.symbols('x y')

        # Define the symbolic function
        self.f_sym = (sp.sin(self.x_sym/8) + sp.cos(self.y_sym/4) -
                     sp.sin(self.x_sym*self.y_sym/16) +
                     sp.cos(self.x_sym**2/16) +
                     sp.sin(self.y_sym**2/8))

        # Create a lambda function for numerical evaluation
        self.f = lambda x, y: (np.sin(x/8) + np.cos(y/4) -
                              np.sin(x*y/16) +
                              np.cos(x**2/16) +
                              np.sin(y**2/8))

    def visualize_sympy(self, x_range=(-10, 10), y_range=(-10, 10)):
        """Visualize the function using sympy"""
        print("Plotting 3D surface with sympy...")
        plot = plot3d(self.f_sym, (self.x_sym, x_range[0], x_range[1]),
                     (self.y_sym, y_range[0], y_range[1]),
                     title="3D Surface of f(x,y)", xlabel="x", ylabel="y")
        return plot

    def visualize_matplotlib(self, x_range=(-10, 10), y_range=(-10, 10), resolution=100):
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

    def evaluate(self, x, y):
        """Evaluate the function at a specific point"""
        return self.f(x, y)

class SimulatedAnnealing:
    """Class to implement the Simulated Annealing Search algorithm"""

    def __init__(self, objective_function, initial_state=(0, 0),
                 max_iterations=10000, initial_temp=100.0,
                 alpha=0.995, step_size=np.pi/32):
        """Initialize the Simulated Annealing algorithm

        Args:
            objective_function: The function to optimize
            initial_state: Starting point (default: (0,0))
            max_iterations: Maximum number of iterations
            initial_temp: Initial temperature
            alpha: Cooling rate for temperature schedule
            step_size: The step size for generating neighbors (π/32)
        """
        self.objective_function = objective_function
        self.current_state = initial_state
        self.best_state = initial_state
        self.current_value = objective_function.evaluate(*initial_state)
        self.best_value = self.current_value
        self.max_iterations = max_iterations
        self.initial_temp = initial_temp
        self.alpha = alpha
        self.step_size = step_size

        # Track all visited states for visualization
        self.path = [initial_state]
        self.values = [self.current_value]

        # Track statistics
        self.accepted_moves = 0
        self.rejected_moves = 0

    def schedule(self, time_step):
        """Enhanced temperature schedule function

        This implements a modified exponential decay that maintains higher
        temperatures longer in the early phases to improve exploration.

        Args:
            time_step: Current iteration number

        Returns:
            Current temperature
        """
        # Slow cooling at the beginning, faster cooling later
        if time_step < self.max_iterations * 0.1:
            return self.initial_temp * (0.998 ** time_step)  # Very slow cooling
        elif time_step < self.max_iterations * 0.5:
            return self.initial_temp * (0.995 ** time_step)  # Medium cooling
        else:
            return self.initial_temp * (0.99 ** time_step)  # Faster cooling

    def generate_neighbor(self, state):
        """Generate a neighbor state by taking a step in x, y or both

        For each state (x, y), we can move with step size 0 or π/32 in any direction.
        This gives us 9 possible neighbors including staying in place.

        Args:
            state: Current state (x, y)

        Returns:
            Neighboring state (x_new, y_new)
        """
        x, y = state

        # Possible steps: -step_size, 0, step_size for both x and y
        possible_steps = [0, self.step_size, -self.step_size]

        # Randomly choose steps for x and y
        dx = random.choice(possible_steps)
        dy = random.choice(possible_steps)

        # Calculate new state
        new_state = (x + dx, y + dy)

        return new_state

    def accept_probability(self, current_value, new_value, temperature):
        """Calculate probability of accepting a move

        If the new value is better (higher), accept with probability 1.
        Otherwise, accept with a probability that decreases as the temperature decreases.

        Args:
            current_value: Value of current state
            new_value: Value of new state
            temperature: Current temperature

        Returns:
            Probability of accepting the move (0 to 1)
        """
        # If new value is better, always accept
        if new_value > current_value:
            return 1.0

        # Otherwise, calculate acceptance probability
        # We're maximizing, so we use (new - current) instead of (current - new)
        try:
            return math.exp((new_value - current_value) / temperature)
        except OverflowError:
            # Handle potential overflow when temperature is very small
            if new_value < current_value:
                return 0.0
            else:
                return 1.0

    def run(self):
        """Run the simulated annealing algorithm

        Returns:
            best_state: Best state found
            best_value: Value of the best state
            path: List of all visited states
            values: Values at all visited states
        """
        print(f"Starting simulated annealing from {self.current_state}...")

        for iteration in range(self.max_iterations):
            # Get current temperature
            temperature = self.schedule(iteration)

            # Stop if temperature is very close to zero
            if temperature < 1e-10:
                print(f"Stopping early at iteration {iteration} as temperature ≈ 0")
                break

            # Generate a neighbor state
            new_state = self.generate_neighbor(self.current_state)

            # Evaluate the new state
            new_value = self.objective_function.evaluate(*new_state)

            # Calculate acceptance probability
            p = self.accept_probability(self.current_value, new_value, temperature)

            # Decide whether to accept the new state
            if random.random() < p:
                self.current_state = new_state
                self.current_value = new_value
                self.accepted_moves += 1

                # Update best state if this is better
                if new_value > self.best_value:
                    self.best_state = new_state
                    self.best_value = new_value
            else:
                self.rejected_moves += 1

            # Record the path (all visited states)
            self.path.append(self.current_state)
            self.values.append(self.current_value)

            # Print progress every 1000 iterations
            if iteration % 1000 == 0:
                print(f"Iteration {iteration}, Temp={temperature:.6f}, " +
                      f"Current value={self.current_value:.6f}, " +
                      f"Best value={self.best_value:.6f}")

        # Print final statistics
        total_moves = self.accepted_moves + self.rejected_moves
        acceptance_rate = self.accepted_moves / total_moves if total_moves > 0 else 0

        print("\nSimulated Annealing completed:")
        print(f"Best state: {self.best_state}")
        print(f"Best value: {self.best_value}")
        print(f"Acceptance rate: {acceptance_rate:.2%}")
        print(f"Total iterations: {len(self.path) - 1}")

        return self.best_state, self.best_value, self.path, self.values

    def visualize_path(self, ax, X, Y, Z):
        """Visualize the search path on the 3D surface

        Args:
            ax: Matplotlib 3D axis
            X, Y, Z: Mesh grid and function values for the surface
        """
        # Extract x and y coordinates from path
        path_x = [state[0] for state in self.path]
        path_y = [state[1] for state in self.path]

        # Calculate z values for each point in the path
        path_z = [self.objective_function.evaluate(x, y) for x, y in zip(path_x, path_y)]

        # Plot the path as a red line
        ax.plot(path_x, path_y, path_z, 'r-', linewidth=2, label='Search Path')

        # Highlight the starting point
        ax.scatter(path_x[0], path_y[0], path_z[0], color='green', s=100, label='Start')

        # Highlight the final point
        ax.scatter(self.best_state[0], self.best_state[1], self.best_value,
                  color='blue', s=100, label='Best Found')

        # Add a legend
        ax.legend()

def main():
    """Main function to run the program"""
    # Create the function object
    function = Function3D()

    # Visualize the function using sympy (optional)
    # sympy_plot = function.visualize_sympy()

    # Visualize the function using matplotlib
    fig, ax, X, Y, Z = function.visualize_matplotlib()

    # Create and run the simulated annealing algorithm
    # Create and run the simulated annealing algorithm
    sa = SimulatedAnnealing(
        objective_function=function,
        initial_state=(0, 0),  # Start at origin
        max_iterations=10000,  # Increased from 5000
        initial_temp=50.0,  # Increased from 10.0
        alpha=0.997,  # Slower cooling (higher alpha)
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