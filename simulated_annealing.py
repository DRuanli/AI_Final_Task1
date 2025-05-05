import random
import math
import numpy as np
from typing import Tuple, List, Any, Optional, Union, Callable
from matplotlib.axes import Axes
from numpy.typing import NDArray

class SimulatedAnnealing:
    """Class to implement the Simulated Annealing Search algorithm"""

    def __init__(self, objective_function: Any, initial_state: Tuple[float, float] = (0, 0),
                 max_iterations: int = 10000, initial_temp: float = 100.0,
                 alpha: float = 0.995, step_size: float = np.pi/32) -> None:
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
        self.path: List[Tuple[float, float]] = [initial_state]
        self.values: List[float] = [self.current_value]

        # Track statistics
        self.accepted_moves: int = 0
        self.rejected_moves: int = 0

    def schedule(self, time_step: int) -> float:
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

    def generate_neighbor(self, state: Tuple[float, float]) -> Tuple[float, float]:
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

    def accept_probability(self, current_value: float, new_value: float, temperature: float) -> float:
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

    def run(self) -> Tuple[Tuple[float, float], float, List[Tuple[float, float]], List[float]]:
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

    def visualize_path(self, ax: Axes, X: NDArray, Y: NDArray, Z: NDArray) -> None:
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