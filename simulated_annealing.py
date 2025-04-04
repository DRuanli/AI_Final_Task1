import numpy as np
import random


class SimulatedAnnealing:
    """
    Implements the Simulated Annealing algorithm to find the maximum
    of a 3D function.
    """

    def __init__(self, function, initial_state=(0, 0), max_iterations=1000,
                 initial_temp=100, step_size=np.pi / 32):
        """
        Initialize the Simulated Annealing algorithm.

        Args:
            function: The function to optimize
            initial_state (tuple): The starting point (x, y)
            max_iterations (int): Maximum number of iterations
            initial_temp (float): Initial temperature
            step_size (float): Step size for neighbor generation
        """
        self.function = function
        self.current_state = initial_state
        self.best_state = initial_state
        self.current_value = function.evaluate(*initial_state)
        self.best_value = self.current_value
        self.max_iterations = max_iterations
        self.initial_temp = initial_temp
        self.step_size = step_size
        self.path = [initial_state]  # To track visited points

    def schedule(self, t):
        """
        Temperature schedule function. This is a custom schedule as required
        by the assignment. We use an exponential cooling schedule that
        starts with high exploration and gradually shifts to exploitation.

        Args:
            t (int): Current iteration

        Returns:
            float: Current temperature
        """
        # Exponential cooling schedule with faster initial cooling
        alpha = 0.95
        return self.initial_temp * (alpha ** t)

    def get_neighbor(self, state):
        """
        Generate a neighbor state by moving step_size in a random direction
        or staying in place.

        Args:
            state (tuple): Current state (x, y)

        Returns:
            tuple: Neighbor state (x', y')
        """
        x, y = state

        # Possible actions: stay in place or move in one of 8 directions
        actions = [(0, 0),  # Stay in place
                   (self.step_size, 0),  # East
                   (self.step_size, self.step_size),  # Northeast
                   (0, self.step_size),  # North
                   (-self.step_size, self.step_size),  # Northwest
                   (-self.step_size, 0),  # West
                   (-self.step_size, -self.step_size),  # Southwest
                   (0, -self.step_size),  # South
                   (self.step_size, -self.step_size)]  # Southeast

        dx, dy = random.choice(actions)
        return (x + dx, y + dy)

    def run(self):
        """
        Run the Simulated Annealing algorithm.

        Returns:
            tuple: The best state found (x, y)
            float: The value at the best state
            list: List of all visited states
        """
        for t in range(self.max_iterations):
            # Get current temperature
            temp = self.schedule(t)

            # Stop if temperature is very low and we've done at least 30% of iterations
            if temp < 1e-10 and t > self.max_iterations * 0.3:
                break

            # Generate a neighbor
            neighbor = self.get_neighbor(self.current_state)

            # Evaluate the neighbor
            neighbor_value = self.function.evaluate(*neighbor)

            # Decide whether to accept the neighbor
            # Since we're maximizing, we flip the sign of delta_e
            delta_e = neighbor_value - self.current_value

            # Always accept if better (higher)
            # Sometimes accept if worse (lower) based on temperature
            if delta_e > 0 or random.random() < np.exp(delta_e / temp):
                self.current_state = neighbor
                self.current_value = neighbor_value
                self.path.append(neighbor)

                # Update best if current is better
                if self.current_value > self.best_value:
                    self.best_state = self.current_state
                    self.best_value = self.current_value

        return self.best_state, self.best_value, self.path