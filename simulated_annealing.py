import numpy as np
import random


class SimulatedAnnealing:
    """
    Implements the Simulated Annealing algorithm to find the maximum
    of a 3D function.
    """

    def __init__(self, function, initial_state=(0, 0), max_iterations=10000,
                 initial_temp=500, step_size=np.pi / 32):
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
        Adaptive temperature schedule function. Slower cooling initially,
        then accelerates for exploitation.

        Args:
            t (int): Current iteration

        Returns:
            float: Current temperature
        """
        if t < self.max_iterations * 0.5:
            alpha = 0.98 - 0.0001 * t
        else:
            alpha = 0.95
        return self.initial_temp * (alpha ** t)

    def get_neighbor(self, state):
        """
        Generate a neighbor state with 50% chance to stay or move.

        Args:
            state (tuple): Current state (x, y)

        Returns:
            tuple: Neighbor state (x', y')
        """
        x, y = state
        if random.random() < 0.5:  # 50% chance to stay
            return (x, y)
        else:  # 50% chance to move
            actions = [(self.step_size, 0), (self.step_size, self.step_size),
                       (0, self.step_size), (-self.step_size, self.step_size),
                       (-self.step_size, 0), (-self.step_size, -self.step_size),
                       (0, -self.step_size), (self.step_size, -self.step_size)]
            dx, dy = random.choice(actions)
            return (x + dx, y + dy)

    def run(self):
        """
        Run the Simulated Annealing algorithm with early stopping.

        Returns:
            tuple: The best state found (x, y)
            float: The value at the best state
            list: List of all visited states
        """
        best_values = [self.best_value]
        for t in range(self.max_iterations):
            temp = self.schedule(t)
            if temp < 1e-10 and t > self.max_iterations * 0.3:
                break

            neighbor = self.get_neighbor(self.current_state)
            neighbor_value = self.function.evaluate(*neighbor)
            delta_e = neighbor_value - self.current_value

            if delta_e > 0 or random.random() < np.exp(delta_e / temp):
                self.current_state = neighbor
                self.current_value = neighbor_value
                self.path.append(neighbor)
                if self.current_value > self.best_value:
                    self.best_state = self.current_state
                    self.best_value = self.current_value

            best_values.append(self.best_value)
            # Early stopping if no improvement in last 1000 iterations
            if t > 1000 and all(v == best_values[-1] for v in best_values[-1000:]):
                break

        return self.best_state, self.best_value, self.path