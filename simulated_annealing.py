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
        Modified temperature schedule with slower cooling and initial high-temperature phase
        """
        # Two-phase cooling: maintain high temperature initially then cool slowly
        if t < self.max_iterations * 0.2:
            return self.initial_temp * 0.9
        else:
            alpha = 0.99  # Much slower cooling (was 0.95)
            return self.initial_temp * (alpha ** (t - 0.2 * self.max_iterations))

    def get_neighbor(self, state):
        """
        Generate neighbor with adaptive exploration strategy
        """
        x, y = state
        temp_ratio = self.current_temp / self.initial_temp

        # Remove "stay in place" option
        actions = [
            (self.step_size, 0),  # East
            (self.step_size, self.step_size),  # Northeast
            (0, self.step_size),  # North
            (-self.step_size, self.step_size),  # Northwest
            (-self.step_size, 0),  # West
            (-self.step_size, -self.step_size),  # Southwest
            (0, -self.step_size),  # South
            (self.step_size, -self.step_size)  # Southeast
        ]

        # Occasionally take larger steps when temperature is high
        if temp_ratio > 0.5 and random.random() < 0.2:
            multiplier = random.randint(2, 5)
            dx, dy = random.choice(actions)
            return (x + dx * multiplier, y + dy * multiplier)

        # Bias toward promising regions when at lower temperatures
        if temp_ratio < 0.3 and self.best_state != (0, 0):
            # Move toward quadrant of best solution
            best_x, best_y = self.best_state
            if random.random() < 0.4:  # 40% chance of biased move
                dx = self.step_size if best_x > x else -self.step_size
                dy = self.step_size if best_y > y else -self.step_size
                return (x + dx, y + dy)

        dx, dy = random.choice(actions)
        return (x + dx, y + dy)

    def run(self):
        """
        Modified run function with restart mechanism
        """
        global_best_state = self.best_state
        global_best_value = self.best_value

        # Store temperature for use in neighbor selection
        self.current_temp = self.initial_temp

        # Main search with multiple restarts
        for restart in range(3):  # Do 3 runs total
            if restart > 0:
                # For restarts, use best position with some random perturbation
                x_random = random.uniform(-2, 2)
                y_random = random.uniform(-2, 2)
                self.current_state = (global_best_state[0] + x_random,
                                      global_best_state[1] + y_random)
                self.current_value = self.function.evaluate(*self.current_state)
                self.current_temp = self.initial_temp * 0.7  # Slightly lower temp for restarts

            # Main simulated annealing loop
            for t in range(self.max_iterations):
                # Update current temperature
                self.current_temp = self.schedule(t)

                # Generate and evaluate neighbor
                neighbor = self.get_neighbor(self.current_state)
                neighbor_value = self.function.evaluate(*neighbor)

                # Decide whether to accept
                delta_e = neighbor_value - self.current_value
                if delta_e > 0 or random.random() < np.exp(delta_e / self.current_temp):
                    self.current_state = neighbor
                    self.current_value = neighbor_value
                    self.path.append(neighbor)

                    # Update best if current is better
                    if self.current_value > self.best_value:
                        self.best_state = self.current_state
                        self.best_value = self.current_value

                # Update global best if needed
                if self.best_value > global_best_value:
                    global_best_state = self.best_state
                    global_best_value = self.best_value

        return global_best_state, global_best_value, self.path