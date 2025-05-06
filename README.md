# Simulated Annealing Search on 3D Surfaces

## Project Description

This project implements a Simulated Annealing Search algorithm to find the maximum value of a 3D function defined as:

```
f(x, y) = sin(x/8) + cos(y/4) - sin((x·y)/16) + cos(x²/16) + sin(y²/8)
```

The implementation visualizes both the 3D surface and the search path taken by the algorithm, providing a comprehensive visualization of how simulated annealing explores the solution space.

## Project Structure

The project is organized using an object-oriented approach with the following key files:

- `function3d.py`: Implements the target function with both symbolic (sympy) and numerical (numpy) implementations
- `simulated_annealing.py`: Contains the core Simulated Annealing algorithm implementation
- `visualization.py`: Provides utilities for visualizing the 3D surface and search path
- `experiment.py`: Framework for tuning algorithm parameters and running experiments
- `main.py`: Entry point for running the complete solution

## Features

1. **3D Surface Visualization**: Multiple rendering options for the target function using both sympy (as required) and matplotlib for enhanced visualizations
2. **Custom Temperature Schedule**: Implements a three-phase cooling schedule that adapts to different stages of the search process
3. **Search Path Tracking**: Records and visualizes all visited states during the search process with a red path line
4. **Parameter Tuning Framework**: Tools for experimenting with different algorithm parameters
5. **Performance Statistics**: Tracks acceptance rates, temperature decay, and other performance metrics
6. **Comprehensive Visualization**: Options to view the search process from different angles and perspectives

## Requirements

- Python 3.8 or higher
- NumPy
- SymPy
- Matplotlib
- SciPy (optional, for some advanced visualizations)

## Installation

1. Ensure Python 3.8+ is installed on your system
2. Install the required dependencies:
   ```
   pip install numpy sympy matplotlib scipy
   ```
3. Clone the repository or extract the provided files to your working directory

## Usage

### Basic Execution

Run the main script to execute the simulated annealing search with default parameters:

```
python main.py
```

This will:
1. Generate the 3D surface visualization
2. Run the simulated annealing algorithm from the origin (0,0)
3. Display the search path and final result
4. Show performance statistics

### Customizing Parameters

You can modify algorithm parameters in the `main.py` file:

```python
# Example parameter configuration
sa = SimulatedAnnealing(
    function=my_function,
    initial_temp=1000,
    cooling_rate=0.995,
    max_iterations=5000,
    step_size=math.pi/32
)
```

### Running Experiments

For parameter tuning and experiments:

```
python experiment.py
```

This will execute multiple runs with different parameter configurations and provide a comparison of results.

## Implementation Details

### Simulated Annealing Algorithm

The implementation follows the standard simulated annealing approach:

1. Start at the initial point (0,0)
2. Generate a neighbor by taking a random step of size π/32 in one of the eight possible directions
3. Evaluate the function at the new point
4. Accept the move if it improves the function value
5. Accept a worsening move with a probability based on the temperature and how much worse the new solution is
6. Decrease temperature according to the schedule function
7. Repeat until termination conditions are met

### Temperature Schedule

The custom temperature schedule implements a three-phase cooling strategy:
- Initial phase (first 10% of iterations): Very slow cooling to allow extensive exploration
- Middle phase (next 40% of iterations): Medium cooling rate to balance exploration and exploitation
- Final phase (last 50% of iterations): Faster cooling to focus on exploitation and convergence

### Visualization Framework

The visualization system provides:
- 3D surface rendering with customizable ranges and resolution
- Search path overlay as a red line
- Start and end points highlighted
- Function value and temperature progression charts
- Option to save images or generate animations of the search process
