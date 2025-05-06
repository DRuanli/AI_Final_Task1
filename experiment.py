import numpy as np
import pandas as pd
import json
import os
import time
import itertools
import math
import random
from typing import List, Tuple, Dict, Any, Optional, Union
from datetime import datetime
from function3d import Function3D
from simulated_annealing import SimulatedAnnealing


class StreamlinedExperimentRunner:
    """Class to run extensive experiments for finding optimal SimulatedAnnealing parameters"""

    def __init__(
            self,
            objective_function: Function3D,
            max_iterations_range: List[int] = [500, 1000, 2000, 3000, 5000, 7500, 10000, 15000, 20000],
            initial_temp_range: List[float] = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0],
            alpha_range: List[float] = [0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.99, 0.995, 0.999],
            runs_per_config: int = 10,
            step_size: float = np.pi / 32,
            initial_state: Tuple[float, float] = (0, 0),
            output_dir: str = "experiment_results",
            sample_mode: Optional[str] = None,
            checkpoint_frequency: int = 10,
            random_seed: Optional[int] = None
    ) -> None:
        """Initialize the experiment runner for extensive parameter testing

        Args:
            objective_function: The function to optimize
            max_iterations_range: Wide range of max_iterations values to test
            initial_temp_range: Wide range of initial_temp values to test
            alpha_range: Wide range of alpha values to test
            runs_per_config: Number of runs for each parameter configuration
            step_size: Step size for the SimulatedAnnealing algorithm
            initial_state: Starting point for all experiments
            output_dir: Directory to save results
            sample_mode: Optional sampling mode to reduce combinations
                         ('random', 'grid', 'focused', 'layered' or None for full grid)
            checkpoint_frequency: How often to save intermediate results (every N configurations)
            random_seed: Optional seed for reproducibility
        """
        self.objective_function = objective_function
        self.max_iterations_range = max_iterations_range
        self.initial_temp_range = initial_temp_range
        self.alpha_range = alpha_range
        self.runs_per_config = runs_per_config
        self.step_size = step_size
        self.initial_state = initial_state
        self.output_dir = output_dir
        self.sample_mode = sample_mode
        self.checkpoint_frequency = checkpoint_frequency

        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Storage for results
        self.results: Dict[Tuple[int, float, float], List[Dict[str, Any]]] = {}
        self.best_configs: List[Tuple[Tuple[int, float, float], Dict[str, Any]]] = []
        self.best_value: float = float('-inf')

        # Experiment metadata
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.total_configs = len(max_iterations_range) * len(initial_temp_range) * len(alpha_range)
        self.configs_tested = 0
        self.start_time = None

        if sample_mode:
            print(f"Using '{sample_mode}' sampling mode to reduce configuration count")

        print(f"Parameter space: {self.total_configs} possible configurations")
        print(
            f"- max_iterations: {len(max_iterations_range)} values from {min(max_iterations_range)} to {max(max_iterations_range)}")
        print(
            f"- initial_temp: {len(initial_temp_range)} values from {min(initial_temp_range)} to {max(initial_temp_range)}")
        print(f"- alpha: {len(alpha_range)} values from {min(alpha_range)} to {max(alpha_range)}")

    def _generate_configurations(self) -> List[Tuple[int, float, float]]:
        """Generate parameter configurations based on sampling mode

        Returns:
            List of parameter configurations to test
        """
        all_configs = list(itertools.product(
            self.max_iterations_range,
            self.initial_temp_range,
            self.alpha_range
        ))

        if not self.sample_mode:
            return all_configs

        if self.sample_mode == 'random':
            # Randomly sample configurations (about 20% of total)
            sample_size = max(20, self.total_configs // 5)
            return sorted(random.sample(all_configs, min(sample_size, len(all_configs))))

        elif self.sample_mode == 'grid':
            # Take every N-th configuration to create a sparser grid
            stride = max(2, self.total_configs // 100)  # Aim for about 100 configs
            return all_configs[::stride]

        elif self.sample_mode == 'focused':
            # Take a finer grid around typical good values and a sparser grid elsewhere
            good_iters = [i for i in self.max_iterations_range if 2000 <= i <= 10000]
            good_temps = [t for t in self.initial_temp_range if 1.0 <= t <= 50.0]
            good_alphas = [a for a in self.alpha_range if 0.9 <= a <= 0.99]

            focused_configs = list(itertools.product(good_iters, good_temps, good_alphas))
            other_configs = [c for c in all_configs if c not in focused_configs]

            # Take all focused configs and a subset of others
            stride = max(2, len(other_configs) // 30)
            return focused_configs + other_configs[::stride]

        elif self.sample_mode == 'layered':
            # Start with a coarse grid, then add finer grids in promising regions
            # This simulates multiple rounds of experiments

            # Layer 1: Very sparse grid across entire space
            stride1 = max(3, self.total_configs // 50)
            layer1 = all_configs[::stride1]

            # Layer 2: Medium grid in moderate-to-good parameter regions
            mid_iters = [i for i in self.max_iterations_range if i >= 1000]
            mid_temps = [t for t in self.initial_temp_range if 0.5 <= t <= 50.0]
            mid_alphas = [a for a in self.alpha_range if 0.85 <= a <= 0.99]

            mid_configs = list(itertools.product(mid_iters, mid_temps, mid_alphas))
            stride2 = max(2, len(mid_configs) // 40)
            layer2 = [c for c in mid_configs[::stride2] if c not in layer1]

            # Layer 3: Fine grid in likely-good parameter regions
            good_iters = [i for i in self.max_iterations_range if 3000 <= i <= 15000]
            good_temps = [t for t in self.initial_temp_range if 1.0 <= t <= 20.0]
            good_alphas = [a for a in self.alpha_range if 0.9 <= a <= 0.99]

            good_configs = list(itertools.product(good_iters, good_temps, good_alphas))
            layer3 = [c for c in good_configs if c not in layer1 and c not in layer2]

            return layer1 + layer2 + layer3

        return all_configs  # Default to full grid

    def _calculate_convergence_speed(self, values: List[float]) -> float:
        """Calculate how quickly the algorithm converges to good solutions

        Args:
            values: List of function values from each iteration

        Returns:
            Convergence speed metric (higher is better)
        """
        if not values:
            return 0.0

        # Find the iteration where we reach 95% of the maximum value
        max_value = max(values)
        target_value = 0.95 * max_value

        for i, value in enumerate(values):
            if value >= target_value:
                # Normalize by the total number of iterations to get a comparable metric
                return float(len(values)) / (i + 1) if i > 0 else float(len(values))

        return 0.0  # Never reached target value

    def _calculate_exploration_rate(self, path: List[Tuple[float, float]]) -> float:
        """Calculate how well the algorithm explored the search space

        Args:
            path: List of states visited during search

        Returns:
            Exploration rate metric (higher means more exploration)
        """
        if len(path) <= 1:
            return 0.0

        # Calculate the unique area covered by the path
        x_coords = [p[0] for p in path]
        y_coords = [p[1] for p in path]

        # Calculate the area of the bounding box of exploration
        x_range = max(x_coords) - min(x_coords)
        y_range = max(y_coords) - min(y_coords)
        area = x_range * y_range

        # Count unique states (with some tolerance for floating point)
        unique_states = set()
        for x, y in path:
            # Round to handle floating point precision
            unique_states.add((round(x, 6), round(y, 6)))

        # Combine both metrics - balance between area covered and unique points visited
        return area * len(unique_states) / len(path)

    def _calculate_stability_score(self, results: List[Dict[str, Any]]) -> float:
        """Calculate stability score based on variation in results

        Args:
            results: List of run results for a configuration

        Returns:
            Stability score (higher is better)
        """
        if not results:
            return 0.0

        best_values = [r['best_value'] for r in results]
        mean_value = sum(best_values) / len(best_values)

        if mean_value == 0:
            return 0.0

        # Calculate coefficient of variation (lower is more stable)
        std_dev = (sum((v - mean_value) ** 2 for v in best_values) / len(best_values)) ** 0.5
        cv = std_dev / mean_value

        # Convert to a score where higher is better
        return 1.0 / (1.0 + cv)

    def _save_checkpoint(self, results_df: pd.DataFrame) -> None:
        """Save current results as a checkpoint

        Args:
            results_df: DataFrame with current results
        """
        checkpoint_path = f"{self.output_dir}/checkpoint_{self.experiment_id}_{self.configs_tested}.csv"
        results_df.to_csv(checkpoint_path, index=False)
        print(f"Checkpoint saved to {checkpoint_path}")

        # Also update the latest checkpoint
        latest_path = f"{self.output_dir}/latest_checkpoint_{self.experiment_id}.csv"
        results_df.to_csv(latest_path, index=False)

    def run_experiments(self) -> Dict[Tuple[int, float, float], List[Dict[str, Any]]]:
        """Run all experiments with different parameter combinations

        Returns:
            Dictionary mapping parameter configurations to results
        """
        # Generate configurations to test
        param_combinations = self._generate_configurations()

        total_experiments = len(param_combinations) * self.runs_per_config
        experiment_count = 0
        self.configs_tested = 0

        print(f"Running {total_experiments} experiments across {len(param_combinations)} parameter configurations...")
        print(f"Each configuration will be run {self.runs_per_config} times for statistical reliability.")

        # Create results dataframe for incremental progress tracking
        results_df = pd.DataFrame(columns=[
            'max_iterations', 'initial_temp', 'alpha', 'run',
            'best_value', 'execution_time', 'convergence_speed',
            'exploration_rate', 'accepted_rate', 'early_stopping',
            'best_state_x', 'best_state_y'
        ])

        self.start_time = time.time()

        try:
            for config_idx, (max_iter, init_temp, alpha) in enumerate(param_combinations):
                config_key = (max_iter, init_temp, alpha)
                self.results[config_key] = []

                print(f"\nTesting configuration {config_idx + 1}/{len(param_combinations)}: "
                      f"max_iterations={max_iter}, initial_temp={init_temp}, alpha={alpha}")
                config_start_time = time.time()

                for run in range(self.runs_per_config):
                    experiment_count += 1
                    run_start_time = time.time()

                    print(f"  Run {run + 1}/{self.runs_per_config} "
                          f"(Experiment {experiment_count}/{total_experiments})")

                    # Create and run the simulated annealing algorithm with current parameters
                    sa = SimulatedAnnealing(
                        objective_function=self.objective_function,
                        initial_state=self.initial_state,
                        max_iterations=max_iter,
                        initial_temp=init_temp,
                        alpha=alpha,
                        step_size=self.step_size
                    )

                    best_state, best_value, path, values = sa.run()

                    execution_time = time.time() - run_start_time

                    # Calculate early stopping - did we reach the maximum iterations?
                    early_stopping = len(path) < max_iter

                    # Calculate performance metrics
                    convergence_speed = self._calculate_convergence_speed(values)
                    exploration_rate = self._calculate_exploration_rate(path)
                    accepted_rate = sa.accepted_moves / (sa.accepted_moves + sa.rejected_moves) if (
                                                                                                               sa.accepted_moves + sa.rejected_moves) > 0 else 0
                    total_iterations = len(path) - 1

                    # Record performance metrics
                    result = {
                        "best_state": best_state,
                        "best_value": best_value,
                        "execution_time": execution_time,
                        "convergence_speed": convergence_speed,
                        "exploration_rate": exploration_rate,
                        "accepted_rate": accepted_rate,
                        "early_stopping": early_stopping,
                        "total_iterations": total_iterations,
                        "accepted_moves": sa.accepted_moves,
                        "rejected_moves": sa.rejected_moves,
                        "final_value": values[-1] if values else 0,
                        "avg_value": sum(values) / len(values) if values else 0
                    }

                    self.results[config_key].append(result)

                    # Add to results dataframe
                    results_df = results_df._append({
                        'max_iterations': max_iter,
                        'initial_temp': init_temp,
                        'alpha': alpha,
                        'run': run + 1,
                        'best_value': best_value,
                        'execution_time': execution_time,
                        'convergence_speed': convergence_speed,
                        'exploration_rate': exploration_rate,
                        'accepted_rate': accepted_rate,
                        'early_stopping': early_stopping,
                        'best_state_x': best_state[0],
                        'best_state_y': best_state[1]
                    }, ignore_index=True)

                    # Update best configuration if this run found a better value
                    if best_value > self.best_value:
                        self.best_value = best_value

                        # Only update best configs if significantly better (0.01% improvement)
                        if not self.best_configs or best_value > self.best_configs[0][1]["best_value"] * 1.0001:
                            self.best_configs = [(config_key, result)]
                        elif abs(best_value - self.best_configs[0][1]["best_value"]) < 0.0001:
                            self.best_configs.append((config_key, result))

                # Report timing for this configuration
                self.configs_tested += 1
                config_time = time.time() - config_start_time
                avg_best_value = sum(run["best_value"] for run in self.results[config_key]) / len(
                    self.results[config_key])
                std_best_value = np.std([run["best_value"] for run in self.results[config_key]])

                print(
                    f"  Completed in {config_time:.2f}s. Average best value: {avg_best_value:.6f} (±{std_best_value:.6f})")

                # Calculate and print ETA
                if self.configs_tested > 1:
                    elapsed_time = time.time() - self.start_time
                    avg_time_per_config = elapsed_time / self.configs_tested
                    remaining_configs = len(param_combinations) - self.configs_tested
                    eta_seconds = avg_time_per_config * remaining_configs

                    hours, remainder = divmod(eta_seconds, 3600)
                    minutes, seconds = divmod(remainder, 60)

                    print(f"  Progress: {self.configs_tested}/{len(param_combinations)} configurations "
                          f"({self.configs_tested / len(param_combinations) * 100:.1f}%)")
                    print(f"  ETA: {int(hours)}h {int(minutes)}m {int(seconds)}s")

                # Save checkpoint if needed
                if self.configs_tested % self.checkpoint_frequency == 0:
                    self._save_checkpoint(results_df)

                    # Find best parameters based on current results
                    self._report_current_best()

        # Handle interruptions gracefully
        except KeyboardInterrupt:
            print("\nExperiment interrupted! Saving current results...")
            self._save_checkpoint(results_df)
            self._save_results(partial=True)
            print("Partial results saved. You can resume experiments later.")

            # Find best parameters based on current results
            self._report_current_best()

            raise

        # Calculate and display total execution time
        total_time = time.time() - self.start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        print(f"\nAll experiments completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")

        # Save final results
        self._save_results()

        return self.results

    def _report_current_best(self) -> None:
        """Report the current best parameters based on results so far"""
        if not self.results:
            print("No results available yet.")
            return

        # Get best configurations
        best_configs = self.find_best_parameters(top_n=3)

        print("\nCurrent Best Parameters:")
        for i, (config, metrics) in enumerate(best_configs[:3], 1):
            max_iter, init_temp, alpha = config
            print(f"{i}. max_iter={max_iter}, init_temp={init_temp:.2f}, alpha={alpha:.4f}")
            print(f"   Value: {metrics['avg_best_value']:.6f} (±{metrics.get('std_best_value', 0):.6f})")

    def _save_results(self, partial: bool = False) -> None:
        """Save experiment results to files

        Args:
            partial: Whether this is a partial save due to interruption
        """
        status = "partial" if partial else "final"

        # Save raw results in JSON format (for potential later analysis)
        results_json = {}
        for config, runs in self.results.items():
            # Convert tuple key to string for JSON
            config_str = f"{config[0]}_{config[1]}_{config[2]}"

            # Clean up results for JSON serialization
            serializable_runs = []
            for run in runs:
                clean_run = {k: v for k, v in run.items() if k != 'best_state'}
                clean_run['best_state_x'] = run['best_state'][0]
                clean_run['best_state_y'] = run['best_state'][1]
                serializable_runs.append(clean_run)

            results_json[config_str] = serializable_runs

        with open(f"{self.output_dir}/{status}_results_{self.experiment_id}.json", 'w') as f:
            json.dump(results_json, f, indent=2)

        # Save processed results as CSV for easy analysis
        rows = []
        for config, runs in self.results.items():
            max_iter, init_temp, alpha = config

            for i, run in enumerate(runs):
                rows.append({
                    'max_iterations': max_iter,
                    'initial_temp': init_temp,
                    'alpha': alpha,
                    'run': i + 1,
                    'best_value': run['best_value'],
                    'execution_time': run['execution_time'],
                    'convergence_speed': run['convergence_speed'],
                    'exploration_rate': run['exploration_rate'],
                    'accepted_rate': run['accepted_moves'] / (run['accepted_moves'] + run['rejected_moves']) if (run[
                                                                                                                     'accepted_moves'] +
                                                                                                                 run[
                                                                                                                     'rejected_moves']) > 0 else 0,
                    'total_iterations': run['total_iterations'],
                    'early_stopping': run.get('early_stopping', False),
                    'best_state_x': run['best_state'][0],
                    'best_state_y': run['best_state'][1]
                })

        results_df = pd.DataFrame(rows)
        results_df.to_csv(f"{self.output_dir}/{status}_results_{self.experiment_id}.csv", index=False)

        # Calculate aggregated results
        agg_results = []
        for config, runs in self.results.items():
            max_iter, init_temp, alpha = config

            best_values = [run['best_value'] for run in runs]
            execution_times = [run['execution_time'] for run in runs]

            agg_results.append({
                'max_iterations': max_iter,
                'initial_temp': init_temp,
                'alpha': alpha,
                'avg_best_value': sum(best_values) / len(best_values),
                'std_best_value': np.std(best_values),
                'min_best_value': min(best_values),
                'max_best_value': max(best_values),
                'avg_execution_time': sum(execution_times) / len(execution_times),
                'std_execution_time': np.std(execution_times),
                'avg_convergence_speed': sum(run['convergence_speed'] for run in runs) / len(runs),
                'avg_exploration_rate': sum(run['exploration_rate'] for run in runs) / len(runs),
                'avg_accepted_rate': sum(run['accepted_rate'] for run in runs) / len(runs),
                'stability_score': self._calculate_stability_score(runs),
                'early_stopping_rate': sum(1 for run in runs if run.get('early_stopping', False)) / len(runs)
            })

        agg_df = pd.DataFrame(agg_results)
        agg_df.to_csv(f"{self.output_dir}/{status}_agg_results_{self.experiment_id}.csv", index=False)

        # Save experiment metadata
        metadata = {
            'experiment_id': self.experiment_id,
            'max_iterations_range': self.max_iterations_range,
            'initial_temp_range': self.initial_temp_range,
            'alpha_range': self.alpha_range,
            'runs_per_config': self.runs_per_config,
            'step_size': float(self.step_size),
            'initial_state': self.initial_state,
            'total_configs_tested': len(self.results),
            'total_runs': sum(len(runs) for runs in self.results.values()),
            'best_configs': [
                {
                    'max_iterations': config[0],
                    'initial_temp': config[1],
                    'alpha': config[2],
                    'best_value': result['best_value'],
                    'execution_time': result['execution_time'],
                    'best_state': [result['best_state'][0], result['best_state'][1]]
                }
                for config, result in self.best_configs[:5]  # Save top 5 configs
            ],
            'status': status,
            'timestamp': datetime.now().isoformat()
        }

        with open(f"{self.output_dir}/{status}_metadata_{self.experiment_id}.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"{status.capitalize()} results saved to {self.output_dir}/")

    def find_best_parameters(self, top_n: int = 10) -> List[Tuple[Tuple[int, float, float], Dict[str, float]]]:
        """Find the best parameter configurations based on multiple criteria

        Args:
            top_n: Number of top configurations to return

        Returns:
            List of (configuration, metrics) tuples sorted by performance
        """
        if not self.results:
            raise ValueError("No experiments have been run yet. Call run_experiments() first.")

        # Calculate average metrics for each configuration
        avg_metrics = {}

        for config, runs in self.results.items():
            best_values = [run["best_value"] for run in runs]

            avg_metrics[config] = {
                "avg_best_value": sum(best_values) / len(best_values),
                "std_best_value": np.std(best_values),
                "min_best_value": min(best_values),
                "max_best_value": max(best_values),
                "avg_execution_time": sum(run["execution_time"] for run in runs) / len(runs),
                "avg_convergence_speed": sum(run["convergence_speed"] for run in runs) / len(runs),
                "avg_exploration_rate": sum(run["exploration_rate"] for run in runs) / len(runs),
                "stability_score": self._calculate_stability_score(runs),
                # Combined score factoring in average value, reliability, and efficiency
                "combined_score": (
                        0.7 * sum(best_values) / len(best_values) +
                        0.2 * self._calculate_stability_score(runs) +
                        0.1 * sum(run["convergence_speed"] for run in runs) / len(runs)
                )
            }

        # Sort configurations by combined score
        sorted_configs = sorted(
            avg_metrics.items(),
            key=lambda x: x[1]["combined_score"],
            reverse=True
        )

        return sorted_configs[:top_n]

    def generate_report(self, top_n: int = 10) -> str:
        """Generate a comprehensive text report of the experiment results

        Args:
            top_n: Number of top configurations to include in the report

        Returns:
            Detailed text report
        """
        if not self.results:
            return "No experiments have been run yet."

        # Find best parameters
        best_configs = self.find_best_parameters(top_n=top_n)

        # Generate report
        report = [
            "========================================================",
            "            SIMULATED ANNEALING PARAMETER STUDY         ",
            "========================================================",
            f"Experiment ID: {self.experiment_id}",
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "EXPERIMENT SUMMARY",
            f"- Parameter Space: {self.total_configs} possible configurations",
            f"- Configurations Tested: {self.configs_tested}",
            f"- Runs Per Configuration: {self.runs_per_config}",
            f"- Total Experiments: {sum(len(runs) for runs in self.results.values())}",
            f"- Sampling Mode: {self.sample_mode if self.sample_mode else 'Full Grid'}",
            "",
            "PARAMETER RANGES",
            f"- max_iterations: {len(self.max_iterations_range)} values from {min(self.max_iterations_range)} to {max(self.max_iterations_range)}",
            f"- initial_temp: {len(self.initial_temp_range)} values from {min(self.initial_temp_range)} to {max(self.initial_temp_range)}",
            f"- alpha: {len(self.alpha_range)} values from {min(self.alpha_range)} to {max(self.alpha_range)}",
            "",
            "TOP CONFIGURATIONS",
        ]

        # Add top configurations
        for i, (config, metrics) in enumerate(best_configs, 1):
            max_iter, init_temp, alpha = config

            config_report = [
                f"{i}. Configuration: max_iterations={max_iter}, initial_temp={init_temp}, alpha={alpha}",
                f"   Average Best Value: {metrics['avg_best_value']:.6f} (±{metrics['std_best_value']:.6f})",
                f"   Range: [{metrics['min_best_value']:.6f}, {metrics['max_best_value']:.6f}]",
                f"   Stability Score: {metrics['stability_score']:.2f}",
                f"   Average Execution Time: {metrics['avg_execution_time']:.2f}s",
                f"   Average Convergence Speed: {metrics['avg_convergence_speed']:.2f}",
                f"   Average Exploration Rate: {metrics['avg_exploration_rate']:.2f}",
                ""
            ]

            report.extend(config_report)

        # Calculate parameter statistics
        report.extend([
            "PARAMETER ANALYSIS",
            "Best values for each parameter:",
        ])

        # Find best values for each parameter
        max_iter_stats = {}
        for max_iter in self.max_iterations_range:
            values = []
            for config, runs in self.results.items():
                if config[0] == max_iter:
                    values.extend([run["best_value"] for run in runs])
            if values:
                max_iter_stats[max_iter] = sum(values) / len(values)

        best_max_iter = max(max_iter_stats.items(), key=lambda x: x[1]) if max_iter_stats else (None, 0)

        temp_stats = {}
        for temp in self.initial_temp_range:
            values = []
            for config, runs in self.results.items():
                if config[1] == temp:
                    values.extend([run["best_value"] for run in runs])
            if values:
                temp_stats[temp] = sum(values) / len(values)

        best_temp = max(temp_stats.items(), key=lambda x: x[1]) if temp_stats else (None, 0)

        alpha_stats = {}
        for alpha in self.alpha_range:
            values = []
            for config, runs in self.results.items():
                if config[2] == alpha:
                    values.extend([run["best_value"] for run in runs])
            if values:
                alpha_stats[alpha] = sum(values) / len(values)

        best_alpha = max(alpha_stats.items(), key=lambda x: x[1]) if alpha_stats else (None, 0)

        report.extend([
            f"- max_iterations: {best_max_iter[0]} (avg value: {best_max_iter[1]:.6f})",
            f"- initial_temp: {best_temp[0]} (avg value: {best_temp[1]:.6f})",
            f"- alpha: {best_alpha[0]} (avg value: {best_alpha[1]:.6f})",
            "",
            "FINAL RECOMMENDATION",
            f"Based on {self.configs_tested} configurations and {sum(len(runs) for runs in self.results.values())} experiments,",
            f"the recommended parameters for the Simulated Annealing algorithm are:",
            f"- max_iterations = {best_configs[0][0][0]}",
            f"- initial_temp = {best_configs[0][0][1]}",
            f"- alpha = {best_configs[0][0][2]}",
            "",
            "This configuration achieved an average best value of:",
            f"{best_configs[0][1]['avg_best_value']:.6f} (±{best_configs[0][1]['std_best_value']:.6f})",
            "",
            "========================================================",
        ])

        # Save report to file
        report_text = "\n".join(report)
        report_path = f"{self.output_dir}/report_{self.experiment_id}.txt"

        with open(report_path, 'w') as f:
            f.write(report_text)

        print(f"Report saved to {report_path}")

        return report_text

    def run_and_report(self) -> Tuple[int, float, float]:
        """Run experiments and generate a comprehensive report

        Returns:
            The best parameter configuration (max_iterations, initial_temp, alpha)
        """
        # Run experiments
        self.run_experiments()

        # Find best parameters
        best_configs = self.find_best_parameters()
        best_config = best_configs[0][0]  # First tuple element is the configuration

        # Generate and print report
        report = self.generate_report()
        print("\n" + "=" * 50)
        print(report)
        print("=" * 50)

        return best_config


import numpy as np
from function3d import Function3D


def main():
    # Create the function to optimize
    function = Function3D()

    # Create experiment runner with wide parameter ranges
    experimenter = StreamlinedExperimentRunner(
        objective_function=function,
        # Extensive parameter ranges
        max_iterations_range=[500, 1000, 2000, 3000, 5000, 7500, 10000],
        initial_temp_range=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0],
        alpha_range=[0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.99, 0.995, 0.999],
        runs_per_config=10,  # Run each config 10 times for statistical reliability
        step_size=np.pi / 32,
        # Use 'layered' sampling to efficiently explore parameter space
        sample_mode='layered',
        # Save checkpoints every 10 configurations
        checkpoint_frequency=10,
        # Set random seed for reproducibility (optional)
        random_seed=42
    )

    # Run all experiments and get comprehensive report
    best_config = experimenter.run_and_report()

    print(f"Optimized parameters: {best_config}")


if __name__ == "__main__":
    main()