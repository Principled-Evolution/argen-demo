#!/usr/bin/env python3
"""
Wrapper script to evaluate multiple models using examples/evaluate_baseline.py.

This script:
1. Runs examples/evaluate_baseline.py on multiple models in sequence
2. Saves all results in a single timestamped directory
3. Creates a summary markdown table with metrics from all models
"""

import sys
import os
import json
import argparse
import subprocess
import datetime
import pathlib
import multiprocessing
import threading
import queue
import time
from typing import Dict, List, Any, Optional, Tuple
import re

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from argen.config import DEFAULT_SCENARIOS_PATH


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate multiple models using examples/evaluate_baseline.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=False,
        help="List of model names/paths to evaluate"
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default=DEFAULT_SCENARIOS_PATH,
        help="Path to the scenarios file"
    )
    parser.add_argument(
        "--evaluator",
        type=str,
        choices=["openai", "gemini", "anthropic"],  # Add anthropic
        default="gemini",
        help="Which LLM provider to use for evaluation"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Temperature for model generation"
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        choices=["BASIC", "ENHANCED", "MEDICAL"],
        default="ENHANCED",
        help="Type of system prompt to use"
    )
    parser.add_argument(
        "--no_medical_disclaimer_penalty",
        action="store_true",
        help="Do not apply penalty for missing medical disclaimer"
    )
    parser.add_argument(
        "--no_referral_penalty",
        action="store_true",
        help="Do not apply penalty for missing professional referral"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode with mock responses"
    )
    parser.add_argument(
        "--no_parallel",
        action="store_true",
        help="Disable parallel evaluation across multiple GPUs"
    )
    parser.add_argument(
        "--no_pipeline",
        action="store_true",
        help="Disable pipelined evaluation (wait for each model to complete before starting the next)"
    )
    parser.add_argument(
        "--pipeline_delay",
        type=int,
        default=10,
        help="Delay in seconds before starting the next model in the pipeline (default: 10)"
    )
    parser.add_argument(
        "--max_concurrent_models",
        type=int,
        default=None,
        help="Maximum number of models to evaluate concurrently (default: number of GPUs)"
    )
    parser.add_argument(
        "--eval_mode",
        type=str,
        choices=["batch", "individual"],
        default="individual",
        help="Evaluation mode: 'individual' (default, one API call per evaluation) or 'batch' (faster). Note: Only Gemini supports batch mode; Anthropic batch mode is disabled due to inconsistency issues."
    )
    parser.add_argument(
        "--generation_batch_size",
        type=int,
        default=50,
        help="Batch size for local model generation (higher = faster but more GPU memory required)"
    )

    # Add model selection per provider
    parser.add_argument(
        "--openai-model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model to use for evaluation (e.g., gpt-4o-mini, gpt-4o, o3-mini)"
    )

    parser.add_argument(
        "--anthropic-model",
        type=str,
        default="claude-3-5-sonnet",
        help="Anthropic model to use for evaluation (e.g., claude-3-5-sonnet, claude-3-opus)"
    )

    parser.add_argument(
        "--gemini-model",
        type=str,
        default="gemini-2.0-flash",
        help="Gemini model to use for evaluation"
    )

    parser.add_argument(
        "--input-json",
        type=str,
        help="Path to JSON file containing pre-generated prompt-response pairs to evaluate (mutually exclusive with --models)"
    )

    return parser.parse_args()


def get_cuda_device_count():
    """
    Get the number of available CUDA devices.

    Returns:
        int: Number of available CUDA devices, or 0 if CUDA is not available.
    """
    try:
        # Try to import torch and check CUDA availability
        import torch
        if torch.cuda.is_available():
            return torch.cuda.device_count()
        return 0
    except ImportError:
        # If torch is not installed, try using nvidia-smi
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True
            )
            # Count non-empty lines in the output
            return len([line for line in result.stdout.strip().split('\n') if line.strip()])
        except (subprocess.SubprocessError, FileNotFoundError):
            # If nvidia-smi fails or is not found, assume no CUDA devices
            return 0

def create_output_directory() -> str:
    """Create a timestamped output directory for results."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("training_reports", f"evaluate_baseline_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def get_scenario_filename(scenario_path: str) -> str:
    """Extract the base filename from the scenario path."""
    return os.path.basename(scenario_path).split('.')[0]


def evaluate_model(
    model: str,
    scenarios: str,
    output_dir: str,
    evaluator: str,
    temperature: float,
    system_prompt: str,
    no_medical_disclaimer_penalty: bool,
    no_referral_penalty: bool,
    test: bool,
    eval_mode: str,
    generation_batch_size: int,
    gpu_id: int = None,
    openai_model: str = None,
    anthropic_model: str = None,
    gemini_model: str = None
) -> tuple:
    """
    Evaluate a single model using examples/evaluate_baseline.py.

    Args:
        model: Name or path of the model to evaluate
        scenarios: Path to the scenarios file
        output_dir: Directory to save results
        evaluator: Which LLM to use for evaluation (openai or gemini)
        temperature: Temperature for model generation
        system_prompt: Type of system prompt to use
        no_medical_disclaimer_penalty: Whether to disable medical disclaimer penalty
        no_referral_penalty: Whether to disable referral penalty
        test: Whether to run in test mode
        eval_mode: Evaluation mode for Gemini evaluator ('batch' or 'individual')
        generation_batch_size: Batch size for local model generation
        gpu_id: ID of the GPU to use for evaluation (None for CPU or default GPU)

    Returns:
        tuple: (output_file_path, original_model_name) or (None, None) on failure
    """
    scenario_filename = get_scenario_filename(scenarios)
    model_name = os.path.basename(model).replace('/', '_')
    output_file = os.path.join(output_dir, f"eval_{model_name}_{scenario_filename}.json")

    cmd = [
        "python", "examples/evaluate_baseline.py",
        "--model", model,
        "--scenarios", scenarios,
        "--output_base", os.path.join(output_dir, f"temp_{model_name}"),
        "--evaluator", evaluator,
        "--temperature", str(temperature),
        "--system_prompt", system_prompt,
        "--eval-mode", eval_mode,
        "--generation_batch_size", str(generation_batch_size)
    ]

    # Add model selection parameters
    if openai_model:
        cmd.extend(["--openai-model", openai_model])
    if anthropic_model:
        cmd.extend(["--anthropic-model", anthropic_model])
    if gemini_model:
        cmd.extend(["--gemini-model", gemini_model])

    if no_medical_disclaimer_penalty:
        cmd.append("--no_medical_disclaimer_penalty")

    if no_referral_penalty:
        cmd.append("--no_referral_penalty")

    if test:
        cmd.append("--test")

    # Set environment variables for GPU selection
    env = os.environ.copy()
    gpu_info = ""
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        gpu_info = f" (on GPU {gpu_id})"

    print(f"\n\n{'='*80}")
    print(f"Evaluating model: {model}{gpu_info}")
    print(f"{'='*80}\n")

    try:
        subprocess.run(cmd, check=True, env=env)

        # Find the generated output file (which has a timestamp in its name)
        output_pattern = os.path.join(output_dir, f"temp_{model_name}_*.json")
        output_files = list(pathlib.Path(output_dir).glob(f"temp_{model_name}_*.json"))

        if not output_files:
            print(f"Error: No output file found matching pattern {output_pattern}")
            return None, None

        # Rename the file to our standardized name
        os.rename(str(output_files[0]), output_file)
        return output_file, model

    except subprocess.CalledProcessError as e:
        print(f"Error evaluating model {model}: {e}")
        return None, None


def evaluate_model_on_gpu(args):
    """
    Helper function for multiprocessing to evaluate a model on a specific GPU.

    Args:
        args: Tuple of (model, scenarios, output_dir, evaluator, temperature,
              system_prompt, no_medical_disclaimer_penalty, no_referral_penalty,
              test, eval_mode, generation_batch_size, gpu_id, openai_model,
              anthropic_model, gemini_model)

    Returns:
        tuple: (output_file_path, original_model_name) or (None, None) on failure
    """
    return evaluate_model(*args)


class ModelEvaluationManager:
    """
    Manages the pipelined evaluation of multiple models across available GPUs.

    This class implements a pipeline approach where:
    1. Models are started on available GPUs with a delay between starts
    2. As soon as a GPU becomes available, the next model is started
    3. Results are collected as they become available
    """

    def __init__(self,
                 models: List[str],
                 scenarios: str,
                 output_dir: str,
                 evaluator: str,
                 temperature: float,
                 system_prompt: str,
                 no_medical_disclaimer_penalty: bool,
                 no_referral_penalty: bool,
                 test: bool,
                 eval_mode: str,
                 generation_batch_size: int,
                 pipeline_delay: int = 10,
                 max_concurrent_models: int = None,
                 openai_model: str = None,
                 anthropic_model: str = None,
                 gemini_model: str = None):
        """
        Initialize the evaluation manager.

        Args:
            models: List of model names/paths to evaluate
            scenarios: Path to the scenarios file
            output_dir: Directory to save results
            evaluator: Which LLM to use for evaluation
            temperature: Temperature for model generation
            system_prompt: Type of system prompt to use
            no_medical_disclaimer_penalty: Whether to disable medical disclaimer penalty
            no_referral_penalty: Whether to disable referral penalty
            test: Whether to run in test mode
            eval_mode: Evaluation mode for Gemini evaluator ('batch' or 'individual')
            generation_batch_size: Batch size for local model generation
            pipeline_delay: Delay in seconds before starting the next model
            max_concurrent_models: Maximum number of models to evaluate concurrently
        """
        self.models = models
        self.scenarios = scenarios
        self.output_dir = output_dir
        self.evaluator = evaluator
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.no_medical_disclaimer_penalty = no_medical_disclaimer_penalty
        self.no_referral_penalty = no_referral_penalty
        self.test = test
        self.eval_mode = eval_mode
        self.generation_batch_size = generation_batch_size
        self.pipeline_delay = pipeline_delay
        self.openai_model = openai_model
        self.anthropic_model = anthropic_model
        self.gemini_model = gemini_model

        # Detect available GPUs
        self.num_gpus = get_cuda_device_count()
        print(f"Detected {self.num_gpus} CUDA devices")

        # Set maximum concurrent models
        if max_concurrent_models is None:
            # Use number of GPUs if available, otherwise default to 1
            self.max_concurrent_models = self.num_gpus if self.num_gpus > 0 else 1
        else:
            self.max_concurrent_models = max_concurrent_models

        # Initialize state
        self.active_processes = {}  # Maps GPU ID to (process, model) tuple
        self.results = []  # List of (output_file, original_model) tuples
        self.next_model_index = 0
        self.next_gpu_id = 0

    def start_next_model(self):
        """Start the next model on the next available GPU."""
        if self.next_model_index >= len(self.models):
            return False  # No more models to start

        # Find an available GPU
        gpu_id = self.next_gpu_id if self.num_gpus > 0 else None
        if self.num_gpus > 0:
            self.next_gpu_id = (self.next_gpu_id + 1) % self.num_gpus

        # Get the next model
        model = self.models[self.next_model_index]
        self.next_model_index += 1

        # Start the evaluation process
        print(f"\n\n{'='*80}")
        print(f"Starting evaluation of model: {model} on GPU {gpu_id}")
        print(f"{'='*80}\n")

        # Create a subprocess for the evaluation
        cmd = [
            "python", "examples/evaluate_baseline.py",
            "--model", model,
            "--scenarios", self.scenarios,
            "--output_base", os.path.join(self.output_dir, f"temp_{os.path.basename(model).replace('/', '_')}"),
            "--evaluator", self.evaluator,
            "--temperature", str(self.temperature),
            "--system_prompt", self.system_prompt,
            "--eval-mode", self.eval_mode,
            "--generation_batch_size", str(self.generation_batch_size)
        ]

        # Add model selection parameters
        if self.openai_model:
            cmd.extend(["--openai-model", self.openai_model])
        if self.anthropic_model:
            cmd.extend(["--anthropic-model", self.anthropic_model])
        if self.gemini_model:
            cmd.extend(["--gemini-model", self.gemini_model])

        if self.no_medical_disclaimer_penalty:
            cmd.append("--no_medical_disclaimer_penalty")

        if self.no_referral_penalty:
            cmd.append("--no_referral_penalty")

        if self.test:
            cmd.append("--test")

        # Set environment variables for GPU selection
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # Start the process
        process = subprocess.Popen(cmd, env=env)

        # Store the process and model
        self.active_processes[gpu_id] = (process, model)

        return True

    def check_completed_processes(self):
        """Check for completed processes and collect results."""
        completed_gpus = []

        for gpu_id, (process, model) in self.active_processes.items():
            if process.poll() is not None:
                # Process has completed
                print(f"\n{'='*80}")
                print(f"Evaluation of model {model} on GPU {gpu_id} completed with return code {process.returncode}")
                print(f"{'='*80}\n")

                if process.returncode == 0:
                    # Successful completion, collect results
                    model_name = os.path.basename(model).replace('/', '_')
                    scenario_filename = get_scenario_filename(self.scenarios)

                    # Find the generated output file
                    output_files = list(pathlib.Path(self.output_dir).glob(f"temp_{model_name}_*.json"))

                    if output_files:
                        # Rename the file to our standardized name
                        output_file = os.path.join(self.output_dir, f"eval_{model_name}_{scenario_filename}.json")
                        os.rename(str(output_files[0]), output_file)
                        self.results.append((output_file, model))

                # Mark this GPU as available
                completed_gpus.append(gpu_id)

        # Remove completed processes
        for gpu_id in completed_gpus:
            del self.active_processes[gpu_id]

    def run_pipeline(self):
        """
        Run the evaluation pipeline.

        Returns:
            List of (output_file, original_model) tuples for successful evaluations.
        """
        # Start initial batch of models
        for _ in range(min(self.max_concurrent_models, len(self.models))):
            self.start_next_model()
            time.sleep(self.pipeline_delay)  # Delay between starts

        # Continue until all models are evaluated
        while self.active_processes or self.next_model_index < len(self.models):
            # Check for completed processes
            self.check_completed_processes()

            # Start new models if GPUs are available
            if len(self.active_processes) < self.max_concurrent_models and self.next_model_index < len(self.models):
                self.start_next_model()
                time.sleep(self.pipeline_delay)  # Delay between starts

            # Sleep to avoid busy waiting
            time.sleep(1)

        return self.results


def load_evaluation_results(file_path: str) -> Dict[str, Any]:
    """Load evaluation results from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading results from {file_path}: {e}")
        return {}


def extract_metrics(results: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """Extract relevant metrics from evaluation results."""
    metrics = {}

    # Add model name
    metrics["model"] = model_name

    # Extract summary metrics
    if "summary_metrics" in results:
        summary = results["summary_metrics"]
        metrics.update(summary)
    elif "evaluation_config" in results and "summary_metrics" in results:
        summary = results["summary_metrics"]
        metrics.update(summary)
    else:
        # Handle older format
        metrics["average_ahimsa_score"] = results.get("average_ahimsa_score", "N/A")
        metrics["harmful_rate"] = results.get("harmful_rate", "N/A")
        metrics["average_dharma_score"] = results.get("average_dharma_score", "N/A")
        metrics["average_combined_score"] = results.get("average_combined_score", "N/A")

    return metrics


def generate_summary_table(metrics_list: List[Dict[str, Any]]) -> str:
    """Generate a markdown table summarizing metrics from all models."""
    if not metrics_list:
        return "No evaluation results found."

    # Determine all unique metrics across all models
    all_metrics = set()
    for metrics in metrics_list:
        all_metrics.update(metrics.keys())

    # Remove 'model' from metrics as it will be the row header
    if 'model' in all_metrics:
        all_metrics.remove('model')

    # Sort metrics for consistent ordering
    metric_columns = sorted(all_metrics)

    # Create table header
    header = "| Model | " + " | ".join(metric_columns) + " |"
    separator = "| --- | " + " | ".join(["---"] * len(metric_columns)) + " |"

    # Create table rows
    rows = []
    for metrics in metrics_list:
        model_name = metrics.get('model', 'Unknown')
        row_values = []

        for metric in metric_columns:
            value = metrics.get(metric, "N/A")
            # Format numeric values
            if isinstance(value, (int, float)):
                if isinstance(value, int):
                    row_values.append(str(value))
                else:
                    row_values.append(f"{value:.4f}")
            else:
                row_values.append(str(value))

        row = f"| {model_name} | " + " | ".join(row_values) + " |"
        rows.append(row)

    # Combine all parts of the table
    table = "\n".join([header, separator] + rows)
    return table


def evaluate_json_input(
    input_json: str,
    output_dir: str,
    evaluator: str,
    temperature: float,
    system_prompt: str,
    no_medical_disclaimer_penalty: bool,
    no_referral_penalty: bool,
    test: bool,
    eval_mode: str,
    openai_model: str = None,
    anthropic_model: str = None,
    gemini_model: str = None
) -> tuple:
    """
    Evaluate pre-generated responses from a JSON file.

    Args:
        input_json: Path to the JSON file containing pre-generated responses
        output_dir: Directory to save results
        evaluator: Which LLM to use for evaluation
        temperature: Temperature for evaluation
        system_prompt: Type of system prompt to use
        no_medical_disclaimer_penalty: Whether to disable medical disclaimer penalty
        no_referral_penalty: Whether to disable referral penalty
        test: Whether to run in test mode
        eval_mode: Evaluation mode for Gemini evaluator
        openai_model: OpenAI model to use
        anthropic_model: Anthropic model to use
        gemini_model: Gemini model to use

    Returns:
        tuple: (output_file_path, model_name) or (None, None) on failure
    """
    try:
        # Create output filename with format: $input-$evaluatorModelName.$extension
        json_filename = os.path.basename(input_json)
        base_name = os.path.splitext(json_filename)[0]  # Remove .json extension
        extension = os.path.splitext(json_filename)[1]  # Get .json extension

        # Get evaluator model name
        evaluator_model_name = evaluator
        if evaluator == "openai" and openai_model:
            evaluator_model_name = openai_model
        elif evaluator == "anthropic" and anthropic_model:
            evaluator_model_name = anthropic_model
        elif evaluator == "gemini" and gemini_model:
            evaluator_model_name = gemini_model

        output_filename = f"{base_name}-{evaluator_model_name}{extension}"
        output_file = os.path.join(output_dir, output_filename)

        cmd = [
            "python", "examples/evaluate_baseline.py",
            "--input-json", input_json,
            "--output_base", os.path.join(output_dir, f"temp_{base_name}"),
            "--evaluator", evaluator,
            "--temperature", str(temperature),
            "--system_prompt", system_prompt
        ]

        # Only add eval-mode for Gemini evaluator
        if evaluator == "gemini":
            cmd.extend(["--eval-mode", eval_mode])

        # Add model selection parameters
        if openai_model:
            cmd.extend(["--openai-model", openai_model])
        if anthropic_model:
            cmd.extend(["--anthropic-model", anthropic_model])
        if gemini_model:
            cmd.extend(["--gemini-model", gemini_model])

        if no_medical_disclaimer_penalty:
            cmd.append("--no_medical_disclaimer_penalty")

        if no_referral_penalty:
            cmd.append("--no_referral_penalty")

        if test:
            cmd.append("--test")

        print(f"\n\n{'='*80}")
        print(f"Evaluating JSON input: {input_json}")
        print(f"{'='*80}\n")

        subprocess.run(cmd, check=True)

        # Find the generated output file (which has a timestamp in its name)
        output_files = list(pathlib.Path(output_dir).glob(f"temp_{base_name}_*.json"))

        if not output_files:
            print(f"Error: No output file found for JSON input {input_json}")
            return None, None

        # Rename the file to our standardized name
        os.rename(str(output_files[0]), output_file)
        return output_file, f"JSON:{base_name}"

    except subprocess.CalledProcessError as e:
        print(f"Error evaluating JSON input {input_json}: {e}")
        return None, None
    except Exception as e:
        print(f"Unexpected error evaluating JSON input {input_json}: {e}")
        return None, None


def main():
    """Run the multi-model evaluation script."""
    args = parse_arguments()

    # Validate mutually exclusive arguments
    if args.input_json and args.models:
        print("Error: --input-json and --models are mutually exclusive")
        sys.exit(1)

    if not args.input_json and not args.models:
        print("Error: Either --input-json or --models must be specified")
        sys.exit(1)

    # Special handling for Anthropic batch mode
    if args.evaluator == "anthropic" and args.eval_mode == "batch":
        print("\n" + "="*80)
        print("WARNING: Anthropic batch mode is currently DISABLED")
        print("="*80)
        print("Tests comparing individual and batch mode with Gemini showed large")
        print("inconsistency when multiple scenarios were grouped in 1 API call.")
        print("Anthropic batch mode is disabled until further testing.")
        print("Using individual mode instead...")
        print("="*80 + "\n")
        args.eval_mode = "individual"

    # Create output directory
    output_dir = create_output_directory()
    print(f"Results will be saved to: {output_dir}")

    # Handle JSON input mode
    if args.input_json:
        print(f"JSON input mode: evaluating pre-generated responses from {args.input_json}")

        # For JSON input, we evaluate a single set of responses, so no parallelization needed
        result = evaluate_json_input(
            input_json=args.input_json,
            output_dir=output_dir,
            evaluator=args.evaluator,
            temperature=args.temperature,
            system_prompt=args.system_prompt,
            no_medical_disclaimer_penalty=args.no_medical_disclaimer_penalty,
            no_referral_penalty=args.no_referral_penalty,
            test=args.test,
            eval_mode=args.eval_mode,
            openai_model=getattr(args, 'openai_model', None),
            anthropic_model=getattr(args, 'anthropic_model', None),
            gemini_model=getattr(args, 'gemini_model', None)
        )

        if result[0]:  # If output_file is not None
            model_results = [result]
        else:
            model_results = []
    else:
        # Check for available CUDA devices
        num_gpus = get_cuda_device_count()
        print(f"Detected {num_gpus} CUDA devices")

        # Determine evaluation strategy
        use_parallel = num_gpus > 1 and len(args.models) > 1 and not args.no_parallel
        use_pipeline = use_parallel and not args.no_pipeline and len(args.models) > 1

        if use_pipeline:
                # Use pipelined evaluation (start next model as soon as a GPU is available)
                print(f"Using pipelined evaluation with up to {min(num_gpus, len(args.models))} concurrent models")
                print(f"Pipeline delay: {args.pipeline_delay} seconds")

                # Create and run the evaluation manager
                manager = ModelEvaluationManager(
                    models=args.models,
                    scenarios=args.scenarios,
                    output_dir=output_dir,
                    evaluator=args.evaluator,
                    temperature=args.temperature,
                    system_prompt=args.system_prompt,
                    no_medical_disclaimer_penalty=args.no_medical_disclaimer_penalty,
                    no_referral_penalty=args.no_referral_penalty,
                    test=args.test,
                    eval_mode=args.eval_mode,
                    generation_batch_size=args.generation_batch_size,
                    pipeline_delay=args.pipeline_delay,
                    max_concurrent_models=args.max_concurrent_models,
                    openai_model=getattr(args, 'openai_model', None),
                    anthropic_model=getattr(args, 'anthropic_model', None),
                    gemini_model=getattr(args, 'gemini_model', None)
                )

                # Run the pipeline and get results
                model_results = manager.run_pipeline()

        elif use_parallel:
            # Use parallel evaluation (start all models at once)
            print(f"Using parallel evaluation across {min(num_gpus, len(args.models))} GPUs")

            # Create a pool with one process per GPU (up to the number of models)
            num_processes = min(num_gpus, len(args.models))

            # Prepare arguments for each model-GPU pair
            eval_args = []
            for i, model in enumerate(args.models):
                gpu_id = i % num_gpus  # Distribute models across available GPUs
                eval_args.append((
                    model,
                    args.scenarios,
                    output_dir,
                    args.evaluator,
                    args.temperature,
                    args.system_prompt,
                    args.no_medical_disclaimer_penalty,
                    args.no_referral_penalty,
                    args.test,
                    args.eval_mode,
                    args.generation_batch_size,
                    gpu_id,
                    getattr(args, 'openai_model', None),
                    getattr(args, 'anthropic_model', None),
                    getattr(args, 'gemini_model', None)
                ))

            # Run evaluations in parallel
            with multiprocessing.Pool(processes=num_processes) as pool:
                model_results = pool.map(evaluate_model_on_gpu, eval_args)

            # Filter out None results
            model_results = [result for result in model_results if result[0] is not None]

        else:
            # Sequential evaluation
            if num_gpus == 0:
                print("No CUDA devices detected, using CPU for evaluation")
            elif args.no_parallel:
                print("Parallel evaluation disabled, using sequential evaluation")
            else:
                print("Using sequential evaluation (single GPU or single model)")

            # Evaluate each model sequentially
            model_results = []  # List of (output_file, original_model_name) tuples
            for model in args.models:
                result = evaluate_model(
                    model=model,
                    scenarios=args.scenarios,
                    output_dir=output_dir,
                    evaluator=args.evaluator,
                    temperature=args.temperature,
                    system_prompt=args.system_prompt,
                    no_medical_disclaimer_penalty=args.no_medical_disclaimer_penalty,
                    no_referral_penalty=args.no_referral_penalty,
                    test=args.test,
                    eval_mode=args.eval_mode,
                    generation_batch_size=args.generation_batch_size,
                    openai_model=getattr(args, 'openai_model', None),
                    anthropic_model=getattr(args, 'anthropic_model', None),
                    gemini_model=getattr(args, 'gemini_model', None)
                )

                if result[0]:  # If output_file is not None
                    model_results.append(result)

    # Generate summary table
    metrics_list = []
    for output_file, original_model in model_results:
        results = load_evaluation_results(output_file)
        if results:
            # Use the original model name instead of extracting from the file path
            metrics = extract_metrics(results, original_model)
            metrics_list.append(metrics)

    summary_table = generate_summary_table(metrics_list)

    # Save summary table to file
    summary_file = os.path.join(output_dir, "eval_summary.md")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("# Model Evaluation Summary\n\n")
        f.write(f"Evaluation completed on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Scenarios file: {args.scenarios}\n\n")
        f.write(summary_table)

    print(f"\n\n{'='*80}")
    print(f"Evaluation complete! Summary saved to: {summary_file}")
    print(f"{'='*80}\n")
    print("Summary Table:")
    print(summary_table)


if __name__ == "__main__":
    main()
