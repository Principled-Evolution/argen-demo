# ArGen Healthcare Agent Alignment Demo

This repository provides an open-source implementation of the ArGen framework's principles for AI alignment through Group Relative Policy Optimization (GRPO) fine-tuning. It demonstrates how to encode two Dharmic ethical principles - Ahimsa (non-maleficence) and Dharma (professional duty) - into programmable reward functions for fine-tuning Large Language Models (LLMs).

## Overview

The ArGen framework, as described in the paper "AI in the Gita's Field: The ArGen Framework for Culturally-Grounded AGI Alignment," proposes a culturally-grounded approach to AI alignment. This implementation focuses on the practical application of Dharmic principles through GRPO fine-tuning with a specific focus on healthcare scenarios.

The project includes:

- Implementation of two reward functions:
  - **Ahimsa (Non-harm)**: Ensures the agent doesn't provide harmful medical advice
  - **Dharma (Professional Duty)**: Ensures the agent stays within its healthcare domain
- Comprehensive datasets with challenging healthcare scenarios and domain violation scenarios
- Local GRPO fine-tuning implementation
- Policy enforcement using OPA with Rego policies
- Comprehensive evaluation framework for comparing baseline and fine-tuned models
- Integration with GOPAL policy structure through a submodule

## Prerequisites

- Python 3.8+
- [Open Policy Agent (OPA)](https://www.openpolicyagent.org/docs/latest/#running-opa) for policy enforcement
- PyTorch and Transformers library for local model training

## Installation

```bash
# Clone the repository
git clone https://github.com/Principled-Evolution/argen-demo.git
cd argen-demo

# Initialize the GOPAL submodule
git submodule update --init --recursive

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Prepare the Datasets

```bash
python examples/prepare_combined_datasets.py
```

This will create several datasets:
- `data/healthcare_scenarios.jsonl`: Challenging healthcare scenarios (Ahimsa principle)
- `data/domain_scenarios.jsonl`: Domain violation scenarios (Dharma principle)
- `data/combined_scenarios.jsonl`: Combined scenarios for comprehensive evaluation
- `data/combined_evaluation.jsonl`: Scenarios formatted for evaluation
- `data/combined_training.jsonl`: Scenarios formatted for training

### 2. Evaluate the Baseline Model

```bash
python examples/evaluate_baseline_comprehensive.py --model llama-3-2-1b-instruct
```

This will evaluate the baseline model on both Ahimsa and Dharma principles and save the results to `data/baseline_comprehensive_results.json`.

### 3. Run GRPO Fine-tuning

```bash
python examples/train_grpo.py --model llama-3-2-1b-instruct --dataset combined_training
```

This will run local GRPO fine-tuning using both Ahimsa and Dharma reward functions.

### 4. Evaluate the Fine-tuned Model

```bash
python examples/evaluate_baseline.py --model path/to/your/fine-tuned-model
```

This will evaluate the fine-tuned model on both Ahimsa and Dharma principles and compare the results with the baseline.

## Project Structure

```
argen-demo/
├── data/                      # Datasets and evaluation results
├── examples/                  # Example scripts
├── gopal/                     # GOPAL submodule with Dharmic principles
├── argen/                     # Source code
│   ├── data/                  # Dataset utilities
│   ├── reward_functions/      # Reward functions for GRPO
│   ├── training/              # Training utilities
│   ├── evaluation/            # Model evaluation utilities
│   └── utils/                 # Utility functions
├── tests/                     # Unit tests
├── README.md                  # This file
└── requirements.txt           # Python dependencies
```

## Reward Functions

This implementation includes two reward functions for GRPO fine-tuning:

### Ahimsa Reward Function

The Ahimsa (non-maleficence) reward function evaluates whether the agent's response adheres to the principle of non-harm in healthcare contexts. It evaluates:

- Harm avoidance score: Whether the response avoids providing harmful medical advice
- Safety consideration score: Whether the response includes appropriate safety considerations
- Presence of medical disclaimers: Whether the response includes appropriate medical disclaimers
- Presence of professional referrals: Whether the response includes referrals to healthcare professionals

### Dharma Reward Function

The Dharma (professional duty) reward function evaluates whether the agent's response stays within its healthcare domain. It evaluates:

- Whether the response provides out-of-domain advice (e.g., financial, fashion, career advice)
- Whether the response appropriately refuses out-of-domain requests
- Whether the response stays focused on healthcare topics

The combination of these reward functions provides a comprehensive approach to aligning a healthcare AI agent with Dharmic principles.

## OPA Policies

This implementation includes two OPA policies for policy enforcement:

### Ahimsa Policy

The Ahimsa policy checks if the agent's response adheres to the principle of non-harm in healthcare contexts. It evaluates:

- Ahimsa score: Whether the response avoids providing harmful medical advice
- Harm avoidance score: Whether the response avoids providing harmful medical advice
- Safety consideration score: Whether the response includes appropriate safety considerations
- Presence of medical disclaimers: Whether the response includes appropriate medical disclaimers
- Presence of professional referrals: Whether the response includes referrals to healthcare professionals

### Dharma Policy

The Dharma policy checks if the agent's response stays within its healthcare domain. It evaluates:

- Whether the response provides out-of-domain advice (e.g., financial, fashion, career advice)
- Whether the response appropriately refuses out-of-domain requests
- Whether the response stays focused on healthcare topics

## GOPAL Integration

This implementation is integrated with the [GOPAL (Governance OPA Library)](https://github.com/Principled-Evolution/gopal) through a submodule. The Dharmic principles are implemented as OPA policies in the GOPAL framework, which are used for policy enforcement in the ArGen environment.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The ArGen framework for providing the conceptual foundation
- The TRL library for GRPO implementation
- The GOPAL repository for policy structure inspiration
