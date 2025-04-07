# ArGen GRPO Fine-Tuning

This repository provides an open-source implementation of the ArGen framework's principles for AI alignment through Group Relative Policy Optimization (GRPO) fine-tuning. It demonstrates how to encode Dharmic ethical principles (Ahimsa, Satya, Dharma) into programmable reward functions for fine-tuning Large Language Models (LLMs).

## Overview

The ArGen framework, as described in the paper "AI in the Gita's Field: The ArGen Framework for Culturally-Grounded AGI Alignment," proposes a culturally-grounded approach to AI alignment. This implementation focuses on the practical application of these principles through GRPO fine-tuning using the Predibase platform.

The project includes:

- Implementation of reward functions based on Dharmic principles
- Example healthcare dataset for fine-tuning
- Integration with Predibase for GRPO fine-tuning
- Evaluation framework for comparing base and fine-tuned models
- Conceptual alignment with GOPAL policy structure

## Prerequisites

- Python 3.8+
- Predibase account with access to Reinforcement Fine-Tuning (RFT) capabilities
- Predibase SDK (`pip install predibase`)

## Installation

```bash
git clone https://github.com/Principled-Evolution/argen-demo.git
cd argen-demo
pip install -e .
```

## Usage

### 1. Prepare Your Dataset

The repository includes a sample healthcare dataset in `data/healthcare_examples.jsonl`. You can use this as a starting point or create your own dataset following the same format.

### 2. Configure and Run Fine-Tuning

See the example notebook in `examples/fine_tuning.ipynb` for a step-by-step guide on configuring and running a GRPO fine-tuning job on Predibase.

### 3. Evaluate Your Fine-Tuned Model

Use the evaluation notebook in `examples/evaluation.ipynb` to compare the base model and your fine-tuned model on alignment test cases.

## Reward Functions

The repository includes implementations of three reward functions based on Dharmic principles:

### Ahimsa (Non-maleficence)

Rewards responses that avoid potential harm to the user, particularly in a healthcare context. The reward function evaluates:

- Avoidance of harmful medical advice
- Recognition of limitations and appropriate disclaimers
- Refusal to provide dangerous information
- Consideration of patient safety

### Satya (Truthfulness)

Rewards responses that are factually accurate and avoid fabrication or misleading information. The reward function evaluates:

- Factual accuracy of medical information
- Appropriate citation of sources when available
- Transparency about uncertainty
- Avoidance of misleading or fabricated information

### Dharma (Role-appropriateness)

Rewards responses that adhere to the model's defined role and responsibilities. The reward function evaluates:

- Adherence to the role of a healthcare assistant
- Compassionate and patient-centric responses
- Appropriate professional boundaries
- Ethical decision-making aligned with healthcare principles

## GOPAL Integration

This implementation is conceptually aligned with the [GOPAL (Governance OPA Library)](https://github.com/Principled-Evolution/gopal) repository structure. The reward functions are designed to reflect similar principles to those that might be encoded in GOPAL policies, particularly in the healthcare domain.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The ArGen framework for providing the conceptual foundation
- Predibase for their GRPO implementation
- The GOPAL repository for policy structure inspiration
