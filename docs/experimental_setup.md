# ArGen Healthcare Agent Alignment Experiment

## Overview

This experiment demonstrates how Group Relative Policy Optimization (GRPO) combined with Open Policy Agent (OPA) can be used to align a healthcare AI agent with two key Dharmic principles:

1. **Ahimsa (Non-harm)**: The agent should not provide harmful medical advice.
2. **Dharma (Professional Duty)**: The agent should stay within its healthcare domain and not provide advice on unrelated topics.

## Experimental Setup

### Base Model

We use the `llama-3-2-1b-instruct` model (1.24B parameters) as our base model. This is the smallest model that supports GRPO on Predibase, making it cost-effective for our demonstration.

### Scenarios

We use two types of scenarios:

1. **Challenging Healthcare Scenarios**: These scenarios test the agent's ability to avoid harmful medical advice (Ahimsa principle).
2. **Domain Violation Scenarios**: These scenarios test the agent's ability to stay within its healthcare domain (Dharma principle).

### Policies

We implement two OPA policies:

1. **Ahimsa Policy**: Checks if the agent's response contains harmful medical advice.
2. **Dharma Policy**: Checks if the agent's response stays within the healthcare domain.

### Reward Functions

We implement two reward functions:

1. **Ahimsa Reward Function**: Evaluates whether the agent's response adheres to the principle of non-harm in healthcare contexts.
2. **Dharma Reward Function**: Evaluates whether the agent's response stays within the healthcare domain.

#### Ahimsa Reward Function

The Ahimsa reward function evaluates responses based on:
- Harm avoidance score
- Safety consideration score
- Presence of medical disclaimers
- Presence of professional referrals

#### Dharma Reward Function

The Dharma reward function evaluates responses based on:
- Whether the response provides out-of-domain advice
- Whether the response appropriately refuses out-of-domain requests
- Whether the response stays focused on healthcare topics

### Evaluation Metrics

We evaluate both baseline and fine-tuned models using the following metrics:

1. **Ahimsa Violation Rate**: Percentage of responses containing harmful medical advice
2. **Dharma Violation Rate**: Percentage of responses providing out-of-domain advice
3. **Combined Score**: Average of Ahimsa and Dharma scores
4. **Qualitative Analysis**: Manual review of responses to ensure they remain helpful for in-domain questions

## Implementation Steps

### 1. Prepare Datasets

We prepare three types of datasets:

1. **Healthcare Scenarios**: Challenging healthcare scenarios that might elicit harmful responses.
2. **Domain Violation Scenarios**: Scenarios that ask the healthcare agent for out-of-domain advice.
3. **Combined Scenarios**: A combination of both types of scenarios for comprehensive evaluation.

### 2. Evaluate Baseline Model

We evaluate the baseline model (`llama-3-2-1b-instruct`) on the combined scenarios to establish a baseline for comparison. We use a high temperature (0.9) to increase the likelihood of generating harmful or out-of-domain responses.

### 3. Run GRPO Fine-tuning

We fine-tune the baseline model using GRPO with both Ahimsa and Dharma reward functions. The fine-tuning process optimizes the model to:

1. Avoid providing harmful medical advice
2. Stay within its healthcare domain
3. Appropriately refuse out-of-domain requests

### 4. Evaluate Fine-tuned Model

We evaluate the fine-tuned model on the same combined scenarios and compare the results with the baseline model. We expect to see:

1. Reduced Ahimsa violation rate
2. Reduced Dharma violation rate
3. Improved Ahimsa and Dharma scores
4. Maintained helpfulness for in-domain questions

## Running the Experiment

### 1. Prepare Datasets

```bash
python examples/prepare_combined_datasets.py
```

### 2. Evaluate Baseline Model

```bash
python examples/evaluate_baseline_comprehensive.py
```

### 3. Run GRPO Fine-tuning

```bash
python examples/run_comprehensive_grpo_finetuning.py
```

### 4. Evaluate Fine-tuned Model

```bash
python examples/evaluate_finetuned_comprehensive.py --model your-finetuned-model-name
```

## Expected Results

We expect the fine-tuned model to show significant improvements over the baseline model:

1. **Ahimsa Principle**: The fine-tuned model should provide safer medical advice, include appropriate disclaimers and referrals, and avoid harmful recommendations.

2. **Dharma Principle**: The fine-tuned model should stay within its healthcare domain, appropriately refuse out-of-domain requests, and avoid providing advice on unrelated topics like finance, fashion, or technology.

3. **Overall Alignment**: The fine-tuned model should demonstrate a better understanding of its role as a healthcare assistant and adhere to both Ahimsa and Dharma principles while maintaining helpfulness for in-domain questions.

## Conclusion

This experiment demonstrates the effectiveness of GRPO and OPA in aligning a healthcare AI agent with Dharmic principles. By combining Ahimsa (non-harm) and Dharma (professional duty) principles, we create a more aligned healthcare assistant that provides safe medical advice while staying within its domain of expertise.
