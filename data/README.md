# Healthcare Dataset for ArGen GRPO Fine-Tuning

This directory contains sample datasets for fine-tuning LLMs using GRPO with Dharmic ethical principles.

## Dataset Format

The datasets are provided in JSONL format with the following fields:

- `prompt`: The user query or instruction
- `role`: The intended role of the AI assistant (e.g., "healthcare_assistant")
- `patient_context`: Additional context about the patient or situation

## Available Datasets

### healthcare_examples.jsonl

A sample dataset containing 20 healthcare-related prompts covering various scenarios:

1. General health inquiries
2. Medication questions
3. Emergency situations
4. Preventive care
5. Potentially harmful requests
6. Mental health concerns
7. Chronic condition management

## Using Custom Datasets

You can create your own datasets following the same format. Ensure that your dataset includes a diverse range of scenarios to effectively train the model on different aspects of the Dharmic principles (Ahimsa, Satya, Dharma).

## Data Considerations

When creating datasets for healthcare applications:

1. Include a mix of straightforward and ethically challenging scenarios
2. Consider patient safety implications
3. Include scenarios where truthfulness is important
4. Include scenarios that test the model's understanding of its role and limitations

## Privacy Notice

These example prompts are fictional and do not contain real patient data. When creating your own datasets, ensure you comply with relevant privacy regulations (e.g., HIPAA) and do not include personally identifiable information.
