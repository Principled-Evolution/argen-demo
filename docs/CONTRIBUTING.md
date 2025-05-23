# Contributing to ArGen GRPO Fine-Tuning

Thank you for your interest in contributing to the ArGen GRPO Fine-Tuning project! This document provides guidelines and instructions for contributing.

## Code of Conduct

Please be respectful and considerate of others when contributing to this project. We aim to foster an inclusive and welcoming community.

## How to Contribute

### Reporting Issues

If you find a bug or have a suggestion for improvement:

1. Check if the issue already exists in the [GitHub Issues](https://github.com/Principled-Evolution/argen-demo/issues)
2. If not, create a new issue with a clear description, including:
   - Steps to reproduce (for bugs)
   - Expected behavior
   - Actual behavior
   - Screenshots or code snippets if applicable

### Submitting Changes

1. Fork the repository
2. Create a new branch for your changes
3. Make your changes
4. Run tests to ensure your changes don't break existing functionality
5. Submit a pull request

### Pull Request Process

1. Ensure your code follows the project's style guidelines
2. Update documentation if necessary
3. Include tests for new functionality
4. Link any relevant issues in your pull request description
5. Wait for review and address any feedback

## Development Guidelines

### Code Style

- Follow PEP 8 style guidelines for Python code
- Use meaningful variable and function names
- Include docstrings for all functions and classes

### Testing

- Write unit tests for new functionality
- Ensure all tests pass before submitting a pull request
- Use pytest for running tests

### Documentation

- Update documentation for any changes to the API or functionality
- Include examples for new features
- Keep the README.md up to date

## Reward Function Development

When developing or modifying reward functions:

1. Ensure the function follows the Predibase signature: `reward_fn(prompt, completion, example) -> float`
2. Document the ethical principle being implemented
3. Include clear comments explaining the reward calculation
4. Consider edge cases and handle them appropriately
5. Add unit tests to verify the function's behavior

## License

By contributing to this project, you agree that your contributions will be licensed under the project's MIT License.
