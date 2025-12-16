# Contributing to TRACE

Thank you for your interest in contributing to TRACE! This document provides guidelines for contributing to the project.

## Getting Started

### Development Setup

1. **Fork and clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/trace.git
cd trace
```

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install in development mode**

```bash
pip install -e ".[dev,docs]"
```

4. **Install pre-commit hooks** (optional but recommended)

```bash
pre-commit install
```

## Development Workflow

### Making Changes

1. **Create a new branch**

```bash
git checkout -b feature/your-feature-name
```

2. **Make your changes**
   - Write clear, documented code
   - Follow the existing code style
   - Add tests for new functionality

3. **Run tests**

```bash
pytest tests/
```

4. **Check code style**

```bash
black src/ tests/
ruff check src/
```

5. **Commit your changes**

```bash
git add .
git commit -m "Add feature: description of your changes"
```

6. **Push to your fork**

```bash
git push origin feature/your-feature-name
```

7. **Open a Pull Request**

## Code Style

### Python Style Guide

- Follow [PEP 8](https://pep8.org/)
- Use [Black](https://black.readthedocs.io/) for formatting (line length: 100)
- Use [Ruff](https://beta.ruff.rs/) for linting
- Write docstrings in [NumPy style](https://numpydoc.readthedocs.io/)

### Example Docstring

```python
def my_function(param1: int, param2: str) -> bool:
    """
    Brief description of the function.

    Longer description if needed, explaining what the function does,
    any important details, or caveats.

    Parameters
    ----------
    param1 : int
        Description of param1
    param2 : str
        Description of param2

    Returns
    -------
    bool
        Description of return value

    Examples
    --------
    >>> result = my_function(42, "test")
    >>> print(result)
    True
    """
    return True
```

## Testing

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use pytest fixtures for common setup
- Aim for high test coverage

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=trace

# Run specific test file
pytest tests/test_simulate.py

# Run specific test
pytest tests/test_simulate.py::test_simulate_conflict_data_basic
```

## Documentation

### Building Documentation

```bash
cd docs
make html
```

View the built documentation at `docs/_build/html/index.html`.

### Documentation Guidelines

- Document all public functions, classes, and modules
- Include examples in docstrings where appropriate
- Update relevant documentation when changing functionality
- Add new tutorials or examples for significant features

## Pull Request Process

1. **Ensure all tests pass**
2. **Update documentation** if needed
3. **Add entry to CHANGELOG.md** (if applicable)
4. **Describe your changes** in the PR description
5. **Link related issues** using keywords (e.g., "Fixes #123")

### PR Checklist

- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] CHANGELOG.md updated (if applicable)
- [ ] PR description is clear and complete

## Reporting Issues

### Bug Reports

When reporting bugs, please include:

- **Description**: Clear description of the bug
- **Steps to reproduce**: Minimal code example
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Environment**: Python version, OS, package versions

### Feature Requests

When requesting features, please include:

- **Use case**: Why is this feature needed?
- **Proposed solution**: How should it work?
- **Alternatives**: Other approaches you've considered

## Code of Conduct

### Our Standards

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on what is best for the community
- Show empathy towards other community members

### Unacceptable Behavior

- Harassment, discrimination, or offensive comments
- Personal attacks or trolling
- Publishing others' private information
- Other conduct that could reasonably be considered inappropriate

## Questions?

If you have questions about contributing, feel free to:

- Open an issue with the "question" label
- Open an issue: https://github.com/OJWatson/trace/issues
- Join our community discussions (if available)

## License

By contributing to TRACE, you agree that your contributions will be licensed under the MIT License.

## Acknowledgments

Thank you for contributing to TRACE! Your efforts help make conflict casualty analysis more accessible and rigorous.
