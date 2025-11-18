# Contributing to Melanoma Detection Project

Thank you for your interest in contributing! This guide will help you get started.

## Development Setup

### 1. Clone Repository

```bash
git clone https://github.com/setab/user-facing-AI-powered-melanoma-detection-system-with-explainability.git
cd melanoma-detection
```

### 2. Create Virtual Environment

```bash
# Using conda (recommended)
conda create -n melanoma-dev python=3.10
conda activate melanoma-dev

# Or using venv
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows
```

### 3. Install in Development Mode

```bash
# Install with all dependencies
pip install -e ".[train,serve,dev]"

# Or install specific requirements
pip install -r requirements/requirements-train.txt
```

### 4. Set Up Pre-commit Hooks (Optional)

```bash
pip install pre-commit
pre-commit install
```

## Project Structure

```
src/
├── inference/     # Inference and XAI
├── training/      # Training and evaluation
├── config.py      # Configuration management
└── serve_gradio.py # Web UI

notebooks/         # Jupyter notebooks for experiments
tests/            # Unit tests
docs/             # Documentation
scripts/          # Executable scripts
```

## Coding Guidelines

### Python Style

- Follow PEP 8 style guide
- Use type hints where appropriate
- Maximum line length: 100 characters
- Use docstrings for all public functions/classes

### Example Function

```python
def calibrate_logits(
    logits: torch.Tensor,
    temperature: float
) -> torch.Tensor:
    """
    Apply temperature scaling to logits.
    
    Args:
        logits: Raw model outputs (B, C)
        temperature: Temperature parameter T > 0
        
    Returns:
        Calibrated logits divided by temperature
    """
    return logits / temperature
```

### Naming Conventions

- Classes: `PascalCase` (e.g., `ModelComparison`)
- Functions: `snake_case` (e.g., `compute_metrics`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_THRESHOLD`)
- Private methods: `_leading_underscore` (e.g., `_load_checkpoint`)

## Testing

### Run All Tests

```bash
# Using pytest
pytest tests/

# With coverage
pytest --cov=src tests/
```

### Add New Tests

Create test files in `tests/` directory:

```python
# tests/test_new_feature.py
import unittest
from src.inference.xai import some_function

class TestNewFeature(unittest.TestCase):
    def test_something(self):
        result = some_function(input_data)
        self.assertEqual(result, expected_output)
```

## Documentation

### Update Documentation

When adding features, update relevant docs in `docs/`:

- `docs/ARCHITECTURE.md` - System design changes
- `docs/MODEL_COMPARISON_GUIDE.md` - New models or metrics
- `README.md` - User-facing features

### Docstring Format

Use Google-style docstrings:

```python
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 10
) -> Dict[str, float]:
    """
    Train a model on training data.
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        epochs: Number of training epochs (default: 10)
        
    Returns:
        Dictionary with training metrics (loss, accuracy)
        
    Raises:
        ValueError: If epochs <= 0
        
    Example:
        >>> model = resnet50(num_classes=7)
        >>> metrics = train_model(model, loader, epochs=20)
        >>> print(metrics['accuracy'])
        0.85
    """
```

## Commit Messages

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style (formatting, missing semicolons, etc.)
- `refactor`: Code change that neither fixes bug nor adds feature
- `test`: Adding or updating tests
- `chore`: Maintenance tasks (dependencies, build, etc.)

### Examples

```bash
# Good commit messages
feat(inference): add batch processing for CLI
fix(calibration): correct temperature scaling formula
docs(readme): update installation instructions
refactor(training): extract validation logic to separate function

# Bad commit messages
fix bug
update code
changes
```

## Pull Request Process

### 1. Create Feature Branch

```bash
git checkout -b feat/your-feature-name
# or
git checkout -b fix/bug-description
```

### 2. Make Changes

- Write clean, documented code
- Add tests for new features
- Update documentation
- Follow coding guidelines

### 3. Test Your Changes

```bash
# Run tests
pytest tests/

# Check code style
flake8 src/ tests/

# Format code
black src/ tests/
```

### 4. Commit Changes

```bash
git add .
git commit -m "feat(scope): descriptive message"
```

### 5. Push and Create PR

```bash
git push origin feat/your-feature-name
```

Then create Pull Request on GitHub with:
- Clear title describing the change
- Description of what changed and why
- Reference any related issues (#123)
- Screenshots for UI changes

### 6. Code Review

- Respond to feedback promptly
- Make requested changes
- Update PR description if scope changes

## Adding New Models

To add a new architecture to model comparison:

1. **Add to architectures list** in `src/training/compare_models.py`:
```python
SUPPORTED_ARCHITECTURES = [
    'resnet50',
    'efficientnet_b3',
    'densenet121',
    'vit_b_16',
    'your_new_model',  # Add here
]
```

2. **Implement model loading**:
```python
elif arch == 'your_new_model':
    model = torchvision.models.your_model(weights='DEFAULT')
    # Modify classifier for 7 classes
    model.fc = nn.Linear(model.fc.in_features, 7)
```

3. **Add documentation** to `docs/MODEL_COMPARISON_GUIDE.md`

4. **Test the model**:
```bash
python src/training/compare_models.py \
  --architectures your_new_model \
  --epochs 5 \
  --test-mode
```

## Adding New Features

### Example: Adding a New Metric

1. **Implement the metric** in appropriate module:
```python
# src/training/metrics.py
def compute_specificity(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """Compute specificity (true negative rate)."""
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0
```

2. **Add tests**:
```python
# tests/test_metrics.py
def test_specificity():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    spec = compute_specificity(y_true, y_pred)
    assert spec == 0.5
```

3. **Integrate into training/evaluation**

4. **Update documentation**

## Questions or Issues?

- Check existing issues: https://github.com/setab/user-facing-AI-powered-melanoma-detection-system-with-explainability/issues
- Create new issue with:
  - Clear title
  - Steps to reproduce (for bugs)
  - Expected vs actual behavior
  - Environment details (OS, Python version, GPU)

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the code, not the person
- Help others learn and grow

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing! 🎉
