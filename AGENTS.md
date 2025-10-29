# SQLBoost Agent Guidelines

## Commands
- **Install**: `pip install -r requirements.txt`
- **Run real ML agent**: `python scripts/run_real_ml_agent.py`
- **Use model**: `python scripts/use_model.py`
- **Lint**: `python -m flake8 src/ scripts/`
- **Format**: `python -m black src/ scripts/`
- **Test**: No dedicated test framework (run scripts manually for validation)

## Code Style
- **Imports**: Standard library → third-party → local imports, one per line
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Types**: Use type hints from `typing` module (Dict, List, Optional, Tuple)
- **Docstrings**: Google-style for classes/methods with Args/Returns
- **Error handling**: Try/except with specific exceptions, use logging
- **Formatting**: 4-space indentation, single quotes for strings, f-strings preferred
- **Logging**: Use `logging.getLogger(__name__)` with appropriate levels
- **Context managers**: Use `@contextmanager` for resource management
- **Security**: Never log or commit secrets, keys, or sensitive database credentials