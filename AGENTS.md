# SQLBoost Agent Guidelines

## Commands
- **Install**: `pip install -r requirements.txt`
- **Train ML agent**: `python scripts/train_ml_agent.py` (trains on collected execution data)
- **Use model**: `python scripts/use_model.py`
- **Lint**: `python -m flake8 src/ scripts/`
- **Format**: `python -m black src/ scripts/`
- **Type check**: `python -m mypy src/ scripts/` (if mypy installed)
- **Test single script**: Run individual scripts manually (no formal test framework)
  - Query generation: `python scripts/generate_queries.py` (connects to DB and generates validated queries)
  - Data insertion: `python scripts/insert_data.py`
  - ML training: `python scripts/train_ml_agent.py` (trains on collected execution data)
  - Model inference: `python scripts/use_model.py`

## Code Style
- **Imports**: Standard library → third-party → local imports, one per line
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Types**: Use type hints from `typing` module (Dict, List, Optional, Tuple, Any)
- **Docstrings**: Google-style for classes/methods with Args/Returns/Raises
- **Error handling**: Try/except with specific exceptions, use logging
- **Formatting**: 4-space indentation, single quotes for strings, f-strings preferred
- **Logging**: Use `logging.getLogger(__name__)` with appropriate levels
- **Context managers**: Use `@contextmanager` for resource management
- **Emojis**: Avoid in code unless explicitly requested
- **Security**: Never log/commit secrets, keys, or sensitive database credentials
- **Database**: Use SQLAlchemy for ORM, psycopg2 for raw connections