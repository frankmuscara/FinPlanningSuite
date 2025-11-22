# Test Suite

## Running Tests

Run all tests:
```bash
pytest
```

Run with coverage report:
```bash
pytest --cov=finplan_suite --cov-report=term-missing
```

Run specific test file:
```bash
pytest tests/test_store.py
pytest tests/test_portfolio.py
pytest tests/test_monte_carlo.py
```

Run tests by marker:
```bash
pytest -m unit           # Run only unit tests
pytest -m integration    # Run only integration tests
pytest -m "not slow"     # Skip slow tests
```

Run with verbose output:
```bash
pytest -v
```

## Test Structure

- `test_store.py` - Client data persistence and CRUD operations
- `test_portfolio.py` - Portfolio optimization and efficient frontier
- `test_monte_carlo.py` - Monte Carlo retirement simulation

## Adding New Tests

1. Create test files with `test_*.py` naming
2. Use descriptive test function names: `test_<what_it_tests>()`
3. Add appropriate markers: `@pytest.mark.unit`, `@pytest.mark.integration`, etc.
4. Use fixtures for setup/teardown (see `test_store.py` for examples)
