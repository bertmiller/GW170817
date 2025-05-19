# Testing Guide

This repository uses **pytest** for all tests. To run the suite, install the
package dependencies and execute `pytest` from the repository root.

```bash
pip install -r gw-siren-pipeline/requirements.txt
pip install -e gw-siren-pipeline
pytest
```

The multi-event pipeline tests are located under `tests/multi_event/`. These
include unit tests for helper functions, integration tests for the orchestrator
script and a validation test that runs a simplified end-to-end workflow with
mock data.

Utilities to generate minimal mock data and configuration files for these tests
are provided in `tests/multi_event/utils.py`.
