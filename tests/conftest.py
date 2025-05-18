import os
import sys
import importlib.util
import types

# Path to the internal tests directory
internal_tests_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'gw-siren-pipeline', 'tests'))
internal_conftest_path = os.path.join(internal_tests_dir, 'conftest.py')

# Provide a stub for analyze_candidates to avoid import errors in internal conftest
sys.modules.setdefault('analyze_candidates', types.ModuleType('analyze_candidates'))

# Load the package's conftest so fixtures are shared
spec = importlib.util.spec_from_file_location('internal_conftest', internal_conftest_path)
internal_conftest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(internal_conftest)
for name in dir(internal_conftest):
    if not name.startswith('_'):
        globals()[name] = getattr(internal_conftest, name)

# Make the h0_e2e_pipeline script importable as a module
scripts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'gw-siren-pipeline', 'scripts'))
pipeline_path = os.path.join(scripts_dir, 'h0_e2e_pipeline.py')
if os.path.exists(pipeline_path):
    spec = importlib.util.spec_from_file_location('h0_e2e_pipeline', pipeline_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules['h0_e2e_pipeline'] = module
