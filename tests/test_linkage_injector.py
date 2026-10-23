"""Independent-trial linkage evaluator sanity checks."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'


class LinkageInjectorTests(unittest.TestCase):
    def test_known_target_identification_and_reproducibility(self):
        previous = list(sys.path)
        try:
            sys.path.insert(0, str(SCRIPTS))
            spec = importlib.util.spec_from_file_location('linkage_injector',
                                                         SCRIPTS/'linkage_injector_experiment.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        finally:
            sys.path[:] = previous
        strong = module.attack_count([1., 1., 1., 0.], 1000, np.random.default_rng(8))
        repeat = module.attack_count([1., 1., 1., 0.], 1000, np.random.default_rng(8))
        weak = module.attack_count([0., 0., 0., 1.], 1000, np.random.default_rng(9))
        self.assertEqual(strong, repeat)
        self.assertGreater(strong, 900)
        self.assertLess(weak, 100)

