"""
Script for testing the installation.

Mainly used for Travis-CI.

Can be expand tof more verbose testing and coverage.  

See NumPy/SciPy examples.
"""
import sys

import pyamg

result = pyamg.test()

if result.wasSuccessful():
    sys.exit(0)
else:
    sys.exit(1)
