"""
Script for testing the installation.

Mainly used for Travis-CI.

Can be expand tof more verbose testing and coverage.

See NumPy/SciPy examples.
"""
import sys

sys.path.pop(0)

import pyamg

result = pyamg.test()

print result

print "--------"
print result.wasSuccessful()

if result.wasSuccessful():
    sys.exit(0)
else:
    sys.exit(1)
