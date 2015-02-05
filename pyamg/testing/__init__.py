from numpy.testing import *


def _show_system_info(self):
    import os
    import sys
    nose = nosetester.import_nose()

    import numpy
    print "NumPy version %s" % numpy.__version__
    npdir = os.path.dirname(numpy.__file__)
    print "NumPy is installed in %s" % npdir

    import scipy
    print "SciPy version %s" % scipy.__version__
    spdir = os.path.dirname(scipy.__file__)
    print "SciPy is installed in %s" % spdir

    pyversion = sys.version.replace('\n', '')
    print "Python version %s" % pyversion

    print "nose version %d.%d.%d" % nose.__versioninfo__

    import pyamg
    print "PyAMG version %s" % pyamg.__version__
    spdir = os.path.dirname(pyamg.__file__)
    print "PyAMG is installed in %s" % spdir


nosetester.NoseTester._show_system_info = _show_system_info

del _show_system_info
