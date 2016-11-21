"""Examples stored in files"""


import os
from glob import glob

from scipy.io import loadmat

__all__ = ['load_example']

base_dir = os.path.split(__file__)[0]
example_dir = os.path.join(base_dir, 'example_data')
example_files = glob(os.path.join(example_dir, '*.mat'))
example_names = [os.path.split(name)[1][:-4] for name in example_files]

example_names.sort()


def load_example(name):
    """Load an example problem by name

    Parameters
    ----------
    name : string (e.g. 'airfoil')
        Name of the example to load

    Notes
    -----
    Each example is stored in a dictionary with the following keys:
        - 'A'        : sparse matrix
        - 'B'        : near-nullspace candidates
        - 'vertices' : dense array of nodal coordinates
        - 'elements' : dense array of element indices

    Current example names are:%s

    Examples
    --------
    >>> from pyamg.gallery import load_example
    >>> ex = load_example('knot')

    """

    if name not in example_names:
        raise ValueError('no example with name (%s)' % name)
    else:
        return loadmat(os.path.join(example_dir, name + '.mat'),
                       struct_as_record=True)


# insert the example names into the docstring
load_example.__doc__ %= ('\n' + ' ' * 8).join([''] + example_names)
