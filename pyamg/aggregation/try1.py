import pyamg
import numpy as np
import matplotlib.pyplot as plt
from aggviz import plotaggs
np.set_printoptions(edgeitems=10,linewidth=180)

data = pyamg.gallery.load_example('unit_square')
G = data['A']
V = data['vertices'][:,:2]
E = data['elements']
G.data[:] = np.ones(len(G.data))
#seed = np.random.randint(1,32768)
#np.random.seed(787888)  # E < S
#seed=7731
seed = 42
np.random.seed(seed)
print(f'seed = {seed}')
#c=np.array([155, 147,  92,  39, 163,  25,  32,  18, 112, 103,  93,  54,  60,  68,  61, 162, 171, 108,  78], dtype=np.int32)
AggOp, c, EP, SI = pyamg.aggregation.aggregate.balanced_lloyd_aggregation(G)
