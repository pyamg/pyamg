import pdb
import numpy as np
import scipy.sparse as sp
from pyamg.relaxation.relaxation import cf_gauss_seidel, \
    fc_gauss_seidel, gauss_seidel


nf = 200
nc = 300
nn = nf+nc
fpts = np.arange(0,nf)
cpts = np.arange(nf,nf+nc)
df1 = np.random.rand(nf) + 0.5
df2 = np.random.rand(nf-1)
df3 = np.random.rand(nf-2)
df4 = np.random.rand(nf-3)
Aff = sp.diags([df1,df2,df3,df4], [0,-1,-2,-3], format='csr')
dc1 = np.random.rand(nc) + 0.5
dc2 = np.random.rand(nc-1)
dc3 = np.random.rand(nc-2)
dc4 = np.random.rand(nc-3)
Acc = sp.diags([dc1,dc2,dc3,dc4], [0,-1,-2,-3], format='csr')
Afc = sp.random(nf,nc, format='csr')
Acf = sp.random(nc,nf, format='csr')

# Create matrices that should yield exact solve with one FC or CF relax.
FC = sp.vstack( [sp.hstack([Aff,sp.csr_matrix(([0],([0],[0])),shape=[nf,nc])]), sp.hstack([Acf,Acc])], format='csr' )
CF = sp.vstack( [sp.hstack([Aff,Afc]), sp.hstack([sp.csr_matrix(([0],([0],[0])),shape=[nc,nf]),Acc])], format='csr' )

# Test FC and CF relax
bb = np.ones((nn,))
xfc = np.zeros((nn,))
xcf = np.zeros((nn,))
fc_gauss_seidel(FC, xfc, bb, cpts, fpts, iterations=1, f_iterations=1, c_iterations=1, sweep='forward')
cf_gauss_seidel(CF, xcf, bb, cpts, fpts, iterations=1, f_iterations=1, c_iterations=1, sweep='forward')

rfc = bb - FC*xfc
rcf = bb - CF*xcf

# Test FC gauss-seidel using GS on subblocks
xxfc = np.zeros((nn,))
xxfc_f = xxfc[fpts]
xxfc_c = xxfc[cpts]
gauss_seidel(Aff, xxfc_f, bb[fpts], iterations=1, sweep='forward')
bc = bb[cpts] - Acf*xxfc_f
gauss_seidel(Acc, xxfc_c, bc, iterations=1, sweep='forward')
xxfc[fpts] = xxfc_f
xxfc[cpts] = xxfc_c
rrfc = bb - FC*xxfc

# Test CF gauss-seidel using GS on subblocks
xxcf = np.zeros((nn,))
xxcf_f = xxcf[fpts]
xxcf_c = xxcf[cpts]
gauss_seidel(Acc, xxcf_c, bb[cpts], iterations=1, sweep='forward')
bf = bb[fpts] - Afc*xxcf_c
gauss_seidel(Aff, xxcf_f, bf, iterations=1, sweep='forward')
xxcf[fpts] = xxcf_f
xxcf[cpts] = xxcf_c
rrcf = bb - CF*xxcf

print("Max abs res: rfc, rcf, rrfc, rrcf")
print(str(np.max(np.abs(rfc)))+", "+str(np.max(np.abs(rcf)))+\
    ", "+str(np.max(np.abs(rrfc)))+", "+str(np.max(np.abs(rrcf))) )


pdb.set_trace()
