import os
import numpy as np
from procar import procar
from unfold import unfold
from unfold import make_kpath, removeDuplicateKpoints, find_K_from_k,\
                    save2VaspKPOINTS
 # The tranformation matrix between supercell and primitive cell.

#M = np.array([[3,3,0],[-1,1,0],[0,0,1]])


# high-symmetry point of a Hexagonal BZ in fractional coordinate
#kpts = [[0.0, 0.0, 0.125],            # G
#        [0.5, 0.0, 0.125],            # M
#        [1./3, 1./3, 0.125],          # K
#        [0.0, 0.0, 0.125]]            # G
kpts = [[0.0, 0.0, 0.125],            # G
        [1./3, 1./3, 0.125]]          # K
        

# basis vector of the primitive cell
#cell = [[ 3.1850, 0.0000000000000000,  0.0],
#        [-1.5925, 2.7582909110534373,  0.0],
#        [ 0.0000, 0.0000000000000000, 35.0]]

#cell = np.array([[ 2.87945,   1.66245123, -0.        ],
#                [-2.87945,   1.66245123, -0.        ],
#                [-0.        , -0.        , 12.1539    ]])
As = np.array([
   [  5.7588999999999997,    0.0000000000000000,    0.000000000000000],
   [ -2.8794499999999990,    4.9873536978542035,    0.000000000000000],
   [  0.0000000000000000,    0.0000000000000000,   12.153900000000000]
])

Au = np.array([
  [ 2.8794499999999998,    1.6624512326180678,   -0.0000000000000000],
  [-2.8794499999999994,    1.6624512326180678,   -0.0000000000000000],
  [-0.0000000000000000,   -0.0000000000000000,   12.1539000000000000]
])

M = np.linalg.solve(Au.T, As.T).T
M = np.round(M).astype(int)
cell = np.linalg.inv(M) @ As
print(cell)

# create band path from the high-symmetry points, 30 points inbetween each pair
# of high-symmetry points
nseg = 60
kpath = make_kpath(kpts, nseg=nseg)
K_in_sup = []
for kk in kpath:
    kg, g = find_K_from_k(kk, M)
    K_in_sup.append(kg)
# remove the duplicate K-points
reducedK, kmap = removeDuplicateKpoints(K_in_sup, return_map=True)

# save to VASP KPOINTS
save2VaspKPOINTS(reducedK)

from unfold import EBS_scatter

if os.path.isfile('WAVECAR'):
    if os.path.isfile('awht.npy'):
        atomic_whts = np.load('awht.npy')
    else:
        p           = procar()
        # The atomic contribution of Ce, Mo, S to each KS states
        atomic_whts = [p.get_pw("1:2")[:,kmap,:], p.get_pw("2:8")[:,kmap,:], p.get_pw("8:20")[:,kmap,:]]
        np.save('awht.npy', atomic_whts)

    if os.path.isfile('sw.npy'):
        sw = np.load('sw.npy')
    else:
        WaveSuper   = unfold(M=M, wavecar='WAVECAR')
        sw = WaveSuper.spectral_weight(kpath)
        np.save('sw.npy', sw)
    print("sw.shape", sw.shape)
    #print("atomic_whts shape", atomic_whts.shape)
    #sw = sw[0] + sw[1]
    #sw = sw[None,...]
    #atomic_whts = atomic_whts[:,0,...]+atomic_whts[0,1,...]
    #atomic_whts = atomic_whts[:,None,...]
    #print("sw.shape", sw.shape)
    #print("atomic_whts = ", atomic_whts[0,:,:,:])
    #print("atomic_whts.shape =", atomic_whts.shape)
    #print("sw[:,0,0,:]] = ", sw[:,0,0,:])
    #atomic_whts[-1], atomic_whts[0] = atomic_whts[0], atomic_whts[-1]
    atomic_whts_Fe = atomic_whts[0].copy()
    atomic_whts_Nb = atomic_whts[1].copy()
    atomic_whts_S = atomic_whts[-1].copy()
    
    atomic_whts = np.asarray([atomic_whts_S, atomic_whts_Nb, atomic_whts_Fe])
    
    
    EBS_scatter(kpath, cell, sw,
                atomic_whts,
                atomic_colors=[ 'green', 'blue', 'red' ],
                nseg=nseg, eref=7.730112,
                ylim=(-4, 0.5), 
                kpath_label = ['K', 'G', "K"],
                factor=50,
                save="ebs_kgk.png",
                combine_spins=False,
                atom_label=[ "S", "Nb", "Co"],
                plot_each_atom=False)
#WaveSuper = unfold(M=M, wavecar='WAVECAR')



#sw = WaveSuper.spectral_weight(kpath)
## show the effective band structure with scatter
#EBS_scatter(kpath, cell, sw, nseg=nseg, eref=7.730112,
#        ylim=(-3.0, 0.5), kpath_label=['G', 'K'], figsize=(3.5, 6.0),
#        factor=10)

#from unfold import EBS_cmaps
#e0, sf = WaveSuper.spectral_function(nedos=4000)
## or show the effective band structure with colormap
#EBS_cmaps(kpath, cell, e0, sf, nseg=nseg, eref=7.8223381662,
#        show=False, kpath_label=['G', 'M', 'K', 'G'], 
#        ylim=(-2.5, 0.5))
