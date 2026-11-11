import numpy as np
import scipy.stats as ss
import nibabel as nib
from brainspace.gradient import GradientMaps
from brainspace.gradient.alignment import procrustes

t='wave3'

mmp_fsLR = np.loadtxt('../../src/fs_LR.64k.mmp_360.txt')
max_mode = 100
sub_list = np.loadtxt(f'../../results/dallas_more/individual/sub_list_{t}.txt', dtype='str')
n_sub = len(sub_list)
print(sub_list)

hcp_gd = np.loadtxt('../../results/hcp/gd_eigen_hcp_group.txt')
hcp_sc = np.loadtxt('../../results/hcp/sc_eigen_hcp_group.txt')
hcp_fc = np.loadtxt('../../results/hcp/fc_eigen_hcp_group.txt')
embedding = 'dm'

gd = [np.loadtxt(f'../../results/dallas_more/individual/{sub}_{t}_gd_mmp360.txt') for sub in sub_list]
gd_eigen = np.zeros((n_sub,360,100))
gd_eigen_align = np.zeros((n_sub,360,100))

for i in range(n_sub):
    gm_l = GradientMaps(kernel='normalized_angle', approach=embedding, n_components=100, random_state=0, alignment='procrustes')
    gm_l.fit(gd[i][:180,:180],reference=hcp_gd[:180], sparsity=0.9)
    gm_r = GradientMaps(kernel='normalized_angle', approach=embedding, n_components=100, random_state=0, alignment='procrustes')
    gm_r.fit(gd[i][180:,180:],reference=hcp_gd[180:], sparsity=0.9)

    gd_eigen[i] = np.concatenate((gm_l.gradients_, gm_r.gradients_))
    for j in range(100):
        if  ss.pearsonr(gd_eigen[i][:180,j],gd_eigen[i][180:,j])[0] < 0:
            gd_eigen[i][:180,j]= -gd_eigen[i][:180,j]
    gd_eigen_align[i] = np.concatenate((gm_l.aligned_, gm_r.aligned_))

    np.savetxt(f'../../results/dallas_more/eigen/gd_eigen_{sub_list[i]}_{t}_mmp_raw.txt', gd_eigen[i])
    np.savetxt(f'../../results/dallas_more/eigen/gd_eigen_{sub_list[i]}_{t}_mmp_hcp.txt', gd_eigen_align[i])
    print('gd...', sub_list[i])

sc = [np.loadtxt(f'../../results/dallas_more/individual/{sub}_{t}_sc_mmp360.txt') for sub in sub_list]
for i in range(n_sub):
    sc[i] = (sc[i] + sc[i].T)/2  # Make symmetric 

sc_eigen = np.zeros((n_sub,360,100))
sc_eigen_align = np.zeros((n_sub,360,100))

for i in range(n_sub):
    gm_l = GradientMaps(kernel='normalized_angle', approach=embedding, n_components=100, random_state=0, alignment='procrustes')
    gm_l.fit(sc[i][:180],reference=hcp_sc[:180], sparsity=0.9)
    gm_r = GradientMaps(kernel='normalized_angle', approach=embedding, n_components=100, random_state=0, alignment='procrustes')
    gm_r.fit(sc[i][180:],reference=hcp_sc[180:], sparsity=0.9)

    sc_eigen[i] = np.concatenate((gm_l.gradients_, gm_r.gradients_))
    for j in range(100):
        if  ss.pearsonr(sc_eigen[i][:180,j],sc_eigen[i][180:,j])[0] < 0:
            sc_eigen[i][:180,j]= -sc_eigen[i][:180,j]

    sc_eigen_align[i] = np.concatenate((gm_l.aligned_, gm_r.aligned_))
    np.savetxt(f'../../results/dallas_more/eigen/sc_eigen_{sub_list[i]}_{t}_mmp_raw.txt',sc_eigen[i])
    np.savetxt(f'../../results/dallas_more/eigen/sc_eigen_{sub_list[i]}_{t}_mmp_hcp.txt',sc_eigen_align[i])
    print('sc...', sub_list[i])

fc = [np.loadtxt(f'../../results/dallas_more/individual/{sub}_{t}_FCpearson_mmp360.txt') for sub in sub_list]
fc_eigen = np.zeros((n_sub,360,100))
fc_eigen_align = np.zeros((n_sub,360,100))

for i in range(n_sub):
    corr = fc[i].copy()
    corr[corr>0.99999] = 1
    tmp = np.arctanh(corr)
    tmp[tmp==np.inf]=0
    gm = GradientMaps(kernel='normalized_angle', approach=embedding, n_components=100, random_state=0, alignment='procrustes')
    gm.fit(tmp, reference=hcp_fc, sparsity=0.9)
    fc_eigen[i] = gm.gradients_
    fc_eigen_align[i] = gm.aligned_
    np.savetxt(f'../../results/dallas_more/eigen/fc_eigen_{sub_list[i]}_{t}_mmp_raw.txt',gm.gradients_)
    np.savetxt(f'../../results/dallas_more/eigen/fc_eigen_{sub_list[i]}_{t}_mmp_hcp.txt',gm.aligned_)
    print('fc grad...', sub_list[i])