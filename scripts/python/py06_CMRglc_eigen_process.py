import numpy as np
import scipy.stats as ss
import nibabel as nib
from brainspace.gradient import GradientMaps
from brainspace.gradient.alignment import procrustes


hcp_gd = np.loadtxt('../../results/hcp/gd_eigen_hcp_group.txt')
hcp_sc = np.loadtxt('../../results/hcp/sc_eigen_hcp_group.txt')
hcp_fc = np.loadtxt('../../results/hcp/fc_eigen_hcp_group.txt')

sub_list = np.loadtxt('../../results/CMRglc/sub_list.txt', dtype='str')
n_sub = len(sub_list)

gd = [np.delete(np.delete(nib.load('../../results/CMRglc/gd/connectome/sub-'+sub+'_ses-open_atlas-glasser-360_GD.shape.gii').agg_data(), 
                          [0,181], axis=0), [0,181], axis=1) for sub in sub_list]

fc = [np.loadtxt('../../results/CMRglc/fc/connectome/fc_'+sub+'_fsLR_mmp.txt') for sub in sub_list]

sc = [np.delete(np.delete(nib.load('../../results/CMRglc/sc/connectome/sub-'+sub+'_ses-open_space-dwi_atlas-glasser-360_desc-iFOD2-40M-SIFT2_full-connectome.shape.gii').agg_data()[49:,49:], 
                          [180], axis=0), [180], axis=1) for sub in sub_list]
for i in range(n_sub):
    for j in range(360):
        sc[i][:,j] = sc[i][j]
    np.fill_diagonal(fc[i], 0) 

gd_eigen = np.zeros((n_sub,360,100))
gd_eigen_align = np.zeros((n_sub,360,100))

embedding = 'dm'

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

    np.savetxt('../../results/CMRglc/gd/eigen/sub-'+sub_list[i]+'_ses-open_mmp_raw.txt', gd_eigen[i])
    np.savetxt('../../results/CMRglc/gd/eigen/sub-'+sub_list[i]+'_ses-open_mmp_hcp.txt', gd_eigen_align[i])


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
    np.savetxt('../../results/CMRglc/sc/eigen/sub-'+sub_list[i]+'_ses-open_mmp_raw.txt',sc_eigen[i])
    np.savetxt('../../results/CMRglc/sc/eigen/sub-'+sub_list[i]+'_ses-open_mmp_hcp.txt',sc_eigen_align[i])

fc_eigen = np.zeros((n_sub,360,100))
fc_eigen_align = np.zeros((n_sub,360,100))

for i in range(n_sub):
    gm = GradientMaps(kernel='normalized_angle', approach=embedding, n_components=100, random_state=0, alignment='procrustes')
    gm.fit(fc[i], reference=hcp_fc, sparsity=0.9)
    fc_eigen[i] = gm.gradients_
    fc_eigen_align[i] = gm.aligned_
    np.savetxt('../../results/CMRglc/fc/eigen/sub-'+sub_list[i]+'_ses-open_mmp_raw.txt',gm.gradients_)
    np.savetxt('../../results/CMRglc/fc/eigen/sub-'+sub_list[i]+'_ses-open_mmp_hcp.txt',gm.aligned_)

