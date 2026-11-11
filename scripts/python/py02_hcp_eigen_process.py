import numpy as np
import scipy.stats as ss
from brainspace.gradient import GradientMaps
import func_plot as fp
import matplotlib.pyplot as plt
from brainspace.gradient.alignment import procrustes

mmp_fsLR = np.loadtxt('../../src/fs_LR.64k.mmp_360.txt')

## Load HCP data
hcp_fc = np.load('../../results/hcp/fc_unrelated.npy')
hcp_sc = np.load('../../results/hcp/sc_unrelated.npy')
hcp_gd = np.load('../../results/hcp/gd_unrelated.npy')

n_indi = hcp_gd.shape[0]

## Zero out the diagonal for FC
for i in range(n_indi):
    np.fill_diagonal(hcp_fc[i], 0)

## Compute eigenvectors for Geodesic distance matrix

embedding='dm'
gd_eigen = np.zeros((360,100))
gm_group_l = GradientMaps(kernel='normalized_angle', approach=embedding, n_components=100, random_state=0)
gm_group_l.fit(hcp_gd.mean(0)[:180,:180], sparsity=0.9)
gm_group_r = GradientMaps(kernel='normalized_angle', approach=embedding, n_components=100, random_state=0)
gm_group_r.fit(hcp_gd.mean(0)[180:,180:], sparsity=0.9)
gd_eigen = np.concatenate((gm_group_l.gradients_, gm_group_r.gradients_))
for j in range(100):
    if ss.pearsonr(gd_eigen[:180,j],gd_eigen[180:,j])[0] < 0:
        gd_eigen[:180,j] = - gd_eigen[:180,j]

np.savetxt('../../results/hcp/gd_eigen_hcp_group.txt', gd_eigen)

gd_eigen_raw = np.zeros((n_indi,360,100))
gd_eigen_align = np.zeros((n_indi,360,100))

for i in range(n_indi):
    eigenvalues_l, gd_eigen_l = np.linalg.eigh(hcp_gd[i][:180,:180])
    eigenvalues_r, gd_eigen_r = np.linalg.eigh(hcp_gd[i][180:,180:])
    gm_group_l = GradientMaps(kernel='normalized_angle', approach=embedding, n_components=100, random_state=0)
    gm_group_l.fit(hcp_gd[i][:180,:180],reference=gd_eigen[:180], sparsity=0.9)
    gm_group_r = GradientMaps(kernel='normalized_angle', approach=embedding, n_components=100, random_state=0)
    gm_group_r.fit(hcp_gd[i][180:,180:],reference=gd_eigen[180:], sparsity=0.9)
    gd_eigen_raw[i] = np.concatenate((gm_group_l.gradients_, gm_group_r.gradients_))
    for j in range(100):
      if ss.pearsonr(gd_eigen_raw[i][:180,j],gd_eigen_raw[i][180:,j])[0] < 0:
        gd_eigen_raw[i][:180,j] = - gd_eigen_raw[i][:180,j]
    gd_eigen_align[i] = procrustes(gd_eigen_raw[i], gd_eigen)
    print('gd...', i+1)
    
np.save('../../results/hcp/gd_eigen_hcp_raw.npy', gd_eigen_raw) 
np.save('../../results/hcp/gd_eigen_hcp_align.npy', gd_eigen_align)    


## plot eigenvectors on the brain for Geodesic distance matrix
vec = [0,1,2]
plot = [None] * len(vec)
for i in range(len(vec)):
  plot[i] = mmp_fsLR.copy()
  for node in range(360):
    plot[i][mmp_fsLR==node+1] = gd_eigen[:,vec[i]][node]
  plot[i][mmp_fsLR==0] = np.nan

fp.plot_surface(data = plot, nan_color=(0.5,0.5,0.5,1), color_range=(-0.1,0.1),
                size = (400, 300*len(vec)), layout_style = 'grid',
                cmap = 'summer', filename = '../../figures/gd_eigen_hcp_group.png',
                display=False)

## plot Geodesic distance matrix
fig, ax = plt.subplots(figsize=(3,3))
matrix = hcp_gd.mean(0)
matrix[matrix==0] = np.nan
ax.imshow(matrix, cmap='summer')
ax.set_xticks([])
ax.set_yticks([])
fig.tight_layout()
fig.savefig('../../figures/gd_hcp_group.png', transparent=True, dpi=300)

## Compute eigenvectors for structural connectivity matrix

sc_eigen = np.zeros((360,100))
gm_group_l = GradientMaps(kernel='normalized_angle', approach=embedding, n_components=100, random_state=0)
gm_group_l.fit(hcp_sc.mean(0)[:180], sparsity=0.9)
gm_group_r = GradientMaps(kernel='normalized_angle', approach=embedding, n_components=100, random_state=0)
gm_group_r.fit(hcp_sc.mean(0)[180:], sparsity=0.9)
sc_eigen = np.concatenate((gm_group_l.gradients_, gm_group_r.gradients_))

for j in range(100):
    if  ss.pearsonr(sc_eigen[:180,j],sc_eigen[180:,j])[0] < 0:
        sc_eigen[:180,j]= -sc_eigen[:180,j]

np.savetxt('../../results/hcp/sc_eigen_hcp_group.txt', sc_eigen)

sc_eigen_raw = np.zeros((n_indi,360,100))
sc_eigen_align = np.zeros((n_indi,360,100))

for i in range(n_indi):
    gm_l = GradientMaps(kernel='normalized_angle', approach=embedding, n_components=100, random_state=0, alignment='procrustes')
    gm_l.fit(hcp_sc[i][:180],reference=sc_eigen[:180], sparsity=0.9)
    gm_r = GradientMaps(kernel='normalized_angle', approach=embedding, n_components=100, random_state=0, alignment='procrustes')
    gm_r.fit(hcp_sc[i][180:],reference=sc_eigen[180:], sparsity=0.9)

    sc_eigen_raw[i] = np.concatenate((gm_l.gradients_, gm_r.gradients_))
    sc_eigen_align[i] = np.concatenate((gm_l.aligned_, gm_r.aligned_))
    for j in range(100):
        if  ss.pearsonr(sc_eigen_raw[i][:180,j],sc_eigen_raw[i][180:,j])[0] < 0:
            sc_eigen_raw[i][:180,j]= -sc_eigen_raw[i][:180,j]
        if  ss.pearsonr(sc_eigen_align[i][:180,j],sc_eigen_align[i][180:,j])[0] < 0:
            sc_eigen_align[i][:180,j]= -sc_eigen_align[i][:180,j]
    print('sc...', i+1)
    
np.save('../../results/hcp/sc_eigen_hcp_raw.npy',sc_eigen_raw)
np.save('../../results/hcp/sc_eigen_hcp_align.npy',sc_eigen_align)

## plot eigenvectors on the brain for structural connectivity matrix
plot = [None] * len(vec)
for i in range(len(vec)):
  plot[i] = mmp_fsLR.copy()
  for node in range(360):
    plot[i][mmp_fsLR==node+1] = sc_eigen[:,vec[i]][node]
  plot[i][mmp_fsLR==0] = np.nan

fp.plot_surface(data = plot, nan_color=(0.5,0.5,0.5,1), color_range=(-0.1,0.1),
                size = (400, 300*len(vec)), layout_style = 'grid',
                cmap = 'plasma', filename = '../../figures/sc_eigen_hcp_group.png',
                display=False)

## plot structural connectivity matrix
fig, ax = plt.subplots(figsize=(3,3))
tmp = hcp_sc.mean(0)
tmp += tmp.T
ax.imshow(tmp, cmap='plasma', vmin=0,vmax=1000)
ax.set_xticks([])
ax.set_yticks([])
fig.tight_layout()
fig.savefig('../../figures/sc_hcp_group.png', transparent=True, dpi=300)

## Compute eigenvectors for functional connectivity matrix
hcp_grad = GradientMaps(kernel='normalized_angle', approach=embedding, n_components=100, random_state=0)
hcp_grad.fit(hcp_fc.mean(0), sparsity=0.9)
fc_eigen = hcp_grad.gradients_

np.savetxt('../../results/hcp/fc_eigen_hcp_group.txt', fc_eigen)

fc_eigen_raw = np.zeros((n_indi,360,100))
fc_eigen_align = np.zeros((n_indi,360,100))

for i in range(n_indi):
    gm = GradientMaps(kernel='normalized_angle', approach=embedding, n_components=100, random_state=0, alignment='procrustes')
    gm.fit(hcp_fc[i], reference=fc_eigen, sparsity=0.9)
    fc_eigen_raw[i] = gm.gradients_
    fc_eigen_align[i] = gm.aligned_
    for j in range(100):
        if  ss.pearsonr(fc_eigen_align[i][:,j], fc_eigen[:,j])[0] < 0:
            fc_eigen_align[i][:,j]= -fc_eigen_align[i][:,j]
    print('fc...', i+1)
np.save('../../results/hcp/fc_eigen_hcp_raw.npy',fc_eigen_raw)
np.save('../../results/hcp/fc_eigen_hcp_align.npy',fc_eigen_align)


## plot eigenvectors on the brain for functional connectivity matrix
plot = [None] * len(vec)
for i in range(len(vec)):
  plot[i] = mmp_fsLR.copy()
  for node in range(360):
    plot[i][mmp_fsLR==node+1] = fc_eigen[:,vec[i]][node]
  plot[i][mmp_fsLR==0] = np.nan

fp.plot_surface(data = plot, nan_color=(0.5,0.5,0.5,1), color_range=(-0.1,0.1),
                size = (400, 300*len(vec)), layout_style = 'grid',
                cmap = 'viridis', filename = '../../figures/fc_eigen_hcp_group.png',
                display=False)

## plot functional connectivity matrix
fig, ax = plt.subplots(figsize=(3,3))
ax.imshow(hcp_fc.mean(0), cmap='viridis',vmin=0,vmax=1)
ax.set_xticks([])
ax.set_yticks([])
fig.tight_layout()
fig.savefig('../../figures/fc_hcp_group.png', transparent=True, dpi=300)
