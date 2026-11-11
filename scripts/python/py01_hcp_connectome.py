import numpy as np
import nibabel as nib
import os
import hcp_utils as hcp

path = '/data/p_02657/data/hcp_bids/derivatives/micapipe_v0.2.0/'
sub_list = np.loadtxt('../../results/hcp/sub_94unrelated_final.txt', dtype='str')

gd = [np.delete(np.delete(nib.load(path+i+'/dist/'+i+'_atlas-glasser-360_GD.shape.gii').agg_data(), 
                          [0,181], axis=0), [0,181], axis=1) for sub in sub_list]

np.save('../../results/hcp/gd_unrelated.npy', np.array(gd))


sc = [np.delete(np.delete(nib.load(path+i+'/dwi/connectomes/'+i+'_space-dwi_atlas-glasser-360_desc-iFOD2-40M-SIFT2_full-connectome.shape.gii').agg_data()[49:,49:], 
                          [180], axis=0), [180], axis=1) for sub in sub_list]

np.save('../../results/hcp/sc_unrelated.npy', np.array(sc))

path_fun = '/data/p_02657/data/human-connectome-project-openaccess/HCP1200/'
fc_sub=[]
for i in sub_list:
  ph = ['rfMRI_REST1_LR','rfMRI_REST1_RL','rfMRI_REST2_LR','rfMRI_REST2_RL']
  tmp = []
  for p in ph:
    try:
      ts = nib.load(path_fun + i[4:] + '/MNINonLinear/Results/'+p+'/'+p+'_Atlas_MSMAll_hp2000_clean.dtseries.nii').get_fdata()
      ts = hcp.parcellate(ts, hcp.mmp)[:,:360]
      fc = np.corrcoef(ts.T)
      fc[fc>0.99999] = 1
    except:
      pass
   
    tmp.append(fc) 
  fc=np.array(tmp).mean(0)
  fc = np.arctanh(fc)
  fc[fc==np.inf] = 0
  fc_sub.append(fc)
  print(i)

np.save('../../results/hcp/fc_unrelated.npy', np.array(fc_sub))
