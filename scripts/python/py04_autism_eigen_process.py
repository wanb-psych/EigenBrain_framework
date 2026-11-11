import numpy as np
import pandas as pd
import pickle
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
import scipy.stats as ss
import os
from brainspace.gradient import GradientMaps

## Load HCP data
hcp_fc = np.loadtxt('../../results/hcp/fc_eigen_hcp_group.txt')
hcp_gd = np.loadtxt('../../results/hcp/gd_eigen_hcp_group.txt')
hcp_sc = np.loadtxt('../../results/hcp/sc_eigen_hcp_group.txt')

sub_list = np.loadtxt('../../results/ABIDE/sub_list.txt', dtype='str')
n_sub = len(sub_list)

fc = [np.loadtxt('../../results/ABIDE/fc/'+sub+'.csv', delimiter=',') for sub in sub_list]

## zero diagonal for gradient processing
for i in range(n_sub):
    np.fill_diagonal(fc[i], 0)

embedding = 'dm'

para = pd.read_csv('../../results/neuromap/Elbows4ThreeModality.txt')
qc = []
qc.append('ID,neurotransmitter,modality,QC_r')

for i in range(n_sub):
    gm = GradientMaps(kernel='normalized_angle', approach=embedding, n_components=100, random_state=0, alignment='procrustes')
    np.fill_diagonal(fc[i], 1)
    gm.fit(fc[i], reference=hcp_fc, sparsity=0.9)
    np.savetxt('../../results/ABIDE/eigen/sub-'+sub_list[i]+'_mmp_raw.txt',gm.gradients_)
    np.savetxt('../../results/ABIDE/eigen/sub-'+sub_list[i]+'_mmp_hcp.txt',gm.aligned_)
    print('Finished fc eigen processing subject:', sub_list[i])
    for trans in para['neurotransmitter'].unique():
        os.makedirs(f'../../results/ABIDE/pred/{trans}', exist_ok=True)
        map = np.loadtxt(f'../../results/neuromap/mmp_{trans}.txt')
        mod='FC'
        glm = pickle.load(open(f'../../results/hcp/glm_{trans}_{mod}.pkl', 'rb'))
        stop = int(para[(para['neurotransmitter']==trans) & (para['modality']==mod)]['elbow'].iloc[0])
        x = sm.add_constant(ss.zscore(gm.aligned_[:, :stop], axis=0))
        sub_pred = glm.predict(x)
        np.savetxt(f'../../results/ABIDE/pred/{trans}/sub-{sub_list[i]}_{trans}_{mod}_pred.txt', sub_pred)
        sub_r = ss.pearsonr(sub_pred, map)[0].round(5)
        qc.append(sub_list[i]+','+trans+','+mod+','+str(sub_r))
        
        mod='GD_SC_FC'
        glm = pickle.load(open(f'../../results/hcp/glm_{trans}_{mod}.pkl', 'rb'))
        stop1 = int(para[(para['neurotransmitter']==trans) & (para['modality']=='GD')]['elbow'].iloc[0])
        stop2 = int(para[(para['neurotransmitter']==trans) & (para['modality']=='SC')]['elbow'].iloc[0])
        stop3 = int(para[(para['neurotransmitter']==trans) & (para['modality']==mod)]['elbow'].iloc[0])
        x = np.concatenate((ss.zscore(hcp_gd[:,:stop1], axis=0),
                            ss.zscore(hcp_sc[:,:stop2], axis=0),
                            ss.zscore(gm.aligned_[:, :stop3], axis=0)),axis=1)
        sub_pred = glm.predict(sm.add_constant(x))
        np.savetxt(f'../../results/ABIDE/pred/{trans}/sub-{sub_list[i]}_{trans}_{mod}_pred.txt', sub_pred)
        sub_r = ss.pearsonr(sub_pred, map)[0].round(5)
        qc.append(sub_list[i]+','+trans+','+mod+','+str(sub_r))
        
        print('Finished subject:', sub_list[i], 'neurotransmitter prediction:', trans)
np.savetxt('../../results/ABIDE/pred_qc_neurotransmitter.txt', qc, fmt='%s')