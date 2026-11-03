import numpy as np
import pandas as pd
import func_plot as fp
import scipy.stats as ss
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
from kneed import KneeLocator
import pickle
import os

mmp_fsLR = np.loadtxt('../../src/fs_LR.64k.mmp_360.txt')
max_mode = 100
sub_list = np.loadtxt('../../results/CMRglc/sub_list.txt', dtype='str')
n_sub = len(sub_list)

glucose = np.zeros((n_sub,360))
gd_eigen = np.zeros((n_sub,360,max_mode))
sc_eigen = np.zeros((n_sub,360,max_mode))
fc_eigen = np.zeros((n_sub,360,max_mode))
for i in range(n_sub):
  glucose[i] = ss.zscore(np.loadtxt('../../results/CMRglc/glucose/individual/'+sub_list[i]+'_fsLR_mmp.txt'))
  gd_eigen[i] = np.loadtxt('../../results/CMRglc/gd/eigen/sub-'+sub_list[i]+'_ses-open_mmp_hcp.txt')
  sc_eigen[i] = np.loadtxt('../../results/CMRglc/sc/eigen/sub-'+sub_list[i]+'_ses-open_mmp_hcp.txt')
  fc_eigen[i] = np.loadtxt('../../results/CMRglc/fc/eigen/sub-'+sub_list[i]+'_ses-open_mmp_hcp.txt')

np.savetxt('../../results/CMRglc/glucose/glucoseMean.txt', glucose.mean(0))

plot = mmp_fsLR.copy()
for node in range(360):
  plot[mmp_fsLR==node+1] = glucose.mean(0)[node]
plot[mmp_fsLR==0] = np.nan
    
fp.plot_surface(data = plot,
                size = (400, 300), color_range=(-3,3), nan_color=(0.5,0.5,0.5,1), layout_style = 'grid',
                cmap = 'CMRmap', filename = '../../figures/CMRglc_mean.png',
                display=False)

## Modeling using hcp group-level gradients
hcp_gd = np.loadtxt('../../results/hcp/gd_eigen_hcp_group.txt')
hcp_sc = np.loadtxt('../../results/hcp/sc_eigen_hcp_group.txt')
hcp_fc = np.loadtxt('../../results/hcp/fc_eigen_hcp_group.txt')

map = glucose.mean(0)
r2 = np.zeros((6,max_mode))
w_tmp = ['GD','SC','FC', 'GD_FC','SC_FC','GD_SC_FC']
elbow_save = []
stop = {}
for j in range(max_mode):
    tmp=[sm.add_constant(ss.zscore(hcp_gd[:,:j+1], axis=0)), 
          sm.add_constant(ss.zscore(hcp_sc[:,:j+1], axis=0)), 
          sm.add_constant(ss.zscore(hcp_fc[:,:j+1], axis=0))]
    for i in range(3): 
        glm = OLS(map, tmp[i]).fit()
        r2[i,j]=glm.rsquared
        
for i in range(3):
    kl_pred = KneeLocator(range(1,101), r2[i], S=6, curve="concave", direction="increasing")
    stop[w_tmp[i]] = kl_pred.knee
    if stop[w_tmp[i]] is None:
        stop[w_tmp[i]] = 100
        
    glm = OLS(map, tmp[i][:,:stop[w_tmp[i]]+1]).fit()
    
    with open(f'../../results/CMRglc/glm_CMRglc_{w_tmp[i]}.pkl', 'wb') as f:
        pickle.dump(glm, f)
    
    #print(' ' + w_tmp[i] + ' : ' +str(stop[w_tmp[i]]))
    elbow_save.append('CMRglc,' + w_tmp[i] + ',' +str(stop[w_tmp[i]]))

for j in range(max_mode):
    tmp=[sm.add_constant(np.concatenate((ss.zscore(hcp_gd[:,:stop['GD']], axis=0),
                                          ss.zscore(hcp_fc[:,:j+1], axis=0)),axis=1)),
          sm.add_constant(np.concatenate((ss.zscore(hcp_sc[:,:stop['SC']], axis=0),
                                          ss.zscore(hcp_fc[:,:j+1], axis=0)),axis=1)),
          sm.add_constant(np.concatenate((ss.zscore(hcp_gd[:,:stop['GD']], axis=0),
                                          ss.zscore(hcp_sc[:,:stop['SC']], axis=0),
                                          ss.zscore(hcp_fc[:,:j+1], axis=0)),axis=1))]
    for i in range(3): 
        glm = OLS(map, tmp[i]).fit()
        r2[i+3,j]=glm.rsquared

for i in range(3):
    kl_pred = KneeLocator(range(1,101), r2[i+3], S=6, curve="concave", direction="increasing")
    stop[w_tmp[i+3]] = kl_pred.knee
    if stop[w_tmp[i+3]] is None:
        stop[w_tmp[i+3]] = 100
    if i < 2:    
        glm = OLS(map, tmp[i][:,:stop[w_tmp[i]]+stop[w_tmp[i+3]]+1]).fit()
    else:
        glm = OLS(map, tmp[i][:,:stop[w_tmp[0]]+stop[w_tmp[1]]+stop[w_tmp[i+3]]+1]).fit()
    
    with open(f'../../results/CMRglc/glm_CMRglc_{w_tmp[i+3]}.pkl', 'wb') as f:
        pickle.dump(glm, f)
    
    #print(' ' + w_tmp[i+3] + ' : ' +str(stop[w_tmp[i+3]]))
    elbow_save.append('CMRglc,' + w_tmp[i+3] + ',' +str(stop[w_tmp[i+3]]))
                
np.savetxt(f'../../results/CMRglc/r2_hcp-group.txt', r2.T)
np.savetxt('../../results/CMRglc/Elbows4ThreeModality.txt', elbow_save, fmt='%s')

with open(f'../../results/CMRglc/glm_CMRglc_GD_SC_FC.pkl', 'rb') as f:
    glm = pickle.load(f)

corr_r=['ID,hcp,self,hcp_self']
for i in range(n_sub):
    x = np.concatenate((ss.zscore(gd_eigen[i][:,:stop['GD']], axis=0),
                        ss.zscore(sc_eigen[i][:,:stop['SC']], axis=0),
                        ss.zscore(fc_eigen[i][:,:stop['GD_SC_FC']], axis=0)),axis=1)
    sub_pred1 = glm.predict(sm.add_constant(x))
    np.savetxt(f'../../results/CMRglc/glucose/pred/{sub_list[i]}_pred_self.txt', sub_pred1)  
    r1=ss.pearsonr(glucose[i], sub_pred1)[0]
    
    x = np.concatenate((ss.zscore(hcp_gd[:,:stop['GD']], axis=0),
                        ss.zscore(hcp_sc[:,:stop['SC']], axis=0),
                        ss.zscore(fc_eigen[i][:,:stop['GD_SC_FC']], axis=0)),axis=1)
    sub_pred2 = glm.predict(sm.add_constant(x))
    np.savetxt(f'../../results/CMRglc/glucose/pred/{sub_list[i]}_pred_hcp.txt', sub_pred2)  
    r2=ss.pearsonr(glucose[i], sub_pred2)[0]
    r12=ss.pearsonr(sub_pred1, sub_pred2)[0]
    corr_r.append(sub_list[i]+','+str(r2.round(5))+','+str(r1.round(5))+','+str(r12.round(5)))
    
np.savetxt(f'../../results/CMRglc/pred_r_hcp_self.txt', corr_r, fmt='%s')

## Individual-level modeling

## Prediction for autism subjects using the same model
sub_list = np.loadtxt('../../results/ABIDE/sub_list.txt', dtype='str')
n_sub = len(sub_list)
qc = ['ID,neurotransmitter,modality,QC_r']
os.makedirs(f'../../results/ABIDE/pred/CMRglc', exist_ok=True)
for i in range(n_sub):
    fc_eigen = np.loadtxt('../../results/ABIDE/eigen/sub-'+sub_list[i]+'_mmp_hcp.txt')
    x = np.concatenate((ss.zscore(hcp_gd[:,:stop['GD']], axis=0),
                        ss.zscore(hcp_sc[:,:stop['SC']], axis=0),
                        ss.zscore(fc_eigen[:,:stop['GD_SC_FC']], axis=0)),axis=1)
    sub_pred = glm.predict(sm.add_constant(x))
    np.savetxt(f'../../results/ABIDE/pred/CMRglc/sub-{sub_list[i]}_CMRglc_GD_SC_FC_pred.txt', sub_pred)
    sub_r = ss.pearsonr(sub_pred, map)[0].round(5)
    qc.append(sub_list[i]+',CMRglc,GD_SC_FC,'+str(sub_r))
    
np.savetxt('../../results/ABIDE/pred_qc_CMRglc.txt', qc, fmt='%s')

## Prediction for scz subjects using the same model
sub_list = np.array(pd.read_csv('../../results/Geneva/demo_sort.csv')['ID'])
n_sub = len(sub_list)
os.makedirs(f'../../results/Geneva/pred/CMRglc', exist_ok=True)
for ses in ['ses-T1','ses-T2']:
    qc = ['ID,neurotransmitter,modality,QChcp_gd_r,QCself_gd_r']
    for i in range(n_sub):
        fc_eigen = np.loadtxt(f'../../results/Geneva/eigen/fc_eigen_{sub_list[i]}_{ses}_mmp_hcp.txt')
        gd_eigen = np.loadtxt(f'../../results/Geneva/eigen/gd_eigen_{sub_list[i]}_{ses}_mmp_hcp.txt')
        x = np.concatenate((ss.zscore(hcp_gd[:,:stop['GD']], axis=0),
                            ss.zscore(hcp_sc[:,:stop['SC']], axis=0),
                            ss.zscore(fc_eigen[:,:stop['GD_SC_FC']], axis=0)),axis=1)
        sub_pred = glm.predict(sm.add_constant(x))
        np.savetxt(f'../../results/Geneva/pred/CMRglc/{sub_list[i]}_{ses}_CMRglc_GD_SC_FC_hcpGD_pred.txt', sub_pred)
        sub_r_hcp = ss.pearsonr(sub_pred, map)[0].round(5)
        
        x = np.concatenate((ss.zscore(gd_eigen[:,:stop['GD']], axis=0),
                            ss.zscore(hcp_sc[:,:stop['SC']], axis=0),
                            ss.zscore(fc_eigen[:,:stop['GD_SC_FC']], axis=0)),axis=1)
        sub_pred = glm.predict(sm.add_constant(x))
        np.savetxt(f'../../results/Geneva/pred/CMRglc/{sub_list[i]}_{ses}_CMRglc_GD_SC_FC_selfGD_pred.txt', sub_pred)
        sub_r_self = ss.pearsonr(sub_pred, map)[0].round(5)

        qc.append(sub_list[i]+',CMRglc,GD_SC_FC,'+str(sub_r_hcp)+','+str(sub_r_self))
        
    np.savetxt(f'../../results/Geneva/pred_qc_{ses}_CMRglc.txt', qc, fmt='%s')