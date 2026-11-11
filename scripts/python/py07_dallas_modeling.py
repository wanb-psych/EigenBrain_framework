import numpy as np
import pandas as pd
import scipy.stats as ss
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
from kneed import KneeLocator
import pickle
import os

t='wave3'

mmp_fsLR = np.loadtxt('../../src/fs_LR.64k.mmp_360.txt')
max_mode = 100
sub_list = np.loadtxt(f'../../results/dallas_more/individual/sub_list_{t}.txt', dtype='str')
n_sub = len(sub_list)

amyloid = np.zeros((n_sub,360))
tau = np.zeros((n_sub,360))
for i in range(n_sub):
  tmp = np.loadtxt(f'../../results/dallas_more/individual/{sub_list[i]}_{t}_ab_mmp360.txt')
  if len(tmp.shape)==2:
    amyloid[i] = ss.zscore(tmp[:,-1])
  else:
    amyloid[i] = ss.zscore(tmp)
  tmp = np.loadtxt(f'../../results/dallas_more/individual/{sub_list[i]}_{t}_tau_mmp360.txt')
  if len(tmp.shape)==2:
    tau[i] = ss.zscore(tmp[:,-1])
  else:
    tau[i] = ss.zscore(tmp)

## Modeling using hcp group-level gradients
hcp_gd = np.loadtxt('../../results/hcp/gd_eigen_hcp_group.txt')
hcp_sc = np.loadtxt('../../results/hcp/sc_eigen_hcp_group.txt')
hcp_fc = np.loadtxt('../../results/hcp/fc_eigen_hcp_group.txt')

gd_eigen = np.zeros((n_sub,360,100))
sc_eigen = np.zeros((n_sub,360,100))
fc_eigen = np.zeros((n_sub,360,100))
for i in range(n_sub):
    gd_eigen[i] = np.loadtxt(f'../../results/dallas_more/eigen/gd_eigen_{sub_list[i]}_{t}_mmp_hcp.txt')
    sc_eigen[i] = np.loadtxt(f'../../results/dallas_more/eigen/sc_eigen_{sub_list[i]}_{t}_mmp_hcp.txt')
    fc_eigen[i] = np.loadtxt(f'../../results/dallas_more/eigen/fc_eigen_{sub_list[i]}_{t}_mmp_hcp.txt')
for map_type in ['amyloid','tau']:
    if map_type == 'amyloid':
        map = amyloid.mean(0)
    else:
        map = tau.mean(0)
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
        if stop[w_tmp[i]] <=20: # not robust estimate
            stop[w_tmp[i]] = 20
            
        glm = OLS(map, tmp[i][:,:stop[w_tmp[i]]+1]).fit()
        
        with open(f'../../results/dallas_more/glm_{map_type}_{w_tmp[i]}.pkl', 'wb') as f:
            pickle.dump(glm, f)
        
        #print(' ' + w_tmp[i] + ' : ' +str(stop[w_tmp[i]]))
        elbow_save.append(map_type + ',' + w_tmp[i] + ',' +str(stop[w_tmp[i]]))

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
        if stop[w_tmp[i+3]] <=20: # not robust estimate
            stop[w_tmp[i+3]] = stop[w_tmp[2]] # use the FC elbow instead
        if i < 2:    
            glm = OLS(map, tmp[i][:,:stop[w_tmp[i]]+stop[w_tmp[i+3]]+1]).fit()
        else:
            glm = OLS(map, tmp[i][:,:stop[w_tmp[0]]+stop[w_tmp[1]]+stop[w_tmp[i+3]]+1]).fit()
        
        with open(f'../../results/dallas_more/glm_{map_type}_{w_tmp[i+3]}.pkl', 'wb') as f:
            pickle.dump(glm, f)
        
        #print(' ' + w_tmp[i+3] + ' : ' +str(stop[w_tmp[i+3]]))
        elbow_save.append(map_type + ',' + w_tmp[i+3] + ',' +str(stop[w_tmp[i+3]]))
                    
    np.savetxt(f'../../results/dallas_more/r2_{map_type}_hcp-group.txt', r2.T)
    np.savetxt(f'../../results/dallas_more/Elbows4ThreeModality_{map_type}.txt', elbow_save, fmt='%s')

    with open(f'../../results/dallas_more/glm_{map_type}_GD_SC_FC.pkl', 'rb') as f:
        glm = pickle.load(f)

    corr_r=['ID,hcp,self,hcp_self']
    for i in range(n_sub):
        x = np.concatenate((ss.zscore(gd_eigen[i][:,:stop['GD']], axis=0),
                            ss.zscore(sc_eigen[i][:,:stop['SC']], axis=0),
                            ss.zscore(fc_eigen[i][:,:stop['GD_SC_FC']], axis=0)),axis=1)
        sub_pred1 = glm.predict(sm.add_constant(x))
        os.makedirs(f'../../results/dallas_more/{map_type}/pred', exist_ok=True)
        np.savetxt(f'../../results/dallas_more/{map_type}/pred/{sub_list[i]}_pred_self.txt', sub_pred1)
        if map_type == 'amyloid':
            indi_map = amyloid
        else:
            indi_map = tau  
        r1=ss.pearsonr(indi_map[i], sub_pred1)[0]
        
        x = np.concatenate((ss.zscore(hcp_gd[:,:stop['GD']], axis=0),
                            ss.zscore(hcp_sc[:,:stop['SC']], axis=0),
                            ss.zscore(fc_eigen[i][:,:stop['GD_SC_FC']], axis=0)),axis=1)
        sub_pred2 = glm.predict(sm.add_constant(x))
        np.savetxt(f'../../results/dallas_more/{map_type}/pred/{sub_list[i]}_pred_hcp.txt', sub_pred2)  
        r2=ss.pearsonr(indi_map[i], sub_pred2)[0]
        r12=ss.pearsonr(sub_pred1, sub_pred2)[0]
        corr_r.append(sub_list[i]+','+str(r2.round(5))+','+str(r1.round(5))+','+str(r12.round(5)))
        
    np.savetxt(f'../../results/dallas_more/{map_type}/pred_r_hcp_self.txt', corr_r, fmt='%s')
    
    ## Prediction for autism subjects using the same model
    sub_asd = np.loadtxt('../../results/ABIDE/sub_list.txt', dtype='str')
    n_asd = len(sub_asd)
    qc = ['ID,neurotransmitter,modality,QC_r']
    os.makedirs(f'../../results/ABIDE/pred/{map_type}', exist_ok=True)
    for i in range(n_asd):
        fc_eigen_asd = np.loadtxt('../../results/ABIDE/eigen/sub-'+sub_asd[i]+'_mmp_hcp.txt')
        x = np.concatenate((ss.zscore(hcp_gd[:,:stop['GD']], axis=0),
                            ss.zscore(hcp_sc[:,:stop['SC']], axis=0),
                            ss.zscore(fc_eigen_asd[:,:stop['GD_SC_FC']], axis=0)),axis=1)
        sub_pred = glm.predict(sm.add_constant(x))
        np.savetxt(f'../../results/ABIDE/pred/{map_type}/sub-{sub_asd[i]}_{map_type}_GD_SC_FC_pred.txt', sub_pred)
        sub_r = ss.pearsonr(sub_pred, map)[0].round(5)
        qc.append(sub_asd[i]+','+map_type+',GD_SC_FC,'+str(sub_r))
        
    np.savetxt(f'../../results/ABIDE/pred_qc_{map_type}.txt', qc, fmt='%s')
    
    ## Prediction for scz subjects using the same model
    sub_scz = np.array(pd.read_csv('../../results/Geneva/demo_sort.csv')['ID'])
    n_scz = len(sub_scz)
    os.makedirs(f'../../results/Geneva/pred/{map_type}', exist_ok=True)
    for ses in ['ses-T1','ses-T2']:
        qc = ['ID,neurotransmitter,modality,QChcp_gd_r,QCself_gd_r']
        for i in range(n_scz):
            fc_eigen_scz = np.loadtxt(f'../../results/Geneva/eigen/fc_eigen_{sub_scz[i]}_{ses}_mmp_hcp.txt')
            gd_eigen_scz = np.loadtxt(f'../../results/Geneva/eigen/gd_eigen_{sub_scz[i]}_{ses}_mmp_hcp.txt')
            x = np.concatenate((ss.zscore(hcp_gd[:,:stop['GD']], axis=0),
                                ss.zscore(hcp_sc[:,:stop['SC']], axis=0),
                                ss.zscore(fc_eigen_scz[:,:stop['GD_SC_FC']], axis=0)),axis=1)
            sub_pred = glm.predict(sm.add_constant(x))
            np.savetxt(f'../../results/Geneva/pred/{map_type}/{sub_scz[i]}_{ses}_{map_type}_GD_SC_FC_hcpGD_pred.txt', sub_pred)
            sub_r_hcp = ss.pearsonr(sub_pred, map)[0].round(5)
            
            x = np.concatenate((ss.zscore(gd_eigen_scz[:,:stop['GD']], axis=0),
                                ss.zscore(hcp_sc[:,:stop['SC']], axis=0),
                                ss.zscore(fc_eigen_scz[:,:stop['GD_SC_FC']], axis=0)),axis=1)
            sub_pred = glm.predict(sm.add_constant(x))
            np.savetxt(f'../../results/Geneva/pred/{map_type}/{sub_scz[i]}_{ses}_{map_type}_GD_SC_FC_selfGD_pred.txt', sub_pred)
            sub_r_self = ss.pearsonr(sub_pred, map)[0].round(5)

            qc.append(sub_scz[i]+','+map_type+',GD_SC_FC,'+str(sub_r_hcp)+','+str(sub_r_self))
            
        np.savetxt(f'../../results/Geneva/pred_qc_{ses}_{map_type}.txt', qc, fmt='%s')