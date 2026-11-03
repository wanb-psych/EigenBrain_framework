from brainsmash.mapgen.base import Base
import numpy as np
import scipy.stats as ss
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm

dist_mat_file_L = '../../src/LeftParcelGeodesicDistmat.txt'
dist_mat_file_R = '../../src/RightParcelGeodesicDistmat.txt'

fc_eigen = np.loadtxt('../../results/hcp/fc_eigen_hcp_group.txt')
sc_eigen = np.loadtxt('../../results/hcp/sc_eigen_hcp_group.txt')
gd_eigen = np.loadtxt('../../results/hcp/gd_eigen_hcp_group.txt')

transmitter_name = ['5HT1a', '5HT1b', '5HT2a', '5HT4','5HT6','5HTT',
                    'D1', 'D2', 'DAT',
                    'NET', 'H3','a4b2', 'M1',
                    'VAChT', 'CB1', 'MOR',
                    'NMDA', 'mGluR5',
                    'GABA', 'KOR']

def get_surrogates(input, output):
    base_L = Base(x=np.loadtxt(input)[:180], D=dist_mat_file_L, kernel='gaussian', pv=25, nh=30, seed=0)
    base_R = Base(x=np.loadtxt(input)[180:], D=dist_mat_file_R, kernel='gaussian', pv=25, nh=30, seed=0)
    surrogates_L = base_L(n=1000)
    surrogates_R = base_R(n=1000)
    surrogates = np.concatenate((surrogates_L, surrogates_R),axis=1)
    np.savetxt(output, surrogates)
    
for i in transmitter_name:
    print(f'Generating nulls for {i} ...')
    get_surrogates(f'../../results/neuromap/mmp_{i}.txt', f'../../results/nulls/{i}_variogram.txt')
    
print('Generating nulls for CMRglc ...')
get_surrogates(f'../../results/CMRglc/glucose/glucoseMean.txt', 
               f'../../results/nulls/CMRglc_variogram.txt')

print('Generating nulls for amyloid ...')
get_surrogates(f'../../results/dallas_more/amyloidMean.txt', 
               f'../../results/nulls/amyloid_variogram.txt')

print('Generating nulls for tau ...')
get_surrogates(f'../../results/dallas_more/tauMean.txt', 
               f'../../results/nulls/tau_variogram.txt')

def get_r2(input, output):
    r2 = np.zeros((1000,3,5))
    map_all = np.loadtxt(input)
    mode = [20,40,60,80,100]
    for p in range(1000):
        map = map_all[p]
        for j in range(5):
            tmp=[sm.add_constant(ss.zscore(gd_eigen[:,:mode[j]], axis=0)), 
                sm.add_constant(ss.zscore(sc_eigen[:,:mode[j]], axis=0)), 
                sm.add_constant(ss.zscore(fc_eigen[:,:mode[j]], axis=0))]
            for i in range(3):
                glm = OLS(map, tmp[i]).fit()
                r2[p,i,j]=glm.rsquared
    np.save(output, r2)
    
for i in transmitter_name + ['CMRglc', 'amyloid', 'tau']:
    print(f'Generating null modeling for {i} ...')
    get_r2(f'../../results/nulls/{i}_variogram.txt', f'../../results/nulls/{i}_r2.npy')