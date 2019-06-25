import glob
from astropy.table import Table 
from tqdm import tqdm

files = glob.glob('data_*.txt')

with open('all_kics.txt','w') as f:

	f.write('kic,period,epoch,sde,impact,depth,niter\n')

	for fname in tqdm(files):
		s = open(fname,'r').read().replace(' ',', ')
		f.write(s+'\n')

print('Done!')