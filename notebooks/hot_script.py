from hot_utils import *

from argparse import ArgumentParser
import matplotlib as mpl

mpl.style.use('seaborn-colorblind')

#To make sure we have always the same matplotlib settings
#(the ones in comments are the ipython notebook settings)

mpl.rcParams['figure.figsize']=(8.0,6.0)    #(6.0,4.0)
mpl.rcParams['font.size']=18               #10 
mpl.rcParams['savefig.dpi']= 200             #72 
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams["font.family"] = "Times New Roman"


if __name__ == '__main__':
    ap = ArgumentParser(description='hot: look for planets around hot stars with iterative sine fitting')
    ap.add_argument('-kic', default=8197761,type=int,help='Target KIC ID')

    do_all(kic)
