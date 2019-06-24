from hot_utils import *

if __name__ == '__main__':
    ap = ArgumentParser(description='hot: look for planets around hot stars with iterative sine fitting')
    ap.add_argument('-kic', default=8197761,type=int,help='Target KIC ID')

    do_all(kic)
