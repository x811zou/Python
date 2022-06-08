import pandas as pd
import gzip
import sys
import numpy as np
#simulate-ase.py <num-genes> <het-sites-per-gene> <read-depth> <theta>

N=int(sys.argv[1])
M=int(sys.argv[2])
D=int(sys.argv[3])
theta=float(sys.argv[4])
outfile=sys.argv[5]

#theta=3
#D=10
#M=5
#N=10
# check if theta is between (0,2)
if theta <= 0:
    raise ValueError("theta should be positive")

with open(outfile,'w') as f:
    #f.write('M')
    for k in range(N):  
        A=np.random.binomial(D, theta/(1+theta), size=M)
        AR = ["%d\t%d" % (A[i], D-A[i]) for i in range(len(A))]
        line = '%d\t%d\t%s\t%s\n' % (k+1,M, "\t".join(AR), theta)
        #print(line)
        f.write(line)
