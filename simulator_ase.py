#!/usr/bin/env python
#=========================================================================
# simulator_ase.py <num-genes> <het-sites-per-gene> <read-depth> <theta>
# Within simulate-ase.py you would simulate N genes, each having M het sites.  
# Each of those would have a total of D reads; theta is the effect size (the amount of ASE). # The number of alternate (A) and reference (R) reads (where A+R=D) as follows:
# A ~ binomial(D,0.5*theta)
# R=D-A
# For now, all the sites can have the same read depth, and we won't worry about distances between them or whether they share the same reads (we can add those things to the simulator later).
# The simulator should output N lines, one gene per line, with fields separated by tabs:
# M   A[1]   R[1]   ...   A[M]   R[M]   theta
#=========================================================================

import pandas as pd
import gzip
import sys
import numpy as np

N=int(sys.argv[1])
M=int(sys.argv[2])
D=int(sys.argv[3])
theta=float(sys.argv[4])
filename=sys.argv[5]
# check if theta is between (0,2)
if theta <= 0:
    raise ValueError("theta should be above zero")

with open(filename,'w') as f:
    for k in range(N):  
        A=np.random.binomial(D, theta/(1+theta), size=M)
        AR = ["%d\t%d" % (A[i], D-A[i]) for i in range(len(A))]
        line = '%d\t%s\t%s\n' % (M, "\t".join(AR), theta)
        #print(line)
        f.write(line)
