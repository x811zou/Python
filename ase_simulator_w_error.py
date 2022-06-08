import pandas as pd
import gzip
import sys
import numpy as np
from numpy.random import uniform
#simulate-ase.py <num-genes> <het-sites-per-gene> <read-depth> <theta>

N=int(sys.argv[1])
M=int(sys.argv[2])
D=int(sys.argv[3])
theta=float(sys.argv[4])
error=float(sys.argv[5]) # 0.1
outfile=sys.argv[6]

#theta=3
#D=10
#M=5
#N=10
if theta <= 0:
    raise ValueError("theta should be positive")

with open(outfile,'w') as f:
    #f.write('M')
    for k in range(N):
#        A=np.random.binomial(D, theta/(1+theta), size=M)
#        AR = ["%d\t%d" % (A[i], D-A[i]) for i in range(len(A))]
        switch_ind = False
        index = []
        AR = []
        Current_Step = ["A","R"]
        for i in range(M):
          A= np.random.binomial(D, theta/(1+theta))
          value = {"A":A,"R":D-A}
          switch_ind = True if uniform() <= error else False
          index.append(switch_ind)
          if switch_ind == True:
              Current_Step = [ Current_Step[1], Current_Step[0]]
              AR.append("%d\t%d" % (value[Current_Step[0]], value[Current_Step[1]] ))
          else:
              AR.append("%d\t%d" % (value[Current_Step[0]], value[Current_Step[1]] ))
        line = '%d\t%d\t%s\t%s\n' % (k+1,M, "\t".join(AR), theta)
        #print(index)
        f.write(line)
