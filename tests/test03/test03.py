#!/usr/bin/env python
"""
 test sensitivity of leakage rate to momentum relaxation factor

"""
import sys
sys.path.append("../../src")

from seal import seal
from seal_funcs import read_parameters
import time
import numpy as np
#import matplotlib.pyplot as plt
#from scipy import optimize

def main():
    
    relax_uv = np.array([0.3, 0.5, 0.7, 0.9])
    q = np.zeros(relax_uv.size, dtype=np.float64)
    param = read_parameters('Kanki01_input.yaml')
    s = seal(param)
    for idx, val in enumerate(relax_uv):
        s.relax_uv = val   
        s.solve_zeroth()        
        q[idx] = s.q
        
    for idx, val in enumerate(relax_uv):    
        print("relax : {a:g} , leakage [cm^3/s] : {b:g}".format(a=val, b= q[idx]/s.rho_s*1.e6) )  
    
    
if __name__ == "__main__":
    start = time.time()
    main()  
    end = time.time()
    print('runtime [s]')
    print(end-start)
