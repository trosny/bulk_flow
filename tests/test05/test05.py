#!/usr/bin/env python
"""
 test05 
 testing incompressible solver with dtu test rig parameter values
 zeroth order solution only
"""
import sys
sys.path.append("../../src")

from seal import seal
from seal_funcs import read_parameters
import time

def main():
    """
    to call run *.py file containing class as script
    """
    # output filename
    param = read_parameters('dtuTestRig_input.yaml')
    s = seal(param)
    s.solve_zeroth()
    s.plot_res()
    
    
if __name__ == "__main__":
    start = time.time()
    main()  
    end = time.time()
    print('runtime [s]')
    print(end-start)
