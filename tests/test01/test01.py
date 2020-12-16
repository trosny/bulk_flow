#!/usr/bin/env python
"""
test01 of seal class
baseline test of zeroth-order solution
- coarse grid
- "standard" relaxation factors
"""
    
import sys
sys.path.append("../../src")

from seal import seal
from seal_funcs import read_parameters
import time
import cProfile

def run_example():
    """
    test01 of seal class
    baseline test of zeroth-order solution
    - coarse grid
    - "standard" relaxation factors
    """
    # output filename
    param = read_parameters('Kanki01_input.yaml')
    s = seal(param)
    s.solve_zeroth()
    s.plot_res()
    
if __name__ == "__main__":
    start = time.time()
    run_example()
    end = time.time()
    print('runtime [s]')
    print(end-start)    
