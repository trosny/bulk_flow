#!/usr/bin/env python
"""
test09
testing of energy equation
"""
    
import sys
sys.path.append("../../src")

from seal import seal
from seal_funcs import read_parameters
import time
import cProfile

def run_example():
    """

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
