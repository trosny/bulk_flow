#!/usr/bin/env python
"""
 test06
 test error handling of missing input parameters
 input parameters are checked both for the mesh base class and the seal
 subclass
"""
import sys
sys.path.append("../../src")

from seal import seal
from seal_funcs import read_parameters

def test_key_error1():
    """
    """
    param = read_parameters('input.yaml')
    
    # missing parameter, should throw KeyError
    s = seal(param)       
  

def test_key_error2():
    """
    """
    param = read_parameters('input.yaml')
    
    # try-except block execution
    # KeyError asserted with try, so except block executed
    try:
        s = seal(param) 
    except:        
        param['uv_src_method'] = 0
        s = seal(param)     
    
    # run zeroth order solve for a few iterations
    s.max_it = 5
    s.solve_zeroth()  
 
    
if __name__ == "__main__":
    test_key_error2()
    test_key_error1()    
