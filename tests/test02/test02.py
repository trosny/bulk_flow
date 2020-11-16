#!/usr/bin/env python
"""
 test02 of seal class
 
 Getting converged solution on refined grid at larger eccentricity ratios.
 
 - For large eccentricity ratios (e.g. 0.8), convergence difficulties
   are encountered for the zeroth-order problem. Notably, the issue seems
   to stem from friction source terms.
 - An alternative treatment of the friction source terms has been implemented
   using a Taylor series linearization to get implicit and explicit
   contributions. However, this alternative scheme does not appear to be any
   more robust than the original, simple implicit/explicit blending.
 - Obtaining a stable, converged solution requires an excessive amount of 
   tuning of the momentum and pressure relaxation factors.
 - A simple solution is to increase the blending factor "uv_src_blend" to a
   large value to increase the diagonal dominance of the momentum equations.
   This can be used to maintain stability at the cost of a slower convergence
   rate. (Note the very large momentum residuals at the start of the 
   computation.)
 - This approach still requires tuning, but the tuning can be limited to
   a single factor, which if made large (1-2 orders-of-magnitude larger
   default value of 1.0) will ensure convergence.
 - Note in this example that the discretization is set to fully upwind,
   "gamma : 0.0", this should always be the first setting to change to enhance
   stability.
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
    param = read_parameters('Kanki01_input.yaml')
    s = seal(param)
    s.solve_zeroth()
    s.plot_res()
    
    
if __name__ == "__main__":
    start = time.time()
    main()  
    end = time.time()
    print('runtime [s]')
    print(end-start)
