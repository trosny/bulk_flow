#!/usr/bin/env python
"""
 test04
 - initial testing of dynamic coefficient prediction
 - water seal (incompressible flow)

"""
import sys
sys.path.append("../../src")

from seal import seal
from seal_funcs import read_parameters
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

def main():
    
    # experimental results
    q_exp = 4634.0 # leakage [cm^3/s]
    kxx_exp = 3.59
    kyx_exp = 10.8
    dxx_exp = 147.0
    dyx_exp = 55.3
    mxx_exp = 221.5
    
    # setup and solve zeroth-order problem
    param = read_parameters('Kanki01_input.yaml')
    s = seal(param)
    s.solve_zeroth()
    
    # solve first-order problem for several perturbation frequencies
    # sub-synchronous frequencies considered 
    pertFreq = np.array([0.0, 25.0, 50.0, 75.0, 100.0, 125.0, 150.0])
    # arrays for saving complex forces
    fx = np.zeros(pertFreq.size, dtype=np.complex128)
    fy = np.zeros(pertFreq.size, dtype=np.complex128)
    for idx_freq, freq in enumerate(pertFreq):
        s.whirl_f = freq     
        s.solve_first()
        fx[idx_freq] = s.fx1
        fy[idx_freq] = s.fy1

    # functions used if curve-fitting    
    def linear_func(x, b):
        # damping, linear w.r.t. frequency
        return ( b * x)
    
    def quad_func(x, a, b):
        # dynamic stiffness, added mass term exhibits frequency^2 behavior
        return (a + b * x ** 2.0)
    
    # curve-fit real and imaginary parts of forces
    # fitting coefficients are the dynamic coefficients    
    px_r,cov = optimize.curve_fit(quad_func, pertFreq, np.real(fx), p0=[0.0, 0.0]) 
    py_r,cov = optimize.curve_fit(quad_func, pertFreq, np.real(fy), p0=[0.0, 0.0])     
    px_i,cov = optimize.curve_fit(linear_func, pertFreq, np.imag(fx), p0=[0.0]) 
    py_i,cov = optimize.curve_fit(linear_func, pertFreq, np.imag(fy), p0=[0.0]) 
    
    # print  leakage and dynamic coefficients
    print('---Values (model | exp)---')
    print("leakage [cm^3/s] : {a:g} | {b:g}".format(a=s.q/s.rho_s*1.e6, b=q_exp)) 
    print("K_xx [MN/m]      : {a:g} | {b:g}".format(a=px_r[0]/1.e6, b=kxx_exp)) 
    print("K_yx [MN/m]      : {a:g} | {b:g}".format(a=py_r[0]/1.e6, b=kyx_exp)) 
    print("D_xx [kN.s/m]    : {a:g} | {b:g}".format(a=px_i[0]/1.e3, b=dxx_exp)) 
    print("D_yx [kN.s/m]    : {a:g} | {b:g}".format(a=py_i[0]/1.e3, b=dyx_exp))   
    print("M_xx [kg]        : {a:g} | {b:g}".format(a=px_r[1], b=mxx_exp))   
    
    # print relative errors in leakage and dynamic coefficients
    print('---Relative errors---')
    print("leakage [%] : {a:g}".format(a=(s.q/s.rho_s*1.e6-q_exp) / q_exp * 100.0 )) 
    print("K_xx [%]    : {a:g}".format(a=(px_r[0]/1.e6-kxx_exp) / kxx_exp * 100.0 )) 
    print("K_yx [%]    : {a:g}".format(a=(py_r[0]/1.e6-kyx_exp) / kyx_exp * 100.0 )) 
    print("D_xx [%]    : {a:g}".format(a=(px_i[0]/1.e3-dxx_exp) / dxx_exp * 100.0 )) 
    print("D_yx [%]    : {a:g}".format(a=(py_i[0]/1.e3-dyx_exp) / dyx_exp * 100.0 ))   
    print("M_xx [%]    : {a:g}".format(a=(px_r[1]-mxx_exp) / mxx_exp * 100.0 ))  
      
    # generate figures of forces as a function of perturbation frequency
    # include data and curve-fits
    # curve-fit coefficients correspond to dynamic coefficients
    plt.figure()
    plt.scatter(pertFreq, np.real(fx)/1.e6, label='fx-data')
    plt.plot(pertFreq, quad_func(pertFreq, *px_r)/1.e6, 'g--', label='fit: a=%5.3f, b=%5.3f' % tuple(px_r))
    plt.scatter(pertFreq, np.real(fy)/1.e6, label = 'fy-data')
    plt.plot(pertFreq, quad_func(pertFreq, *py_r)/1.e6, 'r--', label='fit: a=%5.3f, b=%5.3f' % tuple(py_r))
    plt.xlabel(r'Pert. Freq. [rad/s]')
    plt.ylabel(r'Re(Force) [MN/m]')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('force_real.png')
    plt.close()
    
    plt.figure()
    plt.scatter(pertFreq, np.imag(fx)/1.e6, label='fx-data')
    plt.plot(pertFreq, linear_func(pertFreq, *px_i)/1.e6, 'g--', label='fit: a=%5.3f' % tuple(px_i))
    plt.scatter(pertFreq, np.imag(fy)/1.e6, label='fy-data')
    plt.plot(pertFreq, linear_func(pertFreq, *py_i)/1.e6, 'r--', label='fit: a=%5.3f' % tuple(py_i))
    plt.xlabel(r'Pert. Freq. [rad/s]')
    plt.ylabel(r'Im(Force) [MN/m]')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('force_imag.png')
    plt.close()
    
    
if __name__ == "__main__":
    start = time.time()
    main()  
    end = time.time()
    print('runtime [s]')
    print(end-start)
