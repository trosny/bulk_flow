#!/usr/bin/env python
"""
 val02
 - validation case
 - "Short" water seal (incompressible flow)
 - Kanki and Kawakami, IMechE paper, pp.159-166, 1984

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

    # functions used if curve-fitting    
    def linear_func(x, b):
        # damping, linear w.r.t. frequency
        return ( b * x)
    
    def quad_func(x, a, b):
        # dynamic stiffness, added mass term exhibits frequency^2 behavior
        return (a + b * x ** 2.0)
    
    # experimental results
    q_exp = 9047.0 # leakage [cm^3/s]   
    
    
    K_exp = np.array([[3.96,0.664],[-0.337,4.01]])
    D_exp = np.array([[24.82,12.3],[-10.88,24.46]])
    M_exp = np.array([[0.0,0.0],[0.0,0.0]])
    K = np.zeros((2,2))
    D = np.zeros((2,2))
    M = np.zeros((2,2))
    
    # Nx = np.array([5, 8, 10, 15, 20, 25, 30])
    # Ny = np.array([10, 15, 20, 30, 40, 50, 70])
    
    Nx = np.array([5])
    Ny = np.array([20])
   
    
    # arrays for plotting
    Kxx_p = np.zeros(Nx.size)
    Kxy_p = np.zeros(Nx.size)
    Dxx_p = np.zeros(Nx.size)
    Dxy_p = np.zeros(Nx.size)
    Mxx_p = np.zeros(Nx.size)
    q_p = np.zeros(Nx.size)
    
    param = read_parameters('Kanki02_input.yaml')
    s = seal(param)
    
    #pert_dirs = ['X','Y'] # for testing symmetry
    pert_dirs = ['X']
    #pertFreq = np.array([200.0])
    pertFreq = np.array([0.0, 25.0, 50.0, 100.0])
    fx = np.zeros(pertFreq.size, dtype=np.complex128)
    fy = np.zeros(pertFreq.size, dtype=np.complex128)
    
    for idx,nx in enumerate(Nx):
        
        s.Nx = Nx[idx]
        s.Ny = Ny[idx]
        print(s.Nx)
        print(s.Ny)
        s.update_mesh()
        s.restart_seal()
        s.solve_zeroth()
        
        q_p[idx] = s.q/s.rho_s*1.e6
    
        for pert in pert_dirs:
            s.pert_dir = pert
            #s.update_seal() # need to clean up seal updates, this overwrites zeroth bcs which impacts stiffness
            print(s.pert_dir)
        
            # arrays for saving complex forces
            #fx = fx * 0.0
            #fy = fy * 0.0
            
            for idx_freq, freq in enumerate(pertFreq):
                s.whirl_f = freq     
                s.solve_first()
                fx[idx_freq] = s.fx1
                fy[idx_freq] = s.fy1
                print(freq)


            # curve-fit real and imaginary parts of forces
            # fitting coefficients are the dynamic coefficients    
            px_r,cov = optimize.curve_fit(quad_func, pertFreq, np.real(fx), p0=[0.0, 0.0]) 
            py_r,cov = optimize.curve_fit(quad_func, pertFreq, np.real(fy), p0=[0.0, 0.0])     
            px_i,cov = optimize.curve_fit(linear_func, pertFreq, np.imag(fx), p0=[0.0]) 
            py_i,cov = optimize.curve_fit(linear_func, pertFreq, np.imag(fy), p0=[0.0]) 
            
            if pert == 'X':
                K[0,0] = px_r[0]/1.e6
                K[1,1] = K[0,0]
                K[1,0] = py_r[0]/1.e6
                K[0,1] = -K[1,0]
                D[0,0] = px_i[0]/1.e3
                D[1,1] = D[0,0]
                D[1,0] = -py_i[0]/1.e3
                D[0,1] = -D[1,0]
                M[0,0] = px_r[1]
                M[1,1] = M[0,0]
                M[1,0] = py_r[1]
                M[0,1] = -M[1,0]
            elif pert == 'Y':
                K[0,1] = px_r[0]/1.e6
                K[1,1] = py_r[0]/1.e6
                D[0,1] = px_i[0]/1.e3
                D[1,1] = py_i[0]/1.e3
                M[0,1] = px_r[1]
                M[1,1] = py_r[1]
           
        Kxx_p[idx] = K[0,0]
        Kxy_p[idx] = K[0,1]
        Dxx_p[idx] = D[0,0]
        Dxy_p[idx] = D[0,1]
        Mxx_p[idx] = M[0,0]


  
    print('----q [cm^3/s]-------')
    print('--predicted--')
    print(q_p[0])
    print('--Exp.--')
    print(q_exp)  
    print('----K [MN/m]-------')
    print('--predicted--')
    print(K)
    print('--Exp.--')
    print(K_exp)
    print('----D [kN.s/m]-------')
    print('--predicted--')
    print(D)
    print('--Exp.--')
    print(D_exp)
    print('----M [kg]-------')
    print('--predicted--')
    print(M)
    print('--Exp.--')
    print(M_exp)
    print('---------------')
    
    
    # if param["pert_dir"] == 'X':
    # # # print  leakage and dynamic coefficients
        # print('---Values (model | exp)---')
        # print("leakage [cm^3/s] : {a:g} | {b:g}".format(a=s.q/s.rho_s*1.e6, b=q_exp)) 
        # print("K_xx [MN/m]      : {a:g} | {b:g}".format(a=px_r[0]/1.e6, b=K_exp[0,0])) 
        # print("K_yx [MN/m]      : {a:g} | {b:g}".format(a=py_r[0]/1.e6, b=K_exp[1,0])) 
        # print("D_xx [kN.s/m]    : {a:g} | {b:g}".format(a=px_i[0]/1.e3, b=D_exp[0,0])) 
        # print("D_yx [kN.s/m]    : {a:g} | {b:g}".format(a=py_i[0]/1.e3, b=D_exp[1,0]))   
        # print("M_xx [kg]        : {a:g} | {b:g}".format(a=px_r[1], b=M_exp[0,0])) 
        # print("M_yx [kg]        : {a:g} | {b:g}".format(a=py_r[1], b=M_exp[1,0]))        
        
        # # # print relative errors in leakage and dynamic coefficients
        # # # print('---Relative errors---')
        # # # print("leakage [%] : {a:g}".format(a=(s.q/s.rho_s*1.e6-q_exp) / q_exp * 100.0 )) 
        # # # print("K_xx [%]    : {a:g}".format(a=(px_r[0]/1.e6-kxx_exp) / kxx_exp * 100.0 )) 
        # # # print("K_yx [%]    : {a:g}".format(a=(py_r[0]/1.e6+kxy_exp) / kyx_exp * 100.0 )) 
        # # # print("D_xx [%]    : {a:g}".format(a=(px_i[0]/1.e3-dxx_exp) / dxx_exp * 100.0 )) 
        # # # print("D_yx [%]    : {a:g}".format(a=(py_i[0]/1.e3+dxy_exp) / dyx_exp * 100.0 ))   
        # # # print("M_xx [%]    : {a:g}".format(a=(px_r[1]-mxx_exp) / mxx_exp * 100.0 ))  
        
    # elif param["pert_dir"] == 'Y':
        # print('---Values (model | exp)---')
        # print("leakage [cm^3/s] : {a:g} | {b:g}".format(a=s.q/s.rho_s*1.e6, b=q_exp)) 
        # print("K_xy [MN/m]      : {a:g} | {b:g}".format(a=px_r[0]/1.e6, b=K_exp[0,1])) 
        # print("K_yy [MN/m]      : {a:g} | {b:g}".format(a=py_r[0]/1.e6, b=K_exp[1,1])) 
        # print("D_xy [kN.s/m]    : {a:g} | {b:g}".format(a=px_i[0]/1.e3, b=D_exp[0,1])) 
        # print("D_yy [kN.s/m]    : {a:g} | {b:g}".format(a=py_i[0]/1.e3, b=D_exp[1,1]))   
        # print("M_xy [kg]        : {a:g} | {b:g}".format(a=px_r[1], b=M_exp[0,1]))
        # print("M_yy [kg]        : {a:g} | {b:g}".format(a=py_r[1], b=M_exp[1,1]))        
        # print relative errors in leakage and dynamic coefficients
        # print('---Relative errors---')
        # print("leakage [%] : {a:g}".format(a=(s.q/s.rho_s*1.e6-q_exp) / q_exp * 100.0 )) 
        # print("K_xx [%]    : {a:g}".format(a=(px_r[0]/1.e6-kxx_exp) / kxx_exp * 100.0 )) 
        # print("K_yx [%]    : {a:g}".format(a=(py_r[0]/1.e6-kyx_exp) / kyx_exp * 100.0 )) 
        # print("D_xx [%]    : {a:g}".format(a=(px_i[0]/1.e3-dxx_exp) / dxx_exp * 100.0 )) 
        # print("D_yx [%]    : {a:g}".format(a=(py_i[0]/1.e3-dyx_exp) / dyx_exp * 100.0 ))   
        # print("M_xx [%]    : {a:g}".format(a=(px_r[1]-mxx_exp) / mxx_exp * 100.0 ))  

          
 
    # plt.figure()
    # plt.plot(Nx*Ny, q_p)
    # plt.axhline(y=q_exp, color='r', linestyle='-')
    # plt.xlabel(r'$N_x \times N_y$')
    # plt.ylabel(r'$Leakage [cm$^3$/s]')   
    # plt.tight_layout()
    # plt.savefig('q_grid.png')
    # plt.close()  
 
    # plt.figure()
    # plt.plot(Nx*Ny, Kxx_p)
    # plt.axhline(y=np.mean(K_exp[0,0]), color='r', linestyle='-')
    # plt.xlabel(r'$N_x \times N_y$')
    # plt.ylabel(r'$K_{xx}=K_{yy}$ [MN/m]')   
    # plt.tight_layout()
    # plt.savefig('Kxx_grid.png')
    # plt.close()  
    
    # plt.figure()
    # plt.plot(Nx*Ny, Kxy_p)
    # plt.axhline(y=np.mean(K_exp[0,1]), color='r', linestyle='-')
    # plt.xlabel(r'$N_x \times N_y$')
    # plt.ylabel(r'$K_{xy}=-K_{yx}$ [MN/m]')
    # plt.tight_layout()
    # plt.savefig('Kxy_grid.png')
    # plt.close()  
    
    # plt.figure()
    # plt.plot(Nx*Ny, Dxx_p)
    # plt.axhline(y=np.mean(D_exp[0,0]), color='r', linestyle='-')
    # plt.xlabel(r'$N_x \times N_y$')
    # plt.ylabel(r'$D_{xx}=D_{yy}$ [kN.s/m]')
    # plt.tight_layout()
    # plt.savefig('Dxx_grid.png')
    # plt.close()
    
    # plt.figure()
    # plt.plot(Nx*Ny, Dxy_p)
    # plt.axhline(y=np.mean(D_exp[0,1]), color='r', linestyle='-')
    # plt.xlabel(r'$N_x \times N_y$')
    # plt.ylabel(r'$D_{xy}=-D_{yx}$ [kN.s/m]')
    # plt.tight_layout()
    # plt.savefig('Dxy_grid.png')
    # plt.close()
    
    # plt.figure()
    # plt.plot(Nx*Ny, Mxx_p)
    # plt.axhline(y=np.mean(M_exp[0,0]), color='r', linestyle='-')
    # plt.xlabel(r'$N_x \times N_y$')
    # plt.ylabel(r'$M_{xx}=M_{yy}$ [kg]')
    # plt.tight_layout()
    # plt.savefig('Mxx_grid.png')
    # plt.close()
      
      
    # generate figures of forces as a function of perturbation frequency
    # include data and curve-fits
    # curve-fit coefficients correspond to dynamic coefficients
    # plt.figure()
    # plt.scatter(pertFreq, np.real(fx)/1.e6, label='fx-data')
    # plt.plot(pertFreq, quad_func(pertFreq, *px_r)/1.e6, 'g--', label='fit: a=%5.3f, b=%5.3f' % tuple(px_r))
    # plt.scatter(pertFreq, np.real(fy)/1.e6, label = 'fy-data')
    # plt.plot(pertFreq, quad_func(pertFreq, *py_r)/1.e6, 'r--', label='fit: a=%5.3f, b=%5.3f' % tuple(py_r))
    # plt.xlabel(r'Pert. Freq. [rad/s]')
    # plt.ylabel(r'Re(Force) [MN/m]')
    # plt.legend(loc='best')
    # plt.tight_layout()
    # plt.savefig('force_real.png')
    # plt.close()
    
    # plt.figure()
    # plt.scatter(pertFreq, np.imag(fx)/1.e6, label='fx-data')
    # plt.plot(pertFreq, linear_func(pertFreq, *px_i)/1.e6, 'g--', label='fit: a=%5.3f' % tuple(px_i))
    # plt.scatter(pertFreq, np.imag(fy)/1.e6, label='fy-data')
    # plt.plot(pertFreq, linear_func(pertFreq, *py_i)/1.e6, 'r--', label='fit: a=%5.3f' % tuple(py_i))
    # plt.xlabel(r'Pert. Freq. [rad/s]')
    # plt.ylabel(r'Im(Force) [MN/m]')
    # plt.legend(loc='best')
    # plt.tight_layout()
    # plt.savefig('force_imag.png')
    # plt.close()
    
    
if __name__ == "__main__":
    start = time.time()
    main()  
    end = time.time()
    print('runtime [s]')
    print(end-start)
