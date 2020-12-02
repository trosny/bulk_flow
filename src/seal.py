#!/usr/bin/env python
"""
"""

#import sys
import numpy as np
from scipy.sparse import lil_matrix, save_npz
from scipy.sparse.linalg import spsolve, spilu, LinearOperator, gmres, cg, bicg, minres
#import yaml
import time
from seal_funcs import div, k_frene, f_blasius, f_hirs, f_haaland, f_moody, f_universal, \
    forces, sparse_to_full, read_parameters
import mesh
#from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt


class seal(mesh.mesh):
    '''
    seal is subclass of mesh
    inherit all attributes and methods from mesh using 'super' built-in method
    and pass 'params' when init base class
    '''
    def __init__(self, params):
        super(seal, self).__init__(params)
        #mesh(self).__init__(params)
        
        # reference dictionary
        ref_dict =  {'gamma','relax_mode','relax_uv','relax_uv1','relax_p','u_tol','v_tol',\
                     'm_tol','uv_src_method','uv_src_blend','max_it','max_it_pert',\
                     'nCorrectors','u_s','rho_s','mu_s','rpm_rotor','p_in','xi_in',\
                     'rpm_inlet','p_exit','xi_exit','friction','rotor_roughness',\
                     'stator_roughness','whirl_f','read_restart','save_restart',\
                     'print_residuals','print_output','pert_dir','debug_seal'}
        # simple check if required parameters are present, throw KeyError
        # if not. 
        # TODO : - add type checking
        #        - set default value to parameter if not present
        for key in ref_dict:
            if key not in params:
                raise KeyError(f"'{key}' missing from input parameter dictionary")
        
        # additional attributes
        self.gamma = params.get('gamma')
        self.gamma1 = params.get('gamma1')
        self.relax_mode = params.get('relax_mode')
        self.relax_uv = params.get('relax_uv')
        self.relax_uv1 = params.get('relax_uv1')
        self.relax_p = params.get('relax_p')
        self.u_tol = params.get('u_tol')
        self.v_tol = params.get('v_tol')
        self.m_tol = params.get('m_tol')
        self.uv_src_method = params.get('uv_src_method')
        self.uv_src_blend = params.get('uv_src_blend')
        self.max_it= params.get('max_it')
        self.max_it_pert = params.get('max_it_pert')
        self.nCorrectors = params.get('nCorrectors')
        
        self.u_s = params.get('u_s')
        self.rho_s = params.get('rho_s')
        self.mu_s = params.get('mu_s') 
        self.rpm_rotor = params.get('rpm_rotor')
 
        self.p_in = params.get('p_in')
        self.xi_in = params.get('xi_in')
        self.rpm_inlet =  params.get('rpm_inlet')
        self.p_exit =  params.get('p_exit')        
        self.xi_exit =  params.get('xi_exit')

        self.friction = params.get('friction')
        self.rotor_roughness = params.get('rotor_roughness')
        self.stator_roughness = params.get('stator_roughness')
        self.whirl_f = params.get('whirl_f')    
        
        self.read_restart = params.get('read_restart')
        self.save_restart = params.get('save_restart') 
        self.print_residuals = params.get('print_residuals')       
        self.plot_figs = params.get('print_output')
        self.pert_dir = params.get('pert_dir') 
        self.debug_seal = params.get('debug_seal')
        #
        self.restart_seal()
    
    
    def update_seal(self):
        self._seal_params()
        self._init_zeroth_bcs()
        
    def restart_seal(self):
        self.zeroth_converged = 0      
        self._seal_params()
        self._init_var_arrays()
        self._init_zeroth_bcs()
        self._init_zeroth_pressure()
        self._init_zeroth_massflux()
    
    
    def _seal_params(self):
        ''' derived and non-dimensionalized parameters
        '''
        # derived parameters
        self.v_rotor = self.rpm_rotor * 2. * np.pi * self.R / 60.
        self.v_inlet = self.rpm_inlet * 2. * np.pi * self.R / 60.
        self.Omega = self.rpm_rotor * 2. * np.pi / 60.
        #
        self.rho_init = self.rho_s / self.rho_s
        self.p_i = self.p_in / (self.rho_s * self.u_s ** 2)  # non-dim pressure
        self.p_e = self.p_exit / (self.rho_s * self.u_s ** 2)
        self.u_i = 1.0 # 0.0 / u_s  # non-dim
        self.v_i = self.v_inlet / self.u_s
        self.v_r = self.v_rotor / self.u_s
        self.Re = self.rho_s * self.u_s * self.R / self.mu_s # Reynolds number 
        self.sigma = self.whirl_f * self.R / self.u_s
        
    def _init_var_arrays(self): 
        '''initialize variable arrays
        '''        
        # zeroth order problem
        
        self.A = lil_matrix((self.Nc, self.Nc), dtype=np.float64)
        self.A2 = lil_matrix((self.Nc, self.Nc), dtype=np.float64) # debug
        self.Ap = lil_matrix((self.Nc, self.Nc), dtype=np.float64)
        self.apu = np.zeros(self.Nc, dtype=np.float64)
        self.apv = np.zeros(self.Nc, dtype=np.float64) # debug
        if self.debug_seal:
            self.residual = np.zeros(self.Nc, dtype=np.float64) # debug
        self.Dp = np.zeros(self.Nc, dtype=np.float64)
        self.bu = np.zeros(self.Nc, dtype=np.float64)
        self.bv = np.zeros(self.Nc, dtype=np.float64)
        self.bp = np.zeros(self.Nc, dtype=np.float64)

        self.rho = np.ones(self.Nc , dtype=np.float64) * self.rho_init
        self.rhof = np.ones(self.Nf, dtype=np.float64) * self.rho_init
        self.phi = np.zeros(self.Nf, dtype=np.float64)  # face mass flux
        self.Df = np.zeros(self.Nf, dtype=np.float64) 
        self.u = np.ones(self.Nc, dtype=np.float32) * self.u_i
        self.v = np.ones(self.Nc, dtype=np.float64) * self.v_i
        self.u_star = np.zeros(self.Nc, dtype=np.float64) 
        self.v_star = np.zeros(self.Nc, dtype=np.float64)
        self.press = np.zeros(self.Nc, dtype=np.float64)
        self.p_corr = np.zeros(self.Nc, dtype=np.float64)
        self.ubc = np.zeros(self.Nfbc, dtype=np.float64)
        self.vbc = np.zeros(self.Nfbc, dtype=np.float64)
        self.pbc = np.zeros(self.Nfbc, dtype=np.float64)
        self.p_corr_bc = np.zeros(self.Nfbc, dtype=np.float64)
        self.rhobc = np.ones(self.Nfbc, dtype=np.float64) * self.rho_init
        
        self.grad_p = np.zeros([self.Nc, 2], dtype=np.float64)
        self.grad_u = np.zeros([self.Nc, 2], dtype=np.float64)
        self.grad_v = np.zeros([self.Nc, 2], dtype=np.float64)
        self.grad_p_corr = np.zeros([self.Nc, 2], dtype=np.float64)
        
        # first order problem            
        self.u1 = np.zeros(self.Nc, dtype=np.complex128)
        self.v1 = np.zeros(self.Nc, dtype=np.complex128)
        self.press1 = np.zeros(self.Nc, dtype=np.complex128)
        self.p1_corr = np.zeros(self.Nc, dtype=np.complex128)
        self.phi1 = np.zeros(self.Nf, dtype=np.complex128)
        self.u1bc = np.zeros(self.Nfbc , dtype=np.complex128)
        self.v1bc = np.zeros(self.Nfbc, dtype=np.complex128)
        self.p1bc = np.zeros(self.Nfbc, dtype=np.complex128)  
        self.bu1 = np.zeros(self.Nc, dtype=np.complex128)
        self.bv1 = np.zeros(self.Nc, dtype=np.complex128)
        self.bp1 = np.zeros(self.Nc, dtype=np.complex128)    
        self.p1_corr_bc = np.zeros(self.Nfbc, dtype=np.complex128) 
        self.grad_p1 = np.zeros([self.Nc, 2], dtype=np.complex128)
        self.grad_p1_corr = np.zeros([self.Nc, 2], dtype=np.complex128)
      

    def solve_zeroth(self):
        outer_iter = 0
        m_error = 1.0
        u_error = 1.0
        v_error = 1.0
            
        while (outer_iter < self.max_it and ( m_error > self.m_tol or u_error > self.u_tol or v_error > self.u_tol )  ):
        
            self.grad_p = self._cc_grad(self.grad_p, self.press, self.pbc)
            # coefficient matrix A and ap are same for u and v momentum, evaluate once to improve efficiency
            #self.A = self.A.tolil() * 0.0
            self._setup_zeroth_uv()
            self.u_star = spsolve(self.A.tocsr(), self.bu)
            
            if self.uv_src_method == 1:
                self.v_star = spsolve(self.A2.tocsr(), self.bv)
            else: 
                self.v_star = spsolve(self.A.tocsr(), self.bv)            
            
            self.Dp = self.cell[:,2] * self.hc / self.apu
            self._cc_to_int_faces(self.Df, self.Dp)
            self._cc_to_int_faces(self.rhof, self.rho)
            
            # extrapolate inflow and outflow boundaries
            idx = 0
            for i in range(self.Nfstart[0], self.Nf): # loop over boundary faces
                self.rhof[i] = self.rhobc[idx]
                if self.bc_type[idx] == 1 or self.bc_type[idx] == 2:  # inflow or outflow
                    self.Df[i] = self.Dp[self.owner[i]]
                idx += 1
            self.phi = self._massflux_rhiechow(self.phi, self.rho, self.u, \
                             self.v, self.press, self.grad_p, self.rhobc, \
                             self.pbc, self.ubc, self.vbc)  

            self._setup_zeroth_p()
            self.p_corr = spsolve(self.Ap.tocsr(), self.bp)  
            
            #print(self.p_corr)
            #print(self.grad_p_corr[:,0])
            self.grad_p_corr = self._cc_grad(self.grad_p_corr, self.p_corr, self.p_corr_bc)        

            # implicit under-relaxation
            if self.relax_mode == 'implicit':
                self.u = self.u_star - self.Dp * self.grad_p_corr[:,0]
                self.v = self.v_star - self.Dp * self.grad_p_corr[:,1]
            # explicit under-relaxation
            elif self.relax_mode == 'explicit':
                self.u = ( 1. - self.relax_uv ) * self.u + self.relax_uv * ( self.u_star - self.Dp * self.grad_p_corr[:,0] )
                self.v = ( 1. - self.relax_uv ) * self.v + self.relax_uv * ( self.v_star - self.Dp * self.grad_p_corr[:,1] )
            
            self.press = self.press + self.relax_p * self.p_corr
            #print(self.grad_p_corr[:,0])
     
            self._correct_phi(self.phi, self.rho, self.rhobc, self.p_corr, self.p_corr_bc)
                        
            self._update_zeroth_bcs()
            
            for idx in range(self.nCorrectors):
                self.grad_p = self._cc_grad(self.grad_p, self.press, self.pbc)
                self.u_star, self.v_star = self._solve_uv_explicit()

                self.phi = self._massflux_rhiechow(self.phi, self.rho, self.u, \
                             self.v, self.press, self.grad_p, self.rhobc, \
                             self.pbc, self.ubc, self.vbc)  
                             
                self._setup_zeroth_p()
                self.p_corr = spsolve(self.Ap.tocsr(), self.bp)  
            
                self.grad_p_corr = self._cc_grad(self.grad_p_corr, self.p_corr, self.p_corr_bc)        

                # implicit under-relaxation
                if self.relax_mode == 'implicit':
                    self.u = self.u_star - self.Dp * self.grad_p_corr[:,0]
                    self.v = self.v_star - self.Dp * self.grad_p_corr[:,1]
                # explicit under-relaxation
                elif self.relax_mode == 'explicit':
                    self.u = ( 1. - self.relax_uv ) * self.u + self.relax_uv * ( self.u_star - self.Dp * self.grad_p_corr[:,0] )
                    self.v = ( 1. - self.relax_uv ) * self.v + self.relax_uv * ( self.v_star - self.Dp * self.grad_p_corr[:,1] )
                    
                self.press = self.press + self.relax_p * self.p_corr

                self._correct_phi(self.phi, self.rho, self.rhobc, self.p_corr, self.p_corr_bc)

                self._update_zeroth_bcs()
                      
            
            # inlet and outlet mass flux
            m_out = np.sum( self.phi[self.Nfstart[1]:self.Nfstart[2]] )
            m_in = np.sum( self.phi[self.Nfstart[3]:self.Nf] )
            # residuals / errors
            u_error = np.sum( np.abs( self.bu - self.A.dot(self.u) ) )
            if self.uv_src_method == 1:
                v_error = np.sum( np.abs( self.bv - self.A2.dot(self.v) ) ) 
            else:
                v_error = np.sum( np.abs( self.bv - self.A.dot(self.v) ) ) 
            m_error = np.sum(np.abs(self.bp) )
            
            outer_iter += 1
            if self.print_residuals:
                print("{a:g} {b:g} {c:g} {d:g} {e:g} {f:g} ".format(a=outer_iter, b=m_error, c=u_error, d=v_error, e=m_in,
                                                                    f=m_out))
        
        ps_in = np.mean( self.pbc[(self.Nfstart[3] - self.Nfint):self.Nfbc] )
        ps_in = ps_in * self.rho_s * self.u_s ** 2

        self.q = m_out * self.rho_s * self.u_s * self.R  * self.C

        if self.debug_seal:
            self.residual = self.bv - self.A2.dot(self.v)

        fx, fy = forces(self.press, self.cell)
        self.fx = fx * self.rho_s * self.u_s ** 2 * self.R  ** 2
        self.fy = fy * self.rho_s * self.u_s ** 2 * self.R  ** 2

        print('leakage rate kg/s, fx [N], fy [N], inlet p [kPa]')
        print(self.q, self.fx, self.fy, ps_in/1e3)
        
    def solve_first(self):
        # update seal parameters in case whirl frequency changed
        self._seal_params()
    
        # zeroth-order velocity gradients
        self.grad_u = self._cc_grad(self.grad_u, self.u, self.ubc)
        self.grad_v = self._cc_grad(self.grad_v, self.v, self.vbc)
        
        
        outer_iter = 0
        m_error = 1.0
        u_error = 1.0
        v_error = 1.0
            
        while (outer_iter < self.max_it_pert and ( m_error > self.m_tol or u_error > self.u_tol or v_error > self.u_tol )  ):
        
            self.grad_p1 = self._cc_grad(self.grad_p1, self.press1, self.p1bc)

            self._setup_first_uv()
            self.u_star1 = spsolve(self.A.tocsr(), self.bu1)
            self.v_star1 = spsolve(self.A.tocsr(), self.bv1)              
            
            self.phi1 = self._massflux_rhiechow(self.phi1, self.rho, self.u1, \
                             self.v1, self.press1, self.grad_p1, self.rhobc, \
                             self.p1bc, self.u1bc, self.v1bc)  
                                              

            self._setup_first_p()
            self.p1_corr = spsolve(self.Ap.tocsr(), self.bp1)  

            self.grad_p1_corr = self._cc_grad(self.grad_p1_corr, self.p1_corr, self.p1_corr_bc)        

            if self.relax_mode == 'implicit':
                self.u1 = self.u_star1 - self.Dp * self.grad_p1_corr[:,0]
                self.v1 = self.v_star1 - self.Dp * self.grad_p1_corr[:,1]
             # explicit under-relaxation
            elif self.relax_mode == 'explicit':
                self.u1 = ( 1. - self.relax_uv1 ) * self.u1 + self.relax_uv1 * ( self.u_star1 - self.Dp * self.grad_p1_corr[:,0] )
                self.v1 = ( 1. - self.relax_uv1 ) * self.v1 + self.relax_uv1 * ( self.v_star1 - self.Dp * self.grad_p1_corr[:,1] )
            
            self.press1 = self.press1 + self.relax_p * self.p1_corr

     
            self._correct_phi(self.phi1, self.rho, self.rhobc, self.p1_corr, self.p1_corr_bc)
            self._update_first_bcs()            
            
            # residuals / errors
            u_error = np.sum( np.abs( self.bu1 - self.A.dot(self.u1) ) )
            v_error = np.sum( np.abs( self.bv1 - self.A.dot(self.v1) ) ) 
            m_error = np.sum(np.abs(self.bp1) )
            
            outer_iter += 1
            if self.print_residuals:
                print("{a:g} {b:g} {c:g} {d:g} ".format(a=outer_iter, b=m_error, c=u_error, d=v_error))
        
        
        fx1, fy1 = forces(self.press1, self.cell)
        self.fx1 = fx1 * self.rho_s * self.u_s ** 2 * self.R  ** 2 / self.C
        self.fy1 = fy1 * self.rho_s * self.u_s ** 2 * self.R  ** 2 / self.C
        
        # print("Re(f_x1) : {a:g} Im(f_x1) : {b:g}".format(a=np.real(self.fx1)/1e6,b=np.imag(self.fx1)/1e6))
        # print("Re(f_yu1) : {a:g} Im(f_y1) : {b:g}".format(a=np.real(self.fy1)/1e6,b=np.imag(self.fy1)/1e6))
        
            

    def _init_zeroth_bcs(self):
        '''
        '''
        # cycle1 (bottom)
        idx = 0
        for i in range(self.Nfstart[0], self.Nfstart[1]):
            self.ubc[idx] = 1.0  # slip
            self.vbc[idx] = 0.0
            self.pbc[idx] = 0.0
            self.rhobc[idx] = self.rho_init
            idx += 1
        # outflow (right)
        for i in range(self.Nfstart[1], self.Nfstart[2]):
            self.ubc[idx] = 0.0
            self.vbc[idx] = 0.0
            self.pbc[idx] = self.p_e
            self.rhobc[idx] = self.rho_init
            idx += 1
        # cycle2 (top)
        for i in range(self.Nfstart[2], self.Nfstart[3]):
            self.ubc[idx] = 0.0
            self.vbc[idx] = 0.0
            self.pbc[idx] = 0.0
            self.rhobc[idx] = self.rho_init
            idx += 1
        # inflow (left)
        for i in range(self.Nfstart[3], self.Nf):
            self.ubc[idx] = self.u_i
            self.vbc[idx] = self.v_i
            self.pbc[idx] = self.p_i
            self.rhobc[idx] = self.rho_init
            idx += 1

    def _update_zeroth_bcs(self):
        ''' zeroth order bcs
        '''
        sf = self.sf
        owner = self.owner
        cn = self.cn 
        hf = self.hf
        phi = self.phi
        
        idx = 0
        for i in range(self.Nfstart[0], self.Nf):
            p = owner[i]
            if self.bc_type[idx] == 1:  # total pressure inlet
                area = np.sqrt(sf[i, 0] ** 2 + sf[i, 1] ** 2) * hf[i]
                self.ubc[idx] = phi[i] * (cn[i, 0] / np.abs(cn[i, 0])) / area
                self.pbc[idx] = self.p_i - 0.5 * (1.0 + self.xi_in) * np.abs(self.ubc[idx]) ** 2
            if self.bc_type[idx] == 2:  # outlet
                area = np.sqrt(sf[i, 0] ** 2 + sf[i, 1] ** 2) * hf[i]
                self.ubc[idx] =  phi[i] * (cn[i, 0] / np.abs(cn[i, 0])) / area
                self.vbc[idx] = self.v[p]
                self.pbc[idx] = self.p_e - 0.5 * (1.0 - self.xi_exit) * np.abs(self.ubc[idx]) ** 2  # total pressure bc
                
            if self.bc_type[idx] == 0:  # solid wall
                self.pbc[idx] = self.pbc[idx] + self.relax_p * self.ppbc[idx]
            if self.bc_type[idx] == 3 or self.bc_type[idx] == 4:  # cyclic bcs
                self.pbc[idx] = self.pbc[idx] + self.relax_p * self.p_corr_bc[idx]
                self.ubc[idx] = self.u[p]
                self.vbc[idx] = self.v[p]
            idx += 1

    def _update_first_bcs(self):
        ''' first order bcs
        '''
        sf = self.sf
        owner = self.owner
        cn = self.cn 
        hf = self.hf
        phi1 = self.phi1
        
        idx = 0
        for i in range(self.Nfstart[0], self.Nf):
            p = owner[i]
            if self.bc_type[idx] == 1:  # total pressure inlet
                area = np.sqrt(sf[i, 0] ** 2 + sf[i, 1] ** 2) * hf[i]
                #self.u1bc[idx] = phi1[i] * (cn[i, 0] / np.abs(cn[i, 0])) / area
                self.u1bc[idx] = self.u1[p]
                #self.p1bc[idx] = - (1.0 + self.xi_in) * np.abs(self.ubc[idx]) * (np.abs(np.real(self.u1bc[idx])) + np.abs(np.imag(self.u1bc[idx]))*1j)
                self.p1bc[idx] = (1.0 + self.xi_in) * np.abs(self.ubc[idx]) * self.u1bc[idx]
            if self.bc_type[idx] == 2:  # outlet
                area = np.sqrt(sf[i, 0] ** 2 + sf[i, 1] ** 2) * hf[i]
                #self.u1bc[idx] =  phi1[i] * (cn[i, 0] / np.abs(cn[i, 0])) / area
                self.u1bc[idx] = self.u1[p]
                self.v1bc[idx] = self.v1[p]
                self.p1bc[idx] = (1.0 - self.xi_exit) * np.abs(self.ubc[idx]) * self.u1bc[idx]
                
            if self.bc_type[idx] == 0:  # solid wall
                pass
            if self.bc_type[idx] == 3 or self.bc_type[idx] == 4:  # cyclic bcs
                self.p1bc[idx] = self.p1bc[idx] + self.relax_p * self.p1_corr_bc[idx]
                self.u1bc[idx] = self.u1[p]
                self.v1bc[idx] = self.v1[p]
            idx += 1

    def _init_zeroth_pressure(self):
        '''
        Initialize zeroth order pressure field with average of inlet and 
        exit pressures
        '''
        for i in range(self.Nx):
            for j in range(self.Ny):
                m = lambda i, j: i * self.Ny + j
                p = m(i, j)
                # average
                self.press[p] = 0.5 * ( self.p_i + self.p_e )
        for i in range(self.Nfstart[3] - self.Nfint, self.Nf - self.Nfint): # inflow
            self.pbc[i] = self.p_i
        for i in range(self.Nfstart[1] - self.Nfint, self.Nfstart[2] - self.Nfint): # outflow
            self.pbc[i] = self.p_e
            
    def _init_zeroth_massflux(self):
        '''
        Initial computation of mass flux at the faces for zeroth order problem
        Sign based on outward (for owner cell) normal of face
          'out' of owner cell is (+)
          'in' owner cell is (-)
        '''
        rho = self.rho
        gf = self.gf
        u = self.u
        v = self.v
        owner = self.owner
        neighbor = self.neighbor
        hf = self.hf
        sf = self.sf
        ubc = self.ubc
        vbc = self.vbc
        rhobc = self.rhobc
        cyclic = self.cyclic
        for i in range(self.Nfint):
            # linear interpolation of density and velocity to face
            rhof = gf[i] * rho[owner[i]] + (1. - gf[i]) * rho[neighbor[i]]
            uf = gf[i] * u[owner[i]] + (1. - gf[i]) * u[neighbor[i]]
            vf = gf[i] * v[owner[i]] + (1. - gf[i]) * v[neighbor[i]]
            self.phi[i] = hf[i] * rhof * (uf * sf[i, 0] + vf * sf[i, 1])
            # phi[i] = (uf * sf[i, 0] + vf * sf[i, 1])
        idx = 0  # all bcs counter
        idx2 = 0  # cyclic bc counter
        for i in range(self.Nfstart[0], self.Nf):
            if self.bc_type[idx] == 1:  # inflow
                #phi[i] = hf[i] * rhobc[idx] * (ubc[idx] * sf[i, 0] + vbc[idx] * sf[i, 1])
                self.phi[i] = hf[i] * rhobc[idx] * (ubc[idx] * sf[i, 0] )
                # phi[i] = (ubc[idx] * sf[i, 0] + vbc[idx] * sf[i, 1])
                # phi[i] = rho[owner[i]] * (u[owner[i]] * sf[i, 0] + v[owner[i]] * sf[i, 1])
            if self.bc_type[idx] == 2:  # outflow
                self.phi[i] = hf[i] * rhobc[idx] * (ubc[idx] * sf[i, 0] + vbc[idx] * sf[i, 1])
                # phi[i] = (ubc[idx] * sf[i, 0] + vbc[idx] * sf[i, 1])
                # phi[i] = rho[owner[i]] * (u[owner[i]] * sf[i, 0] + v[owner[i]] * sf[i, 1])
            if self.bc_type[idx] == 4:  # cyclic 2
                rhof = gf[i] * rho[owner[i]] + (1. - gf[i]) * rho[cyclic[idx2, 2]]
                uf = gf[i] * u[owner[i]] + (1. - gf[i]) * u[cyclic[idx2, 2]]
                vf = gf[i] * v[owner[i]] + (1. - gf[i]) * v[cyclic[idx2, 2]]
                self.phi[i] = hf[i] * rhof * (uf * sf[i, 0] + vf * sf[i, 1])  # cycle 1
                # phi[i] =  (uf * sf[i, 0] + vf * sf[i, 1]) # cycle 1
                self.phi[cyclic[idx2, 3]] = - self.phi[i]
                idx2 += 1
            idx += 1        
    
    
    def _cc_grad(self, grad_var, var, bvar):
        '''
        evaluate cell-center gradient, Green-Gauss
        '''
        gf = self.gf
        owner = self.owner
        neighbor = self.neighbor
        cyclic = self.cyclic
        sf = self.sf
        cell = self.cell
        
        grad_var = grad_var * 0.0
        
        # interior
        for i in range(self.Nfint):
            # linear interpolation of density and velocity to face
            varf = gf[i] * var[owner[i]] + (1. - gf[i]) * var[neighbor[i]]
            grad_var[owner[i], 0] += varf * sf[i, 0]
            grad_var[owner[i], 1] += varf * sf[i, 1]
            grad_var[neighbor[i], 0] += - varf * sf[i, 0]
            grad_var[neighbor[i], 1] += - varf * sf[i, 1]

        # idx = 0
        # for i in range(Nfstart[0], Nf):
        #     varf = bvar[idx]
        #     grad_var[owner[i], 0] += varf * sf[i, 0]
        #     grad_var[owner[i], 1] += varf * sf[i, 1]
        #     idx += 1

        idx = 0
        idx2 = 0
        for i in range(self.Nfstart[0], self.Nf):
            if self.bc_type[idx] == 0:  # solid wall
                varf = bvar[idx]
                grad_var[owner[i], 0] += varf * sf[i, 0]
                grad_var[owner[i], 1] += varf * sf[i, 1]
            if self.bc_type[idx] == 1:  # inflow
                varf = bvar[idx]
                grad_var[owner[i], 0] += varf * sf[i, 0]
                grad_var[owner[i], 1] += varf * sf[i, 1]
            if self.bc_type[idx] == 2:  # outflow
                varf = bvar[idx]
                grad_var[owner[i], 0] += varf * sf[i, 0]
                grad_var[owner[i], 1] += varf * sf[i, 1]
            if self.bc_type[idx] == 4:  # cyclic 2
                varf = gf[i] * var[owner[i]] + (1. - gf[i]) * var[cyclic[idx2, 2]]
                # cycle 1 face contributions
                grad_var[owner[i], 0] += varf * sf[i, 0]
                grad_var[owner[i], 1] += varf * sf[i, 1]
                # cycle 2 face contributions
                grad_var[cyclic[idx2, 2], 0] += -varf * sf[i, 0]
                grad_var[cyclic[idx2, 2], 1] += -varf * sf[i, 1]
                idx2 += 1
            idx += 1  
            
            # divide by volume after summing face contributions
        grad_var[:, 0] = grad_var[:, 0] / cell[:, 2]
        grad_var[:, 1] = grad_var[:, 1] / cell[:, 2]
        
        return grad_var

    def _compute_friction(self, u, v):
        U_s = np.sqrt(u ** 2 + v ** 2)
        U_r = np.sqrt(u ** 2 + (v - self.v_r) ** 2)
        # use local velocity magnitude and local film thickness to estimate Reynolds numbers
        Re_r = self.rho_s * self.rho * U_r * self.u_s * self.hc * self.C / self.mu_s
        Re_s = self.rho_s * self.rho * U_s * self.u_s * self.hc * self.C / self.mu_s

        if self.friction == 'blasius':
            f_r = f_blasius(Re_r)
            f_s = f_blasius(Re_s)
        elif self.friction == 'hirs':
            f_r = f_hirs(Re_r)
            f_s = f_hirs(Re_s)
        elif self.friction == 'haaland':
            f_r = f_haaland(Re_r , self.rotor_roughness * 2.0 * self.C * np.ones_like(self.hc), self.hc * self.C)
            f_s = f_haaland(Re_s , self.stator_roughness * 2.0 * self.C * np.ones_like(self.hc), self.hc * self.C)
        elif self.friction == 'moody':
            #  Moody from Childs 1993, expects hydraulic diameter scaled Re
            f_r = f_moody(Re_r * 2.0, self.rotor_roughness * 2.0 * self.C * np.ones_like(self.hc), self.hc * self.C)
            f_s = f_moody(Re_s * 2.0, self.stator_roughness * 2.0 * self.C * np.ones_like(self.hc), self.hc * self.C)
        elif self.friction == 'universal':
            f_r = f_universal(Re_r, self.rotor_roughness * 2.0 * self.C * np.ones_like(self.hc), self.hc * self.C)
            f_s = f_universal(Re_s, self.stator_roughness * 2.0 * self.C * np.ones_like(self.hc), self.hc * self.C)
            
        return U_r, U_s, f_r, f_s
    
    def _setup_zeroth_uv(self):
        '''
        Coefficient matrix and source terms for zeroth-order momentum equations
        '''
        owner = self.owner
        neighbor = self.neighbor
        gf = self.gf
        phi = self.phi
        cyclic = self.cyclic
        hc = self.hc
        
        self.A = self.A * 0.0
        self.A2 = self.A2 * 0.0
        self.bu = self.bu * 0.0
        self.bv = self.bv * 0.0
        self.apu = self.apu * 0.0
        self.apv = self.apv * 0.0

        # fluxes
        for i in range(self.Nfint):
            p = owner[i]
            nb = neighbor[i]
            # convective fluxes
            # first-order upwind
            #fluxp = 0.5 * phi[i] * (np.sign(phi[i]) + 1.)
            #fluxnb = -0.5 * phi[i] * (np.sign(phi[i]) - 1.)
            # mixed upwind/linear
            fluxp = phi[i] * ((1. - self.gamma) * 0.5 * (np.sign(phi[i]) + 1.) + self.gamma * gf[i])
            fluxnb = phi[i] * ((-1. + self.gamma) * 0.5 * (np.sign(phi[i]) - 1.) + self.gamma * (1. - gf[i]))
            # owner
            self.A[p, p] += fluxp
            self.A[p, nb] += fluxnb
            # neighbor
            self.A[nb, p] += -fluxp
            self.A[nb, nb] += -fluxnb
            
            self.A2[p, p] += fluxp
            self.A2[p, nb] += fluxnb
            # neighbor
            self.A2[nb, p] += -fluxp
            self.A2[nb, nb] += -fluxnb
            

        # convective fluxes at inflow and outflow, diffusive fluxes at solid walls
        idx = 0
        idx2 = 0
        for i in range(self.Nfstart[0], self.Nf):
            p = owner[i]
            if self.bc_type[idx] == 1:  # inflow
                self.bu[p] += - phi[i] * self.ubc[idx]   # convective flux
                self.bv[p] += - phi[i] * self.vbc[idx]  # convective flux
                #self.A[p, p] +=  phi[i] # issue with stability
                #self.A2[p, p] +=  phi[i]
            if self.bc_type[idx] == 2:  # outflow
                #bu[p] +=  - phi[i] * ubc[idx]  # convective flux
                #bv[p] +=  - phi[i] * vbc[idx]  # convective flux
                self.A[p, p] +=  phi[i]  
                self.A2[p, p] +=  phi[i]
            if self.bc_type[idx] == 0:  # solid wall
                fluxp = 1. / self.Re * hf[i] * (div(sf[i, 0], cn[i, 0]) + div(sf[i, 1], cn[i, 1])) 
                # owner
                self.A[p, p] += fluxp
                self.bu[p] += fluxp * self.ubc[idx]
                self.bv[p] += fluxp * self.vbc[idx]
                # ap[p] += fluxp
            if self.bc_type[idx] == 4:  # cyclic 2
                nb = cyclic[idx2, 2]
                # convective
                fluxp = phi[i] * ((1. - self.gamma) * 0.5 * (np.sign(phi[i]) + 1) + self.gamma * gf[i]) 
                fluxnb = phi[i] * ((-1. + self.gamma) * 0.5 * (np.sign(phi[i]) - 1) + self.gamma * (1. - gf[i])) 

                self.A[p, p] += fluxp
                self.A[p, nb] += fluxnb
                # neighbor
                self.A[nb, p] += -fluxp
                self.A[nb, nb] += -fluxnb
                
                self.A2[p, p] += fluxp
                self.A2[p, nb] += fluxnb
                # neighbor
                self.A2[nb, p] += -fluxp
                self.A2[nb, nb] += -fluxnb

                idx2 += 1

            idx += 1
            
        #self.A2 = self.A

        U_r, U_s, f_r, f_s = self._compute_friction(self.u, self.v)
        
        m = -0.25

        # source terms
        for i in range(self.Nc):
            self.bu[i] += - hc[i] * self.grad_p[i, 0] * self.cell[i, 2]  # pressure
            self.bv[i] += - hc[i] * self.grad_p[i, 1] * self.cell[i, 2]  # pressure
            # friction factor formulation

            if self.uv_src_method == 0:
                flux = - 0.5 * (self.R / self.C) * self.rho[i] * ( U_r[i] * f_r[i] + U_s[i] * f_s[i]) * self.cell[i, 2] 
                self.A[i, i] += - self.uv_src_blend * flux  # implicit portion
                self.A2[i, i] += - self.uv_src_blend * flux  # implicit portion
                self.bu[i] += (1. - self.uv_src_blend) * flux * self.u[i]  # explicit portion
                self.bv[i] += (1. - self.uv_src_blend) * flux * self.v[i]
                self.bv[i] += 0.5 * (self.R / self.C) * self.v_r * self.rho[i] * (U_r[i] * f_r[i]) * self.cell[i, 2] 

            
            elif self.uv_src_method == 1:
                sx = - 0.5 * (self.R / self.C) * self.rho[i] * ( U_r[i] * f_r[i] + U_s[i] * f_s[i]) * self.u[i] 
                sy = - 0.5 * (self.R / self.C) * self.rho[i] * ( U_r[i] * f_r[i] + U_s[i] * f_s[i]) * self.v[i] \
                    + 0.5 * (self.R / self.C) * self.v_r * self.rho[i] * (U_r[i] * f_r[i]) 
                #sy = - 0.5 * (self.R / self.C) * self.rho[i] * ( U_r[i] * f_r[i] + U_s[i] * f_s[i]) * self.v[i] 
                
                dsx_dvx =  - 0.5 * (self.R / self.C) * self.rho[i] * ( 
                - (1. + m) * f_r[i] * self.u[i] **2 / U_r[i]  \
                + f_r[i] * self.u[i] +  \
                - (1. + m) * f_s[i] * self.u[i] **2 / U_s[i]  \
                + f_s[i] * self.u[i] )
                
                dsy_dvy =  - 0.5 * (self.R / self.C) * self.rho[i] * ( 
                - (1. + m) * f_r[i] * self.v[i] * (self.v[i] - self.v_r ) / U_r[i]  \
                + f_r[i] * self.v[i] +  \
                - (1. + m) * f_s[i] * self.v[i] * (self.v[i] - self.v_r ) / U_s[i]  \
                + f_s[i] * self.v[i] ) + \
                0.5 * (self.R / self.C) * self.rho[i] * self.v_r * ( \
                - (1. + m) * f_r[i] * (self.v[i] - self.v_r ) / U_r[i] )  
                         
                
                Scx = (sx - dsx_dvx * self.u[i])
                Scy = (sy - dsy_dvy * self.v[i])   
                
                self.A[i, i] += - self.uv_src_blend * dsx_dvx * self.cell[i, 2] 
                self.A2[i,i] += - self.uv_src_blend * dsy_dvy * self.cell[i, 2] 
                self.bu[i] += (1. - self.uv_src_blend) * Scx * self.cell[i, 2]  
                self.bv[i] += (1. - self.uv_src_blend) * Scy * self.cell[i, 2] 
                #self.bv[i] += 0.5 * (self.R / self.C) * self.v_r * self.rho[i] * (U_r[i] * f_r[i]) * self.cell[i, 2] 

            if self.relax_mode == 'implicit':
                self.A[i, i] = self.A[i, i] / self.relax_uv  # relax main diagonal
                self.A2[i, i] = self.A2[i, i] / self.relax_uv
        

        self.apu = self.A.diagonal(0) #ap / param['relax_uv']
        self.apv = self.A2.diagonal(0) #ap / param['relax_uv']
        
        if self.relax_mode == 'implicit':
            self.bu += (1. - self.relax_uv) * self.apu * self.u  # relaxation has been applied to ap, i.e. ap = ap / relax
            self.bv += (1. - self.relax_uv) * self.apv * self.v

    def _setup_first_uv(self):
        '''
        Source terms for first-order momentum equations
        '''

        hc = self.hc
        rho = self.rho
        u = self.u
        v = self.v
        u1 = self.u1
        v1 = self.v1
        grad_p = self.grad_p
        grad_u = self.grad_u
        grad_v = self.grad_v
        cell = self.cell
        
        self.bu1 = 0.0 * self.bu1
        self.bv1 = 0.0 * self.bv1

        U_r, U_s, f_r, f_s = self._compute_friction(self.u, self.v)
        
        m = -0.25

        # source terms
        for i in range(self.Nc):
            self.bu1[i] += - hc[i] * self.grad_p1[i, 0] * cell[i, 2]  # pressure
            self.bv1[i] += - hc[i] * self.grad_p1[i, 1] * cell[i, 2]  # pressure
            
            if self.pert_dir == 'X':
                h_psi = np.cos(cell[i,1])
            if self.pert_dir == 'Y':
                h_psi = np.sin(cell[i,1])
            
            if self.uv_src_method == 0:
                flux = - 0.5 * (self.R / self.C) * self.rho[i] * ( U_r[i] * f_r[i] + U_s[i] * f_s[i]) * self.cell[i, 2] 
                
                self.bu1[i] += (1. - self.uv_src_blend) * flux * self.u1[i]  # explicit portion
                self.bv1[i] += (1. - self.uv_src_blend) * flux * self.v1[i]
            
                if self.friction == 'blasius':

                    self.bu1[i] += (- rho[i] * hc[i] * u1[i] * grad_u[i, 0]  \
                             - rho[i] * hc[i] * v1[i] * grad_u[i, 1]  \
                             - rho[i] * h_psi * u[i] * grad_u[i, 0]  \
                             - rho[i] * h_psi * v[i] * grad_u[i, 1] \
                             - grad_p[i, 0] * h_psi \
                             - 0.5 * (self.R / self.C) * rho[i] * u[i] * ( \
                             + (1. + m) * f_r[i] / U_r[i] * ( u[i] * u1[i] + (v[i] - self.v_r) * v1[i] ) \
                             + (1. + m) * f_s[i] / U_s[i] * ( u[i] * u1[i] + v[i] * v1[i] ) \
                             + U_s[i] * m * f_s[i] * h_psi / hc[i]  \
                             + U_r[i] * m * f_r[i] * h_psi / hc[i] ) \
                             + 1j * self.sigma * rho[i] * hc[i] * u1[i] ) * cell[i,2]                                        
                    
                    self.bv1[i] += ( - rho[i] * hc[i] * u1[i] * grad_v[i, 0]  \
                             - rho[i] * hc[i] * v1[i] * grad_v[i, 1]  \
                             - rho[i] * h_psi * u[i] * grad_v[i, 0]  \
                             - rho[i] * h_psi * v[i] * grad_v[i, 1]  \
                             - grad_p[i, 1] * h_psi  \
                             - 0.5 * (self.R / self.C) * rho[i] * v[i]  * ( \
                             + (1. + m) * f_r[i]  / U_r[i] * ( u[i] * u1[i] + (v[i] - self.v_r) * v1[i] ) \
                             + (1. + m) * f_s[i]  / U_s[i] * ( u[i] * u1[i] + v[i] * v1[i] ) \
                             + U_s[i] * m * f_s[i] * h_psi /  hc[i] \
                             + U_r[i] * m * f_r[i] * h_psi / hc[i] ) \
                             + 1j * self.sigma * rho[i] * hc[i] * v1[i] ) * cell[i,2]
                   
                    # # Note that param['v_r'] = param['R'] * param['Omega'] / param['u_s']           
                       
                    self.bv1[i] +=  0.5 * (self.R * self.v_r) / (self.C) * rho[i] * cell[i, 2] * ( \
                             + (1. + m) * f_r[i] / U_r[i] * ( u[i] * u1[i] + (v[i] - self.v_r) * v1[i]) \
                             + m * U_r[i] * f_r[i] * h_psi / hc[i] )                           
                             
                elif param['friction'] == 'moody':
                    pass            
                
                
            elif self.uv_src_method == 1:
                raise ValueError(f"uv_src_method = 1 is not implemented for first-order problem, \
                use uv_src_method = 0")
            
            
            

     
        if self.relax_mode == 'implicit':
            self.bu1 += (1. - self.relax_uv) * self.apu * self.u1  # relaxation has been applied to ap, i.e. ap = ap / relax
            self.bv1 += (1. - self.relax_uv) * self.apu * self.v1

    def _solve_uv_explicit(self):
        '''
        explicit solution of zeroth-order momentum equations
        '''
        owner = self.owner
        neighbor = self.neighbor
        gf = self.gf
        phi = self.phi
        cyclic = self.cyclic
        hc = self.hc
        
        #self.bu = self.bu * 0.0
        #self.bv = self.bv * 0.0
        #self.apu = self.apu * 0.0
        
        bu = np.zeros(self.Nc, dtype=np.float64)
        bv = np.zeros(self.Nc, dtype=np.float64)
        ap = np.zeros(self.Nc, dtype=np.float64)
        if self.uv_src_method == 1:
            ap2 = np.zeros(self.Nc, dtype=np.float64)
        
        u = self.u
        v = self.v
        phi = self.phi

        # fluxes
        for i in range(self.Nfint):
            p = owner[i]
            nb = neighbor[i]

            # convective fluxes
            fluxp = phi[i] * ((1. - self.gamma) * 0.5 * (np.sign(phi[i]) + 1.) + self.gamma * gf[i])
            fluxnb = phi[i] * ((-1. + self.gamma) * 0.5 * (np.sign(phi[i]) - 1.) + self.gamma * (1. - gf[i]))
            # 
            ap[p] += fluxp
            ap[nb] += -fluxnb

            bu[p] += -fluxnb * u[nb]
            bu[nb] += fluxp * u[p]
            bv[p] += -fluxnb * v[nb]
            bv[nb] += fluxp * v[p]


        # convective fluxes at inflow and outflow, diffusive fluxes at solid walls
        idx = 0
        idx2 = 0
        for i in range(self.Nfstart[0], self.Nf):
            p = owner[i]
            if self.bc_type[idx] == 1:  # inflow
                bu[p] += - phi[i] * self.ubc[idx]  # convective flux
                bv[p] += - phi[i] * self.vbc[idx]  # convective flux
                #ap[p] +=  phi[i] # issue with stability
            if self.bc_type[idx] == 2:  # outflow
                #bu[p] +=  - phi[i] * ubc[idx]  # convective flux
                #bv[p] +=  - phi[i] * vbc[idx] #ubc[idx]  # convective flux
                ap[p] +=  phi[i]
            if self.bc_type[idx] == 0:  # solid wall
                fluxp = 1. / self.Re * hf[i] * (div(sf[i, 0], cn[i, 0]) + div(sf[i, 1], cn[i, 1]))
                # owner
                #A[p, p] += fluxp
                bu[p] += fluxp * self.ubc[idx]
                bv[p] += fluxp * self.vbc[idx]
                ap[p] += fluxp

            if self.bc_type[idx] == 4:  # cyclic 2
                nb = cyclic[idx2, 2]
                # convective
                fluxp = phi[i] * ((1. - self.gamma) * 0.5 * (np.sign(phi[i]) + 1.) + self.gamma * gf[i])
                fluxnb = phi[i] * ((-1. + self.gamma) * 0.5 * (np.sign(phi[i]) - 1.) + self.gamma * (1. - gf[i]))
                #
                ap[p] += fluxp
                ap[nb] += -fluxnb

                bu[p] += - fluxnb * u[nb]
                bu[nb] +=  fluxp * u[p]
                bv[p] += - fluxnb * v[nb]
                bv[nb] += fluxp * v[p]

                idx2 += 1

            idx += 1


        if self.uv_src_method == 1:
            ap2 = ap
        
        U_r, U_s, f_r, f_s = self._compute_friction(u, v)

        for i in range(self.Nc):
            bu[i] += - hc[i] * self.grad_p[i, 0] * self.cell[i, 2]  # pressure
            bv[i] += - hc[i] * self.grad_p[i, 1] * self.cell[i, 2]  # pressure
            # friction factor formulation
            
            
            m = -0.25

            if self.uv_src_method == 0:
                flux = - 0.5 * (self.R / self.C) * self.rho[i] * ( U_r[i] * f_r[i] + U_s[i] * f_s[i]) * self.cell[i, 2] 
                ap[i] += - self.uv_src_blend * flux  # implicit portion
                bu[i] += (1. - self.uv_src_blend) * flux * self.u[i]  # explicit portion
                bv[i] += (1. - self.uv_src_blend) * flux * self.v[i]
                bv[i] += 0.5 * (self.R / self.C) * self.v_r * self.rho[i] * (U_r[i] * f_r[i]) * self.cell[i, 2] 

            
            elif self.uv_src_method == 1:
                sx = - 0.5 * (self.R / self.C) * self.rho[i] * ( U_r[i] * f_r[i] + U_s[i] * f_s[i]) * self.u[i] 
                sy = - 0.5 * (self.R / self.C) * self.rho[i] * ( U_r[i] * f_r[i] + U_s[i] * f_s[i]) * self.v[i] \
                    + 0.5 * (self.R / self.C) * self.v_r * self.rho[i] * (U_r[i] * f_r[i]) 
                #sy = - 0.5 * (self.R / self.C) * self.rho[i] * ( U_r[i] * f_r[i] + U_s[i] * f_s[i]) * self.v[i] 
                
                dsx_dvx =  - 0.5 * (self.R / self.C) * self.rho[i] * ( 
                - (1. + m) * f_r[i] * self.u[i] **2 / U_r[i]  \
                + f_r[i] * self.u[i] +  \
                - (1. + m) * f_s[i] * self.u[i] **2 / U_s[i]  \
                + f_s[i] * self.u[i] )
                
                dsy_dvy =  - 0.5 * (self.R / self.C) * self.rho[i] * ( 
                - (1. + m) * f_r[i] * self.v[i] * (self.v[i] - self.v_r ) / U_r[i]  \
                + f_r[i] * self.v[i] +  \
                - (1. + m) * f_s[i] * self.v[i] * (self.v[i] - self.v_r ) / U_s[i]  \
                + f_s[i] * self.v[i] ) + \
                0.5 * (self.R / self.C) * self.rho[i] * self.v_r * ( \
                - (1. + m) * f_r[i] * (self.v[i] - self.v_r ) / U_r[i] )  
                         
                
                Scx = (sx - dsx_dvx * self.u[i])
                Scy = (sy - dsy_dvy * self.v[i])   
                
                ap[i] += - self.uv_src_blend * dsx_dvx * self.cell[i, 2] 
                ap2[i] += - self.uv_src_blend * dsy_dvy * self.cell[i, 2] 
                bu[i] += (1. - self.uv_src_blend) * Scx * self.cell[i, 2]  
                bv[i] += (1. - self.uv_src_blend) * Scy * self.cell[i, 2] 
     

            # flux = - 0.5 * (self.R / self.C) * self.rho[i] * ( U_s[i] * f_s[i] + U_r[i] * f_r[i] ) * self.cell[i, 2] 
            # ap[i] += - self.uv_src_blend * flux  # implicit portion
            # bu[i] += (1. - self.uv_src_blend) * flux * u[i]  # explicit portion
            # bv[i] += (1. - self.uv_src_blend) * flux * v[i]
            # bv[i] += 0.5 * (self.R / self.C) * self.rho[i] * (U_r[i] * f_r[i]) * self.v_r * self.cell[i, 2] 

        #self.u_star = bu / self.apu
        #self.v_star = bv / self.apu
        self.u_star = bu / ap
        
        if self.uv_src_method == 1:
            self.v_star = bv / ap2
        else:
            self.v_star = bv / ap
        #us = bp/ap + b/ap  + (1. - relax_u) * (ap) * u
        return self.u_star, self.v_star

    
    def _cc_to_int_faces(self, valf, val):
        '''
        function interpolates cc values to interior faces
        '''
        # grad_val = cc_gradient_int( val,Nc,Nf,Nfint,Nfstart,owner,neighbor,sf,gf,cyclic,cell )

        for i in range(self.Nfint):
            valf[i] = self.gf[i] * val[self.owner[i]] + (1. - self.gf[i]) * val[self.neighbor[i]]
        idx = 0  # all bcs counter
        idx2 = 0  # cyclic bc counter
        for i in range(self.Nfstart[0], self.Nf):
            if self.bc_type[idx] == 4:  # cyclic 2
                valf[i] = self.gf[i] * val[self.owner[i]] + (1. - self.gf[i]) * val[self.cyclic[idx2, 2]]
                valf[self.cyclic[idx2, 3]] = valf[i]
                idx2 += 1
            idx += 1

    def _massflux_rhiechow(self, phi, rho, u, v, press, gradp, rhobc, pbc, ubc, vbc):
        '''
        Evaluate mass flux at the faces
        Sign based on outward (for owner cell) normal of face
          'out' of owner cell is (+)
          'in' owner cell is (-)
        For information regarding Rhie and Chow implementation, consult...
          1.  "Computational methods for fluid dynamics," Ferziger and Peric, 2002
              pp. 196 - 201
              see eqns. (7.132) and (7.133) for explictly corrected u* and u' equations
              where u* and u' are the intermediate and correction velocities, respectively
          2.  "The finite volume method in CFD", Moukalled et al., Springer, 2016.
              pp. 585 - 587, general
              Notably, the authors provide detailed implementation of momentum boundary
              conditions starting on p. 602 and bcs for the pressure correction equation
              starting on p. 617
              pp. 630 - 636 describes the treatment of under-relaxation, transient, and body
              force terms within a Rhie-Chow "framework"
          3.  "Note on the use of momentum interpolation method for unsteady flows,"
              Choi, S.K., Num. Heat Transfer, vol. 36, pp. 545-550, 1990.
              Note that Choi 1999 noted the dependency of the Rhie-Chow solution on relaxation
              factor and treatment of the unsteady term.  The author provided an alternative
              form for the cell-face velocities (used within the pressure correction equation),
              a similar expression embodied in Eq. (15.216) on p. 636 in Moukalled et al.

        '''

        sf = self.sf
        gf = self.gf
        owner = self.owner
        neighbor = self.neighbor
        cn = self.cn
        cyclic = self.cyclic
        cell = self.cell
        Df = self.Df
        Dp = self.Dp
        hf = self.hf

        for i in range(self.Nfint):
            # linear interpolation of density and velocity to face
            idxp = owner[i]
            idxnb = neighbor[i]
            rhof = gf[i] * rho[owner[i]] + (1. - gf[i]) * rho[idxnb]
            uf = gf[i] * u[owner[i]] + (1. - gf[i]) * u[neighbor[i]]
            vf = gf[i] * v[owner[i]] + (1. - gf[i]) * v[neighbor[i]]
            gradpf_x = gf[i] * gradp[idxp, 0] + (1. - gf[i]) * gradp[idxnb, 0]
            gradpf_y = gf[i] * gradp[idxp, 1] + (1. - gf[i]) * gradp[idxnb, 1]
            coeff = Df[i]
            rx = cell[idxnb, 0] - cell[idxp, 0]
            ry = cell[idxnb, 1] - cell[idxp, 1]
            uadj = coeff * (div((press[idxnb] - press[idxp]), (cell[idxnb, 0] - cell[idxp, 0])) - gradpf_x)
            vadj = coeff * (div((press[idxnb] - press[idxp]), (cell[idxnb, 1] - cell[idxp, 1])) - gradpf_y)  
            # 
            phi[i] = hf[i] * rhof * (uf * sf[i, 0] + vf * sf[i, 1]) - hf[i] * rhof * (uadj * sf[i, 0] + vadj * sf[i, 1])

        # inflow and outflow boundaries
        idx = 0  # all bcs counter
        idx2 = 0  # cyclic bc counter
        for i in range(self.Nfstart[0], self.Nf):
            p = owner[i]
            if self.bc_type[idx] == 1:  # inflow
                rhof = rhobc[idx]
                uf = ubc[idx]
                vf = vbc[idx]
                coeff = Dp[p]
                # average pressure gradient at face taken to be pressure gradient
                # in cell
                gradpf_x = gradp[p, 0]
                gradpf_y = gradp[p, 1]
                uadj = coeff * (div((pbc[idx] - press[p]), (cn[i, 0])) - gradpf_x)
                vadj = coeff * (div((pbc[idx] - press[p]), (cn[i, 1])) - gradpf_y)
                # 
                phi[i] = hf[i] * rhof * (uf * sf[i, 0] + vf * sf[i, 1]) - hf[i] * rhof * (uadj * sf[i, 0] + vadj * sf[i, 1])

            if self.bc_type[idx] == 2:  # outflow
                rhof = rhobc[idx]
                uf = ubc[idx]
                vf = vbc[idx]
                coeff = Dp[p]
                gradpf_x = gradp[p, 0]
                gradpf_y = gradp[p, 1]
                uadj = coeff * (div((pbc[idx] - press[p]), (cn[i, 0])) - gradpf_x)
                vadj = coeff * (div((pbc[idx] - press[p]), (cn[i, 1])) - gradpf_y)
                # 
                phi[i] = hf[i] * rhof * (uf * sf[i, 0] + vf * sf[i, 1]) - hf[i] * rhof * (uadj * sf[i, 0] + vadj * sf[i, 1])
            if self.bc_type[idx] == 4:  # cyclic 2
                idxp = owner[i]
                idxnb = cyclic[idx2, 2]
                uf = gf[i] * u[idxp] + (1. - gf[i]) * u[idxnb]
                vf = gf[i] * v[idxp] + (1. - gf[i]) * v[idxnb]
                rhof = gf[i] * rho[idxp] + (1. - gf[i]) * rho[idxnb]
                gradpf_x = gf[i] * gradp[idxp, 0] + (1. - gf[i]) * gradp[idxnb, 0]
                gradpf_y = gf[i] * gradp[idxp, 1] + (1. - gf[i]) * gradp[idxnb, 1]
                coeff = Df[i]
                rx = - cn[cyclic[idx2, 3], 0] + cn[i, 0]
                ry = - cn[cyclic[idx2, 3], 1] + cn[i, 1]
                uadj = coeff * (div((press[idxnb] - press[idxp]), rx) - gradpf_x)
                vadj = coeff * (div((press[idxnb] - press[idxp]), ry) - gradpf_y)
                #
                phi[i] = hf[i] * rhof * (uf * sf[i, 0] + vf * sf[i, 1]) - hf[i] * rhof * (
                            uadj * sf[i, 0] + vadj * sf[i, 1])
                phi[cyclic[idx2, 3]] = - phi[i]
                idx2 += 1
            idx += 1
        return phi

    def _setup_zeroth_p(self):
        '''
        coefficient matrix and source term for zeroth order pressure correction
        '''
        sf = self.sf
        gf = self.gf
        owner = self.owner
        neighbor = self.neighbor
        cn = self.cn
        cyclic = self.cyclic
        cell = self.cell
        Df = self.Df
        Dp = self.Dp
        hf = self.hf
        phi = self.phi
        rhof = self.rhof
        
        self.Ap = self.Ap * 0.0
        self.bp = self.bp * 0.0
        
        # interior
        for i in range(self.Nfint):
            p = owner[i]
            nb = neighbor[i]
            fluxp = rhof[i] * Df[i] * hf[i] * (div(sf[i, 0], (cell[nb, 0] - cell[p, 0])) +
                                                div(sf[i, 1], (cell[nb, 1] - cell[p, 1])))
            fluxnb = -fluxp
            # owner        
            self.Ap[p, p] += fluxp
            self.Ap[p, nb] += fluxnb
            # neighbor
            self.Ap[nb, p] += -fluxp
            self.Ap[nb, nb] += -fluxnb
            self.bp[p] += - phi[i] 
            self.bp[nb] += phi[i] 

        idx = 0
        idx2 = 0
        for i in range(self.Nfstart[0], self.Nf):
            p = owner[i]
            if self.bc_type[idx] == 1:  # inflow
                fluxp = rhof[i] * Df[i] * hf[i] * (div(sf[i, 0], cn[i, 0]) + div(sf[i, 1], cn[i, 1])) 
                self.Ap[p, p] += fluxp
                self.bp[p] += - phi[i] 
            if self.bc_type[idx] == 2:  # outflow, specified pressure
                fluxp = rhof[i] * Df[i] * hf[i] * (div(sf[i, 0], cn[i, 0]) + div(sf[i, 1], cn[i, 1])) 
                self.Ap[p, p] += fluxp
                self.bp[p] += - phi[i] 
            if self.bc_type[idx] == 0:  # solid wall, mass flux and mass flux corrections are zero
                pass
            if self.bc_type[idx] == 4:  # cyclic 2
                nb = cyclic[idx2, 2]
                rx = - cn[cyclic[idx2, 3], 0] + cn[i, 0]
                ry = - cn[cyclic[idx2, 3], 1] + cn[i, 1]
                fluxp = rhof[i] * Df[i] * hf[i] * (div(sf[i, 0], rx) + div(sf[i, 1], ry)) 
                fluxnb = -fluxp
                # owner
                self.Ap[p, p] += fluxp
                self.Ap[p, nb] += fluxnb
                # neighbor
                self.Ap[nb, p] += -fluxp
                self.Ap[nb, nb] += -fluxnb
                #
                self.bp[p] += -phi[i]
                self.bp[nb] += phi[i]
                #
                idx2 += 1
            idx += 1
            
    def _setup_first_p(self):
        '''
        coefficient matrix and source term for zeroth order pressure correction
        '''
        owner = self.owner
        neighbor = self.neighbor
        cyclic = self.cyclic
        cell = self.cell
        
        phi1 = self.phi1
        self.bp1 = self.bp1 * 0.0
        
        # interior
        for i in range(self.Nfint):
            p = owner[i]
            nb = neighbor[i]
            self.bp1[p] += - phi1[i] 
            self.bp1[nb] += phi1[i] 

        idx = 0
        idx2 = 0
        for i in range(self.Nfstart[0], self.Nf):
            p = owner[i]
            if self.bc_type[idx] == 1:  # inflow
                self.bp1[p] += - phi1[i] 
            if self.bc_type[idx] == 2:  # outflow, specified pressure
                self.bp1[p] += - phi1[i] 
            if self.bc_type[idx] == 0:  # solid wall, mass flux and mass flux corrections are zero
                pass
            if self.bc_type[idx] == 4:  # cyclic 2
                nb = cyclic[idx2, 2]
                #
                self.bp1[p] += -phi1[i]
                self.bp1[nb] += phi1[i]
                #
                idx2 += 1
            idx += 1    
            
        for i in range(self.Nc):
            if self.pert_dir == 'X':
                h_psi = np.cos(cell[i,1])
                h_psi_grad = - np.sin(cell[i,1]) 
            if self.pert_dir == 'Y':
                h_psi = np.sin(cell[i,1])
                h_psi_grad = np.cos(cell[i,1]) 
            #
            self.bp1[i] += (- self.rho[i] * h_psi * self.grad_u[i,0]  \
                    - self.rho[i] * h_psi * self.grad_v[i,1]  \
                    - self.rho[i] * self.v[i] * h_psi_grad  \
                    + 1j * self.sigma * self.rho[i] * h_psi ) * cell[i,2]  
            #self.bp1[i] = - self.bp1[i]         
            

    def _correct_phi(self, phi, rho, rhobc, p_corr, ppbc):
        '''
        Apply corrections to face "mass" fluxes
        '''
        sf = self.sf
        gf = self.gf
        owner = self.owner
        neighbor = self.neighbor
        cn = self.cn
        cyclic = self.cyclic
        cell = self.cell
        Df = self.Df
        Dp = self.Dp
        hf = self.hf

        for i in range(self.Nfint):
            idxp = owner[i]
            idxnb = neighbor[i]
            rhof = gf[i] * rho[owner[i]] + (1. - gf[i]) * rho[idxnb]
            # pressure correction gradient at cell face
            dpx = div((p_corr[idxnb] - p_corr[idxp]), (cell[idxnb, 0] - cell[idxp, 0]))
            dpy = div((p_corr[idxnb] - p_corr[idxp]), (cell[idxnb, 1] - cell[idxp, 1]))
            phi[i] += - hf[i] * rhof * Df[i] * (dpx * sf[i, 0] + dpy * sf[i, 1])
        idx = 0  # all bcs counter
        idx2 = 0  # cyclic bc counter
        for i in range(self.Nfstart[0], self.Nf):
            p = owner[i]
            if self.bc_type[idx] == 1:  # inflow
                #pass
                rhof = rhobc[idx]
                dpx = div((ppbc[idx] - p_corr[p]), ( cn[i, 0] ))
                dpy = div((ppbc[idx] - p_corr[p]), (cn[i, 1]  ))
                phi[i] += - hf[i] * rhof * Df[i] * (dpx * sf[i, 0] + dpy * sf[i, 1])
            if self.bc_type[idx] == 2:  # outflow
                rhof = rhobc[idx]
                dpx = div((ppbc[idx] - p_corr[p]), ( cn[i, 0] ))
                dpy = div((ppbc[idx] - p_corr[p]), (cn[i, 1]  ))
                phi[i] += - hf[i] * rhof * Df[i] * (dpx * sf[i, 0] + dpy * sf[i, 1])
            if self.bc_type[idx] == 4:  # cyclic 2
                idxp = owner[i]
                idxnb = cyclic[idx2, 2]
                rhof = gf[i] * rho[idxp] + (1. - gf[i]) * rho[idxnb]
                dpx = div((p_corr[idxnb] - p_corr[idxp]), (cell[idxnb, 0] - cell[idxp, 0]))
                dpy = div((p_corr[idxnb] - p_corr[idxp]), (cell[idxnb, 1] - cell[idxp, 1]))
                phi[i] += - hf[i] * rhof * Df[i] * (dpx * sf[i, 0] + dpy * sf[i, 1])
                phi[cyclic[idx2, 3]] +=  hf[i] * rhof * Df[i] * (dpx * sf[i, 0] + dpy * sf[i, 1])
                idx2 += 1
            idx += 1
        return phi

    def plot_res(self):
        self.cell[:, 0] = self.cell[:, 0] * self.R
        self.cell[:, 1] = self.cell[:, 1] * self.R
        self.cell[:, 2] = self.cell[:, 2] * self.R ** 2
    
        self.hc = self.hc * self.C
        self.u = self.u * self.u_s
        self.v = self.v * self.u_s
        self.press = self.press * self.rho_s * self.u_s ** 2
      
        x_cc, y_cc, u_cc = sparse_to_full(self.Nx, self.Ny, self.u, self.cell)
        x_cc, y_cc, v_cc = sparse_to_full(self.Nx, self.Ny, self.v, self.cell)
        x_cc, y_cc, p_cc = sparse_to_full(self.Nx, self.Ny, self.press, self.cell)
        x_cc, y_cc, h_cc = sparse_to_full(self.Nx, self.Ny, self.hc, self.cell)
        

        plt.figure()
        # density = 1 is 30 x 30 grid
        plt.streamplot(x_cc[:, 0], y_cc[0, :], u_cc.T, v_cc.T, density=1, linewidth=0.5, arrowsize=0.5)
        plt.gca().set_aspect("equal")
        plt.xlabel(r'Axial position [m]')
        plt.ylabel(r'Circumferential position [m]')
        plt.title(r'Flow streamlines')
        plt.savefig('streamlines.png')
        plt.close()

        plt.figure()
        plt.contourf(x_cc, y_cc, h_cc*1e6)
        plt.colorbar()
        plt.gca().set_aspect("equal")
        plt.xlabel(r'Axial position [m]')
        plt.ylabel(r'Circumferential position [m]')
        plt.title(r'Film thickness [\mu m]')
        plt.tight_layout()
        plt.savefig('film_thickness_contour.png')
        plt.close()

        plt.figure()
        plt.contourf(x_cc, y_cc, p_cc/1e5)
        plt.colorbar()
        plt.gca().set_aspect("equal")
        plt.xlabel(r'Axial position [m]')
        plt.ylabel(r'Circumferential position [m]')
        plt.title(r'Pressure [bars]')
        plt.tight_layout()
        plt.savefig('pressure_contour.png')
        plt.close()

        plt.figure()
        plt.contourf(x_cc, y_cc, u_cc)
        plt.colorbar()
        plt.gca().set_aspect("equal")
        plt.xlabel(r'Axial position [m]')
        plt.ylabel(r'Circumferential position [m]')
        plt.title(r'Axial velocity [m/s]')
        plt.tight_layout()
        plt.savefig('u_contour.png')
        plt.close()

        plt.figure()
        plt.contourf(x_cc, y_cc, v_cc)
        plt.colorbar()
        plt.gca().set_aspect("equal")
        plt.xlabel(r'Axial position [m]')
        plt.ylabel(r'Circumferential position [m]')
        plt.title(r'Circumferential velocity [m/s]')
        plt.tight_layout()
        plt.savefig('v_contour.png')
        plt.close()
        
        if self.debug_seal:
            x_cc, y_cc, r_cc = sparse_to_full(self.Nx, self.Ny, self.residual, self.cell)
            plt.figure()
            plt.contourf(x_cc, y_cc, r_cc)
            plt.colorbar()
            plt.gca().set_aspect("equal")
            plt.xlabel(r'Axial position [m]')
            plt.ylabel(r'Circumferential position [m]')
            plt.title(r'Residuals')
            plt.tight_layout()
            plt.savefig('residuals.png')
            plt.close()
    
    
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
        
