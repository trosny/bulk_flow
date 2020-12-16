#!/usr/bin/env python
# seal_funcs

import numpy as np
import yaml

def div(n, d):
    # return zero if division by zero
    return n / d if d else 0

def k_frene(Re):
    '''
    friction 'k' parameters from Frene, Arghir, and Constantinescu 2006
    
    sample numpy style doc
    
    Parameters
    ----------
    foo : int, float, str, or tf.Tensor
        The foo to bar, which has a really really, reeeeeeeeeeeeeeeeally
        unnecessarily long multiline description.
    bar : str
        Bar to use on foo
        :
        :class: `numpy.ndarray`
    baz : float
        Baz to frobnicate

    Returns
    -------
    float
        The frobnicated baz
    
    '''
    k_p = np.zeros_like(Re); k_c = np.zeros_like(Re)
    for i, val in enumerate(Re):
        k_p[i] = 0.0 * 1.0 / 6.8 * ( (div(1.0, Re[i])) ** 0.681 ) # form to handle division by zero
        k_c[i]= 12.0 + 0.0044 * Re[i] ** 0.96
    return k_p, k_c

def f_blasius(Re, n, m):
    '''
    Blasius fanning friction factor / skin friction coefficient
    Matches Black/Yamada equation exactly in the absence of rotation, i.e. the rotational
    Reynolds number is zero.
    Valid for smooth pipes for Re up to 100,000
    '''
    n = n  # 0.316 / 4.0
    m = m
    f = np.zeros_like(Re)
    for i, val in enumerate(Re):
        f[i] = n * ( Re[i] ) ** m
    # divide by 4 to return skin friction coefficient rather than Darcy friction factor
    # cf = f/4.0
    return f

def f_hirs(Re):
    '''
    Hirs friction factor formula.  Same functional form as Blasius.
    Re length scale is film thickness, h.
    See eqn.6f on p. 140 in Hirs 1973
    See n and m values following eqn. 13 b on p. 143 in Hirs 1973
    See also Table 1 in Hirs 1970 thesis
    
    Obsolete -- Can just use blasius and with change to m and n coefficients
    '''
    n = 0.066
    m = -0.25
    f = np.zeros_like(Re)
    for i, val in enumerate(Re):
        f[i] = n * ( Re[i] ) ** m  # form to handle division by zero
    # divide by 4 to return skin friction coefficient rather than Darcy friction factor
    # cf = f/4.0
    return f

def f_haaland(Re,ar,h):
    '''
    Haaland equation approximation to Moody friction factor

    Re : array, Reynolds number, dimensionless
    ar : array, absolute surface roughness, [m]
    h  : array, film thickness, [m]
    '''
    f = np.zeros_like(Re)
    for i, val in enumerate(Re):
        # Haaland equation, factor of 0.25 to return fanning rather than Darcy friction factor
        f[i] = 0.25 * ( -1.8 * np.log10( div(ar[i], 3.7 * 2.0 * h[i]) ** 1.11 + div(6.9, Re[i]) ) ) ** (-2.0)
        # Swamee-Jain
        #f[i] = 0.25 * 0.25 * ( np.log10(div(rr, 3.7)  + div(5.47, Re[i] ** 0.9) )) ** (-2.0)
    # return fanning friction factor
    return f

def f_moody(Re,ar,h):
    '''
    Moody friction factor, Childs 1993, eq. 4.15, p. 235
    Expression matches eqs. 9 and 10 on p. 193 in Nelson and Nguyen, 1987
    Note that Nelson and Nguyen, 1987 determined an optimized set of Hirs 
    parameters n = 0.0148 and m = -0.03971 to match smooth Moody.  They used 
    Re based a length scale of 2*h in both their definitions of Hirs and Moody 
    friction factor formulas.
    What is the original source for this specific form of the friction factor
    equation? "Mechanics of Fluids" by Massey (see p. 254) is referenced by 
    Nelson but the Haaland equation is presented.
    
    #
    Reynolds length scale based on hydraulic diameter, 2*h expected
    Re : array, Reynolds number, dimensionless
    ar : array, absolute surface roughness, [m]
    h  : array, film thickness, [m]
    '''
    a1 = 1.375e-3
    b2 = 2e4
    b3 = 10**6
    f = a1 * ( 1.0 + ( ( b2 * ar / (2.0 * h) ) + (b3 / Re) ) ** (1. / 3.) ) # vectorized
    # f = np.zeros_like(Re)
    # for i, val in enumerate(Re):
    #     f[i] = a1 * (1. + ( div(b2*ar[i],2.0*h[i]) + div(b3, Re[i]) )  ** (1./3.)  )
    return f

def f_universal(Re,ar,h):
    '''
    Universal friction factor formula
    Artiles, ICYl/IFACE documentation, NASA CR 2004-213199 VOL 4
    Zirkelback and San Andres, 1996
    Re based on h
    #
    Reynolds length scale based on hydraulic diameter, 2*h expected
    Re : array, Reynolds number, dimensionless
    ar : array, absolute surface roughness, [m]
    h  : array, film thickness, [m]
    '''
    f = np.zeros_like(Re)
    # a1 = 1.375e-3
    # b2 = 1e4
    # b3 = 10**6
    a1 = 1.375e-3
    b2 = 2e4
    b3 = 10 ** 6
    f_moody = a1 * (1.0 + ((b2 * ar / (2.0 * h)) + (b3 / Re)) ** (1. / 3.))  # vectorized
    for i, val in enumerate(Re):
        #f_moody = a1 * (1. + ( div(b2*ar[i],h[i]) + div(b3, 2.0 * Re[i]) )  ** (1./3.)  )
        if Re[i] <= 1000.0:
            f[i] = 12.0 / Re[i]
        elif Re[i]  > 1000.0 and Re[i] <= 3000.0:
            xi = ( Re[i] - 1000.0 ) / 2000.0
            f[i] = 12.0 / Re[i] * (1.0 - 3.0 * xi **2 + 2.0 * xi ** 3) + f_moody[i] * (3.0 * xi ** 2 - 2.0 * xi ** 3)
        elif Re[i] > 3000.0:
            f[i] = f_moody[i]
    return f

def forces(press, cell):
    fx = 0.0
    fy = 0.0
    for i, val in enumerate(press):
        fx += press[i] * np.cos(cell[i,1]) * cell[i,2]
        fy += press[i] * np.sin(cell[i,1]) * cell[i,2]
    return fx, fy
    
# def forces_shear(press, cell):
    # fx = 0.0;
    # fy = 0.0
    # for i, val in enumerate(press):
        # fx += press[i] * np.cos(cell[i,1]) * cell[i,2]
        # fy += press[i] * np.sin(cell[i,1]) * cell[i,2]
    # return fx, fy    

def sparse_to_full(Nx, Ny, phi, cell):
    phicc = np.zeros([Nx, Ny])
    xcc = np.zeros([Nx, Ny])
    ycc = np.zeros([Nx, Ny])
    for i in range(Nx):
        for j in range(Ny):
            mc = lambda i, j: i * (Ny) + j
            phicc[i, j] = phi[mc(i, j)]
            xcc[i, j] = cell[mc(i, j), 0]
            ycc[i, j] = cell[mc(i, j), 1]
    return xcc, ycc, phicc

def read_parameters(filename):
    """
    read parameter file
    """
    with open(filename) as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        if yaml.__version__ >= '5.1':
            paramDict = yaml.load(file, Loader=yaml.FullLoader)
        else:
            paramDict = yaml.load(file)
    return paramDict

