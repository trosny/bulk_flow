#!/usr/bin/env python
"""
"""

import numpy as np
import yaml

def div(n, d):
    # return zero if division by zero
    return n / d if d else 0

def film(param, x, y):
    #h = 1.0 + param['er'] * np.cos(y)
    # with misalignment, see Snyder thesis for equation
    # re-visit
    # if param['transient']:
        # if param['whirl_type'] == 'Circular':
            # #ex = param['er'] + param['whirl_a'] * np.cos( param['whirl_f'] * param['t'] )
            # #ey = param['whirl_a'] * np.sin( param['whirl_f'] * param['t']  )
            # ex = param['whirl_a'] * np.cos(param['t'])
            # ey = param['whirl_a'] * np.sin( param['t'])
        # elif param['whirl_type'] == 'X':
            # ex = param['er'] + param['whirl_a'] * np.sin(param['t'] )
            # ey = 0.0
        # elif param['whirl_type'] == 'Y':
            # ex = param['er']
            # ey = param['whirl_a'] * np.sin(param['t'] )
        # else:
            # raise SystemExit('Invalid "whirl_type" entered.  Case-sensitive available options are "Circular", "X", or "Y"')
    # else:
    
    ex = param['ex']
    ey = param['ey']
    
    att_ang = param['t']  # np.arctan( ey / ex )   
    
    # with axial misalignment
    # h = 1.0 + ( ex + param['angX'] * (x - param['Lx'] / 2.0 ) * param['R'] / param['C'] ) * np.cos( y ) + \
    #           ( ey + param['angY'] * (x - param['Lx'] / 2.0 ) * param['R'] / param['C'] ) * np.sin( y )
    h = 1.0 + ex  * np.cos( y ) + ey  * np.sin( y )
    xdot = param['whirl_a'] * np.sin(param['t'])
    ydot = - param['whirl_a'] * np.cos(param['t'])
    return h, att_ang, xdot, ydot

def dfilm(param, x, y, att_ang, xdot, ydot):
    '''
    compute squeeze term, dhdt
    '''
    # dh = 0.0
    if param['transient']:
        if param['whirl_type'] == 'Circular':
            #dh = (xdot * np.cos(att_ang) + ydot * np.sin(att_ang) ) * np.cos(y) + \
            #     (xdot * np.sin(att_ang) - ydot * np.cos(att_ang) ) * np.sin(y)
            dh = xdot * np.cos(y) + ydot * np.sin(y)
            #vt = param['whirl_a'] * param['whirl_f'] * param['C'] / param['u_s'] * np.abs( np.cos( y - param['t'] ) )
            # vt = (param['R'] - param['whirl_a'] * param['C']) * param['whirl_f'] / param['u_s'] * np.maximum(np.cos(y - param['t']),0.0) + \
                 # (param['R'] + param['whirl_a'] * param['C']) * param['whirl_f'] / param['u_s'] * np.maximum(
                # -np.cos(y - param['t']), 0.0)
            vt = 0.0    
            # dh = - param['whirl_f'] * param['whirl_a'] * np.sin(param['whirl_f'] * param['t']) * np.cos(y) + \
            #      param['whirl_f'] * param['whirl_a'] * np.cos(param['whirl_f'] * param['t']) * np.sin(y)
        elif param['whirl_type'] == 'X':
            dh = param['whirl_a'] * np.cos( param['t'] ) * np.cos(y)
            vt =0.0 # placeholder
        elif param['whirl_type'] == 'Y':
            dh = param['whirl_a'] * np.cos( param['t'] ) * np.sin(y)
            vt =0.0 # placeholder
    else:
        dh = 0.0
        vt = 0.0
    return dh, vt

def compute_points(param, x, y):
    '''
    rectangular grid
    '''
    points = np.zeros([param['Np'], 2], dtype=np.double)
    #hn = np.zeros(param['Np'], dtype=np.double)
    for i in range(param['Nx'] + 1):
        for j in range(param['Ny'] + 1):
            m = lambda i, j: i * (param['Ny'] + 1) + j
            p = m(i, j)
            points[p, 0] = x[i]
            points[p, 1] = y[j]
            #hn[p] = film(param, x[i], y[j])
    return points

def compute_mesh(param, points):
    '''
    '''
    idxg = 0
    faces = np.zeros([param['Nf'], 2], dtype=int)
    sf = np.zeros([param['Nf'], 2], dtype=np.double)
    owner = np.zeros(param['Nf'], dtype=int)
    neighbor = np.zeros(param['Nfint'], dtype=int)
    cell = np.zeros([param['Nc'], 3], dtype=np.double)  # store cell center x, cell center y, and cell volume
    cyclic = np.zeros([param['Nfcycle1'], 4], dtype=int)  # owner (north) cell, owner face, nb cell, nb face
    #hc = np.zeros(param['Nc'], dtype=np.double)
    #hf = np.zeros(param['Nf'], dtype=np.double)  # average film thickness for face, approximate
    # populate interior faces, loop over cells
    for i in range(param['Nx']):
        for j in range(param['Ny']):
            # global node and cell indices increment differently
            m = lambda i, j: i * (param['Ny'] + 1) + j
            mc = lambda i, j: i * (param['Ny']) + j
            #
            xsw = points[m(i, j), 0]
            ysw = points[m(i, j), 1]
            xnw = points[m(i, j + 1), 0]
            ynw = points[m(i, j + 1), 1]
            xse = points[m(i + 1, j), 0]
            yse = points[m(i + 1, j), 1]
            xne = points[m(i + 1, j + 1), 0]
            yne = points[m(i + 1, j + 1), 1]
            xf = [xsw, xse, xne, xnw, xsw]
            yf = [ysw, yse, yne, ynw, ysw]
            vol = 0.5 * ((xne - xsw) * (ynw - yse) - (yne - ysw) * (xnw - xse))
            xc = 0.0
            yc = 0.0
            for idx in range(0, 4):
                xc += (xf[idx] + xf[idx + 1]) * (xf[idx] * yf[idx + 1] - xf[idx + 1] * yf[idx])
                yc += (yf[idx] + yf[idx + 1]) * (xf[idx] * yf[idx + 1] - xf[idx + 1] * yf[idx])
            xc = xc / (6. * vol)
            yc = yc / (6. * vol)
            #h = film(param, xc, yc)
            cell[mc(i, j), 0] = xc
            cell[mc(i, j), 1] = yc
            cell[mc(i, j), 2] = vol
            #hc[mc(i, j)] = h
            #
            if j == param['Ny'] - 1 and i == param['Nx'] - 1:
                pass
            elif i == param['Nx'] - 1:
                faces[idxg, 0] = m(i + 1, j + 1)
                faces[idxg, 1] = m(i, j + 1)
                owner[idxg] = mc(i, j)  # owner cell id
                neighbor[idxg] = mc(i, j + 1)
                sf[idxg, 0] = ynw - yne  # n_x
                sf[idxg, 1] = -(xnw - xne)  # n_y
                #hf[idxg] = 0.5 * (hn[m(i + 1, j + 1)] + hn[m(i, j + 1)])
                idxg += 1
            elif j == param['Ny'] - 1:
                faces[idxg, 0] = m(i + 1, j)
                faces[idxg, 1] = m(i + 1, j + 1)
                owner[idxg] = mc(i, j)  # owner cell id
                neighbor[idxg] = mc(i + 1, j)
                sf[idxg, 0] = yne - yse  # e_x
                sf[idxg, 1] = -(xne - xse)  # e_y
                #hf[idxg] = 0.5 * (hn[m(i + 1, j)] + hn[m(i + 1, j + 1)])
                idxg += 1
            else:
                faces[idxg, 0] = m(i + 1, j)
                faces[idxg, 1] = m(i + 1, j + 1)
                faces[idxg + 1, 0] = m(i + 1, j + 1)
                faces[idxg + 1, 1] = m(i, j + 1)
                owner[idxg] = mc(i, j)  # owner cell id
                neighbor[idxg] = mc(i + 1, j)
                owner[idxg + 1] = mc(i, j)  # owner cell id
                neighbor[idxg + 1] = mc(i, j + 1)
                sf[idxg, 0] = yne - yse  # e_x
                sf[idxg, 1] = -(xne - xse)  # e_y
                sf[idxg + 1, 0] = ynw - yne  # n_x
                sf[idxg + 1, 1] = -(xnw - xne)  # n_y
                #hf[idxg] = 0.5 * (hn[m(i + 1, j)] + hn[m(i + 1, j + 1)])
                #hf[idxg + 1] = 0.5 * (hn[m(i + 1, j + 1)] + hn[m(i, j + 1)])
                idxg += 2
    # boundaries
    # bc 1
    j = 0
    for i in range(param['Nx']):
        m = lambda i, j: i * (param['Ny'] + 1) + j
        mc = lambda i, j: i * (param['Ny']) + j
        xsw = points[m(i, j), 0];
        ysw = points[m(i, j), 1];
        # xnw = points[m(i,j+1), 0];   ynw = points[m(i,j+1), 1];
        xse = points[m(i + 1, j), 0];
        yse = points[m(i + 1, j), 1];
        # xne = points[m(i+1,j+1), 0];   yne = points[m(i+1,j+1), 1];
        faces[idxg, 0] = m(i, j)
        faces[idxg, 1] = m(i + 1, j)
        owner[idxg] = mc(i, j)  # owner cell id
        sf[idxg, 0] = yse - ysw  # s_x
        sf[idxg, 1] = -(xse - xsw)  # s_y
        #hf[idxg] = 0.5 * (hn[m(i, j)] + hn[m(i + 1, j)])
        cyclic[i, 2] = owner[idxg]
        cyclic[i, 3] = idxg
        idxg += 1
    # bc 2
    i = param['Nx'] - 1
    for j in range(param['Ny']):
        m = lambda i, j: i * (param['Ny'] + 1) + j
        mc = lambda i, j: i * (param['Ny']) + j
        # xsw = points[m(i,j), 0];     ysw = points[m(i,j), 1];
        # xnw = points[m(i,j+1), 0];   ynw = points[m(i,j+1), 1];
        xse = points[m(i + 1, j), 0];
        yse = points[m(i + 1, j), 1];
        xne = points[m(i + 1, j + 1), 0];
        yne = points[m(i + 1, j + 1), 1];
        faces[idxg, 0] = m(i + 1, j)
        faces[idxg, 1] = m(i + 1, j + 1)
        owner[idxg] = mc(i, j)  # owner cell id
        sf[idxg, 0] = yne - yse  # e_x
        sf[idxg, 1] = -(xne - xse)  # e_y
        #hf[idxg] = 0.5 * (hn[m(i + 1, j)] + hn[m(i + 1, j + 1)])
        idxg += 1
        # bc 3
    j = param['Ny'] - 1
    for i in range(param['Nx']):
        m = lambda i, j: i * (param['Ny'] + 1) + j
        mc = lambda i, j: i * (param['Ny']) + j
        # xsw = points[m(i,j), 0];     ysw = points[m(i,j), 1];
        xnw = points[m(i, j + 1), 0];
        ynw = points[m(i, j + 1), 1];
        # xse = points[m(i+1,j), 0];     yse = points[m(i+1,j), 1];
        xne = points[m(i + 1, j + 1), 0];
        yne = points[m(i + 1, j + 1), 1];
        faces[idxg, 0] = m(i + 1, j + 1)
        faces[idxg, 1] = m(i, j + 1)
        owner[idxg] = mc(i, j)  # owner cell id
        sf[idxg, 0] = ynw - yne  # n_x
        sf[idxg, 1] = -(xnw - xne)  # n_y
        #hf[idxg] = 0.5 * (hn[m(i + 1, j + 1)] + hn[m(i, j + 1)])
        cyclic[i, 0] = owner[idxg]
        cyclic[i, 1] = idxg
        idxg += 1
        # bc 4
    i = 0
    for j in range(param['Ny']):
        m = lambda i, j: i * (param['Ny'] + 1) + j
        mc = lambda i, j: i * (param['Ny']) + j
        xsw = points[m(i, j), 0];
        ysw = points[m(i, j), 1];
        xnw = points[m(i, j + 1), 0];
        ynw = points[m(i, j + 1), 1];
        # xse = points[m(i+1,j), 0];     yse = points[m(i+1,j), 1];
        # xne = points[m(i+1,j+1), 0];   yne = points[m(i+1,j+1), 1];
        faces[idxg, 0] = m(i, j + 1)
        faces[idxg, 1] = m(i, j)
        owner[idxg] = mc(i, j)  # owner cell id
        sf[idxg, 0] = ysw - ynw  # e_x
        sf[idxg, 1] = -(xsw - xnw)  # e_y
        #hf[idxg] = 0.5 * (hn[m(i, j)] + hn[m(i, j + 1)])
        idxg += 1
    return faces, owner, neighbor, cell, sf, cyclic

def compute_hf(param, hn, hf):
    '''
    '''
    idxg = 0
    # populate interior faces, loop over cells
    for i in range(param['Nx']):
        for j in range(param['Ny']):
            # global node and cell indices increment differently
            m = lambda i, j: i * (param['Ny'] + 1) + j
            if j == param['Ny'] - 1 and i == param['Nx'] - 1:
                pass
            elif i == param['Nx'] - 1:
                hf[idxg] = 0.5 * (hn[m(i + 1, j + 1)] + hn[m(i, j + 1)])
                idxg += 1
            elif j == param['Ny'] - 1:
                hf[idxg] = 0.5 * (hn[m(i + 1, j)] + hn[m(i + 1, j + 1)])
                idxg += 1
            else:
                hf[idxg] = 0.5 * (hn[m(i + 1, j)] + hn[m(i + 1, j + 1)])
                hf[idxg + 1] = 0.5 * (hn[m(i + 1, j + 1)] + hn[m(i, j + 1)])
                idxg += 2
    # boundaries
    # bc 1
    j = 0
    for i in range(param['Nx']):
        m = lambda i, j: i * (param['Ny'] + 1) + j
        hf[idxg] = 0.5 * (hn[m(i, j)] + hn[m(i + 1, j)])
        idxg += 1
    # bc 2
    i = param['Nx'] - 1
    for j in range(param['Ny']):
        m = lambda i, j: i * (param['Ny'] + 1) + j
        hf[idxg] = 0.5 * (hn[m(i + 1, j)] + hn[m(i + 1, j + 1)])
        idxg += 1
        # bc 3
    j = param['Ny'] - 1
    for i in range(param['Nx']):
        m = lambda i, j: i * (param['Ny'] + 1) + j
        hf[idxg] = 0.5 * (hn[m(i + 1, j + 1)] + hn[m(i, j + 1)])
        idxg += 1
        # bc 4
    i = 0
    for j in range(param['Ny']):
        m = lambda i, j: i * (param['Ny'] + 1) + j
        hf[idxg] = 0.5 * (hn[m(i, j)] + hn[m(i, j + 1)])
        idxg += 1
    return hf

def compute_h(param, points, cell, hn, hc, dhdt, vt):
    for i in range(param['Nx'] + 1):
        for j in range(param['Ny'] + 1):
            m = lambda i, j: i * (param['Ny'] + 1) + j
            p = m(i, j)
            hn[p], att_ang, xdot, ydot = film(param, points[p,0], points[p,1])
            if i < param['Nx'] and j < param['Ny']:
                mc = lambda i, j: i * (param['Ny']) + j
                hc[mc(i, j)], att_ang, xdot, ydot = film(param, cell[mc(i, j), 0], cell[mc(i, j), 1])
                dhdt[mc(i, j)], vt[mc(i, j)] = dfilm(param, cell[mc(i, j), 0], cell[mc(i, j), 1], att_ang, xdot, ydot)
    return hn, hc, dhdt, vt


def k_frene(Re):
    '''
    friction 'k' parameters from Frene, Arghir, and Constantinescu 2006
    '''
    k_p = np.zeros_like(Re); k_c = np.zeros_like(Re)
    for i, val in enumerate(Re):
        k_p[i] = 0.0 * 1.0 / 6.8 * ( (div(1.0, Re[i])) ** 0.681 ) # form to handle division by zero
        k_c[i]= 12.0 + 0.0044 * Re[i] ** 0.96
    return k_p, k_c

def f_blasius(Re):
    '''
    Blasius fanning friction factor / skin friction coefficient
    Matches Black/Yamada equation exactly in the absence of rotation, i.e. the rotational
    Reynolds number is zero.
    Valid for smooth pipes for Re up to 100,000
    '''
    n = 0.079  # 0.316 / 4.0
    m = -0.25
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
    fx = 0.0;
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

def setup_parameters(param):
    "  "
    #param['e'] = param['er'] * param['C']
    #param['e_x'] = param['ex'] * param['C']
    #param['e_y'] = param['ey'] * param['C']
    param['v_rotor'] = param['rotor_rpm'] * 2. * np.pi * param['R'] / 60.
    param['Omega'] = param['rotor_rpm'] * 2. * np.pi / 60.
    # non-dimensionalize parameters
    param['rho_init'] = param['rho_s'] / param['rho_s']
    param['p_i'] = param['p_in'] / (param['rho_s'] * param['u_s'] ** 2)  # non-dim pressure
    param['p_e'] = param['p_exit'] / (param['rho_s'] * param['u_s'] ** 2)
    param['u_i'] = 1.0 # 0.0 / u_s  # non-dim
    param['v_i'] = param['sr'] * param['v_rotor'] / param['u_s']
    param['v_r'] = param['v_rotor'] / param['u_s']
    param['Lx'] = param['L'] / param['R']
    param['Ly'] = 2. * np.pi * param['R'] / param['R']
    param['Np'] = (param['Nx'] + 1) * (param['Ny'] + 1)  # number of points
    param['Nc'] = param['Nx'] * param['Ny']  # number of cells
    param['Nf'] = param['Ny'] * (param['Nx']  + 1) + param['Nx']  * (param['Ny'] + 1)  # total number of faces, 4 faces per cell
    #Nbc = 4
    param['Nfcycle1'] = param['Nx']
    param['Nfcycle2'] = param['Nx']
    param['Nfinlet'] = param['Ny']
    param['Nfoutlet'] = param['Ny']
    param['Nfbc'] = param['Nfcycle1'] + param['Nfcycle2'] + param['Nfinlet'] + param['Nfoutlet']
    param['Nfint'] = param['Nf'] - param['Nfbc']
    param['Nfstart'] = [param['Nfint'], param['Nfint'] + param['Nfcycle1'], param['Nfint'] + param['Nfcycle1'] + param['Nfoutlet'],
               param['Nfint'] + param['Nfcycle1'] + param['Nfoutlet'] + param['Nfcycle2']]  # array of starting indices for bcs
    param['Re'] = param['rho_s'] * param['u_s'] * param['R'] / param['mu_s'] # Reynolds number resulting from non-dim. eqns.
    #param['dt'] = param['dt'] * param['u_s'] / param['R']
    param['dt'] = param['delta_t'] * param['whirl_f'] * 2.0 * np.pi
    #param['whirl_f'] = param['whirl_f'] * param['R'] / param['u_s']
    param['sigma'] = param['whirl_f'] * param['R'] / param['u_s']
    #param['whirl_a'] = param['whirl_a'] / param['C']
    param['t'] = 0.0 # starting time
    return param
