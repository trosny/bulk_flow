#!/usr/bin/env python
"""
"""
import numpy as np

class mesh(object):
    def __init__(self, params):
        # input parameters
        self.Nx = params.get('Nx')
        self.Ny = params.get('Ny')

        #
        self.L = params.get('L')
        self.R = params.get('R') 
        self.C = params.get('C') 
        self.ex = params.get('ex') 
        self.ey =  params.get('ey')
        self.angX = params.get('angX') 
        self.angY =  params.get('angY')     

        self.Np = (self.Nx + 1) * (self.Ny + 1)  # number of points
        self.Nc = self.Nx * self.Ny  # number of cells
        self.Nf = self.Ny * (self.Nx  + 1) + self.Nx  * (self.Ny + 1)  # total number of faces, 4
        self.Nfcycle1 = self.Nx
        self.Nfcycle2 = self.Nx
        self.Nfinlet = self.Ny
        self.Nfoutlet = self.Ny
        self.Nfbc = self.Nfcycle1 + self.Nfcycle2 + self.Nfinlet + self.Nfoutlet
        self.Nfint = self.Nf - self.Nfbc
        self.Nfstart = [self.Nfint, self.Nfint + self.Nfcycle1, \
                        self.Nfint + self.Nfcycle1 + self.Nfoutlet, \
                        self.Nfint + self.Nfcycle1 + self.Nfoutlet + self.Nfcycle2]  #
        #
        self._non_dim_mesh()
        self._init_mesh_arrays()
        self._generate_points()
        self._generate_mesh()
        self._compute_h()
        self._compute_hf()
        self._compute_int()
        self._compute_cyclic_int()
    
     
    def _non_dim_mesh(self):
        ''' non-dimensionalize parameters
        '''
        self.Lx = self.L / self.R
        self.Ly = 2. * np.pi * self.R / self.R

        
    def _init_mesh_arrays(self): 
        self.x = np.linspace(0, self.Lx, self.Nx  + 1 , dtype=np.float64)  # points in axial dir.
        self.y = np.linspace(0, self.Ly, self.Ny + 1 , dtype=np.float64)  # points in circ. dir. dir.
        self.points = np.zeros([self.Np, 2], dtype=np.float64)
        self.faces = np.zeros([self.Nf, 2], dtype=int)
        self.sf = np.zeros([self.Nf, 2], dtype=np.float64)
        self.owner = np.zeros(self.Nf, dtype=int)
        self.neighbor = np.zeros(self.Nfint, dtype=int)
        # store cell center x, cell center y, and cell volume
        self.cell = np.zeros([self.Nc, 3], dtype=np.float64) 
        # owner (north) cell, owner face, nb cell, nb face   
        self.cyclic = np.zeros([self.Nfcycle1, 4], dtype=int) 
        # film thickness at mesh nodes
        self.hn = np.zeros(self.Np, dtype=np.float64)
        # film thickness at cell center      
        self.hc = np.zeros(self.Nc, dtype=np.float64)
        # average film thickness at faces, approximate
        self.hf = np.zeros(self.Nf, dtype=np.float64)  
        # array containing bc type flags
        self.bc_type = np.zeros(self.Nfbc, dtype=int)  # array to contain flags for bc type
        # populate array of bc type flags
        # 0: solid wall, 1: inflow, 2: outflow, 3: cyclic 1, 4: cyclic 2 (owner)
        for i in range(0, self.Nfcycle1):
            self.bc_type[i] = 3  # cyclic 1
        for i in range(self.Nfcycle1, self.Nfcycle1 + self.Nfoutlet):
            self.bc_type[i] = 2  # outlet
        for i in range(self.Nfcycle1 + self.Nfoutlet, self.Nfcycle1 + self.Nfoutlet + self.Nfcycle2):
            self.bc_type[i] = 4  # cyclic 2
        for i in range(self.Nfcycle1 + self.Nfoutlet + self.Nfcycle2, self.Nfbc):
            self.bc_type[i] = 1  # inlet
        self.gf = np.zeros(self.Nf, dtype=np.float64)  # geometric weighting factor, zero for bcs
        self.t = np.zeros(self.Nf, dtype=np.float64)  # distance from owner to neighbor, zero for bcs
        self.cn = np.zeros([self.Nf, 2], dtype=np.float64)  # vector pointing from owner cell centroid to face centroid
        self.cf = np.zeros([self.Nf, 2], dtype=np.float64)  # face centroid
        self.fa = np.zeros(self.Nf, dtype=np.float64)  # area (length) of each face    
            

    def film(self, x, y):  
        h = 1.0 + self.ex  * np.cos( y ) + self.ey  * np.sin( y )
        return h

    def _generate_points(self):
        '''
        rectangular grid
        '''
        #hn = np.zeros(param['Np'], dtype=np.double)
        for i in range(self.Nx + 1):
            for j in range(self.Ny + 1):
                m = lambda i, j: i * (self.Ny + 1) + j
                p = m(i, j)
                self.points[p, 0] = self.x[i]
                self.points[p, 1] = self.y[j]

    def _generate_mesh(self):
        '''
        '''
        points = self.points
        idxg = 0
        # populate interior faces, loop over cells
        for i in range(self.Nx):
            for j in range(self.Ny):
                # global node and cell indices increment differently
                m = lambda i, j: i * (self.Ny + 1) + j
                mc = lambda i, j: i * self.Ny + j
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
                self.cell[mc(i, j), 0] = xc
                self.cell[mc(i, j), 1] = yc
                self.cell[mc(i, j), 2] = vol
                #hc[mc(i, j)] = h
                #
                if j == self.Ny - 1 and i == self.Nx - 1:
                    pass
                elif i == self.Nx - 1:
                    self.faces[idxg, 0] = m(i + 1, j + 1)
                    self.faces[idxg, 1] = m(i, j + 1)
                    self.owner[idxg] = mc(i, j)  # owner cell id
                    self.neighbor[idxg] = mc(i, j + 1)
                    self.sf[idxg, 0] = ynw - yne  # n_x
                    self.sf[idxg, 1] = -(xnw - xne)  # n_y
                    #hf[idxg] = 0.5 * (hn[m(i + 1, j + 1)] + hn[m(i, j + 1)])
                    idxg += 1
                elif j == self.Ny - 1:
                    self.faces[idxg, 0] = m(i + 1, j)
                    self.faces[idxg, 1] = m(i + 1, j + 1)
                    self.owner[idxg] = mc(i, j)  # owner cell id
                    self.neighbor[idxg] = mc(i + 1, j)
                    self.sf[idxg, 0] = yne - yse  # e_x
                    self.sf[idxg, 1] = -(xne - xse)  # e_y
                    #hf[idxg] = 0.5 * (hn[m(i + 1, j)] + hn[m(i + 1, j + 1)])
                    idxg += 1
                else:
                    self.faces[idxg, 0] = m(i + 1, j)
                    self.faces[idxg, 1] = m(i + 1, j + 1)
                    self.faces[idxg + 1, 0] = m(i + 1, j + 1)
                    self.faces[idxg + 1, 1] = m(i, j + 1)
                    self.owner[idxg] = mc(i, j)  # owner cell id
                    self.neighbor[idxg] = mc(i + 1, j)
                    self.owner[idxg + 1] = mc(i, j)  # owner cell id
                    self.neighbor[idxg + 1] = mc(i, j + 1)
                    self.sf[idxg, 0] = yne - yse  # e_x
                    self.sf[idxg, 1] = -(xne - xse)  # e_y
                    self.sf[idxg + 1, 0] = ynw - yne  # n_x
                    self.sf[idxg + 1, 1] = -(xnw - xne)  # n_y
                    #hf[idxg] = 0.5 * (hn[m(i + 1, j)] + hn[m(i + 1, j + 1)])
                    #hf[idxg + 1] = 0.5 * (hn[m(i + 1, j + 1)] + hn[m(i, j + 1)])
                    idxg += 2
        # boundaries
        # bc 1
        j = 0
        for i in range(self.Nx):
            m = lambda i, j: i * (self.Ny + 1) + j
            mc = lambda i, j: i * (self.Ny) + j
            xsw = points[m(i, j), 0];
            ysw = points[m(i, j), 1];
            # xnw = points[m(i,j+1), 0];   ynw = points[m(i,j+1), 1];
            xse = points[m(i + 1, j), 0];
            yse = points[m(i + 1, j), 1];
            # xne = points[m(i+1,j+1), 0];   yne = points[m(i+1,j+1), 1];
            self.faces[idxg, 0] = m(i, j)
            self.faces[idxg, 1] = m(i + 1, j)
            self.owner[idxg] = mc(i, j)  # owner cell id
            self.sf[idxg, 0] = yse - ysw  # s_x
            self.sf[idxg, 1] = -(xse - xsw)  # s_y
            #hf[idxg] = 0.5 * (hn[m(i, j)] + hn[m(i + 1, j)])
            self.cyclic[i, 2] = self.owner[idxg]
            self.cyclic[i, 3] = idxg
            idxg += 1
        # bc 2
        i = self.Nx- 1
        for j in range(self.Ny):
            m = lambda i, j: i * (self.Ny + 1) + j
            mc = lambda i, j: i * (self.Ny) + j
            # xsw = points[m(i,j), 0];     ysw = points[m(i,j), 1];
            # xnw = points[m(i,j+1), 0];   ynw = points[m(i,j+1), 1];
            xse = points[m(i + 1, j), 0];
            yse = points[m(i + 1, j), 1];
            xne = points[m(i + 1, j + 1), 0];
            yne = points[m(i + 1, j + 1), 1];
            self.faces[idxg, 0] = m(i + 1, j)
            self.faces[idxg, 1] = m(i + 1, j + 1)
            self.owner[idxg] = mc(i, j)  # owner cell id
            self.sf[idxg, 0] = yne - yse  # e_x
            self.sf[idxg, 1] = -(xne - xse)  # e_y
            #hf[idxg] = 0.5 * (hn[m(i + 1, j)] + hn[m(i + 1, j + 1)])
            idxg += 1
            # bc 3
        j = self.Ny - 1
        for i in range(self.Nx):
            m = lambda i, j: i * (self.Ny + 1) + j
            mc = lambda i, j: i * (self.Ny) + j
            # xsw = points[m(i,j), 0];     ysw = points[m(i,j), 1];
            xnw = points[m(i, j + 1), 0];
            ynw = points[m(i, j + 1), 1];
            # xse = points[m(i+1,j), 0];     yse = points[m(i+1,j), 1];
            xne = points[m(i + 1, j + 1), 0];
            yne = points[m(i + 1, j + 1), 1];
            self.faces[idxg, 0] = m(i + 1, j + 1)
            self.faces[idxg, 1] = m(i, j + 1)
            self.owner[idxg] = mc(i, j)  # owner cell id
            self.sf[idxg, 0] = ynw - yne  # n_x
            self.sf[idxg, 1] = -(xnw - xne)  # n_y
            #hf[idxg] = 0.5 * (hn[m(i + 1, j + 1)] + hn[m(i, j + 1)])
            self.cyclic[i, 0] = self.owner[idxg]
            self.cyclic[i, 1] = idxg
            idxg += 1
            # bc 4
        i = 0
        for j in range(self.Ny):
            m = lambda i, j: i * (self.Ny + 1) + j
            mc = lambda i, j: i * (self.Ny) + j
            xsw = points[m(i, j), 0];
            ysw = points[m(i, j), 1];
            xnw = points[m(i, j + 1), 0];
            ynw = points[m(i, j + 1), 1];
            # xse = points[m(i+1,j), 0];     yse = points[m(i+1,j), 1];
            # xne = points[m(i+1,j+1), 0];   yne = points[m(i+1,j+1), 1];
            self.faces[idxg, 0] = m(i, j + 1)
            self.faces[idxg, 1] = m(i, j)
            self.owner[idxg] = mc(i, j)  # owner cell id
            self.sf[idxg, 0] = ysw - ynw  # e_x
            self.sf[idxg, 1] = -(xsw - xnw)  # e_y
            #hf[idxg] = 0.5 * (hn[m(i, j)] + hn[m(i, j + 1)])
            idxg += 1

    def _compute_hf(self):
        '''
        '''
        hn = self.hn
        idxg = 0
        # populate interior faces, loop over cells
        for i in range(self.Nx):
            for j in range(self.Ny):
                # global node and cell indices increment differently
                m = lambda i, j: i * (self.Ny + 1) + j
                if j == self.Ny - 1 and i == self.Nx - 1:
                    pass
                elif i == self.Nx - 1:
                    self.hf[idxg] = 0.5 * (hn[m(i + 1, j + 1)] + hn[m(i, j + 1)])
                    idxg += 1
                elif j == self.Ny - 1:
                    self.hf[idxg] = 0.5 * (hn[m(i + 1, j)] + hn[m(i + 1, j + 1)])
                    idxg += 1
                else:
                    self.hf[idxg] = 0.5 * (hn[m(i + 1, j)] + hn[m(i + 1, j + 1)])
                    self.hf[idxg + 1] = 0.5 * (hn[m(i + 1, j + 1)] + hn[m(i, j + 1)])
                    idxg += 2
        # boundaries
        # bc 1
        j = 0
        for i in range(self.Nx):
            m = lambda i, j: i * (self.Ny + 1) + j
            self.hf[idxg] = 0.5 * (hn[m(i, j)] + hn[m(i + 1, j)])
            idxg += 1
        # bc 2
        i = self.Nx - 1
        for j in range(self.Ny):
            m = lambda i, j: i * (self.Ny+ 1) + j
            self.hf[idxg] = 0.5 * (hn[m(i + 1, j)] + hn[m(i + 1, j + 1)])
            idxg += 1
            # bc 3
        j = self.Ny- 1
        for i in range(self.Nx):
            m = lambda i, j: i * (self.Ny + 1) + j
            self.hf[idxg] = 0.5 * (hn[m(i + 1, j + 1)] + hn[m(i, j + 1)])
            idxg += 1
            # bc 4
        i = 0
        for j in range(self.Ny):
            m = lambda i, j: i * (self.Ny + 1) + j
            self.hf[idxg] = 0.5 * (hn[m(i, j)] + hn[m(i, j + 1)])
            idxg += 1

    def _compute_h(self):
        for i in range(self.Nx + 1):
            for j in range(self.Ny + 1):
                m = lambda i, j: i * (self.Ny + 1) + j
                p = m(i, j)
                self.hn[p] = self.film(self.points[p,0], self.points[p,1])
                if i < self.Nx and j < self.Ny:
                    mc = lambda i, j: i * self.Ny + j
                    self.hc[mc(i, j)] = self.film(self.cell[mc(i, j), 0], self.cell[mc(i, j), 1])
                    
    def _compute_int(self):
        '''compute interpolation factors and cell-center to cell-face vectors
        '''
        points = self.points
        faces = self.faces
        cell = self.cell
        sf = self.sf
        owner = self.owner
        neighbor = self.neighbor
        for i in range(self.Nfint):
            fxc = 0.5 * (points[faces[i, 0], 0] + points[faces[i, 1], 0])  # face centroid
            fyc = 0.5 * (points[faces[i, 0], 1] + points[faces[i, 1], 1])
            xnb = cell[neighbor[i], 0]  # neighbor centroid
            ynb = cell[neighbor[i], 1]
            xp = cell[owner[i], 0]  # owner centroid
            yp = cell[owner[i], 1]
            rnb = np.sqrt((fxc - xnb) ** 2. + (fyc - ynb) ** 2)
            rp = np.sqrt((xp - xnb) ** 2. + (yp - ynb) ** 2)
            self.cn[i, 0] = fxc - xp
            self.cn[i, 1] = fyc - yp
            self.cf[i, 0] = fxc
            self.cf[i, 1] = fyc
            self.gf[i] = rnb / rp
            self.t[i] = rp
            self.fa[i] = np.sqrt(np.abs(sf[i, 0]) ** 2 + np.abs(sf[i, 1]) ** 2)
        # add bcs
        for i in range(self.Nfstart[0], self.Nf):
            fxc = 0.5 * (points[faces[i, 0], 0] + points[faces[i, 1], 0])  # face centroid
            fyc = 0.5 * (points[faces[i, 0], 1] + points[faces[i, 1], 1])
            xp = cell[owner[i], 0]  # owner centroid
            yp = cell[owner[i], 1]
            self.cn[i, 0] = fxc - xp
            self.cn[i, 1] = fyc - yp
            self.cf[i, 0] = fxc
            self.cf[i, 1] = fyc
            self.gf[i] = 1.0
            self.fa[i] = np.sqrt(np.abs(sf[i, 0]) ** 2 + np.abs(sf[i, 1]) ** 2)

    def _compute_cyclic_int(self):
        '''
        update interpolation factors for cyclic bcs
        '''
        cn = self.cn
        cyclic = self.cyclic
        sf = self.sf
        idx = 0
        idx2 = 0
        for i in range(self.Nfstart[0], self.Nf):
            if self.bc_type[idx] == 3:  # cyclic 1
                nbx = np.abs(cn[cyclic[idx2, 1], 0])  # cyclic 2
                nby = np.abs(cn[cyclic[idx2, 1], 1])
                px = np.abs(cn[i, 0])
                py = np.abs(cn[i, 1])
                rnb = np.sqrt((nbx) ** 2. + (nby) ** 2)
                rp = np.sqrt((nbx + px) ** 2. + (nby + py) ** 2)
                self.gf[i] = rnb / rp
                self.t[i] = rp
                self.fa[i] = np.sqrt(np.abs(sf[i, 0]) ** 2 + np.abs(sf[i, 1]) ** 2)
                self.gf[cyclic[idx2, 1]] = 1.0 - (rnb / rp)
                self.t[cyclic[idx2, 1]] = rp
                self.fa[cyclic[idx2, 1]] = np.sqrt(np.abs(sf[cyclic[idx2, 1], 0]) ** 2 + np.abs(sf[cyclic[idx2, 1], 1]) ** 2)
                idx2 += 1
                #
            idx += 1
