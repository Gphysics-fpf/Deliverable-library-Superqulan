import numpy as np 
from scipy import sparse
import itertools, operator
import numpy as np
from numpy import floor, sqrt
import scipy.sparse as sp
import scipy.sparse.linalg
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from math import pi as π
import math
import copy
import sympy as sym
from bosons import construct_basis_superfast, concatenate_bases, diagonals_with_energies 


class Architecture: 
    '''class that includes the basic features that every experiment should have. Then the specific experiments inherit
    all the features from this class.'''
    

class Exp2purcells2excitations(Architecture):
    
    def __init__(self, δ1=2*π*8.406, δ2=2*π*8.406, g1=2*π*0.0086, g2=2*π*0.0086, gp1=2*π*0.025, gp2=2*π*0.025,
                 ω1=2*π*8.406, ω2=2*π*8.406, ωp1= 2*π*8.406, ωp2 = 2*π*8.406,
                 κ1=2*π*0.0086, κ2=2*π*0.0086, l=5, N_WGmodes=5,
                 Gconstant=False, δLamb=0.0, Nexcitations = 2, Nfilters = 2, include_vacuum = False):
        
        """Experiment with 2 qubits, 2 cavities, 2 Purcell filters and waveguide.
        
        extra functionality of being able to choose one or two excitations.
        ..."""
        self.δ1 = δ1
        self.δ2 = δ2
        self.δLamb = δLamb
        self.g1 = g1
        self.g2 = g2
        self.gp1 = gp1
        self.gp2 = gp2
        self.ω1 = ω1
        self.ω2 = ω2
        self.ωp1 = ωp1
        self.ωp2 = ωp2
        self.κ1 = κ1
        self.κ2 = κ2 
        self.l = l
        self.Gconstant = Gconstant
        self.N_WGmodes = N_WGmodes
        self.Nexcitations = Nexcitations
        self.Nqubits = 2
        self.Ncavs = 2
        self.Nfilters = Nfilters
        self.qubit_indices = np.arange(0,self.Nqubits) 
        self.cav_indices = np.arange(self.Nqubits,self.Nqubits + self.Ncavs) 
        self.filter_indices = np.arange(self.Nqubits + self.Ncavs, self.Nqubits + self.Ncavs + self.Nfilters )


        if include_vacuum: 
            self.Basis = concatenate_bases(Nqubits = self.Nqubits, Ncavs = self.Ncavs, Nfilters = self.Nfilters, N_WGmodes = self.N_WGmodes, Up_to_Nexcitations = self.Nexcitations)
            self.first_WGmode = len(self.Basis) - 1 - self.N_WGmodes
        
        else:     
            self.Basis = construct_basis_superfast(Nqubits = self.Nqubits, Ncavs = self.Ncavs, Nfilters = self.Nfilters, N_WGmodes = self.N_WGmodes, Nexcitations = self.Nexcitations)
            self.first_WGmode = len(self.Basis) - self.N_WGmodes
            

        self.which_to_WG = [self.Ncavs + self.Nfilters, self.Ncavs + self.Nfilters+1]
        self.compute_matrices_()

    def compute_matrices_(self):
        # We then set the parameters of the waveguide a = width
        m = np.arange(self.l*80)                               # Number of modes in the bandwidth
        a = 0.02286                                            # Width of our waveguide (in principle fixed)
        c = 299792458                                          # Speed of light in vacuum
        self.ω = (2*π)*(c*np.sqrt((1/(2*a))**2 + (m/(2*self.l))**2))/10**9 # ω of the modes (in GHz)

        #we obtain the mode that is resonant with cavity 1
        self.mcentral = np.abs(self.ω-self.ω1).argmin()
        mmin = max(self.mcentral - int(self.N_WGmodes/2), 0)
        self.mrelevant = np.arange(mmin, mmin + self.N_WGmodes)
        self.ωrelevant = self.ω[mmin:mmin+self.N_WGmodes]
        
        νrelevant = self.ωrelevant*1e9/(2*π)
        vgroup = c*np.sqrt(1 - (c/(2*a*νrelevant))**2)
        
        # Index of the mode that is resonant with the qubit 1
        resonant_ndx = np.abs(self.ωrelevant-self.ω1).argmin()
        G1 = np.sqrt(((self.κ1*10**9)*vgroup)/(2*self.l))/10**9
        G2 = np.sqrt(((self.κ2*10**9)*vgroup)/(2*self.l))/10**9
        
        if self.Gconstant: 
            self.G1 = G1[resonant_ndx]
            self.G2 = G2[resonant_ndx]
        else:
            self.G1 = G1[resonant_ndx]*np.sqrt(self.ωrelevant/self.ωrelevant[resonant_ndx])
            self.G2 = G2[resonant_ndx]*np.sqrt(self.ωrelevant/self.ωrelevant[resonant_ndx])
            
        
        self.tprop = (self.l/(vgroup[resonant_ndx]))*10**9
        
        ''' Construct the vector of energies E for filling in the diagonal entries. '''
        self.E = np.array([self.δ1 + self.δLamb, self.δ2 + self.δLamb, self.ω1, self.ω2, self.ωp1, self.ωp2] + list(self.ωrelevant))
        
        H_diag = diagonals_with_energies(self.Basis, self.E)
        
        '''Fill in the entries for cavity- waveguide couplings (in this case purcell-waveguide)'''
        H_cav_WG = cav_WG_with_couplings(self, self.G1, self.G2, self.Basis)
        
        H_cav1_Purcell1, H_cav2_Purcell2 = cav_Purcell_with_couplings(self, self.Basis)
        
        H_cav1_Purcell1 *= self.gp1
        H_cav2_Purcell2 *= self.gp2

        ''' Up untill here this is the whole static part'''
        self.H = H_diag + H_cav_WG + H_cav1_Purcell1 + H_cav2_Purcell2 
        
        
        
        self.size = len(self.Basis)

        
        ''' Here we initialize the dynamical part, but only for later use in Hamiltonian() '''
        H_qb1_cav1, H_qb2_cav2 = qubit_cav_with_couplings(self, self.Basis)

        self.H_qb1_cav1 = H_qb1_cav1
        self.H_qb2_cav2 = H_qb2_cav2

        
    def Hamiltonian(self, g1=None, g2=None):
        """Adds the qubit part to the hamiltonian matrix"""
        
        # This remains the same, Each step the trotter cals the function Hamiltonian and changes g1, g2, dynamically.
        
        return self.H + (self.g1 if g1 is None else g1) * self.H_qb1_cav1 \
                      + (self.g2 if g2 is None else g2) * self.H_qb2_cav2

    
   
    def mode(self, m, l, x):
        """Return waveguide mode evaluated at position x for all frequencies"""
        return np.cos((π*m/l)*x)
    
    
    def cavity_excited(self, which=0):
        if which not in [0,1]:
            raise Exception(f'Cavity number {which} not in [0,1]')
        C = np.zeros(self.size)
        C[which + 2] = 1
        return C  # state with cavity 'include_vacuumwhich' excited with 1 photon 
    
    
    
    def qubit_excited(self, which=0):
        if which not in [0,1]:
            raise Exception(f'Qubit number {which} not in [0,1]')
        C = np.zeros(self.size)
        C[which] = 1
        return C  # state with qubit 'which' excited with 1 photon

    
    def change_parameters(self, **kwdargs):
        output = copy.deepcopy(self)
        for k in kwdargs:
            if k not in output.__dict__:
                raise Exception(f'Unknown parameter {k}')
            output.__dict__[k] = kwdargs[k]
        output.compute_matrices_()
        return output