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
#%matplotlib widget

''' Routines for the construction of the Operators in Hilbert space '''

def construct_basis_superfast(Nqubits: int = 2, Ncavs: int = 0, Nfilters: int = 0, N_WGmodes: int = 30, Nexcitations: int = 2) -> dict:

    ''' Function that takes as inputs the number of components of the setup and Nexcitations
    constructs the allowed physical states.'''
    
    N_modes = Nqubits + Ncavs + Nfilters + N_WGmodes

    ''' filterfalse discards the true outputs of unphysical_states when run over the iterable produced by combinations with replacements. '''      
            
    Basis = {}
    for i,v in enumerate(list(itertools.filterfalse(lambda x: unphysical_states(x, Nqubits), itertools.combinations_with_replacement(np.arange(N_modes),Nexcitations)))):
        Basis[v] = i
        
    return Basis


def unphysical_states(Basis_element: tuple, Nqubits: int) -> bool:

    ''' Function to be combined with filterfalse. It returns false 
    whenever the state is physical (allowed)'''

    if Basis_element[0] >= Nqubits:  # If the fist component is not qubit, then there no qubit excitations in this state. Therefore physical.

        return False

    for i in range(len(Basis_element)-1):  # If there are qubits run through the components.

        if Basis_element[i] == Basis_element[i+1] and Basis_element[i] < Nqubits and Basis_element[i+1] < Nqubits:  # If two qubit excitations are consecutive, unphysical. return true.

            return True


def transform_new(add: int, remove: int, Basis: dict) -> sp.csr_matrix:

    ''' Function that takes as inputs the locations where the excitation is removed
    and where it is added. Returns a Matrix representing that operation (plus an addequate coefficient)
    in the order dictated by Basis'''

    row = []
    column = []
    coefficient = []
    
    for v in Basis:  # we run through the elements of the basis, now well labeled, the vectors are the keys.
        
        count = v.count(remove)  # Count how many times the excitation is located in the remove box
        
        if count:  # Whenever count is not zero, transform the vector
            
            ndx = v.index(remove)  # Obtain the position in the vector of the first excitation to remove.
            
            transformed = tuple(sorted(v[:ndx]+v[ndx+1:]+(add,)))  # Construct the tupple without remove adding instead add. 
                                                                   # Sort it before looking for it in basis VEERY important.
            if transformed in Basis:

                row.append(Basis[v]), column.append(Basis[transformed]), coefficient.append(np.sqrt(count))

    return sp.csr_matrix( (coefficient, (row, column)), shape=(len(Basis), len(Basis)) )




def concatenate_bases(Nqubits: int = 2, Ncavs: int = 0, Nfilters: int = 0, N_WGmodes: int = 30, Up_to_Nexcitations: int = 2) -> dict:
        
    ''' Function that takes as inputs the number of components of the setup and Nexcitations
    constructs the basis resulting from the concatenation of subspaces up to these Nexcitations'''
    
    N_modes = Nqubits + Ncavs + Nfilters + N_WGmodes 
        
    Basis = {}  # We initialize the variable Basis, a dictionary.

    for Nexcitations in range(Up_to_Nexcitations+1):  # Loop that runs over all the subspaces
        
        if Nexcitations == 0:
            
            Basis[()] = 0  # For the subspace of 0 excitations we manually create the empty tuple corresponding to vacuum.
            
        else:  # For all the non-trivial subspaces we compute a basis by calling the itertools tool
     
            ''' filterfalse discards the true outputs of unphysical_states when run over the iterable produced by combinations with replacements. '''              
            
            Basis_subspace = {}  # We initialize the variable Basis for a particular subspace
            
            index_0 = len(Basis)  # Very important. The indexation of the new subspace must begin where the previous left
            
            for i,v in enumerate(list(itertools.filterfalse(lambda x: unphysical_states(x, Nqubits), itertools.combinations_with_replacement(np.arange(N_modes),Nexcitations)))):

                Basis_subspace[v] = i + index_0

            Basis.update(Basis_subspace)  # Update is a sort of append for dictionaries.
    
    return Basis


def erase(remove: int, Basis: dict) -> sp.csr_matrix:

    row = []
    column = []
    coefficient = []
    
    for v in Basis:  # we run through the elements of the basis, now well labeled, the vectors are the keys.
        
        count = v.count(remove)  # Count how many times the excitation is located in the remove box
        
        if count:  # Whenever count is not zero, transform the vector
            
            ndx = v.index(remove)  # Obtain the position in the vector of the first excitation to remove.
            
            transformed = tuple(sorted(v[:ndx]+v[ndx+1:]))  # Construct the tupple without remove (erasing the lelement we wanted to).
                                                                   # Sort it before looking for it in basis VEERY important.
            if transformed in Basis:
                
                print('Maps the element',Basis[v],'onto the element',Basis[transformed], 'with coefficient',np.sqrt(count))
                
                # row.append(Basis[v]), column.append(Basis[transformed]), coefficient.append(np.sqrt(count))
                row.append(Basis[transformed]), column.append(Basis[v]), coefficient.append(np.sqrt(count))
                
    return sp.csr_matrix( (coefficient, (row, column)), shape=(len(Basis), len(Basis)) )
