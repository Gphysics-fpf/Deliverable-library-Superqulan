import itertools
from typing import Iterable, Tuple, Dict
import numpy as np
import scipy.sparse as sp

"""State: A sorted tuple of integers, denoted which modes are occupied with excitations"""
State = tuple[int, ...]

"""Basis: A dictionary mapping states to positions in a vector state"""
Basis = dict[State, int]


def construct_basis(
    qubits: int,
    bosons: int,
    excitations: int,
) -> Basis:
    """Construct the basis for a given number of qubits and bosons with a
    fixed number of excitations.

    Creates the basis of all possible occupied modes for the given number of
    `excitations`. Each state is represented by a sorted tuple of the modes that
    are occupied. Modes 0 up to (qubits-1) are hard-core boson modes and thus
    can only appear once. All other modes are ordinary bosons and may host 0 up
    to `excitations`.

    Args:
        qubits (int): Number of qubits >= 0
        bosons (int): Number of bosonic modes >= 0
        excitations (int): Number of excitations >= 0

    Returns:
        basis: Map from configurations to an index in the Hilbert space basis
    """

    def make_bosonic_states(n_modes: int, excitations: int):
        return itertools.combinations_with_replacement(np.arange(n_modes), excitations)

    def select_hardcore_boson_states(qubits: int, states: Iterable) -> Iterable:
        return itertools.filterfalse(lambda x: unphysical_state(x, qubits), states)

    return {
        v: i
        for i, v in enumerate(
            select_hardcore_boson_states(
                qubits, make_bosonic_states(qubits + bosons, excitations)
            )
        )
    }


def unphysical_state(configuration: State, qubits: int) -> bool:
    """Given a sorted list of occupied modes, check whether a qubit mode
    appears more than once.

    Args:
        configuration (State): Sorted tuple with the occupied modes
        qubits (int): Number of hard-core boson modes >= 0

    Returns:
        bool: False if state is physical, True if it is not.
    """
    last = -1
    for mode in configuration:
        if mode >= qubits:
            return False
        if last == mode:
            return True
        last = mode


def move_excitation_operator(
    origin_mode: int, destination_mode: int, basis: Basis
) -> sp.csr_matrix:
    """
    Create a sparse matrix representation of an operator that moves an
    excitation from mode 'origin' to mode 'destination'.

    Args:
        origin (int): Index of the origin mode
        destination (int): Index of the destination mode
        basis (dict): Collection of physical states (see: construct_basis)

    Returns:
        Operator (sp.csr_matrix): Matrix representation of the quantum operator
    """

    row = []
    column = []
    coefficient = []

    for state in basis:
        origin_occupation = state.count(origin_mode)
        if origin_occupation:

            ndx = state.index(origin_mode)

            transformed_state = tuple(
                sorted(state[:ndx] + state[ndx + 1 :] + (destination_mode,))
            )
            # Construct the tupple without remove adding instead add.
            # Sort it before looking for it in basis VEERY important.
            if transformed_state in basis:

                destination_occupation = transformed_state.count(destination_mode)
                row.append(basis[state])
                column.append(basis[transformed_state])
                coefficient.append(np.sqrt(origin_occupation * destination_occupation))

    return sp.csr_matrix((coefficient, (row, column)), shape=(len(basis), len(basis)))


def diagonals_with_energies(basis: Basis, frequencies: np.ndarray) -> sp.dia_matrix:
    """_summary_

    Args:
        basis (dict): _description_
        frequencies (np.ndarray): _description_

    Returns:
        sp.dia_matrix: _description_
    """
    energy = np.empty(len(basis))  # initialize the energy coresponding to each vector.

    for occupation, pos in basis.items():

        energy[pos] = np.sum(frequencies[list(occupation)])

    return sp.diags(energy)


def concatenate_bases(
    qubits: int = 2,
    bosons: int = 30,
    excitations: int = 2,
) -> Basis:

    """Function that takes as inputs the number of components of the setup and Nexcitations
    constructs the basis resulting from the concatenation of subspaces up to these Nexcitations'.

    Args:
        qubits (int): Number of qubits >= 0
        bosons (int): Number of bosonic modes >= 0
        excitations (int): Number of excitations of the biggest subspace >= 0

    Returns:
        basis: Collection of all the states that constitute the basis properly sorted.
    """
    Basis = {}

    for excitations in range(excitations + 1):

        if excitations == 0:

            Basis[
                ()
            ] = 0  # For the subspace of 0 excitations we manually create the empty tuple corresponding to vacuum.

        else:
            Basis_subspace = (
                {}
            )  # We initialize the variable Basis for a particular subspace

            index_0 = len(
                Basis
            )  # Very important. The indexation of the new subspace must begin where the previous left

            def make_bosonic_states(n_modes: int, excitations: int):
                return itertools.combinations_with_replacement(
                    np.arange(n_modes), excitations
                )

            def select_hardcore_boson_states(qubits: int, states: Iterable) -> Iterable:
                return itertools.filterfalse(
                    lambda x: unphysical_state(x, qubits), states
                )

            for i, v in enumerate(
                select_hardcore_boson_states(
                    qubits, make_bosonic_states(qubits + bosons, excitations)
                )
            ):
                Basis_subspace[v] = i + index_0

                Basis.update(
                    Basis_subspace
                )  # Update is a sort of append for dictionaries.

    return Basis


def erase(remove: int, Basis: dict) -> sp.csr_matrix:

    """Creates a sparse matrix representation of an operator that erases an excitation
    from 'remove'.

    Args:
        remove (int): Index of the mode from which the excitation is going to be removed
        destination (int): Index of the destination mode
        basis (dict): Collection of physical states (see: construct_basis)

    Returns:
        Operator (sp.csr_matrix): Matrix representation of the quantum operator"""
    row = []
    column = []
    coefficient = []

    for (
        v
    ) in (
        Basis
    ):  # we run through the elements of the basis, now well labeled, the vectors are the keys.

        count = v.count(
            remove
        )  # Count how many times the excitation is located in the remove box

        if count:  # Whenever count is not zero, transform the vector

            ndx = v.index(
                remove
            )  # Obtain the position in the vector of the first excitation to remove.

            transformed = tuple(
                sorted(v[:ndx] + v[ndx + 1 :])
            )  # Construct the tupple without remove (erasing the lelement we wanted to).
            # Sort it before looking for it in basis VEERY important.
            if transformed in Basis:

                print(
                    "Maps the element",
                    Basis[v],
                    "onto the element",
                    Basis[transformed],
                    "with coefficient",
                    np.sqrt(count),
                )

                # row.append(Basis[v]), column.append(Basis[transformed]), coefficient.append(np.sqrt(count))
                row.append(Basis[transformed]), column.append(
                    Basis[v]
                ), coefficient.append(np.sqrt(count))

    return sp.csr_matrix((coefficient, (row, column)), shape=(len(Basis), len(Basis)))
