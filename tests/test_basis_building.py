from unittest import TestCase
from superqulan.bosons import construct_basis, Basis, State


class TestBosonicBasis(TestCase):
    def assertEqualBasis(self, basis: Basis, state_list: list[State]):
        """Verify that the basis contains the given list of configurations."""
        self.assertTrue(len(basis) == len(state_list))
        self.assertTrue(all(s in basis for s in state_list))

    def assertEqualExcitations(self, basis: Basis, excitations: int):
        """Verify that all states have the same number of `excitations`"""
        self.assertTrue(all(len(a) == excitations for a in basis))

    def assertBasisContainsAllIndices(self, basis: Basis):
        """Verify that the basis contains indices to all states in the basis,
        without duplicates."""
        L = len(basis)
        indices = sorted(b for _, b in basis.items())
        self.assertTrue(all(i == j for i, j in zip(indices, range(L))))

    def test_qubits_and_bosons_same_for_1_excitation(self):
        for modes in range(4):
            last = None
            for qubits in range(modes):
                basis = construct_basis(
                    qubits=qubits, bosons=modes - qubits, excitations=1
                )
                if last is not None:
                    self.assertEqualBasis(basis, last)

    def test_1_qubit_0_bosons_0_excitations(self):
        one_qubit_0 = [()]
        self.assertEqualBasis(
            construct_basis(qubits=1, bosons=0, excitations=0), one_qubit_0
        )

    def test_1_qubit_0_bosons_1_excitations(self):
        one_qubit_1 = [(0,)]
        self.assertEqualBasis(
            construct_basis(qubits=1, bosons=0, excitations=1), one_qubit_1
        )

    def test_0_qubits_1_bosons_0_excitations(self):
        one_boson_0 = [()]
        self.assertEqualBasis(
            construct_basis(qubits=0, bosons=1, excitations=0), one_boson_0
        )

    def test_0_qubits_1_bosons_1_excitations(self):
        one_boson_1 = [(0,)]
        self.assertEqualBasis(
            construct_basis(qubits=0, bosons=1, excitations=1), one_boson_1
        )
