import unittest
from superqulan.bosons import construct_basis, number_operator
from .common import *


class TestNumberOperator(unittest.TestCase):
    def assertEqualSparse(self, a, b):
        return self.assertTrue(equal_sparse_matrices(a, b))

    def run_for_all_combinations(self, function, excitation_range=[1], max_modes=4):
        for excitations in excitation_range:
            for modes in range(max_modes + 1):
                for qubits in range(modes):
                    basis = construct_basis(
                        qubits=qubits, bosons=modes - qubits, excitations=excitations
                    )
                    for i in range(modes):
                        function(modes, excitations, i, basis)

    def test_number_operator_is_0_for_0_excitations(self):
        def test(modes, excitations, i, basis):
            self.assertTrue(is_zero_sparse_matrix(number_operator(basis, i)))

        self.run_for_all_combinations(test, excitation_range=[0], max_modes=10)

    def test_exact_number_operator_for_1_excitations(self):
        self.run_for_all_combinations(
            lambda modes, excitations, i, basis: self.assertTrue(
                equal_sparse_matrices(
                    number_operator(basis, i),
                    sp_from_coordinates([(1.0, i, i)], (len(basis), len(basis))),
                )
            ),
            excitation_range=[1],
        )

    def test_exact_number_operator_for_2_qubits_2_bosons_2_excitations(self):
        basis = construct_basis(qubits=2, bosons=2, excitations=2)
        self.assertEqualSparse(
            number_operator(basis, 0), sp.diags([1, 1, 1, 0, 0, 0, 0, 0])
        )
        self.assertEqualSparse(
            number_operator(basis, 1), sp.diags([1, 0, 0, 1, 1, 0, 0, 0])
        )
        self.assertEqualSparse(
            number_operator(basis, 2), sp.diags([0, 1, 0, 1, 0, 2, 1, 0])
        )
        self.assertEqualSparse(
            number_operator(basis, 3), sp.diags([0, 0, 1, 0, 1, 0, 1, 2])
        )
