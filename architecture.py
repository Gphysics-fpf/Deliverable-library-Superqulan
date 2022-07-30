from typing import Optional, Union, Callable
from dataclasses import dataclass
from math import pi as π
from numbers import Number
import numpy as np
import scipy.sparse as sp
from waveguide import Waveguide
from bosons import (
    construct_basis,
    move_excitation_operator,
    diagonals_with_energies,
    concatenate_bases,
)


@dataclass
class Exp_2qubits_2cavities_2purcells:
    """Architecture with 2 qubits, 2 cavities, 2 Purcell filters and waveguide.
    The disposition of this setup is 1qb + 1cav + 1 Pfil + WG and the same at the other end
    ..."""

    δ1: float = 2 * π * 8.406
    δ2: float = 2 * π * 8.406
    g1: Union[Number, Callable] = 2 * π * 0.0086
    g2: Union[Number, Callable] = 2 * π * 0.0086
    gp1: float = 2 * π * 0.025
    gp2: float = 2 * π * 0.025
    ω1: float = 2 * π * 8.406
    ω2: float = 2 * π * 8.406
    ωp1: float = 2 * π * 8.406
    ωp2: float = 2 * π * 8.406
    κ1: float = 2 * π * 0.0086
    κ2: float = 2 * π * 0.0086
    δLamb: float = 0.0
    Nexcitations: int = 1
    Up_to_Nexcitations: Optional[int] = None
    waveguide: Optional[Waveguide] = None
    Nqubits: int = 2
    Nfilters: int = 2
    Ncavs: int = 2

    def __post_init__(self):

        self.qubit_indices = np.arange(0, self.Nqubits)
        self.cav_indices = np.arange(self.Nqubits, self.Nqubits + self.Ncavs)
        self.filter_indices = np.arange(
            self.Nqubits + self.Ncavs, self.Nqubits + self.Ncavs + self.Nfilters
        )

        if self.waveguide is None:
            self.waveguide = Waveguide(
                frequency=self.ω1 + self.δLamb, length=5, modes=351
            )
        waveguide = WG = self.waveguide

        if self.Up_to_Nexcitations is not None:
            self.Basis = concatenate_bases(
                qubits=self.Nqubits,
                bosons=self.Ncavs + self.Nfilters + waveguide.modes,
                Up_to_Nexcitations=self.Up_to_Nexcitations,
            )
            self.first_WGmode = len(self.Basis) - 1 - waveguide.modes

        else:
            self.Basis = construct_basis(
                qubits=self.Nqubits,
                bosons=self.Ncavs + self.Nfilters + waveguide.modes,
                excitations=self.Nexcitations,
            )
            self.first_WGmode = len(self.Basis) - waveguide.modes

        self.size = len(self.Basis)

        """ Construct the vector of energies E for filling in the diagonal entries. """

        self.E = np.array(
            [
                self.δ1 + self.δLamb,
                self.δ2 + self.δLamb,
                self.ω1,
                self.ω2,
                self.ωp1,
                self.ωp2,
            ]
            + list(WG.frequencies)
        )

        H_diag = diagonals_with_energies(self.Basis, self.E)

        """Fill in the entries for cavity- waveguide couplings (in this case purcell-waveguide)"""

        H_purcell_waveguide_x0 = sum(
            G
            * move_excitation_operator(
                origin_mode=self.filter_indices[0],
                destination_mode=self.first_WGmode + k,
                basis=self.Basis,
            )
            for k, G in enumerate(waveguide.coupling_strength(self.ω1, self.κ1, 0))
        )
        H_purcell_waveguide_xl = sum(
            G
            * move_excitation_operator(
                origin_mode=self.filter_indices[1],
                destination_mode=self.first_WGmode + k,
                basis=self.Basis,
            )
            for k, G in enumerate(
                waveguide.coupling_strength(self.ω2, self.κ2, waveguide.length)
            )
        )

        gp = np.array([self.gp1, self.gp2])
        H_cavity_purcell = sum(
            [
                g
                * move_excitation_operator(
                    origin_mode=self.cav_indices[i],
                    destination_mode=self.filter_indices[i],
                    basis=self.Basis,
                )
                for i, g in enumerate(gp)
            ]
        )

        """ Up untill here this is the whole static part"""

        H_off_diag = H_cavity_purcell + H_purcell_waveguide_x0 + H_purcell_waveguide_xl

        self.H = H_diag + H_off_diag + H_off_diag.H

        """ Here we initialize the dynamical part, but only for later use in Hamiltonian() """

        self.H_qb1_cav1 = move_excitation_operator(
            origin_mode=self.qubit_indices[0],
            destination_mode=self.cav_indices[0],
            basis=self.Basis,
        )
        self.H_qb2_cav2 = move_excitation_operator(
            origin_mode=self.qubit_indices[1],
            destination_mode=self.cav_indices[1],
            basis=self.Basis,
        )

    def make_Hamiltonian(self, g1, g2):
        """Adds the qubit part to the hamiltonian matrix"""

        Ht = g1 * self.H_qb1_cav1 + g2 * self.H_qb2_cav2
        return self.H + Ht + Ht.H

    def Hamiltonian(self, t=0.0):
        """Adds the qubit part to the hamiltonian matrix"""

        g1 = self.g1
        if not np.isscalar(g1):
            g1 = g1(t)
        g2 = self.g2
        if not np.isscalar(g2):
            g2 = g2(t)
        return self.make_Hamiltonian(g1, g2)

    def qubit_excited(self, which=0):
        if which not in [0, 1]:
            raise Exception(f"Qubit number {which} not in [0,1]")
        C = np.zeros(self.size)
        C[which] = 1
        return C


class Exp_2qubits_2cavities:

    δ1: float = 2 * π * 8.406
    δ2: float = 2 * π * 8.406
    g1: Union[Number, Callable] = 2 * π * 0.0086
    g2: Union[Number, Callable] = 2 * π * 0.0086
    ω1: float = 2 * π * 8.406
    ω2: float = 2 * π * 8.406
    κ1: float = 2 * π * 0.0086
    κ2: float = 2 * π * 0.0086
    δLamb: float = 0.0
    Nexcitations: int = 1
    Up_to_Nexcitations: Optional[int] = None
    waveguide: Optional[Waveguide] = None
    Nqubits: int = 2
    Ncavs: int = 2
    Nfilters: int = 0

    def __post_init__(self):

        """Architecture with 2 qubits, 2 cavities, 2 Purcell filters and waveguide.
        The disposition of this setup is 1qb + 1cav + 1 Pfil + WG and the same at the other end
        ..."""
        self.qubit_indices = np.arange(0, self.Nqubits)
        self.cav_indices = np.arange(self.Nqubits, self.Nqubits + self.Ncavs)

        if waveguide is None:
            waveguide = Waveguide(frequency=self.ω1 + self.δLamb, length=5, modes=351)
        self.waveguide = WG = waveguide

        if self.Up_to_Nexcitations is not None:
            self.Basis = concatenate_bases(
                qubits=self.Nqubits,
                bosons=self.Ncavs + self.Nfilters + waveguide.modes,
                Up_to_Nexcitations=self.Up_to_Nexcitations,
            )
            self.first_WGmode = len(self.Basis) - 1 - waveguide.modes

        else:
            self.Basis = construct_basis(
                qubits=self.Nqubits,
                bosons=self.Ncavs + self.Nfilters + waveguide.modes,
                excitations=self.Nexcitations,
            )
            self.first_WGmode = len(self.Basis) - waveguide.modes

        """ Construct the vector of energies E for filling in the diagonal entries. """

        self.E = np.array(
            [
                self.δ1 + self.δLamb,
                self.δ2 + self.δLamb,
                self.ω1,
                self.ω2,
            ]
            + list(WG.frequencies)
        )

        H_diag = diagonals_with_energies(self.Basis, self.E)

        """Fill in the entries for cavity- waveguide couplings (in this case purcell-waveguide)"""

        H_cavity_waveguide_x0 = sum(
            G
            * move_excitation_operator(
                origin_mode=self.cav_indices[0],
                destination_mode=self.first_WGmode + k,
                basis=self.Basis,
            )
            for k, G in enumerate(waveguide.coupling_strength(self.ω1, self.κ1, 0))
        )
        H_cavity_waveguide_xl = sum(
            G
            * move_excitation_operator(
                origin_mode=self.cav_indices[1],
                destination_mode=self.first_WGmode + k,
                basis=self.Basis,
            )
            for k, G in enumerate(
                waveguide.coupling_strength(self.ω2, self.κ2, waveguide.length)
            )
        )

        """ Up untill here this is the whole static part"""

        H_off_diag = H_cavity_waveguide_x0 + H_cavity_waveguide_xl

        self.H = H_diag + H_off_diag + H_off_diag.H

        """ Here we initialize the dynamical part, but only for later use in Hamiltonian() """

        self.H_qb1_cav1 = move_excitation_operator(
            origin_mode=self.qubit_indices[0],
            destination_mode=self.cav_indices[0],
            basis=self.Basis,
        )
        self.H_qb2_cav2 = move_excitation_operator(
            origin_mode=self.qubit_indices[1],
            destination_mode=self.cav_indices[1],
            basis=self.Basis,
        )

    def make_Hamiltonian(self, g1, g2):
        """Adds the qubit part to the hamiltonian matrix"""

        Ht = g1 * self.H_qb1_cav1 + g2 * self.H_qb2_cav2
        return self.H + Ht + Ht.H

    def Hamiltonian(self, t=0.0):
        """Adds the qubit part to the hamiltonian matrix"""

        g1 = self.g1
        if not np.isscalar(g1):
            g1 = g1(t)
        g2 = self.g2
        if not np.isscalar(g2):
            g2 = g2(t)
        return self.make_Hamiltonian(g1, g2)

    def qubit_excited(self, which=0):
        if which not in [0, 1]:
            raise Exception(f"Qubit number {which} not in [0,1]")
        C = np.zeros(self.size)
        C[which] = 1
        return C
