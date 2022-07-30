# Deliverable-library-Superqulan

Library created within the context of the european project Superqulan whose purpose is to offer a simple way to do simulations in distributed quantum computing architechtures.

We propose an efficient way to represent quantum states and operators that allows for the simulations of quantum systems of relativelyh big sizes without a ver extensive memory consumption. The aim of the library is also to be flexible and allow the user to switch from one kind of architecture to another as well as from different kinds of evolution routines.


## Dependences

This package needs the following packages:

```bash
pip install numpy
pip install scipy.sparse
pip install joblib
pip install itertools
pip install math
```

## Structure 
This github repository is organized ad follows:

There are four main documents with the complete information needed to run any experiment. 

- `bosons.py` contains the routines that build up a Hilbert space basis together with the most important operators, the information required is the number of fermionic modes, the number of bosonic modes and the amount of excitations.
- `waveguide.py` constructs the object that joins the different quantum nodes. One can specify the length as well as the number of modes.
- `simulator.py` contains the function to implement time evolution.
- `architecture.py` makes use of bosons.py and waveguide.py to create full distributed quantum architectures consisting of nodes and links.

In the foder [examples](https://github.com/Gphysics-fpf/Deliverable-library-Superqulan/tree/main/examples) there are several notebooks that show how to make use of the library tu run quantum experiments. 

## Usage

## References
[1] Phys. Rev. Applied 17, 054038 – (2022)

## TO DO list 
- [ ] Fill in the usage subsection with a minimal example of use of the library.
- [ ] Extend the type declaration to whole repository. 
- [ ] Think what is the best place to accomodate the controls, in particular the routines to compute complex controls.
- [ ] Upgrade waveguide such that it allows you to imlement different dispersion relations.
- [ ] Think how to fit the quantum trajectories rutines in all of this, imo together with the evolution rutines is the best place.
- [ ] Think of further examples to ilustrate the library, besides the ones we already have. GOOD CANDIDATES: State transfer with transmons, multipartite entanglement (maybe this one making use of two excitations).


