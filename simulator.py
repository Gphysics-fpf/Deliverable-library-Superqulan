import numpy as np
import scipy.sparse.linalg


def Trotter_solver_dynamics(time: np.ndarray, v0: np.ndarray, H: callable):

    vt = [v0]
    last_t = time[0]

    for t in time[1:]:
        dt = t - last_t

        v0 = scipy.sparse.linalg.expm_multiply((-1j * dt) * H(t), v0)
        vt.append(v0)

        last_t = t

    return np.array(vt).T
