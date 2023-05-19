import numpy as np

import cegis.verifier as verifier
from config import VerifierType


class Mode:
    def __init__(self, flow, invariant, disturbance=None) -> None:
        """Represents a single mode in a linear hybrid automaton"""
        self.A, self.c = flow
        self.ndim = self.A.shape[0]
        self.P = invariant
        if disturbance is None:
            self.disturbance = np.zeros((self.ndim))
        else:
            self.disturbance = np.array(disturbance)
        self.rand = np.random.default_rng()

    def flow_upper(self, x):
        """Returns the upper bound of the mode's output for a given input"""
        return (
            self.A @ np.array(x).T
            + self.c.reshape(-1, 1)
            + np.array(self.disturbance).reshape(-1, 1)
        )

    def flow_lower(self, x):
        """Returns the lower bound of the mode's output for a given input"""
        return (
            self.A @ np.array(x).T
            + self.c.reshape(-1, 1)
            - np.array(self.disturbance).reshape(-1, 1)
        )

    def flow_random(self, x):
        """Returns the mode's output for a given input with noise"""
        d = self.rand.uniform(0, self.disturbance.max(), self.disturbance.shape)
        return (self.A @ np.array(x) + self.c + d).tolist()

    def flow_no_disturbance(self, x):
        """Returns the mode's output for a given input without disturbance"""
        return (self.A @ np.array(x).T) + self.c.reshape(-1, 1)

    def flow_smt(self, x):
        """Returns the flow without disturbance in SMT format"""
        return (self.A @ np.array(x) + self.c.reshape(1, -1)).tolist()

    def flow_disturbance_smt(self, x):
        """Returns the flow with disturbance in SMT format"""
        return (
            self.A @ np.array(x)
            + self.c.reshape(1, -1)
            + self.disturbance.reshape(1, -1)
        ).tolist()

    def contains(self, x):
        """Checks if a given input is in the mode's domain"""
        return self.P.contains(x)

    def as_smt(self, x, And):
        """Represent the mode as SMT formula for the invariant and flow"""
        return self.P.as_smt(x, And), self.flow_smt(x)

    def check_disturbance(self, benchmark, S, ver_type=verifier.DRealVerifier):
        S = S[self.P.contains(S)].numpy()
        if S.shape[0] == 0:
            self.disturbance = list(self.disturbance)
            return
        stop = False
        attempts = 0
        improve_attempt = False
        Sdot = benchmark.f(S).T  # np.array(list(map(benchmark.f, S))).T
        y = self.flow_no_disturbance(S)
        e = abs((Sdot - y)).max(axis=1)
        e = (e / 0.9).tolist()
        ver_type = (
            verifier.Z3Verifier
            if ver_type == VerifierType.Z3
            else verifier.DRealVerifier
        )
        x = ver_type.new_vars(self.ndim)
        fs = np.array(benchmark.f(x)).reshape(-1, 1)
        (
            Ps,
            Ms,
        ) = self.as_smt(x, ver_type.solver_fncts()["And"])
        Ms = np.array(Ms).reshape(-1, 1)
        ver = ver_type(x, self.ndim, lambda *args: Ps, verbose=False)
        while not stop and attempts < 20:
            e = [round(max(ei, 0.005), ndigits=3) for ei in e]
            res, cex = ver.verify(fs, Ms, e)
            if res:
                # print(self.disturbance - e)
                self.disturbance = e
                improve_attempt = True  # After the first successful attempt, we try to improve the disturbance until we fail
                e = [round(max(ei * 0.95, 0.005), ndigits=3) for ei in e]
                if e == [0.005 for i in range(self.ndim)]:
                    stop = True

            else:
                if cex == []:
                    # Z3 returned unknown, what to do?
                    e = [round(ei * 1.05, ndigits=3) for ei in e]
                else:
                    S = np.vstack((S, cex))
                    Sdot = benchmark.f(S).T  # np.array(list(map(benchmark.f, S))).T
                    y = self.flow_no_disturbance(S)
                    e = abs((Sdot - y)).max(axis=1)
                    e = (e / 0.9).tolist()
                    e = [round(max(ei, 0.005), ndigits=3) for ei in e]
                if improve_attempt:
                    # Failed to improve after initial success, so we stop
                    stop = True
            attempts += 1

    def inv_str(self, vx=[], vu=[], sep="\n"):
        """Returns the invariant as a string"""
        s = self.P.to_str(vx=vx, sep=sep) + sep
        vu = vu if vu else ["u" + str(i) for i in range(self.ndim)]
        for i, ui in enumerate(vu):
            s += ui + " <={}".format(self.disturbance[i])
            s += sep + ui + " >= -{}".format(self.disturbance[i])
            s += sep
        s = s[:-2]  # remove final &
        return s

    def flow_str(self, vx=[], vu=[], sep="\n"):
        """Get string representation of flow"""
        s = ""
        vx = vx if vx else ["x" + str(i) for i in range(self.ndim)]
        vu = vu if vu else ["u" + str(i) for i in range(self.ndim)]
        if len(vx) != self.ndim:
            raise ValueError("Number of variables does not match")
        if len(vu) != self.ndim:
            raise ValueError("Number of inputs does not match")
        if not all(isinstance(item, str) for item in vx):
            raise ValueError("vx must be a list of strings")
        s = ""
        for i, var in enumerate(vx[:-1]):
            A_row = self.A[i, :]
            b = self.c[i]
            s += var + "'=="
            fi = " + ".join([str(A_row[j]) + " * " + vx[j] for j in range(self.ndim)])
            s += fi + "+ " + str(b) + " + " + vu[i]
            s += sep

        A_row = self.A[-1, :]
        b = self.c[-1]
        t = "+ ".join([str(A_row[j]) + " * " + vx[j] for j in range(self.ndim)])

        s += vx[-1] + "'=="
        s += t + "+ " + str(b) + " + " + vu[-1]
        return s
