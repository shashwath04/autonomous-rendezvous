import numpy as np

# FOR BOTH THE KF AND EKF ADD NIS GATING FOR BAD MEASUREMENTS

class KalmanFilter:
    def __init__(self, Ad, Bd, Cd, Q, R, P0=None):
        self.Fk = np.asarray(Ad, dtype=float)
        self.Bk = np.asarray(Bd, dtype=float)
        self.Hk = np.asarray(Cd, dtype=float)

        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)

        self.n = self.Fk.shape[0]
        self.m = self.Hk.shape[0]

        self.P_post = (0.1 * np.eye(self.n) if P0 is None else np.asarray(P0, dtype=float).copy())

        self.P_pred = None
        self.x_pred = None

    def predict(self, x, u):
        x = np.asarray(x, dtype=float).reshape(self.n, 1)
        u = np.asarray(u, dtype=float).reshape(self.Bk.shape[1], 1)

        self.x_pred = self.Fk @ x + self.Bk @ u
        self.P_pred = (self.Fk @ self.P_post @ self.Fk.T + self.Q)

        return self.x_pred.copy()

    def update(self, measurement):
        if self.x_pred is None: raise RuntimeError("Call predict() before update()")

        measurement = np.asarray(measurement, dtype=float).reshape(self.m, 1)
        residual = measurement - self.Hk @ self.x_pred

        S = self.Hk @ self.P_pred @ self.Hk.T + self.R

        PHt = self.P_pred @ self.Hk.T
        K = np.linalg.solve(S.T, PHt.T).T

        x_update = self.x_pred + K @ residual

        I_KH = np.eye(self.n) - K @ self.Hk
        self.P_post = (I_KH @ self.P_pred @ I_KH.T + K @ self.R @ K.T) # using jacob's form to make sure there isn't any instability

        self.P_post = 0.5 * (self.P_post + self.P_post.T)

        self.x_pred = None
        self.P_pred = None

        return x_update



import numpy as np


class ExtendedKalmanFilter:
    def __init__(self, Ad, Bd, Q, x0, P0):
        self.Ad = np.asarray(Ad, dtype=float)
        self.Bd = np.asarray(Bd, dtype=float)
        self.Q = np.asarray(Q, dtype=float)

        self.x = np.asarray(x0, dtype=float).reshape(6, 1)
        self.P = np.asarray(P0, dtype=float).reshape(6, 6).copy()

    def predict(self, u):
        u = np.asarray(u, dtype=float).reshape(self.Bd.shape[1], 1)

        self.x = self.Ad @ self.x + self.Bd @ u
        self.P = self.Ad @ self.P @ self.Ad.T + self.Q
        self.P = 0.5 * (self.P + self.P.T)

    def _update(self, z, predicted_measurement, H, R, angular=False):
        z = np.asarray(z, dtype=float).reshape(-1, 1)
        predicted_measurement = np.asarray(
            predicted_measurement, dtype=float
        ).reshape(-1, 1)

        H = np.asarray(H, dtype=float)
        R = np.asarray(R, dtype=float)

        residual = z - predicted_measurement

        if angular:
            residual = (residual + np.pi) % (2 * np.pi) - np.pi

        S = H @ self.P @ H.T + R
        PHt = self.P @ H.T
        K = np.linalg.solve(S.T, PHt.T).T

        self.x = self.x + K @ residual

        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)

    def update_gps(self, measurement, R):
        H = np.eye(6)
        self._update(measurement, H @ self.x, H, R)

    def update_ffrf(self, measurement, R):
        H = np.hstack((np.eye(3), np.zeros((3, 3))))
        self._update(measurement, H @ self.x, H, R)

    def update_vbs(self, measurement, R, C_camera_orbital):

        if measurement is None:
            return False

        C = np.asarray(C_camera_orbital, dtype=float).reshape(3, 3)

        target_camera = -C @ self.x[:3]
        xc, yc, zc = target_camera.ravel()

        if zc <= 1e-9:
            return False

        predicted_angles = np.array([
            [np.arctan2(xc, zc)],
            [np.arctan2(yc, zc)],
        ])

        J = np.array([
            [zc / (xc**2 + zc**2),
             0.0,
             -xc / (xc**2 + zc**2)],

            [0.0,
             zc / (yc**2 + zc**2),
             -yc / (yc**2 + zc**2)],
        ])

        H = np.zeros((2, 6))
        H[:, :3] = J @ (-C)

        self._update(
            measurement, predicted_angles, H, R, angular=True
        )
        return True