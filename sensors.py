import numpy as np

class PhoeinxS:
    def __init__(self, position_std, velocity_std, seed=None):
        self.type = 'GPS'
        self.rng = np.random.default_rng(seed)

        self.std = np.concatenate((
            np.broadcast_to(position_std, (3,)),
            np.broadcast_to(velocity_std, (3,)),
        )).astype(float)

        if not np.all(np.isfinite(self.std)) or np.any(self.std < 0):
            raise ValueError("Noise standard deviations must be finite and nonnegative")

        self.H = np.eye(6)
        self.R = np.diag(self.std**2)


    def measure(self, true_state):
        state = np.asarray(true_state, dtype=float).reshape(6, 1)
        noise = self.rng.normal(
            loc = 0.0,
            scale = self.std[:, None],
            size = (6, 1)
        )

        return self.H @ state + noise

class FFRF():
    def __init__(self, position_std, seed=None):
        self.type = "FFRF"
        self.rng = np.random.default_rng(seed)

        self.std = np.broadcast_to(
            position_std, (3,)
        ).astype(float).copy()

        if not np.all(np.isfinite(self.std)) or np.any(self.std < 0):
            raise ValueError("Noise standard deviations must be finite and nonnegative")

        self.H = np.hstack((np.eye(3), np.zeros((3, 3))))
        self.R = np.diag(self.std**2)

    def measure(self, true_state):
        state = np.asarray(true_state, dtype=float).reshape(6, 1)
        noise = self.rng.normal(
            0.0, self.std[:, None], size=(3, 1)
        )

        return self.H @ state + noise

class VBS():
    def __init__(self, angle_std, horizontal_fov_deg=60.0, vertical_fov_deg=60.0, seed=None):
        self.type = "VBS"
        self.rng = np.random.default_rng(seed)

        self.std = np.broadcast_to(
            angle_std, (2,)
        ).astype(float).copy()

        if not np.all(np.isfinite(self.std)) or np.any(self.std < 0):
            raise ValueError("Noise standard deviations must be finite and nonnegative")


        fov = np.array([horizontal_fov_deg, vertical_fov_deg])
        if not np.all(np.isfinite(fov)) or np.any((fov <= 0) | (fov >= 180)):
            raise ValueError("FOV must be between 0 and 180 degrees")

        self.half_fov = np.deg2rad(fov) / 2
        self.R = np.diag(self.std**2)

    def measure(self, target_position_camera):
        x, y, z = np.asarray(
            target_position_camera, dtype=float
        ).reshape(3)

        if not np.all(np.isfinite([x, y, z])):
            raise ValueError("Target position must be finite")

        if z <= 0:
            return None  

        angles = np.array([
            np.arctan2(x, z),
            np.arctan2(y, z),
        ])

        if np.any(np.abs(angles) > self.half_fov):
            return None

        noise = self.rng.normal(0.0, self.std, size=2)
        return (angles + noise).reshape(2, 1)







    


