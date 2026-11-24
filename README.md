# Satellite Rendezvous

The closed-loop example estimates and reduces the separation between a chaser
and a chief in a circular Earth orbit. It combines CW-based discrete LQR,
GPS/FFRF/VBS measurements fused with an EKF, and attitude control that transitions
from thruster detumbling to wheel pointing and thruster-assisted wheel unloading.

## Run the example

Use Python 3.10 and the dependencies in `requirements.txt`. Run notebooks
from the repository root and use the same Python environment for the kernel.

```sh
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter notebook simulation_test.ipynb
```

The notebook runs Basilisk and saves its results to
`saved_runs/rendezvous_basilisk.npz`. The standalone reference is also
available through the same call and output format.

```python
from basilisk_runner import run_rendezvous
from dynamics import orbital_rate
from simulation import simulate

n = orbital_rate(6.778e6, 3.986004418e14)
initial_state = [100, 0, 0, 0, 0, 0]
basilisk = run_rendezvous(n, initial_state, dt=0.1, T=1000)
reference = simulate(n, initial_state, dt=0.1, T=1000)
```

The initial state is the chaser's position in meters and rotating-frame velocity
in meters per second, ordered radial, along-track, and cross-track. The returned
tuple contains time, true relative state, applied acceleration, estimated
relative state, attitude, wheel reaction torque on the body, and RCS torque.
State, estimate, and attitude include the initial sample. Control histories
describe the intervals between samples. The attitude history contains a
body-to-orbital quaternion in xyzw order, body angular velocity expressed in
body axes, and wheel momentum.

## Code layout

- `rendezvous/flight_control.py` shares control, actuator allocation, and
  sensor update timing between the two backends.
- `rendezvous/simulation.py` runs the standalone CW/RK4 reference.
- `rendezvous/basilisk/closed_loop.py` runs the sampled controller against
  Basilisk's two-body spacecraft dynamics and native balanced reaction wheels.
  A Basilisk force/torque effector applies the thruster allocator's duty-averaged
  body wrench during each control interval.
- `rendezvous/basilisk/orbit.py` provides the existing J2/drag orbit and pulsed
  thruster scenario through `basilisk_runner.run_orbit`. The numbered notebooks
  cover open-loop validation, actuator tests, and momentum dumping.

Orbital transfer, RRT, LQT, and genetic-algorithm cost tuning are future work.
