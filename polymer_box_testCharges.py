from polymer.main import run_sim
import os

n = 6
for k in range(n):
    fname = os.getcwd() + f"configs/tests/box_testCharges_100beads_1Msteps_{k:d}.toml"
    run_sim(type="box", fname_config=fname)