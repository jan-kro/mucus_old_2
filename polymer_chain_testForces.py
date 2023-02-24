from polymer.main import run_sim
import os

n = 4
for k in range(n):
    fname = os.getcwd() + f"/configs/tests/chain_testForces_50beads_1Msteps_{k:d}.toml"
    run_sim(type="box", fname_config=fname)