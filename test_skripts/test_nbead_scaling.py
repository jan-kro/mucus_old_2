from polymer.main import run_sim
import os

cwd = os.getcwd()
cfg_path = cwd + "/configs/tests"
n = 5
for i in range(n):
    fin = cfg_path + f"/cfg_test_time_nbeads_2_{i:d}.toml"
    
    print(f"\n\nRunning simulation {i+1:d}/{n:d} ...\n")
    run_sim(fname_config=fin)

print("\nAll simulations finished.")