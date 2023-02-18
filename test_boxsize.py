from polymer.main import run_sim
import os

cwd = os.getcwd()
cfg_path = cwd + "/configs/tests"
cfg_name = "test_boxsize"
n = 5
for i in range(n):
    fin = cfg_path + f"/cfg_{cfg_name:s}_{i:d}.toml"
    
    print(f"\n\nRunning simulation {i+1:d}/{n:d} ...\n")
    run_sim(fname_config=fin)

print("\nAll simulations finished.")