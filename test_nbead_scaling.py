from polymer.main import run_sim
import os

cwd = os.getcwd()
cfg_path = cwd + "/configs/tests"

for i in range(6):
    fin = cfg_path + f"/cfg_test_time_nbeads_{i:d}.toml"
    
    print(f"\n\nRunning simulation {i+1:d}/5 ...\n")
    run_sim(fname_config=fin)

print("\nAll simulations finished.")