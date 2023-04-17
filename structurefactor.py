import numpy as np
from polymer import Config, Polymer
import os
import time
import datetime

cfg = Config.from_toml("/home/jan/Documents/masterthesis/project/mucus/cfg_structurefactor.toml")
p = Polymer(cfg)
p.create_box()

print("starting a simulation with following parameters:")
p.print_sim_info()

now = time.localtime()
now_str = f"{now.tm_mon:d}.{now.tm_mday}.{now.tm_year}  {now.tm_hour}:{now.tm_min}:{now.tm_sec}"
print("\nsimulation started ", now_str)

p.simulate()

now = time.localtime()
now_str = f"{now.tm_mon:d}.{now.tm_mday}.{now.tm_year}  {now.tm_hour}:{now.tm_min}:{now.tm_sec}"

print("simulation finished ", now_str)

print("simulation time: ", datetime.timedelta(seconds=round(p.config.simulation_time)))

print("\nsave system...")
p.save_system()

os.mkdir(p.config.dir_output+f"/Sq")

print("\ncalculate structure factor")
Q, S_q = p.get_structure_factor_rdf()
np.save(p.config.dir_output+f"/Sq/Q.npy", Q)
np.save(p.config.dir_output+f"/Sq/Sq.npy", S_q)

print("\ndone")