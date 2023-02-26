import numpy as np
from polymer import Config, Polymer

import sys
import time
import datetime


def create_wind(nc, d):
    nch = nc//2

    pc1 = np.array((np.arange(nc)*d, np.zeros(nc),np.zeros(nc))).T
    pc1 += (0, nc, nc)

    pc2 = np.zeros((nc, 3))
    pc2[:nch, 1] = np.arange(nch)*d
    pc2[:nch, 2] += d

    pc2[nch, 1] = pc2[nch-1, 1]
    pc2[nch+1, 1] = pc2[nch-1, 1]
    pc2[nch+1, 2] = -d

    pc2[nch+2:, 1] = d*np.fliplr([np.arange(nch-2)]) + d
    pc2[nch+2:, 2] -= d 

    pc2 += (nc, nch, nc)

    p = np.append(pc1, pc2, axis=0)
    return p

# SIMULATE

# (1) close chains
#~~~~~~~~~~~~~~~~~
config = Config.from_toml(sys.argv[1])
p = Polymer(config)

p.create_box()

# put chains closter together
pos = p.positions
pos[60:, 2] -= 38
p.set_positions(pos)

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
print("\ndone")

# (2) wound chains
# ~~~~~~~~~~~~~~~~
p = Polymer(config)

p.create_box()

# wind chains
pos = create_wind(60, 2)
p.set_positions(pos)

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
print("\ndone")
