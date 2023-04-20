import numpy as np
from polymer import Config, Polymer
import time
import datetime

# here i want to create four intertwined sinusioidal chains in a mesh like (#) shape

nbpc = 200 # number of beads per chain
r0 = 2 # beed-beed distance
lbox = r0*nbpc # naive box length

omega = 2*np.pi/(r0*nbpc) # approximate wavelength as the chain length (works only well for long chains)

# create one chain and then copy it
# chain lies in the yz plane and goes in y direction 

# start by creating a sine wave containing 10 times the amount of points  
x = np.linspace(0, r0*nbpc, 10*nbpc)
y = 0.5*r0*np.sin(omega*x)
r_sin = np.array([np.zeros(len(x)), x, y]).T


pos1 = np.zeros((nbpc, 3)) # positions

idx_sin = np.arange(len(r_sin))
# the first position is [0 0 0] so the loop starts at idx 1
for i in range(1, nbpc):
    r = r_sin - pos1[i-1, :]
    d = np.linalg.norm(r, axis=1)
    L = np.abs(d-r0) == min(np.abs(d-r0))
    pos1[i, :] = r_sin[L, :]
    # cut r sin so that the next point can only go in one direction
    r_sin = r_sin[int(idx_sin[L]):]
    idx_sin = np.arange(len(r_sin))
    
# copy and shift to grid shape
pos2 = np.array([pos1[:, 0],pos1[:, 1], -pos1[:, 2]]).T
pos2[:, 0] += lbox/2 + r0*nbpc/4

pos3 = np.array([pos2[:, 1],pos2[:, 0], pos2[:, 2]]).T
pos4 = np.array([pos1[:, 1],pos1[:, 0], pos1[:, 2]]).T

pos1[:, 0] += r0*nbpc/4
pos3[:, 1] = r0*nbpc/4
pos4[:, 1] = r0*nbpc/4 + lbox/2

# concatenate arrays
pos = np.concatenate([pos1, pos2, pos3, pos4])

# shift so everything is centered
lx_chain = np.abs(pos1[0,1] - pos1[nbpc-1,1])
l_diff = lbox - lx_chain
pos += np.array((l_diff/2, l_diff/2, lbox/2))

# create bond list
bonds = list(())
for i in range(4):
    bonds.append((i*nbpc, i*nbpc+1))
    for k in range(i*nbpc + 1, (i+1)*nbpc-1):
        bonds.append((k, k-1))
        bonds.append((k, k+1))
    bonds.append((k+1, k))

    
# create config file
config_dict = {'steps': 1000000, 
               'stride': 40, 
               'number_of_beads': nbpc*4, 
               'nbeads': nbpc, 
               'nchains': 4, 
               'mobility': 5e-05,
               'qbead': 0,
               'dir_output': '/storage/janmak98/masterthesis/ouput/mesh',
               'bonds': bonds}

cfg = Config.from_dict(config_dict)
p = Polymer(cfg)
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

# os.mkdir(p.config.dir_output+f"/Sq")

# print("\ncalculate structure factor")
# Q, S_q = p.get_structure_factor_rdf()
# np.save(p.config.dir_output+f"/Sq/Q.npy", Q)
# np.save(p.config.dir_output+f"/Sq/Sq.npy", S_q)

print("\ndone")

