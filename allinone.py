import numpy as np
from polymer import Config, Polymer
from polymer.main import run_sim
import sys
import os
import time
import datetime

# what I want to do

# create trajectories and distance trajectories
#   Chains
#       NN
#       NN+LJ
#       NN+LJ+Deb
#   Box
#       many chains
# out of this calculate

#   rg
#   r_end_end
#   rdf
#   s(q) using debye(?)
 
#   test for 1 chain and 1 box

# (0) DEFINE NECESSARY FUNCTIONS

# calculate distance matrix trajectory
def distances(traj):
    n = len(traj[0]) # number of atoms
    distances = np.zeros((len(traj), n , n))
    
    r_left = np.tile(traj[0], (n, 1, 1)) # repeats vector along third dimension len(a) times
    r_right = np.reshape(np.repeat(traj[0], n, 0), (n, n, 3)) # does the same but "flipped"

    directions = r_left - r_right # this is right considering the mesh method. dir[i, j] = r_j - r_i
    distances[0] = np.linalg.norm(directions, axis=2)
    
    for k, frame in enumerate(traj[1:]):
        r_left = np.tile(frame, (n, 1, 1)) # repeats vector along third dimension len(a) times
        r_right = np.reshape(np.repeat(frame, n, 0), (n, n, 3)) # does the same but "flipped"

        directions = r_left - r_right # this is right considering the mesh method. dir[i, j] = r_j - r_i
        distances[k+1] = np.linalg.norm(directions, axis=2)
        
    return distances

def allinone(dist_traj):
    # radius of gyration
    n = len(dist_traj[0])
    rg = np.sum(dist_traj, axis=0)/len(dist_traj)
    rg = np.sum(np.triu(rg))*2/(n**2 -n)
    
    ree = list()   # end-end dist
    gr = list()     # RDF
    for frame in dist_traj:
        ree.append(frame[0,-1])
        gr.append(np.triu(frame).flatten())
        
    return rg, ree, gr

# (1) CREATE FILE PATHNAMES AND CONFIGS

descriptors = ("#chain nn",
               "#chain nn lj",
               "#chain nn lj deb",
               "#box no charge",
               "#box normal charge",
               "#box double charge",
               "#box 40mM salt",
               "#box 25mM salt")

# create config paths
dir_data = '/storage/janmak98/masterthesis/pres'
dirs_cfg = dir_data + "/configs"
dirs_traj = dir_data + "/trajectories"
dirs_calc = dir_data + "/calculations"

cfg1 = {'steps':              300000,
        'stride':             60,
        'number_of_beads':    50,
        'nbeads':             50,
        'nchains':            1,
        'mobility':           5e-04,
        'qbead':              0, # 2.08
        'force_constant':     100.0,
        'epsilon_LJ':         0,
        'lB_debye':           1, # 3.077
        'c_S':                10.0,
        'fname_sys':          dirs_cfg+f"/cfg1.toml",
        'fname_traj':         dirs_traj+f"/traj1.gro"}

cfg2 = {'steps':              300000,
        'stride':             60,
        'number_of_beads':    50,
        'nbeads':             50,
        'nchains':            1,
        'mobility':           5e-04,
        'qbead':              0, # 2.08
        'force_constant':     100.0,
        'lB_debye':           1, # 3.077
        'c_S':                10.0,
        'fname_sys':          dirs_cfg+f"/cfg2.toml",
        'fname_traj':         dirs_traj+f"/traj2.gro"}

cfg3 = {'steps':              300000,
        'stride':             60,
        'number_of_beads':    50,
        'nbeads':             50,
        'nchains':            1,
        'mobility':           5e-04,
        'force_constant':     100.0,
        'c_S':                10.0,
        'fname_sys':          dirs_cfg+f"/cfg3.toml",
        'fname_traj':         dirs_traj+f"/traj3.gro"}

cfg4 = {'steps':              600000,
        'stride':             60,
        'number_of_beads':    160,
        'nbeads':             20,
        'nchains':            8,
        'mobility':           5e-04,
        'qbead':              0, # 2.08
        'force_constant':     100.0,
        'lB_debye':           1,
        'c_S':                10.0,
        'fname_sys':          dirs_cfg+f"/cfg4.toml",
        'fname_traj':         dirs_traj+f"/traj4.gro"}

cfg5 = {'steps':              600000,
        'stride':             60,
        'number_of_beads':    160,
        'nbeads':             20,
        'nchains':            8,
        'mobility':           5e-04,
        'force_constant':     100.0,
        'c_S':                10.0,
        'fname_sys':          dirs_cfg+f"/cfg5.toml",
        'fname_traj':         dirs_traj+f"/traj5.gro"}

cfg6 = {'steps':              500000,
        'stride':             60,
        'number_of_beads':    160,
        'nbeads':             20,
        'nchains':            8,
        'mobility':           5e-04,
        'qbead':              2*1.524, # 2.08
        'force_constant':     100.0,
        'c_S':                10.0,
        'fname_sys':          dirs_cfg+f"/cfg6.toml",
        'fname_traj':         dirs_traj+f"/traj6.gro"}

cfg7 = {'steps':              600000,
        'stride':             60,
        'number_of_beads':    160,
        'nbeads':             20,
        'nchains':            8,
        'mobility':           5e-04,
        'force_constant':     100.0,
        'c_S':                40.0,
        'fname_sys':          dirs_cfg+f"/cfg7.toml",
        'fname_traj':         dirs_traj+f"/traj7.gro"}

cfg8 = {'steps':              600000,
        'stride':             60,
        'number_of_beads':    160,
        'nbeads':             20,
        'nchains':            8,
        'mobility':           5e-04,
        'force_constant':     100.0,
        'c_S':                40.0,
        'fname_sys':          dirs_cfg+f"/cfg7.toml",
        'fname_traj':         dirs_traj+f"/traj7.gro"}

# cfgs = (cfg1, cfg2, cfg3, cfg4, cfg5, cfg6, cfg7)
# cfgs = (cfg4, cfg5, cfg6, cfg7)
cfgs = (cfg1, cfg2, cfg4, cfg7)

# (2) SIMULATE

# loop through systems
# indexes = np.arange(len(cfgs))
indexes = (6, 8)
for k, cfg_dict in zip(indexes, cfgs):
    if k == 7:
        break
    cfg = Config.from_dict(cfg_dict)
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
    
    cfg_fname = cfg_dict['fname_sys']
    
    # add descriptors
    f = open(cfg_fname)
    output = descriptors[k]+"\n"
    for line in f:
        output += line
    f.close()

    f = open(cfg_fname, "w")
    f.write(output)
    f.close()
    
    # calculate distances for every 200 steps
    dist = distances(p.trajectory[::200])
    np.save(dirs_calc+f"/dist{k:d}.npy", dist)
    
    # calculate allinone
    rg, ree, gr = allinone(dist)
    
    np.save(dirs_calc+f"/rg{k:d}.npy", rg)
    np.save(dirs_calc+f"/ree{k:d}.npy", np.array(ree))
    np.save(dirs_calc+f"/gr{k:d}.npy", np.array(gr))
    
    # k += 1
    
    now = time.localtime()
    now_str = f"{now.tm_mon:d}.{now.tm_mday}.{now.tm_year}  {now.tm_hour}:{now.tm_min}:{now.tm_sec}"

    print("calculation finished ", now_str)
    print("\ndone")
    
