import numpy as np
from polymer import Config, Polymer
import os
import time

names = ('chain_testForces_50beads_1Msteps', 'box_testCharges_100beads_1Msteps')
dir_data = '/storage/janmak98/masterthesis'

cfg_names = list(())
cfg_names.append(dir_data+'/configs/box/test/cfg_box_twoChains_140beads_2Msteps.toml')
traj_names = list(())
traj_names.append(dir_data + '/trajectories/box/test/traj_box_twoChains_140beads_2Msteps.gro')
dist_names = list(())
dist_names.append(dir_data + '/distances/box/test/dist_box_twoChains_140beads_2Msteps.npy')

for name in names:
    for i in range(4):
        cfg_fname = f'{dir_data:s}/configs/box/test/cfg_{name:s}_{i:d}.toml'
        traj_fname = f'{dir_data:s}/trajectories/box/test/traj_{name:s}_{i:d}.gro'
        dist_fname = f'{dir_data:s}/distances/box/test/dist_{name:s}_{i:d}.npy'
        
        cfg_names.append(cfg_fname)
        traj_names.append(traj_fname)
        dist_names.append(dist_fname)
        
for fname in cfg_names:
    if not os.path.exists(fname):
        raise NameError('File '+fname+' does not exist!')
    
for fname in cfg_names:
    if not os.path.exists(fname):
        raise NameError('File '+fname+' does not exist!')

# delete lbox = None in .toml file
def delete_lbox(cfg_fname):
    f = open(cfg_fname)
    output = ""
    for line in f:
        if line != "lbox = None\n":
            output += line
    f.close()

    f = open(cfg_fname, "w")
    f.write(output)
    f.close()
    return

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

N = len(cfg_names)
i = 1
n_frames = 1000

now = time.localtime()
now_str = f"{now.tm_mon:d}.{now.tm_mday}.{now.tm_year}  {now.tm_hour}:{now.tm_min}:{now.tm_sec}"
print("\nCalculation started ", now_str)

for cfg_fname, traj_fname, dist_fname in zip(cfg_names, traj_names, dist_names):
    
    dist_fname = dist_fname[:-4] + f"_{n_frames:d}.npy"
    
    delete_lbox(cfg_fname)

    cfg = Config.from_toml(cfg_fname)
    p = Polymer(cfg)
    p.load_traj_gro(traj_fname, overwrite=True)

    stride = np.round(len(p.trajectory)/n_frames)
    
    d = distances(p.trajectory[::stride])
    np.save(dist_fname, d)
    
    print(f'\n{i:d}/{N:d} distances calculated')
    i += 1
    
now = time.localtime()
now_str = f"{now.tm_mon:d}.{now.tm_mday}.{now.tm_year}  {now.tm_hour}:{now.tm_min}:{now.tm_sec}"

print("\nCalculation finished ", now_str)