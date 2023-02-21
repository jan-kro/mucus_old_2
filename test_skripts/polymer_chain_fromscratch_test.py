from polymer_chain import Polymer
from time import time
import os


n_beeds = 20 #50
mobility = 0.00005

steps = 1000 #100000
stride = 50

#fname = f"/storage/janmak98/masterthesis/trajectories/single_chain/traj_{n_beeds:d}beads_{steps:d}steps_{mobility:.6f}mu_{stride:d}stride.gro"

p = Polymer(n_beeds=n_beeds,
            mobility=mobility,
            cwd=os.getcwd())

t1 = time()
p.simulate(steps=steps, 
           stride=stride)
t2 = time()

print("simulation time")
print(f"{(t2-t1)//60:.0f}:{(t2-t1)%60:2.0f} min")

#p.plot_distance_distribution()
#p.save_traj_gro(fname_traj=fname)
p.save_traj_gro()

