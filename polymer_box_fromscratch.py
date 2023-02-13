from polymer_chain import Polymer
from time import time
import os




n_beeds = 10
n_chains = 8
mobility = 0.0005
r_0 = 2
l_box = n_beeds*r_0

steps = 500000
stride = 50

fname = f"/storage/janmak98/masterthesis/trajectories/box/Traj_{n_chains:d}Chains_{n_beeds:d}BeadsPerChain_{steps:d}Steps_{mobility:.6f}mu_{stride:d}Stride.gro"

p = Polymer(mobility=mobility,
            cwd=os.getcwd())
p.create_box(N=n_beeds,
             M=n_chains,
             L=l_box)

t1 = time()
p.simulate(steps=steps, 
           stride=stride)
t2 = time()

print("simulation time")
print(f"{(t2-t1)//60:.0f}:{(t2-t1)%60:2.0f} min")

#p.plot_distance_distribution()
#p.save_traj_gro(fname_traj=fname)
print("Done")

