from polymer_chain import Polymer
import time
import datetime
import os


n_beeds = 10
n_chains = 8
mobility = 0.0005
r_0 = 2
l_box = n_beeds*r_0

steps = 5000
stride = 50

fname = f"/storage/janmak98/masterthesis/trajectories/box/Traj_{n_chains:d}Chains_{n_beeds:d}BeadsPerChain_{steps:d}Steps_{mobility:.6f}mu_{stride:d}Stride.gro"

now = time.localtime()
now_str = f"{now.tm_mon:d}.{now.tm_mday}.{now.tm_year}  {now.tm_hour}:{now.tm_min}:{now.tm_sec}"
print("simulation started ", now_str)

p = Polymer(mobility=mobility,
            cwd=os.getcwd())
p.create_box(N=n_beeds,
             M=n_chains,
             L=l_box)

t1 = time.time()
p.simulate(steps=steps, 
           stride=stride)
t2 = time.time()

now = time.localtime()
now_str = f"{now.tm_mon:d}.{now.tm_mday}.{now.tm_year}  {now.tm_hour}:{now.tm_min}:{now.tm_sec}"

print("simulation finished ", now_str)

print("simulation time: ", datetime.timedelta(seconds=round(t2-t1)))
#print(f"{(t2-t1)//60:.0f}:{(t2-t1)%60:2.0f} min")
#p.plot_distance_distribution()
#p.save_traj_gro(fname_traj=fname)
print("save trajectroy ...")
p.save_traj_gro()
print("Done")

