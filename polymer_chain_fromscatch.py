from polymer_chain import Polymer
import time
import os


fname = "/home/jan/Desktop/traj_test.gro"

n_beeds = 20 #50
mobility = 0.00005

steps = 1000 #100000
stride = 50

p = Polymer(n_beeds=n_beeds,
            mobility=mobility,
            cwd=os.getcwd())

t1 = time()
p.simulate(steps=steps, 
           stride=stride)
t2 = time()

print("simulation time")
print(f"{(t2-t1)//60:.0f}:{(t2-t1)%60:2.0f} min")

p.plot_distance_distribution()
p.save_traj_gro(fname_traj=fname)

