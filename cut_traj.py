import numpy as np
from polymer import Config, Polymer
import sys

cfg = Config.from_toml(sys.argv[1])
p = Polymer(cfg)
p.load_traj_gro(sys.argv[2], overwrite=True)

traj = p.trajectory

p.load_traj_ndarray(traj[-1000:])
p.save_system()