import sys
import time
import datetime

from .config import Config
from polymer import Polymer


def run_sim(type: str = "box"):
    config = Config.from_toml(sys.argv[1])
    p = Polymer(config)
    
    if type == "box":
        p.create_box()
    if type == "chain":
        p.create_chain()
    
    print("starting a simulation with following parameters:")
    p.print_sim_info()
    
    now = time.localtime()
    now_str = f"{now.tm_mon:d}.{now.tm_mday}.{now.tm_year}  {now.tm_hour}:{now.tm_min}:{now.tm_sec}"
    print("\nsimulation started ", now_str)
    
    t1 = time.time()
    p.simulate()
    t2 = time.time()

    now = time.localtime()
    now_str = f"{now.tm_mon:d}.{now.tm_mday}.{now.tm_year}  {now.tm_hour}:{now.tm_min}:{now.tm_sec}"

    print("simulation finished ", now_str)

    print("simulation time: ", datetime.timedelta(seconds=round(t2-t1)))
    
    print("\nsave system...")
    p.save_system()
    print("\ndone")