import re
import numpy as np
import toml
from pydantic import BaseModel, root_validator
import os
import time
from typing import Optional


class Config(BaseModel, arbitrary_types_allowed=True):
    """
    The config file contains all the information of the system and where everything is saved
    """
    
    steps:              int
    stride:             int
    number_of_beads:    int
    nbeads:             int
    nchains:            int
    mobility:           float
    rbead:              float               = 1.0
    qbead:              float               = 1.524 # 2.08
    force_constant:     float               = 100.0
    epsilon_LJ:         float               = 0.25
    sigma_LJ:           float               = 2.0
    cutoff_LJ:          float               = 2.0
    lB_debye:           float               = 36.737 # 3.077
    c_S:                float               = 10.0
    cutoff_debye:       float               = None
    lbox:               Optional[float]     = None
    pbc:                bool                = True
    cutoff_pbc:         Optional[float]     = None
    write_traj:         bool                = True
    cwd:                Optional[str]       = os.getcwd()
    name_sys:           Optional[str]       = None
    dir_output:         Optional[str]       = None
    simulation_time:    Optional[float]     = None
    bonds:              Optional[np.ndarray]= None 

    # TODO add current time and date to properly track everything
    
    # TODO maybe create a seperate setup function that creates all the necesarry directories and shit
    
    @classmethod
    def from_toml(cls, path):
        data = toml.load(open(path, encoding="UTF-8"))
        return cls(**data)
    
    @classmethod
    def from_dict(cls, dict):
        return cls(**dict)
    
    @root_validator
    def default_values(cls, values):
        for key, item in values.items():
            
            if key == "cwd":
                values[key] = os.getcwd()
                
            # TODO delete this bs again
            if key == "lbox":
                values[key] = values["nbeads"]*2*values["rbead"]
            
            if key == "name_sys":
                if item is None:
                    " create default system name"
                    
                    now = time.localtime()
                    now_str = f"{now.tm_year}-{now.tm_mon}-{now.tm_mday}_{now.tm_hour}h{now.tm_min}m{now.tm_sec}s"
                    n_beads = values["n_beads"]
                    n_steps = values["steps"]//values["stride"]
                    mobility = values["mobility"]
                    
                    name = f"{n_beads:d}beads_{n_steps:d}frames_{mobility:.5f}mu_" + now_str
                    
                    values[key] = name
            
            # create output directory    
            if key == "dir_output":
                # if outdir is not specified create folder in cdw
                if item is None:
                    
                    now = time.localtime()
                    now_str = f"{now.tm_year}-{now.tm_mon}-{now.tm_mday}_{now.tm_hour}h{now.tm_min}m{now.tm_sec}s"
                    name = values["name_sys"]
                    
                    dir_out = os.getcwd() + f"/systems/sys_{name:s}"                        
                    os.makedirs(dir_out)
                    values[key] = dir_out
                    
                else:
                    dir_out = values[key]
                    
                    # if dir doesn't exist, create new dir
                #     if not os.path.exists(dir_out):
                #         os.makedirs(dir_out)
                
                # # create necessary children folders
                # # TODO delete again at some point
                # if not os.path.exists(dir_out + "/trajectories"):
                #     os.makedirs(dir_out + "/trajectories")
                # if not os.path.exists(dir_out + "/topologies"):
                #     os.makedirs(dir_out + "/topologies")
                # if not os.path.exists(dir_out + "/configs"):
                #     os.makedirs(dir_out + "/configs")
                # if not os.path.exists(dir_out + "/bonds"):
                #     os.makedirs(dir_out + "/bonds")
                # if not os.path.exists(dir_out + "/results"):
                #     os.makedirs(dir_out + "/results")

                # rename system, if system already exists
                # name = values["name_sys"]
                # name_tmp = values["name_sys"]
                                
                # k = 0
                # while os.path.exists(dir_out + "/configs/cfg_" + name_tmp + ".toml"):
                #     name_tmp = name + f"_v{k:d}"
                #     k += 1
                # values["name_sys"] = name_tmp
            
        return values
    
    @root_validator(pre=True)
    def validate_ndarrays(cls, values):
        """
        Iterates through the whole config dictionary
        
        for the bonds key, either a list, is accepted, which is then turend into a ndarray, or a str is accepted, which specifies a path leading to a saved numpy array
        """
        for key, item in values.items():
            data_type = cls.__annotations__[key]
            if data_type == Optional[np.ndarray] or data_type == np.ndarray:
                if item is not None:
                    if isinstance(item, str):
                        values[key] = np.load(item)
                    else:
                        values[key] = np.array(item)
                else:
                    if data_type is np.ndarray:
                        raise ValueError(f"We expected array for {key} but found None.")
            
        return values

    def __format__(self, __format_spec: str) -> str:
        """Format the config as a toml file such that atrributes
        are on multiple lines. Activate if __format_spec is 'fancy'
        """
        output = str(self)
        if __format_spec == "'fancy'" or __format_spec == "fancy":
            # find all spaces that are followed by an attribute of
            # the dataclass and replace them with a newline
            output = re.split("( [a-z|_]+=)", output)
            for i, item in enumerate(output):
                if item.startswith(" "):
                    output[i] = f"\n{item[1:]}"
            output = "".join(output)
            output = output.replace("=", " = ")
        return output
    
    def save_config(self, fout: str = None):
        """
        saves current self.config in a .toml file 
        """
        
        # check if different pathout is specified, than in config
        if fout is None:
            fname_sys = self.fname_sys
        else: 
            fname_sys = fout
        
        if fname_sys is None:
            fname_sys = os.path.join(self.cwd, f"configs/sys_{self.nbeads:d}beeds_{self.lbox:.2f}lbox_{self.mobility:.5f}mu.toml")
            k = 1
            while os.path.exists(fname_sys):
                fname_sys = os.path.join(self.cwd, f"configs/sys_{self.nbeads:d}beeds_{self.lbox:.2f}lbox_{self.mobility:.5f}mu_v{k:d}.toml")
                k += 1
        else:
            #check if path is given with or without filename
            # if not the input will be interpreted as a directory
            if fname_sys[-5:] != ".toml":
                # take care of input ambiguity
                if fname_sys[-1:] == "/":
                    fname_sys = fname_sys[:-1]
                
                path_sys = fname_sys
                
                fname_sys = path_sys + f"/sys_{self.nbeads:d}beeds_{self.lbox:.2f}lbox_{self.mobility:.5f}mu.toml"
                # don't overwrite trajectories 
                k = 1
                while os.path.exists(fname_sys):
                    fname_sys = path_sys + f"/sys_{self.nbeads:d}beeds_{self.lbox:.2f}lbox_{self.mobility:.5f}mu_v{k:d}.toml"
                    k += 1
        
        if fout is None:
            # if system is saved in self.fname_sys, upodate self.fname_sys
            self.fname_sys = fname_sys
        
        output = str(self)
        output = output.replace(" ", "\n")
        output = output.replace("=", " = ")

        f = open(fname_sys, "w")
        f.write(output)
        f.close()
        
        return


if __name__ == "__main__":
    config = Config.from_toml("/home/jan/Documents/masterthesis/project/mucus/configs/tests/cfg_test_box_10_12_0.toml")
    print(config.dir_output)
    print(config.fname_sys)
    print(config.fname_traj)


