import re
import numpy as np
import toml
from pydantic import BaseModel, root_validator
import os
from typing import Optional


class Config(BaseModel, arbitrary_types_allowed=True):
    steps:              int
    stride:             int
    number_of_beads:    int
    nbeads:             int
    nchains:            int
    mobility:           float
    rbead:              float   = 1.0
    qbead:              float   = 1.524 # 2.08
    force_constant:     float   = 100.0
    epsilon_LJ:         float   = 0.25
    sigma_LJ:           float   = 2.0
    cutoff_LJ:          float   = 2.0
    lB_debye:           float   = 36.737 # 3.077
    c_S:                float   = 10.0
    cutoff_debye:       float               = None
    lbox:               Optional[float]     = None
    pbc:                bool                = True
    cutoff_pbc:         Optional[float]     = None
    save_traj:          bool                = True
    write_traj:         bool                = True
    cwd:                Optional[str]       = os.getcwd()
    fname_traj:         Optional[str]       = None
    fname_sys:          Optional[str]       = None
    simulation_time:    Optional[float]     = None
    bonds:              Optional[np.ndarray]= None 

    @classmethod
    def from_toml(cls, path):
        data = toml.load(open(path, encoding="UTF-8"))
        return cls(**data)
    
    @classmethod
    def from_dict(cls, dict):
        return cls(**dict)
    
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
            if key == "cwd":
                values[key] = os.getcwd()
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


# if __name__ == "__main__":
#     config = Config.from_toml("/home/jan/Documents/masterthesis/project/mucus/configs/tests/cfg_test_box_10_12_0.toml")
#     print(config)
