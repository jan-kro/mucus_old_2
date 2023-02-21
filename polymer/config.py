import re
import toml
from pydantic import BaseModel
import os

class Config(BaseModel):
    steps:              int
    stride:             int
    number_of_beads:    int
    nbeads:             int
    nchains:            int
    mobility:           float
    rbead:              float   = 1.0
    qbead:              float   = 2.08
    force_constant:     float   = 100.0
    epsilon_LJ:         float   = 0.25
    sigma_LJ:           float   = 2.0
    cutoff_LJ:          float   = 2.0
    lB_debye:           float   = 3.077
    c_S:                float   = 10
    cutoff_debye:       float   = 4.0
    lbox:               float   = None
    pbc:                bool    = True
    cutoff_pbc:         float   = None
    save_traj:          bool    = True
    write_traj:         bool    = True
    cwd:                str     = "/home/jan/Documents/masterthesis/project/mucus"
    fname_traj:         str     = None
    fname_sys:          str     = None
    simulation_time:    float   = None
    bonds:              list    = None # add this again after testing

    @classmethod
    def from_toml(cls, path):
        data = toml.load(open(path, encoding="UTF-8"))
        return cls(**data)
    
    @classmethod
    def from_dict(cls, dict):
        return cls(**dict)

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
