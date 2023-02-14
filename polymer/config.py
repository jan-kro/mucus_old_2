import re
import toml
from pydantic import BaseModel

class Config(BaseModel):
    steps:              int
    stride:             int
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
    bonds:              list    = None

    @classmethod
    def from_toml(cls, path):
        data = toml.load(open(path, encoding="UTF-8"))
        return cls(**data)

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
