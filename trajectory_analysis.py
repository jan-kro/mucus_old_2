# CALCULATE THE RDF AND S(Q) OUT OF A TRAJECTORY

import numpy as np
from polymer import Config, Polymer
import sys
import time
import datetime


def Sq_rdf(g_r, radii, rho, qmax=1.0, n=512):
    """
    Compute structure factor S(q) from a radial distribution function g(r).
    The calculation of S(q) can be further simplified by moving some of 
    the terms outside the sum. I have left it in its current form so that 
    it matches S(q) expressions found commonly in books and in literature.
    
    Parameters
    ----------
    
    g_r : np.ndarray
        Radial distribution function, g(r).
    radii : np.ndarray
        Independent variable of g(r).
    rho : float
        .
    qmax : floatAverage number density of particles
        Maximum value of momentum transfer (the independent variable of S(q)).
    n : int
        Number of points in S(q).
        
    Returns
    -------
    S_q : np.ndarray
        Structure factor
    Q : np.ndarray
        Momentum transfer (the independent variable of S(q)).
    
    """
    n_r = len(g_r)
    Q = np.linspace(0.0, qmax, n)
    S_q = np.zeros_like(Q)
    
    dr = radii[1] - radii[0]
    
    g_r -= 1
    
    S_q[0] = np.sum(radii**2*g_r)
    
    for q_idx, q in enumerate(Q[1:]):
        S_q[q_idx+1] = np.sum(radii*g_r*np.sin(q*radii)/q)
    
    S_q = 1 + 4*np.pi*rho*dr * S_q / n_r
    
    return S_q, Q

def rdf(traj, lbox, cutoff, max_frames=10000, n_bins=300):
    
    # rdf(p.trajectory, p.box_length, p.cutoff_pbc, max_frames, nbins)
        
    lbox2 = lbox/2
    natoms = len(traj[0])
    atom_indices = np.arange(natoms)    
    dr = np.sqrt(3)*cutoff/n_bins       # NOTE sqrt(3)*r_c because of cubic geometry

    if max_frames > len(traj):
        max_frames = len(traj)
    
    gr = np.zeros(n_bins)
    r = np.linspace(dr/2, np.sqrt(3)*cutoff-dr/2, n_bins)
    
    for k, frame in enumerate(traj[len(traj)-max_frames:]): #take the last max_frames frames
        
        for i in range(natoms-1):

            # 1 shift atom i to center
            frame_shift = frame - frame[i]

            # 2 apply pbc
            frame_shift[frame_shift>lbox2] -= lbox
            frame_shift[frame_shift<-lbox2] += lbox
            
            # 3 calculate distance within cutoff
            
            # create index list that only contains atoms within cutoff
            atom_indices_in = atom_indices[np.all(np.logical_and(frame_shift < cutoff, frame_shift > -cutoff), axis=1)]
            
            # loop over all other atoms but self
            for j in atom_indices_in[atom_indices_in != i]:
                
                # since atom j now sits in 0 we dont deed to calculate the distance to it
                # but only the norm of the position of the surrounding atoms 
                d = np.sqrt(frame_shift[j, 0]**2 + frame_shift[j, 1]**2 + frame_shift[j, 2]**2)
                
                # 4 bin into rdf array
                gr[int(d/dr)] += 2
    
    gr = gr/np.sum(dr*gr)
                
    return r, gr

def report_time(name = "caluclation", report_type = "start", starttime = None):
    if report_type == "start":
        now = time.localtime()
        now_str = f"{now.tm_mon:d}.{now.tm_mday}.{now.tm_year}  {now.tm_hour}:{now.tm_min}:{now.tm_sec}"
        print(f"\n{name:s} started ", now_str)
        return time.time() 

    if report_type == "end":
        now = time.localtime()
        now_str = f"{now.tm_mon:d}.{now.tm_mday}.{now.tm_year}  {now.tm_hour}:{now.tm_min}:{now.tm_sec}"
        print(f"{name:s} finished ", now_str)
        print(f"{name:s} time: ", datetime.timedelta(seconds=round(time.time()-starttime)))
        return
    

# create polymer object and load trajectory

cfg = Config.from_toml(sys.argv[1])
fname_traj = cfg.fname_traj

p = Polymer(cfg)
p.load_traj_gro(fname_traj, overwrite=True)

# caluclate RDF
cutoff = p.box_length/3 # set cutoff for distance pairs

starttime = report_time()

r, gr = rdf(p.trajectory, p.box_length, cutoff, max_frames=50000, n_bins=500)

report_time(report_type="end", starttime=starttime)


# calculate S(q)
Sq, q = Sq_rdf(gr+1, r, 1/100, qmax=1.5, n=2*512)


SQ = np.append([q], [Sq], axis = 0).T
RDF = np.append([r], [gr], axis = 0).T

np.save(f"/home/janmak98/mucus/SQ_{sys.argv[2]}.npy", SQ)
np.save(f"/home/janmak98/mucus/RDF_{sys.argv[2]}.npy", RDF)

