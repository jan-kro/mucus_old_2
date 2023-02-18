import numpy as np
import matplotlib.pyplot as plt
import mdtraj as md
import tidynamics as tid
import os

from polymer.config import Config
from time import time

from copy import deepcopy

class Polymer:
    
    def __init__(self, config: Config):
        
        self.config = config
        
        self.r_beed                 = None
        self.q_beed                 = None
        self.mobility               = None
        self.force_constant_nn      = None
        self.epsilon_LJ             = None
        self.sigma_LJ               = None
        self.cutoff_LJ              = None
        self.lB_debye               = None
        self.c_S                    = None
        self.cutoff_debye           = None
        self.cutoff_pbc             = None
        self.pbc                    = None
        self.bonds                  = None
        self.cwd                    = None  
        self.r0_beeds               = None
        self.A_debye                = None
        self.B_debye                = None        
        self.n_beeds                = None
        self.box_length             = None
        self.shifts                 = None
        self.positions              = None
        self.forces                 = None
        self.energy                 = None
        self.trajectory             = None
        self.distances              = None
        self.directions             = None
        self.cm_trajectory          = None
        self.msd                    = None
        self.structure_factor       = None
        self.indices                = None
        self.idx_table              = None
        self.idx_interactions       = None
        
        self.setup(config)
        
    # TODO: redo the get_bonds() method so that every bond pair only exists once
    
    def setup(self, config):
        self.r_beed             = config.rbead
        self.q_beed             = config.qbead
        self.mobility           = config.mobility
        self.force_constant_nn  = config.force_constant
        self.epsilon_LJ         = config.epsilon_LJ
        self.sigma_LJ           = config.sigma_LJ
        self.cutoff_LJ          = config.cutoff_LJ
        self.lB_debye           = config.lB_debye # units of beed radii
        self.c_S                = config.c_S # salt concentration [c_S] = mM
        self.cutoff_debye       = config.cutoff_debye
        self.cutoff_pbc         = config.cutoff_pbc
        if self.cutoff_pbc is None:
            # NOTE here the cutoff of the force with the longest range is used
            self.cutoff_pbc = np.max((self.cutoff_debye, self.cutoff_LJ))
            self.config.cutoff_pbc = self.cutoff_pbc
        self.pbc                = config.pbc
        self.bonds              = config.bonds
        self.cwd                = config.cwd       
        
        self.r0_beeds           = 2*self.r_beed
        self.A_debye            = self.q_beed**2*self.lB_debye
        self.B_debye            = 1/(self.r_beed*38.46153*self.c_S)
    
    def print_sim_info(self):
        output = str(self.config)
        output = output.replace(" ", "\n")
        output = output.replace("=", " = ")
        print(output)
        return
    
    def create_chain(self,
                     N = 10, 
                     axis: int = 0):
        """
        Create linear chain along specified axis, centered in the origin.
        The get_bonds() method is called here automatically.
        
        Arguments:
            axis (int): specify on which axis the chain lies on (0,1,2) -> (x,y,z)
        """
        self.n_beeds = N
        self.box_length = 2*self.r0_beeds*self.n_beeds
        
        self.positions = np.zeros((self.n_beeds, 3))
        
        for k in range(self.n_beeds):
            self.positions[k, axis] = k*self.r0_beeds
        
        #center chain around 0
        if self.n_beeds != 1:
            self.positions[:, axis] -= self.r0_beeds*(self.n_beeds-1)/2
        
        # create index list
        self.indices = np.arange(self.n_beeds)
        
        # create nxn index table 
        self.idx_table = np.zeros((2, self.n_beeds, self.n_beeds), dtype=int)
        for i in range(self.n_beeds):
            for j in range(self.n_beeds):
                self.idx_table[0, i, j] = i
                self.idx_table[1, i, j] = j

        self.positions += np.array((self.box_length/2, self.box_length/2, self.box_length/2)) 
        
        # set first trajecory frame to initial position
        self.trajectory = np.zeros((1, self.n_beeds, 3)) # first dim is the frame
        self.trajectory[0,:,:] = deepcopy(self.positions)
        
        self.create_shifts()
        self.apply_pbc()
        self.get_bonds()
        
        return
    
    def create_box(self):
        # create box with M chains consisting of N atoms
        
        N = self.config.nbeads
        M = self.config.nchains
        r0 = self.r0_beeds
        L = self.box_length
        
        self.n_beeds = N*M
        
        if M == 1:
            self.create_chain(N=N*M)
            return
        
        if L is None:
            L = N*r0 # box length

        self.box_length = L
        self.config.lbox = L
        
        n_row = int(np.round(np.sqrt(M)))

        # check how many cols are needed
        n_col = 0
        while n_col*n_row < M:
            n_col += 1

        d_row = L/(n_row+1)
        d_col = L/(n_col+1)

        positions = np.zeros((M*N, 3))
        bonds = list(())

        particle_idx = 0
        # make centered mesh
        for i in range(n_col):
            for j in range(n_row):
                if particle_idx == N*M:
                    break
                # create chain
                n = 0
                for k in range(particle_idx, N+particle_idx):
                    positions[k, (i+1)%2] = (L-r0*(N-1))/2 + n*r0
                    positions[k, i%2] = j*d_row + d_row 
                    positions[k, 2] = i*d_col + d_col 
                    n += 1
                # add bonds to bond list
                bonds.append((particle_idx, particle_idx+1))
                for k in range(N-2):
                    particle_idx += 1
                    bonds.append((particle_idx, particle_idx-1))
                    bonds.append((particle_idx, particle_idx+1))
                particle_idx += 1
                bonds.append((particle_idx, particle_idx-1))
                particle_idx += 1
                
        self.positions = deepcopy(positions)
        self.bonds = np.array(bonds)
        
        self.trajectory = np.zeros((1, self.n_beeds, 3))
        self.trajectory[0,:,:] = deepcopy(self.positions)
        
        # create index list
        self.indices = np.arange(self.n_beeds)
        
        # create nxn index table 
        self.idx_table = np.zeros((2, self.n_beeds, self.n_beeds), dtype=int)
        for i in range(self.n_beeds):
            for j in range(self.n_beeds):
                self.idx_table[0, i, j] = i
                self.idx_table[1, i, j] = j
        
        self.create_shifts()
        self.apply_pbc() # unnecessary but feels wrong not to do it
        
        return
    
    def create_shifts(self):
        # array that shifts box for pbc
        self.shifts = np.array(((self.box_length,   0,                0),
                                (0,                 self.box_length,  0),
                                (0,            0,                     self.box_length),
                                (self.box_length,   self.box_length,  0),
                                (self.box_length,   0,                self.box_length),
                                (0,                 self.box_length,  self.box_length),
                                (self.box_length,   self.box_length,  self.box_length),
                                (-self.box_length,  0,                0),
                                (0,                -self.box_length,  0),
                                (0,            0,                    -self.box_length),
                                (-self.box_length,  self.box_length,  0),
                                (-self.box_length,  0,                self.box_length),
                                (0,                -self.box_length,  self.box_length),
                                (self.box_length,  -self.box_length,  0),
                                (self.box_length,   0,               -self.box_length),
                                (0,                 self.box_length, -self.box_length),
                                (-self.box_length, -self.box_length,  0),
                                (-self.box_length,  0,               -self.box_length),
                                (0,                -self.box_length, -self.box_length),
                                (-self.box_length,  self.box_length,  self.box_length),
                                (self.box_length,  -self.box_length,  self.box_length),
                                (self.box_length,   self.box_length, -self.box_length),
                                (-self.box_length, -self.box_length,  self.box_length),
                                (-self.box_length,  self.box_length, -self.box_length),
                                (self.box_length,  -self.box_length, -self.box_length),
                                (-self.box_length, -self.box_length, -self.box_length)))
        return
    
    def create_test_chain(self, 
                          sigma: float = 0.05, 
                          axis: int = 0,
                          centered = False):
        """
        Create linear chain along specified axis, centered in the origin.
        The positions of the linear chain are deviated wit a gaussian noise 
        of standard deviation sigma.
        The get_bonds() method is called here automatically.
        
        Arguments:
            sigma (float): standard deviation for the displacement
            axis (int): specify on which axis the chain lies on (0,1,2) -> (x,y,z)
        """
        if centered == True:
            axis = 0
        
        self.positions = np.zeros((self.n_beeds, 3))
        
        for k in range(self.n_beeds):
            self.positions[k, axis] = k*self.r0_beeds
        
        #center chain around 0
        if self.n_beeds != 1:
            self.positions[:, axis] -= self.r0_beeds*(self.n_beeds-1)/2
        
        if centered == True:   
            self.positions += np.array((self.box_length/2, self.box_length/2, self.box_length/2))
        else:
            self.positions[:, 1] += self.box_length/2
            self.positions[:, 2] += self.box_length/2
        
        # create index list
        self.indices = np.arange(self.n_beeds)
        
        # create nxn index table 
        self.idx_table = np.zeros((2, self.n_beeds, self.n_beeds), dtype=int)
        for i in range(self.n_beeds):
            for j in range(self.n_beeds):
                self.idx_table[0, i, j] = i
                self.idx_table[1, i, j] = j
        
        # to test forces
        self.positions += np.random.randn(self.n_beeds, 3)*sigma

        # set first trajecory frame to initial position
        self.trajectory[0,:,:] = deepcopy(self.positions)
        
        self.apply_pbc()
        self.get_bonds()
        
        return
    
    def get_bonds(self):
        """
        Create list of bond pair indices. 
        Bonds are saved twice, i.e.: (..., (i, j), ... , (j, i), ...)
        """
        #define bonds
        if self.bonds is None: 
            # if bonds are not already defined, create chain
            self.bonds =list(())
            self.bonds.append((0,1))
            for k in range(1, self.n_beeds-1):
                self.bonds.append((k, k-1))
                self.bonds.append((k, k+1))
            self.bonds.append((self.n_beeds-1, self.n_beeds-2))
            
            self.bonds = np.array(self.bonds)
            #self.config.bonds = self.bonds                             # TODO implement this here again after testing
        
        # calculate distances and directions for every bond tuple
        self.get_distances_directions()
        
        return
    
    def get_distances_directions(self):
        """
        Get distances and directions for every combination of atom within a certain cutoff distance to each other.
        The directions are not normalized, since the normalization happens in the forcefiled calculation.
        A list of index tuples is also calculated to assign the interactions to the specific atoms.
        """
        
        directions, distances, idx_table = self.dist_dir_box()
        
        # calculate dist dir for box edge w.r.t. cutoff
        directions_edges, distances_edges, idx_table_edges = self.dist_dir_edges()

        # check if there are edge interactions
        if idx_table_edges.size == 0:
            # if there are no edge interactions only use the values of the box
            self.directions = deepcopy(directions)
            self.distances = deepcopy(distances)
            self.idx_interactions = idx_table.T
        else:
            # combine dist/dir to one list
            self.directions = np.append(directions, directions_edges, axis=0)
            self.distances = np.append(distances, distances_edges, axis=0)
            self.idx_interactions = np.append(idx_table, idx_table_edges, axis=1).T
        
        
        return
    
    def dist_dir_edges(self):
        """
        This uses the concept of the "inner-" and "outer edge". Particles are defined to lie in the inner edge, 
        when thethey are inside the box within the a cutoff distance to the edge of the box. Similarly outer edges
        are particles, that are outside of the box within a cutoff distance to the box edge.
        
        This function calculates all the distances and directions of particles in the inner edge w.r.t. particles 
        in the outer edge and the filters out only particlies which lie in a cutoff distance towards each other.
        """

        # 3d logical lists
        L_left = self.positions < self.cutoff_pbc
        L_right = self.positions > (self.box_length-self.cutoff_pbc)

        #indexes all atoms inside the inner edge
        L_in = np.logical_or(np.sum(L_left, axis=1, dtype=bool), np.sum(L_right, axis=1, dtype=bool))

        edges_in = self.positions[L_in]
        edges_in_indices = self.indices[L_in]

        # only continue if there are edges_in 
        if edges_in_indices.size != 0:
            # shift system and append to array
            edges_out = self.positions + self.shifts[0]
            edges_out_indices = np.arange(self.n_beeds) # NOTE maybe use deepcopy instead? idk what is faster
            for shift in self.shifts[1:]:
                edges_out = np.append(edges_out, self.positions+shift, axis=0)
                edges_out_indices = np.append(edges_out_indices, self.indices)

            # check which particles lie in the outer edge
            L_left_out = np.logical_and((edges_out < 0), (edges_out >= -self.cutoff_pbc))
            L_right_out = np.logical_and((edges_out >= self.box_length), (edges_out < (self.box_length+self.cutoff_pbc)))
            L_out = np.logical_or(np.sum(L_left_out, axis=1, dtype=bool), np.sum(L_right_out, axis=1, dtype=bool))

            edges_out = edges_out[L_out]
            edges_out_indices = edges_out_indices[L_out]
            
            # only continue if there are edges_out
            if edges_out_indices.size != 0:
                # create distance matrix for edges in/out
                n_in = len(edges_in_indices)
                n_out = len(edges_out_indices)

                # create mesh of vectors
                A_mesh = np.tile(edges_in, (n_out, 1, 1)) # repeats vector along third dimension len(b) times
                B_mesh = np.reshape(np.repeat(edges_out, n_in, 0), (n_out, n_in, 3)) # does the same but "flipped"

                directions_edges = B_mesh-A_mesh
                distances_edges = np.linalg.norm(directions_edges, axis=2)

                idx_in, idx_out = np.meshgrid(edges_in_indices, edges_out_indices)
                idx_table_edges = np.append([idx_in], [idx_out], axis=0)
                
                # only output distances smaller than  cutoff
                L_edges = distances_edges < self.cutoff_pbc

                dirs = directions_edges[L_edges]
                dists = distances_edges[L_edges]
                idxs = idx_table_edges[:, L_edges]
                
            # if there are no interactions in the edges, output dummy variables since these are not used anyways
            else:
                dirs = np.array(())
                dists = np.array(())
                idxs = np.array(())
        else:
            dirs = np.array(())
            dists = np.array(())
            idxs = np.array(())
        
        return dirs, dists, idxs
    
    def dist_dir_box(self):
        """
        Calculates all the distances and directions of atoms that are distanced with a certain cutoff radius.
        
        Returns a list of unnormalized directions, a list of distances and a list of atom index tuples w.r.t
        the self.positions variable
        """
    
        # TODO could be faster by only calculating triu matrix

        # make 3d verion of meshgrid
        r_left = np.tile(self.positions, (self.n_beeds, 1, 1)) # repeats vector along third dimension len(a) times
        r_right = np.reshape(np.repeat(self.positions, self.n_beeds, 0), (self.n_beeds, self.n_beeds, 3)) # does the same but "flipped"

        directions = r_left - r_right # this is right considering the mesh method. dir[i, j] = r_j - r_i
        distances = np.linalg.norm(directions, axis=2)
        
        # use cutoff to index values 
        # NOTE this could probably be done in a smarter way
        distances += self.cutoff_pbc*np.eye(self.n_beeds) # add cutoff to disregard same atoms

        L_box = distances < self.cutoff_pbc # NOTE the "<" is important, because if it was "<=" the diagonal values would be included
        
        dirs = directions[L_box]
        dists = distances[L_box]
        idxs = self.idx_table[:, L_box]
        
        return dirs, dists, idxs

    def get_forces(self):
        """
        Delete all old forces and add all forces occuring in resulting from the self.positions configuration
        
        This only works because the forces are defined in a way where they are directly added to the self.forces variable
        """
        
        # delete old forces
        self.forces = np.zeros((self.n_beeds, 3))
        
        # add new forces
        self.force_NearestNeighbours()
        self.force_LennardJones_cutoff()
        self.force_Debye()
        
        return

    def get_forces_test(self):
        """
        for testing the forces
        """
        print("Position")
        print(self.positions)
        
        #delete old forces
        self.forces = np.zeros((self.n_beeds, 3))
        self.force_NearestNeighbours()
        print("nearest neighbours")
        print(self.forces)
        
        self.forces = np.zeros((self.n_beeds, 3))
        self.force_LennardJones_cutoff()
        print("Lennard Jones")
        print(self.forces)
        
        self.forces = np.zeros((self.n_beeds, 3))
        self.force_Debye()
        print("Debye")
        print(self.forces)
        
        return
    
    
    def force_NearestNeighbours(self):
        """
        harmonice nearest neighhbour interactions
        """
        # loop through index list and see if any tuple corresponds to bond
        # NOTE if there were more NN forces in the system this should be done outside of the function
        
        L_nn = list(())
        for idx in self.idx_interactions:
            if np.any(np.logical_and(idx[0] == self.bonds[:, 0], idx[1] == self.bonds[:, 1])):
                L_nn.append(True)
            else:
                L_nn.append(False)
                
        idxs = self.idx_interactions[L_nn]
        distances = self.distances[L_nn].reshape(-1,1)
        directions = self.directions[L_nn]
        
        # calculate the force of every bond at once
        forces_temp = 2*self.force_constant_nn*(1-self.r0_beeds/distances)*directions

        for i, force in zip(idxs[:, 0], forces_temp):
            self.forces[i, :] += force
        
        return

    def force_LennardJones_cutoff(self):
        """
        LJ interactions using a cutoff
        """
        
        L_lj = self.distances < self.cutoff_LJ
        
        idxs = self.idx_interactions[L_lj]
        distances = self.distances[L_lj].reshape(-1, 1)
        directions = self.directions[L_lj]
        
        forces_temp = 4*self.epsilon_LJ*(-12*self.sigma_LJ**12/distances**14 + 6*self.sigma_LJ**7/distances**8)*directions

        for i, force in zip(idxs[:, 0], forces_temp):
            self.forces[i, :] += force
            
        return

    def force_Debye(self):
            """
            non bonded interaction (debye screening)
            """
            distances = self.distances.reshape(-1,1)
            # since the debye cutoff is used for the dist/dir cutoff the distances dont have to be checked
            forces_temp = -self.A_debye*(1+self.B_debye*distances)*np.exp(-self.B_debye*distances)*self.directions/distances**3
            
            for i, force in zip(self.idx_interactions[:, 0], forces_temp):
                self.forces[i, :] += force
            
            return

    
    def force_Random(self):
        """
        Gaussian random Force with a per particle standard deviation of sqrt(6 mu_0) w.r.t. its absolute value
        """
        
        # since the std of the foce should be sqrt(6*mu) but the std of the absolute randn vector is sqrt(3)
        # the std used here is sqrt(2*mu)
        
        return np.sqrt(2*self.mobility)*np.random.randn(self.n_beeds, 3)
    
    
    def analytical_potential_function(self, 
                                      r = None):
        """
        Calculates the analytical interatomic potential of two bonded beeds, using 
        the parameters specified in __init__.
        This function is used for plotting.
        
        Parameters:
            r (ndarray):
                vector of distances between two beeds
        
        Returns:
            r (ndarray):
                vector of distances between two beeds
            U (ndarray):
                theoretical potential U(r)
        """
        
        if r is None:
            r = np.linspace(self.r0_beeds-self.r_beed,self.r0_beeds+self.r_beed, 100)
        
        L_s = 1/(38.46153*self.c_S)
        
        u_nn = self.force_constant_nn*(r-self.r0_beeds)**2
        u_LJ = 4*self.epsilon_LJ*(self.sigma_LJ**12/r**12 - self.sigma_LJ**7/r**6 + 1)
        u_Deb = self.A_debye*np.exp(-self.B_debye*r)/r
        
        # apply cutoff
        u_LJ[r>2*self.r_beed] = 0
        u_Deb[r>self.cutoff_debye] = 0
   
        return r, u_nn + u_LJ + u_Deb
    
    def get_CM_trajectory(self):
        """
        Calculates the center of mass for every frame in the trajectory, assuming equal mass for every 
        beed and saves it into a trajecorty. 
        """
        
        traj_cm = list(())

        #calculate trajectory for center of mass
        for frame in self.trajectory:
            traj_cm.append(np.sum(frame, axis=0))
            
        self.cm_trajectory = np.array(traj_cm)/self.n_beeds
            
        return 
    
    def get_CM_MSD(self, 
                   plot: bool = False):
        """
        Calculates the mead squared deviation of the center of mass to its original position.
        If plot is True, a log-log plot of the absolute MSD and the MSD for every CM coordinate is calculated
        and compared to the theoretical curve resulting from the mu_0 input.
        
        Parameters:
            plot (bool):
                wether to print the plot
        """
        
        # check if cm_traj has already been calculated
        if self.cm_trajectory is None:
            self.get_CM_trajectory()
            traj_cm = self.cm_trajectory
        else:
            traj_cm = self.cm_trajectory
        
        # TODO: saving cm trajectory into new array is unnecessary, clean all unnecessary variables
        
        msd_tid = tid.msd(traj_cm)
        self.msd = msd_tid
        
        if plot is True:
            msd_tid_x = tid.msd(traj_cm[:,0])
            msd_tid_y = tid.msd(traj_cm[:,1])
            msd_tid_z = tid.msd(traj_cm[:,2])
            
            n_frames = len(traj_cm)
            t = np.arange(n_frames)
        
            plt.figure()
            plt.loglog(t[1:], 2*t[1:]*self.mobility, label=r"$2t\tilde{\mu}_0$", linestyle="--")
            plt.loglog(t[1:], 6*t[1:]*self.mobility, label=r"$6t\tilde{\mu}_0$", linestyle="--")
            plt.loglog(t[1:], msd_tid[1:]*self.n_beeds, label=r"$\langle \Delta\tilde{r}^2\rangle$")
            plt.loglog(t[1:], msd_tid_x[1:]*self.n_beeds, label=r"$\langle \Delta\tilde{r}_x^2\rangle$")
            plt.loglog(t[1:], msd_tid_y[1:]*self.n_beeds, label=r"$\langle \Delta\tilde{r}_y^2\rangle$")
            plt.loglog(t[1:], msd_tid_z[1:]*self.n_beeds, label=r"$\langle \Delta\tilde{r}_z^2\rangle$")
            plt.xlabel(r"$t/\Delta t$")
            plt.ylabel("MSD (center of mass)")
            plt.legend()
            plt.show()
            
        return
    
    
    def plot_distance_distribution(self, 
                                   n_bins: int = 100, 
                                   bin_interval: tuple = None):
        """
        Plots a normalized distribution of all bonded distances in the trajectory
        and compares it to the Bolzmann distribution.
        
        Parameters:
            n_bins (int):
                number of bins in the histogram
            bin interval (tuple):
                left and right bin boundaries
        """
    
        distances_nn = list(())
        # for every frame calculate the distance between neighbouring atoms is the chain
        for frame in self.trajectory:
            distances_nn.append(np.sqrt(np.sum((frame[:self.n_beeds-1] - frame[1:])**2, axis=1)))

        distances_nn = np.array(distances_nn).ravel()
        
        if bin_interval is None:
            bin_interval = (np.min(distances_nn), np.max(distances_nn))

        # create normalized histogram
        histogram_nn, edges_nn = np.histogram(distances_nn, bins=n_bins, range=bin_interval, density=True)
        
        r, E_pot = self.analytical_potential_function(edges_nn[:n_bins])
        E_pot_normalize = np.sum((edges_nn[1:]-edges_nn[:n_bins])*np.exp(-E_pot) * 4*np.pi*edges_nn[:n_bins]**2) # here the jacobi determinant for spherical coordinates is added

        plt.figure()
        plt.plot(r - (r[1]-r[0]), histogram_nn, label=r"$p(\Delta\tilde{r})$")
        plt.plot(r, 4*np.pi*r**2*np.exp(-E_pot)/E_pot_normalize, label=r"$\frac{1}{Z}\mathrm{exp}\{-\tilde{U}(\tilde{\Delta r})\}$")
        plt.legend()
        plt.xlabel(r"$\Delta\tilde{r}$")
        plt.ylabel(r"$p(\tilde{r})$")
        plt.grid()
        plt.show()
        
        return
    
    def get_structure_factor(self,
                        q = None):
        """
        Calculates the structure factor of an isotropic system for different wavenumbers q
        Averaged over the whole trajectory
        
        Parametetrs
            q (ndarray):
                1D array of wavenumbers
        """ 
        if q is None:
            q = np.linspace(0.01, 0.8, 300) # kinda arbitrary right now
            # q_vec = q_i - q_s (i=incident, s=scattered), |q_i|=|q_s|=2pi n/lambda
            # q = 4 pi n/lamda * sin(theta/2)
            # n... refractive index of the solution
            # lambda... wavelength of the beam
            # theta... angle between incident beam and scattered beam
            
        # NOTE q could also be calculated using a scattering angle and a wavelength vector:
        # lam = np.linspace(0.5, 20, 300) # wavelength in reduced units
        # n = 1.1 # refractive index
        # theta = 0.05 #scattering angle
        # q = 4*np.pi*n/lam * np.sin(theta/2)
        
        # NOTE for small angle scattering one could use approximations for better performance
            
        q = np.array(q)
        
        self.structure_factor = np.zeros(len(q))
               
        for frame in self.trajectory:
            for i in range(self.n_beeds-1):
                for j in range(i+1, self.n_beeds):
                    r = frame[j] - frame[i] 
                    qr = q*np.sqrt(np.dot(r, r))
                    self.structure_factor += np.sin(qr)/qr
        
        #factor 2 because of double sum reduction from i,j to i<j
        self.structure_factor = self.structure_factor*2/self.n_beeds**2/len(self.trajectory)
        
        return
    
    def create_topology_pdb(self):
        """
        Creates a pdb topology of the current system
        """
        
        
        file_name = f"polymer_{self.n_beeds:d}_beeds"

        # create pdb file
        pdb_file = "topologies/"+file_name+".pdb"


        with open(pdb_file, "w") as f:
            f.write("HEADER\t"+file_name+"\n")
            f.write(f"CRYST1   60.000   60.000   60.000  90.00  90.00  90.00 P 1           1 \n")
            
            # create chain along the x-axis
            for k in range(self.n_beeds):
                #f.write(f"HETATM{k+1:5d}	 CA	 HET X       {k*chain_beed_distance+chain_beed_distance:6.3f}   0       0  1.00  0.00          Ca  \n")
                f.write(f"HETATM{k+1:5d} CA   HET X{k+1:4d}    {k*self.r0_beeds+self.r0_beeds:8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{0.0:6.2f}           C  \n")
            #terminate chain
            f.write(f"TER    {k+2:4d}      HET X {k+1:3d}\n")
            
            # add bonds
            f.write(f"CONECT{1:5d}{2:5d}\n") #first beed
            for k in range(2, self.n_beeds):
                f.write(f"CONECT{k:5d}{k-1:5d}{k+1:5d}\n") #middle beeds
            f.write(f"CONECT{self.n_beeds:5d}{self.n_beeds-1:5d}\n") #last beed
            
            f.write("END\n")
            f.close()
        
        return

    def save_traj_gro(self):
        """
        Saves trajectory in a gromacs trajectory, using mdtraj
        """
        fname_traj = self.config.fname_traj
        
        fname_top = os.path.join(self.cwd, f'topologies/polymer_{self.n_beeds:d}_beeds.pdb')

        if os.path.exists(fname_top) == False:
            self.create_topology_pdb()
        
        if fname_traj is None:
            fname_traj = os.path.join(self.cwd, f"trajectories/traj_{self.n_beeds:d}beeds_{len(self.trajectory):d}frames_{self.mobility:.5f}mu.gro")
            k = 1
            while os.path.exists(fname_traj):
                fname_traj = os.path.join(self.cwd, f"trajectories/traj_{self.n_beeds:d}beeds_{len(self.trajectory):d}frames_{self.mobility:.5f}mu_v{k:d}.gro")
                k += 1
        else:
            #check if path is given with or without filename
            # if not the input will be interpreted as a directory
            if fname_traj[-4:] != ".gro":
                # take care of input ambiguity
                if fname_traj[-1:] == "/":
                    fname_traj = fname_traj[:-1]
                path_traj = fname_traj
                
                fname_traj = path_traj + f"/traj_{self.n_beeds:d}beeds_{len(self.trajectory):d}frames_{self.mobility:.5f}mu.gro"
                # don't overwrite trajectories 
                k = 1
                while os.path.exists(fname_traj):
                    fname_traj = path_traj + f"/traj_{self.n_beeds:d}beeds_{len(self.trajectory):d}frames_{self.mobility:.5f}mu_v{k:d}.gro"
                    k += 1
        
        # save to config
        self.config.fname_traj = fname_traj
        
        # save trajectory in mdtraj to create .gro simulation trajectory
        topology = md.load(fname_top).topology
        trajectory = md.Trajectory(self.trajectory, topology)

        # save as gromacs file
        trajectory.save_gro(filename=fname_traj)
        
        return
    
    def load_traj_gro(self, 
                      fname: str = None,
                      overwrite: bool = False):
         
        if fname is None:
            raise TypeError("Error: provide filename")
        if os.path.exists(fname) != True:
            raise TypeError(f"Error: file \"{fname:s}\" does not exist")
        
        if overwrite == False:
            # TODO remove the "or" once the initialisation is fixed
            if (self.trajectory is None) or (len(self.trajectory<2)):
                raise Exception("Error: Polymer object already has a trajectory")
            else:
                self.trajectory = md.load(fname).xyz
        else:
            self.trajectory = md.load(fname).xyz
        
        
        self.positions = self.trajectory[-1]
        self.get_bonds()
        
    def save_config(self):
        """
        saves current self.config in a .toml file 
        """
        fname_sys = self.config.fname_sys
        
        if fname_sys is None:
            fname_sys = os.path.join(self.cwd, f"configs/sys_{self.n_beeds:d}beeds_{self.box_length:.2f}lbox_{self.mobility:.5f}mu.toml")
            k = 1
            while os.path.exists(fname_sys):
                fname_sys = os.path.join(self.cwd, f"configs/sys_{self.n_beeds:d}beeds_{self.box_length:.2f}lbox_{self.mobility:.5f}mu_v{k:d}.toml")
                k += 1
        else:
            #check if path is given with or without filename
            # if not the input will be interpreted as a directory
            if fname_sys[-5:] != ".toml":
                # take care of input ambiguity
                if fname_sys[-1:] == "/":
                    fname_sys = fname_sys[:-1]
                
                path_sys = fname_sys
                
                fname_sys = path_sys + f"/sys_{self.n_beeds:d}beeds_{self.box_length:.2f}lbox_{self.mobility:.5f}mu.toml"
                # don't overwrite trajectories 
                k = 1
                while os.path.exists(fname_sys):
                    fname_sys = path_sys + f"/sys_{self.n_beeds:d}beeds_{self.box_length:.2f}lbox_{self.mobility:.5f}mu_v{k:d}.toml"
                    k += 1
        
        self.config.fname_sys = fname_sys
        
        output = str(self.config)
        output = output.replace(" ", "\n")
        output = output.replace("=", " = ")

        f = open(fname_sys, "w")
        f.write(output)
        f.close()
        
        return
                
    def save_system(self):
        """
        saves both trajectory and system config
        """
        self.save_traj_gro()
        self.save_config()
        return
    
    
    def apply_pbc(self):
        """
        Repositions all atoms outside of the box to the other end of the box
        """
        
        # calculate modulo L
        self.positions = np.mod(self.positions, self.box_length)
        
        return
        
    def simulate(self):
        """
        Simulates the brownian motion of the system with the defined forcefield using forward Euler.
        """
        
        if self.positions is None:
            self.create_chain()
        
        if self.bonds is None:
            self.get_bonds()
        
        t_start = time()
        
        for step in range(self.config.steps-1):
            
            # get distances for interactions
            self.get_distances_directions()
            
            # get forces
            #self.get_forces_test()
            self.get_forces()
            
            # integrate
            # NOTE: the timestep of the integrator is already implicitly contained in the particle mobility
            self.positions = self.positions + self.mobility*self.forces + self.force_Random()
            
            # apply periodic boundary conditions
            self.apply_pbc()
            
            # This would be needed if the velovity is calculated:
            #self.positions_new = self.positions + self.mobility*self.forces + self.force_Random()
            #self.velocities = (self.positions - self.positions_new)
            #self.positions = deepcopy(self.positions_new)
            
            # write trajectory for every stride frames
            if (self.config.write_traj==True) and (step%self.config.stride==0):
                self.trajectory = np.append(self.trajectory, [self.positions], axis=0)
            
            # if np.any(self.distances_bonded > 5):
            #     print("System exploded")
            #     print("simulation Step", step)
            #     #print(self.forces)
            #     #print(self.distances_bonded)
            #     break
        t_end = time()
        
        self.config.simulation_time = t_end - t_start

        return