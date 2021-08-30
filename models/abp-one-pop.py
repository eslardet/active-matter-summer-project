"""
One population ABP model
Euler or SRK method for solving SDEs
Uses a combined Verlet/ linked list neighbour list method for efficiency of the force function and collision calculations
"""

import numpy as np
import csv
from numba import jit

@jit(nopython=True)
def pbc_wrap(x, L):
    """
    ## Apply periodic boundary conditions

    Input:
    # x: unwrapped value (scalar or array)
    # L: box length

    Output:
    # x_wrap: wrapped value of x after applying periodic BCs
    """
    x_wrap = (x + 0.5*L) % L - 0.5*L
    return x_wrap

@jit(nopython=True)
def linked_grids(grid_number):
    """
    ## Finds the 9 grid neighbours (including the grid itself) for each grid in which to search for neighbouring
    ## particles for a Verlet neighbour list

    Input:
    # grid_number: number of grids on one box length (so total number of grids is the square of this)

    Output:
    # grids_to_check: array of the 9 grid neighbours (including the grid itself) for each grid
    
    """
    grids_to_check = np.zeros((grid_number**2, 9))
    for i in range(grid_number):
        for j in range(grid_number):
            grids_to_check[i + j*grid_number][0] = i + (j) * grid_number
            grids_to_check[i + j*grid_number][1] = (i-1)%grid_number + ((j-1)%grid_number) * grid_number
            grids_to_check[i + j*grid_number][2] = (i-1)%grid_number + j * grid_number
            grids_to_check[i + j*grid_number][3] = (i-1)%grid_number + (j+1)%grid_number * grid_number
            grids_to_check[i + j*grid_number][4] = i + (j+1)%grid_number * grid_number
            grids_to_check[i + j*grid_number][5] = (i+1)%grid_number + (j+1)%grid_number * grid_number
            grids_to_check[i + j*grid_number][6] = (i+1)%grid_number + j * grid_number
            grids_to_check[i + j*grid_number][7] = (i+1)%grid_number + (j-1)%grid_number * grid_number
            grids_to_check[i + j*grid_number][8] = i + (j-1)%grid_number * grid_number
    return grids_to_check


@jit(nopython=True)
def linked_to_verlet(x, y, N_particles, L, rskin, grid_size, grid_number, grids_to_check):
    """
    ## Create linked neighbour list from particle coordinates, search for grid neighbours
    ## and create Verlet neighbour list with a skin radius rskin

    Input:
    # x: x coordinates of particles
    # y: y coordinates of particles
    # N_particles: total number of particles
    # L: box length
    # rskin: skin radius (from centre) for Verlet neighbour list
    # grid_size: box length of grids in sub-box system
    # grid_number: number of grids per length L in sub-box system
    # grids_to_check: array of the 9 grid neighbours (including the grid itself) for each grid

    Output:
    # nlist: all neighbours of for each particle with index higher than current particle 
    # npoint: points to first neighbour in nlist for each particle
    """

    ## Create linked list
    # 'head' contains first particle number in each grid in sub-box system
    # 'linked_list' contains index of next particle in the same grid in
    # sub-box sysem, or -1 if it is the last particle in the grid
    head = np.zeros(grid_number**2)
    head.fill(-1)
    linked_list = np.zeros(N_particles)
    linked_list.fill(-1)
    for i in range(N_particles):
        grid = int((pbc_wrap(x[i],L)+L/2) // grid_size + ((pbc_wrap(y[i],L)+L/2) // grid_size) * grid_number)
        linked_list[i] = head[grid]
        head[grid] = i
    
    ## Create Verlet list by searching through grid neighbours using the linked list
    rskin_sq = rskin**2
    nlist = []
    npoint = []
    for i in range(N_particles):
        npoint.append(len(nlist)) # add starting point index to point array
        current_grid = int((pbc_wrap(x[i],L)+L/2) // grid_size + ((pbc_wrap(y[i],L)+L/2) // grid_size) * grid_number)
        nei_grid = grids_to_check[current_grid]
        for grid in nei_grid:
            j = int(head[int(grid)])
            while j != -1: # when j=-1, we have reached end of linked list
                j = int(j)
                if j > i: # ignore particles with index lower than i by Newton's third law
                    xij = pbc_wrap(x[i] - x[j], L)
                    if abs(xij) <= rskin:
                        yij = pbc_wrap(y[i] - y[j], L)
                        if abs(yij) <= rskin:
                            r_sq = xij**2 + yij**2
                            if r_sq <= rskin_sq: # if within rskin distance add point to verlet list
                                nlist.append(j) 
                j = linked_list[j] # go to next particle in linked list
    nlist = np.array(nlist)
    npoint = np.array(npoint)
    return nlist, npoint

@jit(nopython=True)
def force(x, y, N_particles, L, rcut, nlist, npoint):
    """
    ## Calculate forces between particles using Verlet neighbour list
    ## Force is a purely repulsive force given by the derivative of a WCA potential

    Input:
    # x: x coordinates of particles
    # y: y coordinates of particles
    # N_particles: total number of particles
    # L: box length
    # rcut: cut off distance for force interactions
    # nlist: all neighbours of for each particle with index higher than current particle 
    # npoint: points to first neighbour in nlist for each particle

    Output:
    # Fx: x coordinates of forces
    # Fy: y coordinates of forces
    """
    rcut_sq = rcut**2
    Fx = np.zeros(N_particles)
    Fy = np.zeros(N_particles)

    for i in range(N_particles-1): # last particle has no neighbours in nlist as all neighbours already accounted for
        for k in range(npoint[i], npoint[i+1]):
            j = nlist[k] 
            xij = pbc_wrap(x[i] - x[j], L)
            if abs(xij) <= rcut:
                yij = pbc_wrap(y[i] - y[j], L)
                if abs(yij) <= rcut:
                    r_sq = xij**2 + yij**2
                    if r_sq <= rcut_sq: # if within rcut distance calculate force
                        r = np.sqrt(r_sq)
                        gamma = 24 / r_sq * (2*(1/r)**12 - (1/r)**6)

                        Fji_x = gamma * xij # x force from j to i
                        Fx[i] += Fji_x
                        Fx[j] -= Fji_x # by Newton's 3rd law

                        Fji_y = gamma * yij # y force from j to i
                        Fy[i] += Fji_y
                        Fy[j] -= Fji_y # by Newton's 3rd law
    return Fx, Fy

@jit(nopython=True)
def max_displacement(dxnei, dynei, N_particles):
    """
    ## Find the two maximal displacements of all particles since last neighbour list update 
    ## for automatic neighbour list update

    Input:
    # dxnei: x coordinates of displacements of all particles since last update
    # dynei: y coordinates of displacemetns of all particles since last update
    # N_particles: total number of particles

    Output:
    # drneimax1: largest displacement of a single particle since last update
    # drneimax2: second largest displacement of a single particle since last update
    """
    drneimax1 = 0
    drneimax2 = 0
    for i in range(N_particles):
        drnei = dxnei[i]**2 + dynei[i]**2
        if drnei > drneimax1:
            drneimax2 = drneimax1
            drneimax1 = drnei
        elif drnei > drneimax2:
            drneimax2 = drnei
    drneimax1 = np.sqrt(drneimax1)
    drneimax2 = np.sqrt(drneimax2)
    return drneimax1, drneimax2


def abp_one_population(N_particles, phi, Pe, rdiff, t, sample, delta_t, method, file):
    """
    ## Performs an Euler or SRK method to solve the overdamped Langevin equations (SDEs) and saves x, y and theta
    ## values to a csv file every sample tau
    
    Input:
    # N_particles: total number of particles
    # phi: volume fraction
    # Pe: Peclet number
    # rdiff: difference between rcut (set at 2^(1/6)) and rskin for Verlet list
    # t: total time to run simulation for
    # sample: time between samples for csv file
    # delta_t: timestep
    # method: 'euler' or 'srk'
    # file: name of csv file to write data into
    """
    p = int(np.sqrt(N_particles))
    N_particles = p**2

    # Constants for main loop
    C_rot = np.sqrt(6*delta_t)
    C_trans = np.sqrt(2*delta_t)
    
    rcut = 2**(1/6)
    rskin = rcut + rdiff

    L = np.sqrt(N_particles*np.pi / phi)/2  # box length for desired phi

    N_timesteps = int(t // delta_t)  # total number of time steps
    sample_timestep = round(sample / delta_t) # number of time steps between each sample
    
    theta = np.random.random(N_particles) * 2*np.pi  # random initial orientational angles
    
    ## Generate grid of particles in box for initial conditions
    xs = np.linspace(-L/2, L/2, p, endpoint=False)
    xv, yv = np.meshgrid(xs, xs, sparse=False, indexing='xy')
    x = xv.flatten()
    y = yv.flatten()

    ## Neighbour list initialisations
    dxnei = np.zeros(N_particles)
    dynei = np.zeros(N_particles)
    grid_number = int(L // rskin)
    grid_size = L / grid_number
    grids_to_check = linked_grids(grid_number)
    nlist, npoint = linked_to_verlet(x, y, N_particles, L, rskin, grid_size, grid_number, grids_to_check)

    with open(file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(np.concatenate((x, y, theta)))
        
        ## Main loop ##        
        for n in range(1, N_timesteps+1):
            ## If max displacement threshold reached, update neighbour lists
            drneimax1, drneimax2 = max_displacement(dxnei=dxnei, dynei=dynei, N_particles=N_particles)
            if drneimax1 + drneimax2 > rdiff:
                nlist, npoint = linked_to_verlet(x, y, N_particles, L, rskin, grid_size, grid_number, grids_to_check)
                dxnei = np.zeros(N_particles)
                dynei = np.zeros(N_particles)

            if method == 'euler':
                # Calculate forces
                Fx, Fy = force(x=x, y=y, N_particles=N_particles, L=L, rcut=rcut, nlist=nlist, npoint=npoint)

                # Location update
                dx = delta_t * (Fx + Pe*np.cos(theta)) + C_trans*np.random.normal(0, 1, size=N_particles)
                dy = delta_t * (Fy + Pe*np.sin(theta)) + C_trans*np.random.normal(0, 1, size=N_particles)
                x += dx
                y += dy
                dxnei += dx
                dynei += dy

                # Rotational diffusion update
                theta = theta + C_rot*np.random.normal(0, 1, size=N_particles)

            if method == 'srk':
                # Interaction forces for particles at x, y
                Frx, Fry = force(x=x, y=y, N_particles=N_particles, L=L, rcut=rcut, nlist=nlist, npoint=npoint)

                # First force stage
                Gax = Frx + Pe*np.cos(theta)
                Gay = Fry + Pe*np.sin(theta)

                # Find X, Y, and white noise variables for particle i
                wx = C_trans*np.random.normal(0, 1, size=N_particles)
                wy = C_trans*np.random.normal(0, 1, size=N_particles)
                X = x + delta_t*Gax + wx
                Y = y + delta_t*Gay + wy

                # Interaction forces for particles at X, Y
                FRx, FRy = force(x=X, y=Y, N_particles=N_particles, L=L, rcut=rcut, nlist=nlist, npoint=npoint)

                ### Rotational diffusion update
                theta = theta + C_rot*np.random.normal(0, 1, size=N_particles)

                # Second force stage
                Gbx = FRx + Pe*np.cos(theta)
                Gby = FRy + Pe*np.sin(theta)

                ### Location update
                dx = delta_t/2 * (Gax + Gbx) + wx
                dy = delta_t/2 * (Gay + Gby) + wy
                x += dx
                y += dy
                dxnei += dx
                dynei += dy

            if n % sample_timestep == 0:
                writer.writerow(np.concatenate((x, y, theta)))

### Example paramters and implementation

# N_particles = 100
# t = 10
# sample = 1
# delta_t = 2e-5
# phi = 0.6
# Pe = 120
# rdiff = 0.5
# method = 'euler'

# name = 'N' + str(N_particles) +  '_phi' + str(phi) + '_Pe' + str(Pe) + '_s' + str(sample)
# file = name + '.csv'

# abp_one_population(N_particles, phi, Pe, rdiff, t, sample, delta_t, method, file)