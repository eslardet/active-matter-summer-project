"""
Two population ABP model with a decrease in persistence in particle type B after an A/B collision
Euler method for solving SDEs
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
def force(x, y, N_particles, L, rcut, nlist, npoint, M):
    """
    ## Calculate forces between particles using Verlet neighbour list and update collision matrix M
    ## Force is a purely repulsive force given by the derivative of a WCA potential

    Input:
    # x: x coordinates of particles
    # y: y coordinates of particles
    # N_particles: total number of particles
    # L: box length
    # rcut: cut off distance for force interactions
    # nlist: all neighbours of for each particle with index higher than current particle 
    # npoint: points to first neighbour in nlist for each particle
    # M: collision matrix; particles type A+B currently in contact (no=0; yes=1)

    Output:
    # Fx: x coordinates of forces
    # Fy: y coordinates of forces
    # M: collision matrix; particles type A+B currently in contact (no=0; yes=1)
    """
    rcut_sq = rcut**2
    Fx = np.zeros(N_particles)
    Fy = np.zeros(N_particles)
    
    N_A = N_particles // 2

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
                        
                        # If A-B collision, update collision matrix
                        if i < N_A:
                            if j >= N_A:
                                M[i, j-N_A] = 1
    return Fx, Fy, M

@jit(nopython=True)
def coll_calc(x, y, L, N_A, N_B, M, t_star, t, rcut):
    """
    ## Update collision info by searching through A-B particles currently in contact and checking
    ## if they are no longer in contact

    Input:
    # x: x coordinates of particles
    # y: y coordinates of particles
    # L: box length
    # N_A: number of particles of type A
    # N_B: number of particles of type B
    # M: collision matrix; particles type A+B currently in contact (no=0; yes=1)
    # t_star: last collision time vector
    # t: current time
    # rcut: cut off distance for force interactions

    Output:
    # Fx: x coordinates of forces
    # Fy: y coordinates of forces
    # M: collision matrix; particles type A+B currently in contact (no=0; yes=1)
    """
    rcut_sq = rcut**2
    for i in range(N_A):
        for j in range(N_B):
            if M[i,j] == 1:
                xij = pbc_wrap(x[i] - x[N_A+j], L)
                yij = pbc_wrap(y[i] - y[N_A+j], L)
                r_sq = xij**2 + yij**2
                if r_sq > rcut_sq:
                    M[i,j] = 0
                    t_star[j] = t
    return M, t_star

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

def two_populations_alpha_change(N_particles, phi, Pe, alpha_A, alpha_norm_B, alpha_low_B, t_recovery_B, rdiff, 
                                t, delta_t, sample, file_points, file_alpha):
    """
    ## Performs an Euler method to solve the overdamped Langevin equations (SDEs) and saves x, y and theta
    ## values to a csv file every sample tau

    Input:
    # N_particles: total number of particles
    # phi: volume fraction
    # Pe: Peclet number
    # alpha_A: alpha value of population A particles
    # alpha_norm_B: normal alpha value of population B particles without collisions with A particles
    # alpha_low_B: low alpha value of population B particles immediately after collisions with A particles
    # t_recovery_B: time to recover (linearly) to normal alpha B value
    # rdiff: difference between rcut (set at 2^(1/6)) and rskin for Verlet list
    # t: total time to run simulation for
    # delta_t: timestep
    # sample: time between samples for csv file
    # file_points: name of csv file to write points data into
    # file_alpha: name of csv file to write alpha data into
    """
    p = int(np.sqrt(N_particles))
    N_particles = p**2
    N_A = N_particles // 2
    N_B = N_particles - N_A
    
    # gradient of alpha_B recovery slope
    alpha_gradient_B = (alpha_norm_B - alpha_low_B) / t_recovery_B
    
    # Function to calculate updated alpha_B value
    def alpha_B(t, t_star, 
            alpha_norm=alpha_norm_B, alpha_low=alpha_low_B, 
            alpha_gradient=alpha_gradient_B, t_recovery=t_recovery_B):
        if (t-t_star) < t_recovery:
            alpha = alpha_gradient*(t-t_star) + alpha_low
        else:
            alpha = alpha_norm
        return alpha
    
    L = np.sqrt(N_particles*np.pi / phi)/2  # box length for desired phi
    
    N_timesteps = round(t / delta_t)  # total number of time steps
    sample_timestep = round(sample / delta_t) # number of time steps between each sample
    
    # Initialise vectors/ matrices
    t_star = np.full(N_B, -np.inf) # last collision time
    M = np.zeros((N_A, N_B)) # collision matrix; particles type A+B currently in contact (no=0; yes=1)
    alphas = np.zeros(N_particles)
    alphas[:N_A] = alpha_A
    alphas[N_A:] = alpha_norm_B
    
    # Constants for main loop
    C_trans = np.sqrt(2*delta_t)
    C_rot = np.zeros(N_particles)
    ca = np.sqrt(alpha_A) * C_trans
    C_rot[:N_A].fill(ca)

    rcut = 2**(1/6)
    rskin = rcut + rdiff
    
    theta = np.random.random(N_particles) * 2*np.pi  # random initial orientational angles
    
    ## Generate grid of particles in box for initial conditions
    x = np.linspace(-L/2, L/2, p, endpoint=False)
    x, y = np.meshgrid(x, x, sparse=False, indexing='xy')
    x = x.flatten()
    y = y.flatten()
    j = np.random.permutation(N_particles)
    x = x[j]
    y = y[j]
    
    ## Neighbour list initialisations
    dxnei = np.zeros(N_particles)
    dynei = np.zeros(N_particles)
    grid_number = int(L // rskin)
    grid_size = L / grid_number
    grids_to_check = linked_grids(grid_number)
    nlist, npoint = linked_to_verlet(x, y, N_particles, L, rskin, grid_size, grid_number, grids_to_check)

    with open(file_points, 'w', newline='') as f:
        writer1 = csv.writer(f)
        writer1.writerow(np.concatenate((x, y, theta)))
        with open(file_alpha, 'w', newline='') as g:
            writer2 = csv.writer(g)
        
            ## Main loop ##        
            for n in range(1, N_timesteps+1):
                current_t = n*delta_t # current time for this loop

                ## If max displacement threshold reached, update neighbour lists
                drneimax1, drneimax2 = max_displacement(dxnei=dxnei, dynei=dynei, N_particles=N_particles)
                if drneimax1 + drneimax2 > rdiff:
                    nlist, npoint = linked_to_verlet(x, y, N_particles, L, rskin, grid_size, grid_number, grids_to_check)
                    dxnei = np.zeros(N_particles)
                    dynei = np.zeros(N_particles)

                # Update collision info
                M, t_star = coll_calc(x=x, y=y, L=L, N_A=N_A, N_B=N_B, M=M, t_star=t_star, t=current_t, rcut=rcut)
                
                # Calculate forces and update collision info
                Fx, Fy, M = force(x=x, y=y, N_particles=N_particles, L=L, rcut=rcut, nlist=nlist, npoint=npoint, M=M)
                
                # Update alpha_B values
                for j in range(N_B):
                    alphas[N_A+j] = alpha_B(t=current_t, t_star=t_star[j])
                
                C_rot[N_A:] = np.sqrt(alphas[N_A:]) * C_trans

                # Euler method
                dx = delta_t * (Fx + Pe*np.cos(theta)) + C_trans*np.random.normal(0, 1, size=N_particles)
                dy = delta_t * (Fy + Pe*np.sin(theta)) + C_trans*np.random.normal(0, 1, size=N_particles)
                x += dx
                y += dy
                dxnei += dx
                dynei += dy

                theta = theta + C_rot*np.random.normal(0, 1, size=N_particles)
                                     
                if n % sample_timestep == 0:
                    writer1.writerow(np.concatenate((x, y, theta)))
                    writer2.writerow(alphas)
    return x, y, theta