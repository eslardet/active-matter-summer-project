import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
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
def local_density(x, y, N_particles, L, grid_size, grid_number):
    """
    ## Calculate local number densities within each smaller grid using sub-box method

    Input:
    # x: x coordinates of particles
    # y: y coordinates of particles
    # N_particles: total number of particles
    # L: box length
    # grid_size: box length of grids in sub-box system
    # grid_number: number of grids per length L in sub-box system

    Output:
    # n_density: number density of particles within each grid
    # count_A: number of type A particles in each grid
    # count_B: number of type B particles in each grid
    """
    N_A = N_particles // 2
    count_A = np.zeros(grid_number**2)
    count_B = np.zeros(grid_number**2)
    for i in range(N_A):
        grid = int((pbc_wrap(x[i],L)+L/2) // grid_size + ((pbc_wrap(y[i],L)+L/2) // grid_size) * grid_number)
        count_A[grid] += 1
    for i in range(N_A, N_particles):
        grid = int((pbc_wrap(x[i],L)+L/2) // grid_size + ((pbc_wrap(y[i],L)+L/2) // grid_size) * grid_number)
        count_B[grid] += 1
    n_density = (count_A + count_B) / grid_size**2
    return n_density, count_A, count_B

def snapshot_local_density(file, N_particles, phi, frame, min_grid_size=5):
    """
    ## Calculate local number densities within each smaller grid using sub-box method from csv file

    Input:
    # file: csv file in which a single row contains all the particle x-coordinates, then y-coordinates and 
    theta angles at a single sample time (output file from relevant abp model functions)
    # N_particles: total number of particles
    # phi: volume fraction
    # frame: row of csv file to read
    # min_grid_size: minimum grid length to use for sub-box method (default: 5)

    Output:
    # n_density: number density of particles within each grid
    # count_A: number of type A particles in each grid
    # count_B: number of type B particles in each grid
    """
    with open(file, "r", encoding="utf-8", errors="ignore") as scraped:
        final_line = scraped.readlines()[frame]
        
    L = np.sqrt(N_particles*np.pi / (4*phi))
    grid_number = int(L // min_grid_size)
    grid_size = L / grid_number
    
    r = np.fromstring(final_line, sep=',')
    x = pbc_wrap(r[:N_particles], L)
    y = pbc_wrap(r[N_particles:2*N_particles], L)
    
    n_density, count_A, count_B = local_density(x, y, N_particles, L, grid_size, grid_number)
    return n_density, count_A, count_B

def plot_densities(file, N_particles, phi, frames, label, color, min_grid_size=5, ax=None):
    """
    ## Plot a probability distribution (kernel density estimation from histogram) plot from local number densities

    Input:
    # file: csv file in which a single row contains all the particle x-coordinates, then y-coordinates and 
    theta angles at a single sample time (output file from relevant abp model functions)
    # N_particles: total number of particles
    # phi: volume fraction
    # frames: rows of csv file to read to add to histogram
    # label: label for plot
    # color: colour for line plot
    # min_grid_size: minimum grid length to use for sub-box method (default: 5)
    # ax: axis on which to plot (if None then a figure and axis is created)

    Output:
    # ax: axis with plotted probability distribution
    """
    all_density = []
    all_density = np.array(all_density)

    for f in frames:
        current_density, count_A, count_B = snapshot_local_density(file, N_particles, phi, f, min_grid_size)
        all_density = np.append(all_density, current_density)
        
    if ax == None:
        fig, ax = plt.subplots()
        
    kde = sps.gaussian_kde(all_density)
    x_plot = np.linspace(0, 1.5, 1000)
    ax.plot(x_plot, kde.pdf(x_plot), label=label, color=color)
    
    return ax

def plot_gas_frac(file, N_particles, phi, frames, sample=1, min_grid_size=5, ax=None):
    """
    ## Plot fraction of particle in gas phase (defined as N/V<0.9 in this case) over time from csv file

    Input:
    # file: csv file in which a single row contains all the particle x-coordinates, then y-coordinates and 
    theta angles at a single sample time (output file from relevant abp model functions)
    # N_particles: total number of particles
    # phi: volume fraction
    # frames: rows of csv file to read to add to histogram
    # sample: time between samples for csv file
    # min_grid_size: minimum grid length to use for sub-box method (default: 5)
    # ax: axis on which to plot (if None then a figure and axis is created)

    Output:
    # ax: axis with plotted line
    """
    if ax == None:
        fig, ax = plt.subplots()
    
    gas_frac = []
    for f in frames:
        current_density, count_A, count_B = snapshot_local_density(file, N_particles, phi, f, min_grid_size)
        gas_frac.append(sum(i < 0.9 for i in current_density) / len(current_density))
        
    ax.plot(frames*sample, gas_frac, '-')
    
    return ax

def plot_density_frac(file, N_particles, phi, frames, min_grid_size=5, ax=None):
    """
    ## Plot fraction of particles in each grid against grid number density from csv file for type A and type B particles

    Input:
    # file: csv file in which a single row contains all the particle x-coordinates, then y-coordinates and 
    theta angles at a single sample time (output file from relevant abp model functions)
    # N_particles: total number of particles
    # phi: volume fraction
    # frames: rows of csv file to read to add to histogram
    # min_grid_size: minimum grid length to use for sub-box method (default: 5)
    # ax: two axes ax[0] and ax[1] on which to plot (if None then a figure and axis is created)

    Output:
    # ax: axes with plotted scatter and line plots
    """
    all_density = []
    all_density = np.array(all_density)
    all_frac_A = []
    all_frac_A = np.array(all_frac_A)
    all_frac_B = []
    all_frac_B = np.array(all_frac_B)

    for f in frames:
        current_density, count_A, count_B = snapshot_local_density(file, N_particles, phi, f, min_grid_size)
        count_total = count_A + count_B
        n_grids = len(count_total)
        for i in range(n_grids):
            if count_total[i] != 0:
                all_density = np.append(all_density, current_density[i])
                all_frac_A = np.append(all_frac_A, count_A[i] / (count_total[i]))
                all_frac_B = np.append(all_frac_B, count_B[i] / (count_total[i]))

    if ax == None:
        fig, ax = plt.subplots(1, 2)
    
    x_plot = np.linspace(0, 1.5, 100)
    m_A, b_A = np.polyfit(all_density, all_frac_A, 1)
    m_B, b_B = np.polyfit(all_density, all_frac_B, 1)
    ax[0].plot(all_density, all_frac_A, 'x', color='tab:blue')
    ax[1].plot(all_density, all_frac_B, 'x', color='tab:orange')
    
    ax[0].plot(x_plot, m_A*x_plot + b_A, color='tab:blue', label="A")
    ax[1].plot(x_plot, m_B*x_plot + b_B, color='tab:orange', label="B")
    
    return ax