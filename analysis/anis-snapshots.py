import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv

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

def csv_snapshot_1pop(file, N_particles, phi, sample, row_to_read, arrows=False):
    """
    ## Plot a snapshot of a one population simulation from csv file

    Input:
    # file: csv file in which a single row contains all the particle x-coordinates, then y-coordinates and 
    theta angles at a single sample time (output file from relevant abp model functions)
    # N_particles: total number of particles
    # phi: volume fraction
    # sample: time between samples for csv file
    # row_to_read: row of csv file to use for snapshot
    # arrows: whether to display quiver arrows for particle orientation angles (True or False)
    """
    with open(file, "r", encoding="utf-8", errors="ignore") as scraped:
        final_line = scraped.readlines()[row_to_read]
    L = np.sqrt(N_particles*np.pi / (4*phi))
    
    r = np.fromstring(final_line, sep=',')
    x = pbc_wrap(r[:N_particles], L)
    y = pbc_wrap(r[N_particles:2*N_particles], L)
    
    fig, ax = plt.subplots(figsize=(5,5), dpi=72)

    diameter = (ax.get_window_extent().width * 72/fig.dpi) /L
    ax.plot(x, y, 'o', ms=diameter, alpha=0.3, zorder=1)
    if arrows == True:
        theta = r[2*N_particles:3*N_particles]
        ax.quiver(x, y, np.cos(theta), np.sin(theta), zorder=2)
    ax.set_xlim(-L/2, L/2)
    ax.set_ylim(-L/2, L/2)
    ax.set_aspect('equal')
    ax.set_title("t=" + str(int(round(row_to_read * sample, 0))) + r"$\tau$")
    plt.show()

def csv_snapshot_2pop(file, N_particles, phi, sample, row_to_read, arrows=False):
    """
    ## Plot a snapshot of a two population simulation from csv file

    Input:
    # file: csv file in which a single row contains all the particle x-coordinates, then y-coordinates and 
    theta angles at a single sample time (output file from relevant abp model functions)
    # N_particles: total number of particles
    # phi: volume fraction
    # sample: time between samples for csv file
    # row_to_read: row of csv file to use for snapshot
    # arrows: whether to display quiver arrows for particle orientation angles (True or False)
    """
    N_A = N_particles // 2
    with open(file, "r", encoding="utf-8", errors="ignore") as scraped:
        final_line = scraped.readlines()[row_to_read]
    L = np.sqrt(N_particles*np.pi / (4*phi))
    
    r = np.fromstring(final_line, sep=',')
    x = pbc_wrap(r[:N_particles], L)
    y = pbc_wrap(r[N_particles:2*N_particles], L)
    
    fig, ax = plt.subplots(figsize=(5,5), dpi=72)

    diameter = (ax.get_window_extent().width * 72/fig.dpi) /L
    ax.plot(x[:N_A], y[:N_A], 'o', ms=diameter, zorder=1, alpha=0.3)
    ax.plot(x[N_A:], y[N_A:], 'o', ms=diameter, zorder=1, alpha=0.3)
    if arrows == True:
        theta = r[2*N_particles:3*N_particles]
        ax.quiver(x, y, np.cos(theta), np.sin(theta), zorder=3)
    ax.set_xlim(-L/2, L/2)
    ax.set_ylim(-L/2, L/2)
    ax.set_aspect('equal')
    ax.set_title("t=" + str(int(round(row_to_read * sample, 0))) + r"$\tau$")
    return fig, ax

def csv_make_ani_1pop(file, N_particles, phi, sample, frames, arrows=False):
    """
    ## Plot a snapshot of a one population simulation from csv file

    Input:
    # file: csv file in which a single row contains all the particle x-coordinates, then y-coordinates and 
    theta angles at a single sample time (output file from relevant abp model functions)
    # N_particles: total number of particles
    # phi: volume fraction
    # sample: time between samples for csv file
    # frames: scalar or array for animation frames corresponding to rows of the csv file
    # arrows: whether to display quiver arrows for particle orientation angles (True or False)
    """
    with open(file) as f:
        csv_reader = csv.reader(f)
        rows = list(csv_reader)
    L = np.sqrt(N_particles*np.pi / (4*phi))
    
    N_A = N_particles // 2
    
    ## If using Jupyter notebook
    # %matplotlib notebook
    # plt.rcParams["animation.html"] = "jshtml"
    # plt.ioff()
    # plt.rcParams['animation.embed_limit'] = 2**128

    fig, ax = plt.subplots(figsize=(5,5))

    diameter = (ax.get_window_extent().width * 72/fig.dpi) /L

    points, = plt.plot([], [], 'o', alpha=0.3, ms=diameter, zorder=1)

    if arrows == True:
        r = np.asarray(rows[0], dtype=np.float64)
        x = pbc_wrap(r[:N_particles], L)
        y = pbc_wrap(r[N_particles:2*N_particles], L)
        theta = r[2*N_particles:3*N_particles]
        arrows = plt.quiver(x, y, np.cos(theta), np.sin(theta), zorder=3)

    def init():
        ax.set_xlim(-L/2, L/2)
        ax.set_ylim(-L/2, L/2)
        if arrows == True:
            return arrows, points
        else:
            return points

    def update(n):
        ax.set_title("t = " + str(round(n*sample, 1)) + r"$\tau$", fontsize=10, loc='left')
        r = np.asarray(rows[n], dtype=np.float64)
        x = pbc_wrap(r[:N_particles], L)
        y = pbc_wrap(r[N_particles:2*N_particles], L)
        
        points.set_data(x, y)
        if arrows == True:
            theta = r[2*N_particles:3*N_particles]
            arrows.set_offsets(np.c_[x, y])
            arrows.set_UVC(np.cos(theta), np.sin(theta))
            return arrows, points
        else:
            return points

    ani = FuncAnimation(fig, update, init_func=init, frames=frames, interval=10, blit=True)
    return ani

def csv_make_ani_2pop(file, N_particles, phi, sample, frames, arrows=False):
    """
    ## Plot a snapshot of a two population simulation from csv file

    Input:
    # file: csv file in which a single row contains all the particle x-coordinates, then y-coordinates and 
    theta angles at a single sample time (output file from relevant abp model functions)
    # N_particles: total number of particles
    # phi: volume fraction
    # sample: time between samples for csv file
    # frames: scalar or array for animation frames corresponding to rows of the csv file
    # arrows: whether to display quiver arrows for particle orientation angles (True or False)
    """
    with open(file) as f:
        csv_reader = csv.reader(f)
        rows = list(csv_reader)
    L = np.sqrt(N_particles*np.pi / (4*phi))
    
    N_A = N_particles // 2
    
    ## If using Jupyter notebook
    # %matplotlib notebook
    # plt.rcParams["animation.html"] = "jshtml"
    # plt.ioff()
    # plt.rcParams['animation.embed_limit'] = 2**128

    fig, ax = plt.subplots(figsize=(5,5))

    diameter = (ax.get_window_extent().width * 72/fig.dpi) /L

    points_A, = plt.plot([], [], 'o', alpha=0.3, ms=diameter, zorder=1)
    points_B, = plt.plot([], [], 'o', alpha=0.3, ms=diameter, zorder=2)

    if arrows == True:
        r = np.asarray(rows[0], dtype=np.float64)
        x = pbc_wrap(r[:N_particles], L)
        y = pbc_wrap(r[N_particles:2*N_particles], L)
        theta = r[2*N_particles:3*N_particles]
        arrows = plt.quiver(x, y, np.cos(theta), np.sin(theta), zorder=3)

    def init():
        ax.set_xlim(-L/2, L/2)
        ax.set_ylim(-L/2, L/2)
        if arrows == True:
            return arrows, points_A, points_B,
        else:
            return points_A, points_B,

    def update(n):
        ax.set_title("t = " + str(round(n*sample, 1)) + r"$\tau$", fontsize=10, loc='left')
        r = np.asarray(rows[n], dtype=np.float64)
        x = pbc_wrap(r[:N_particles], L)
        y = pbc_wrap(r[N_particles:2*N_particles], L)
        
        points_A.set_data(x[:N_A], y[:N_A])
        points_B.set_data(x[N_A:], y[N_A:])
        if arrows == True:
            theta = r[2*N_particles:3*N_particles]
            arrows.set_offsets(np.c_[x, y])
            arrows.set_UVC(np.cos(theta), np.sin(theta))
            return arrows, points_A, points_B
        else:
            return points_A, points_B

    ani = FuncAnimation(fig, update, init_func=init, frames=frames, interval=10, blit=True)
    return ani