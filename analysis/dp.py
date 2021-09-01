import numpy as np
import freud

def find_DP(file, N_particles, phi, row_to_read):
    """
    ## Calculate demixing parameter value from row (snapshot) of csv file

    Input:
    # file: csv file in which a single row contains all the particle x-coordinates, then y-coordinates and 
    theta angles at a single sample time (output file from relevant abp model functions)
    # N_particles: total number of particles
    # phi: volume fraction
    # sample: time between samples for csv file
    # row_to_read: row of csv file to use for snapshot

    Output:
    # 
    """
    with open(file, "r", encoding="utf-8", errors="ignore") as scraped:
        final_line = scraped.readlines()[row_to_read]
    
    L = np.sqrt(N_particles*np.pi / phi)/2
    box = freud.Box.square(L)
    points = np.zeros((N_particles, 3))
    r = np.fromstring(final_line, sep=',')
    points[:,0] = r[:N_particles]
    points[:,1] = r[N_particles:2*N_particles]
    
    points_w = box.wrap(points)
    voro = freud.locality.Voronoi()
    voro.compute((box, points_w))
    nl = voro.nlist
    n_count = np.zeros((N_particles, 2))
    
    types = np.empty(N_particles, dtype=str)
    N_A = N_particles // 2
    types[:N_A] = 'A'
    types[N_A:] = 'B'
    
    for (i,j) in nl:
        if types[i] == types[j]:
            n_count[i,0] += 1
        n_count[i,1] += 1
    DP = 2 * ((n_count[:,0] / n_count[:,1]) - 1/2)
    DP_av = np.mean(DP)
    return DP_av