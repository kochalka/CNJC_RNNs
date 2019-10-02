import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.gridspec import GridSpec

def animate_trajectory(points, targ_points=None, keep_last=1, carray=None, interval=5):
    def update(i, data, scat):
        scat.set_offsets(data[max(0,i-keep_last):i])
        if carray is not None:
            scat.set_array(carray[max(0,i-keep_last):i])
        return scat

    fig, ax = plt.subplots(figsize=(8,4))
    scat = ax.scatter(*points[0].T, s=25)
    
    if targ_points is not None:
        ax.scatter(*targ_points, marker='x', s=0.1)
        ax.set_xlim([-0.6,0.4])
        ax.set_ylim([-0.2,0.4])
    else:
        mins = points.min(0)
        maxs = points.max(0)
        ax.set_xlim(mins[0]-0.1, maxs[0]+0.1)
        ax.set_ylim(mins[1]-0.1, maxs[1]+0.1)

    if carray is not None:
        scat.set_clim(vmin=carray.min(), 
                      vmax=carray.max())

    _ = ax.grid(True)
    plt.close()
    
    ani = animation.FuncAnimation(fig, update, frames=range(1,points.shape[0]),
                                  interval = interval, blit=True,
                                  fargs=(points, scat))
    return ani

def animate_activity(activity, trajectory, keep_last=1, targ_points=None, carray=None, interval=5):
    np.random.seed(42)
    positions = gen_positions(activity.shape[1])
    
    def update(i, activity, trajectory, scat0, scat1):
        scat0.set_array(activity[i])
        scat1.set_offsets(trajectory[max(0,i-keep_last):i])
        if carray is not None:
            scat1.set_array(carray[max(0,i-keep_last):i])
        return scat0, scat1

    fig = plt.figure(figsize=(9,3))
    gs = GridSpec(1,5)
    ax0 = fig.add_subplot(gs[:2])
    scat0 = ax0.scatter(*positions.T, s=10, c=activity[0])
    v0, v1 = np.percentile(activity, [2.5, 97.5])
    scat0.set_clim(vmin=-1, vmax=1)
    #scat0.set_clim(vmin=v0, vmax=v1)
    ax0.axis('square')
    ax0.set_xticklabels([])
    ax0.set_yticklabels([])
    ax0.set_title('Hidden Unit Activity')
    
    ax1 = fig.add_subplot(gs[2:])    
    scat1 = ax1.scatter(*trajectory[0].T, s=25)
    
    if targ_points is not None:
        ax1.scatter(*targ_points, marker='x', s=0.1)
        ax1.set_xlim([-0.6,0.4])
        ax1.set_ylim([-0.2,0.4])
    else:
        mins = points.min(0)
        maxs = points.max(0)
        ax1.set_xlim(mins[0]-0.1, maxs[0]+0.1)
        ax1.set_ylim(mins[1]-0.1, maxs[1]+0.1)

    if carray is not None:
        scat1.set_clim(vmin=carray.min(), 
                       vmax=carray.max())

    _ = ax1.grid(True)
    
    plt.close()
    
    ani = animation.FuncAnimation(fig, update, frames=range(activity.shape[0]),
                                  interval = interval, blit=True,
                                  fargs=(activity, trajectory, scat0, scat1))
    return ani

def gen_positions(n,scale=1):
    r = scale * np.sqrt(np.random.rand(n))
    theta =  2 * np.pi * np.random.rand(n)
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    return np.c_[x,y]