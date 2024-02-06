import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import sys
from swe_utils.fv_solver import read_val_profile_from_python_sol_and_nondim

plt.rcParams['savefig.dpi']=300

# FUNCTIONS FOR PAPER PLOTS

def add_legend_labels(vartype="var"):    
    plt.legend()
    plt.set_xlabel("x", fontsize=12)
    plt.set_ylabel(vartype, fontsize=12)
    plt.set_xticks(fontsize=14)
    plt.set_yticks(fontsize=14)
    return

def add_labels(ax, vartype="var", xlabel=True):   
    if (xlabel): 
        ax.set_xlabel("x (m)", fontsize=11)
        #ax.set_xticks(fontsize=14)
    ax.set_ylabel(vartype, fontsize=11)
    #ax.set_yticks(fontsize=14)
    return

def set_axes_limits(ax, xrange=[-1,1], yrange=[0, 1], xset=False):
    if (xset):
        ax.set_xlim(xrange)
    ax.set_ylim(yrange)
    return

def get_val_filenames(prb, nsteps_plot, val_root="val_t_"):
    step = (prb.tmax - prb.tstart) / (nsteps_plot-1)
    t_val_loop = np.arange(prb.tstart, prb.tmax+step, step)
    val_filenames = []
    for i in range(0, t_val_loop.size):
        ts_val = t_val_loop[i]       
        val_filenames.append(val_root + f"{t_val_loop[i]:.2f}")
    return val_filenames

def update_min_max_arrays(array1, array2, v_range):
    v_min = min(np.amin(array1), np.amin(array2))
    if (v_min<v_range[0]):
        v_range[0]=v_min
    v_max = max(np.amax(array1), np.amax(array2))
    if (v_max>v_range[1]):
        v_range[1]=v_max   
    return v_range

def fix_range(v_range):
    delta = (v_range[1] - v_range[0])*0.05
    v_range[0] = v_range[0] - delta
    v_range[1] = v_range[1] + delta
    return v_range

def read_val_data_from_npz(val_file, prb, nd_params, val_dir):
    # load data from validators folder
    data_2 = np.load(val_dir + val_file + ".npz", allow_pickle=True)
    data_2 = np.atleast_1d(data_2.f.arr_0)[0]
    # read all useful variables
    xcoord = data_2["x"][:, 0] * nd_params.L0 + prb.xdisc
    h_true = data_2["true_h"][:,0] * nd_params.H0
    h_pred = data_2["pred_h"][:,0] * nd_params.H0
    u_true = data_2["true_u"][:,0] * nd_params.U0
    u_pred = data_2["pred_u"][:,0] * nd_params.U0
    try:
        z_true = data_2["true_z"][:,0] * nd_params.H0
        z_pred = data_2["pred_z"][:,0] * nd_params.H0
    except:
        z_true = np.zeros(np.shape(xcoord))
        z_pred = np.zeros(np.shape(xcoord))
    return xcoord, h_true, h_pred, u_true, u_pred, z_true, z_pred   

# function dedicated to multiple plots for different times for unsteady problems
def plot_unsteady_1_times_panels(test_name, prb, nd_params, val_dir, plot_subdir, val_filenames, figure_title, csv_filenames, csv_labels):
    
    n_csv = len(csv_filenames)
    colors=["c--", "g--"]
    n_rows = len(val_filenames)
    f, axs = plt.subplots(nrows=n_rows, ncols=2, sharex=True, figsize=(8, 3*n_rows), dpi=100)
    
    #f.suptitle(figure_title)
    h_range=[999, -999]
    u_range=[999, -999]

    for i in range(0, n_rows):
        val_file = val_filenames[i]
        # get data to plot from validator files (and re-scale to dimensional values)
        xcoord, h_true, h_pred, u_true, u_pred, z_true, z_pred = read_val_data_from_npz(val_file=val_file, prb=prb, nd_params=nd_params, val_dir=val_dir)
        # compute ranges (shared for subplots in the same column)
        h_range = update_min_max_arrays(h_true, h_pred, h_range) 
        u_range = update_min_max_arrays(u_true, u_pred, u_range)

        axs[0].plot(xcoord, h_true, "k-", label="exact")
        axs[0].plot(xcoord, h_pred, "r--", label="PINN_augm")
        add_labels(axs[0], "h (m)", xlabel=False)
        
        axs[1].plot(xcoord, u_true, "k-", label="exact")
        axs[1].plot(xcoord, u_pred, "r--", label="PINN_augm")
        add_labels(axs[1], "u (m/s)", xlabel=False)    

        for aa in range(0, n_csv):
            csv_file = csv_filenames[aa]
            xa, ta, ha, ua, qa = read_val_profile_from_python_sol_and_nondim(csv_file, nd_params, 1., prb.xdisc)
            axs[0].plot(xa, ha, colors[aa], label=csv_labels[aa])
            axs[1].plot(xa, ua, colors[aa], label=csv_labels[aa])
            h_range = update_min_max_arrays(ha, ha, h_range) 
            u_range = update_min_max_arrays(ua, ua, u_range)

    h_range = fix_range(h_range)
    u_range = fix_range(u_range)

    axs[0].set_title("a) depth")
    axs[1].set_title("b) velocity")
    
    for i in range(2): 
        axs[i].set_xlabel("x (m)", fontsize=11)

    for i in range(0, n_rows):        
        set_axes_limits(axs[0], yrange=h_range, xset=False)
        set_axes_limits(axs[1], yrange=u_range, xset=False)

    # add legend at the most appropriate location
    axs[1].legend(bbox_to_anchor=(1.01, 0.15), loc='lower left')

    f.tight_layout()
    return f, axs

def read_npz_write_to_csv(csv_filename, timeval, prb, nd_params, val_dir, val_labels):
    xcoord, h_true, h_pred, u_true, u_pred, z_true, z_pred = read_val_data_from_npz(val_file=val_labels[0], prb=prb, nd_params=nd_params, val_dir=val_dir)
    q_ = np.multiply(h_pred, u_pred)
    A = np.stack((xcoord,h_pred,u_pred,q_), axis=1)
    np.savetxt(csv_filename, A, fmt='%1.5f')
    return 

if __name__ == "__main__":

    #import the "setup" file for non augm (already contains test_name, paths, etc.)
    from setup_db_nonaugm import *
    nsteps_plot = 1
    tvals = ["1"]
    #val_labels = get_val_filenames(prb, nsteps_plot, val_root="val_t_")
    val_labels = ["val_t_1.00"]
    csv_filenames = [sol_dir +"dambreak_nonaugm_1.00000.txt"]
    csv_labels = ["PINN_non_augm"]
    read_npz_write_to_csv(csv_filenames[0], 1., prb, nd_params, val_dir, val_labels)
    
    from setup_dambreak import *
    
    # call the plot function
    f, axs = plot_unsteady_1_times_panels(test_name, prb, nd_params, val_dir, plot_subdir, val_labels, "Dam break over horizontal bottom",csv_filenames, csv_labels)

    # save the figure in the plots' subfolder
    f.savefig(test_name + "_nonaugm_for_paper.png")
