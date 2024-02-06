import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import sys

plt.rcParams['axes.formatter.useoffset'] = False
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
    z_true = data_2["true_z"][:,0] * nd_params.H0
    z_pred = data_2["pred_z"][:,0] * nd_params.H0
    return xcoord, h_true, h_pred, u_true, u_pred, z_true, z_pred   

# function dedicated to multiple plots for different times for unsteady problems
def plot_unsteady_multiple_times_panels(test_name, prb, nd_params, val_dir, plot_subdir, val_filenames, figure_title, wse_max=True, shift_wse_range=0., add_error_z=False):
    
    n_rows = len(val_filenames)
    f, axs = plt.subplots(nrows=n_rows, ncols=4, sharex=True, figsize=(12, 3*n_rows), dpi=100)
    
    f.suptitle(figure_title)
    h_range=[999, -999]
    u_range=[999, -999]
    z_range=[999, -999]

    for i in range(0, n_rows):
        val_file = val_filenames[i]
        # get data to plot from validator files (and re-scale to dimensional values)
        xcoord, h_true, h_pred, u_true, u_pred, z_true, z_pred = read_val_data_from_npz(val_file=val_file, prb=prb, nd_params=nd_params, val_dir=val_dir)
        # compute ranges (shared for subplots in the same column)
        h_range = update_min_max_arrays(h_true, h_pred, h_range) 
        u_range = update_min_max_arrays(u_true, u_pred, u_range) 
        z_range = update_min_max_arrays(z_true, z_pred, z_range) 

        axs[i,0].plot(xcoord, h_true, "k-", label="exact")
        axs[i,0].plot(xcoord, h_pred, "r--", label="PINN")
        add_labels(axs[i,0], "h (m)", xlabel=False)
        
        axs[i,1].plot(xcoord, u_true, "k-", label="exact")
        axs[i,1].plot(xcoord, u_pred, "r--", label="PINN")
        add_labels(axs[i,1], "u (m/s)", xlabel=False)    
         
        if (add_error_z):
            dz = z_pred - z_true
            axs[i,2].plot(xcoord, dz, "c:", label="error")        
            add_labels(axs[i,2], "\u0394 z (m)", xlabel=False) 
        else:
            axs[i,2].plot(xcoord, z_true, "k-", label="exact")
            axs[i,2].plot(xcoord, z_pred, "r--", label="PINN")
            add_labels(axs[i,2], "z (m)", xlabel=False)

        #axs[i,2].plot(xcoord, z_true, "k-", label="exact")
        #axs[i,2].plot(xcoord, z_pred, "r--", label="PINN")
        #add_labels(axs[i,2], "z (m)", xlabel=False)

        axs[i,3].plot(xcoord,z_true, "k-", label="exact")
        axs[i,3].plot(xcoord,z_pred, "r--", label="PINN")
        axs[i,3].plot(xcoord,z_true + h_true, "k-") 
        axs[i,3].plot(xcoord,z_pred + h_pred, "r--")
        add_labels(axs[i,3], "z, wse (m)", xlabel=False)

    h_range = fix_range(h_range)
    u_range = fix_range(u_range)
    z_range = fix_range(z_range)
    if (wse_max):
        wse_range = [z_range[0], z_range[1]+h_range[1]+shift_wse_range]
    else:
        wse_range = [z_range[0], max(h_range[1], z_range[1])+shift_wse_range]
    axs[0,0].set_title("a) depth")
    axs[0,1].set_title("b) velocity")
    if (add_error_z):
       axs[0,2].set_title("c) bed elev. error") 
    else:
        axs[0,2].set_title("c) bed elev.")
    axs[0,3].set_title("d) water surface elev.")
    for i in range(4): 
        axs[n_rows-1,i].set_xlabel("x (m)", fontsize=11)

    for i in range(0, n_rows):        
        set_axes_limits(axs[i,0], yrange=h_range, xset=False)
        set_axes_limits(axs[i,1], yrange=u_range, xset=False)
        if (not add_error_z):
            set_axes_limits(axs[i,2], yrange=z_range, xset=False)
        set_axes_limits(axs[i,3], yrange=wse_range, xset=False)

    f.tight_layout()
    return f, axs

# function dedicated to plots for just one time (equal to the final time) for static/steady problems
def plot_static_or_steady_tmax_1_panels(test_name, prb, nd_params, val_dir, plot_subdir, val_file, figure_title, add_error_z=False):
    
    # get data to plot from validator files (and re-scale to dimensional values)
    xcoord, h_true, h_pred, u_true, u_pred, z_true, z_pred = read_val_data_from_npz(val_file=val_file, prb=prb, nd_params=nd_params, val_dir=val_dir)

    # make plot with subplots
    f, axs = plt.subplots(nrows=1, ncols=4, figsize=(12, 3), dpi=100)
    f.suptitle(figure_title)

    axs[0].set_title("a) depth")
    axs[0].plot(xcoord, h_true, "k-", label="exact")
    axs[0].plot(xcoord, h_pred, "r--", label="PINN")
    add_labels(axs[0], "h (m)")

    axs[1].set_title("b) velocity")
    axs[1].plot(xcoord, u_true, "k-", label="exact")
    axs[1].plot(xcoord, u_pred, "r--", label="PINN")
    add_labels(axs[1], "u (m/s)")    
    
    if (add_error_z):
        dz = z_pred - z_true
        axs[2].set_title("c) bed elev. error")
        axs[2].plot(xcoord, dz, "c:", label="error")        
        add_labels(axs[2], "\u0394 z (m)") 
    else:
        axs[2].set_title("c) bed elev.")
        axs[2].plot(xcoord, z_true, "k-", label="exact")
        axs[2].plot(xcoord, z_pred, "r--", label="PINN")
        add_labels(axs[2], "z (m)")

    axs[3].set_title("d) water surface elev.")
    axs[3].plot(xcoord,z_true, "k-", label="exact")
    axs[3].plot(xcoord,z_pred, "r--", label="PINN")
    axs[3].plot(xcoord,z_true + h_true, "k-") 
    axs[3].plot(xcoord,z_pred + h_pred, "r--")
    add_labels(axs[3], "z, wse (m)")
    
    f.tight_layout()
    return f, axs

if __name__ == "__main__":

    # pass the case study name as argument of the command line
    try:
        case_study = sys.argv[1]
    except:
        case_study = "Please provide the case name in the command line"

    if (case_study == "bumpsteady"):
        #import the "setup" file (already contains test_name, paths, etc.)
        from setup_bumpsteady import *
        val_label = "val_t_" + f"{prb.tmax:.2f}"
        # call the plot function
        f, axs = plot_static_or_steady_tmax_1_panels(test_name, prb, nd_params, val_dir, plot_subdir, val_file = val_label, figure_title="Steady flow over bump")
        # add annotations in the fourth panel (if necessary)
        axs[3].annotate("z" , (12, 0.15) )
        axs[3].annotate("wse" , (12, 2.15) )
        # add legend at the most appropriate location
        axs[3].legend(bbox_to_anchor=(0.99, 0.85), loc='upper right')
        # save the figure in the plots' subfolder
        f.savefig(test_name + "_validation_for_paper.png")
    elif (case_study == "dambreak"):
        #import the "setup" file (already contains test_name, paths, etc.)
        from setup_dambreak import *
        nsteps_plot = 3
        tvals = ["0", "0.5", "1"]
        val_labels = get_val_filenames(prb, nsteps_plot, val_root="val_t_")
        # call the plot function
        f, axs = plot_unsteady_multiple_times_panels(test_name, prb, nd_params, val_dir, plot_subdir, val_labels, "Dam break over horizontal bottom")
        # add annotations in the fourth panel (if necessary)
        for i in range(nsteps_plot):
            axs[i,3].annotate("z" , (4.6, 0.05) )
            axs[i,3].annotate("wse" , (4.6, 0.85) )
            # add text with time of each row at the most appropriate location
            for j in range(4):
                if (j==1 or j==2):
                    at = AnchoredText("t = "+ tvals[i], prop=dict(size=11), frameon=False, loc='upper left')
                else:
                    at = AnchoredText("t = "+ tvals[i], prop=dict(size=11), frameon=False, loc='upper right')
                axs[i,j].add_artist(at)
            # add legend at the most appropriate location
            axs[i,3].legend(bbox_to_anchor=(0.01, 0.15), loc='lower left')
        # save the figure in the plots' subfolder
        f.savefig(test_name + "_validation_for_paper.png")
    elif (case_study == "dbstep"):
        from setup_dbstep import *
        nsteps_plot = 3
        tvals = ["0", "0.5", "1"]
        val_labels = get_val_filenames(prb, nsteps_plot, val_root="val_t_")
        f, axs = plot_unsteady_multiple_times_panels(test_name, prb, nd_params, val_dir, plot_subdir, val_labels, "Dam break over bottom step", wse_max=False)
        for i in range(nsteps_plot):
            axs[i,3].annotate("z" , (4.8, 0.8) )
            axs[i,3].annotate("wse" , (4.6, 1.7) )
            for j in range(4):
                if (j==1 or j==2):
                    at = AnchoredText("t = "+ tvals[i], prop=dict(size=11), frameon=False, loc='upper left')
                else:
                    at = AnchoredText("t = "+ tvals[i], prop=dict(size=11), frameon=False, loc='upper right')
                axs[i,j].add_artist(at)
            axs[i,3].legend(bbox_to_anchor=(0.01, 0.4), loc='lower left')
        f.savefig(test_name + "_validation_for_paper.png")
    elif (case_study == "smallpert"):
        from setup_smallpert import *
        nsteps_plot = 3
        tvals = ["0", "0.5", "1"]
        val_labels = get_val_filenames(prb, nsteps_plot, val_root="val_t_")
        f, axs = plot_unsteady_multiple_times_panels(test_name, prb, nd_params, val_dir, plot_subdir, val_labels, "Small amplitude wave over horizontal bottom", shift_wse_range=0.1)
        for i in range(nsteps_plot):
            axs[i,3].annotate("z" , (-5.5, 0.025) )
            axs[i,3].annotate("wse" , (-5.5, 0.7) )
            for j in range(4):
                at = AnchoredText("t = "+ tvals[i], prop=dict(size=11), frameon=False, loc='upper left')
                axs[i,j].add_artist(at)
            axs[i,3].legend(loc='center right')
        f.savefig(test_name + "_validation_for_paper.png")
    elif (case_study == "staticbump"):
        from setup_staticbump import *
        val_label = "val_t_" + f"{prb.tmax:.2f}"
        f, axs = plot_static_or_steady_tmax_1_panels(test_name, prb, nd_params, val_dir, plot_subdir, val_file = val_label, figure_title="Still water over bottom with bump")
        axs[3].annotate("z" , (-9, 0.025) )
        axs[3].annotate("wse" , (-9, 0.45) )
        axs[3].legend(bbox_to_anchor=(0.99, 0.85), loc='upper right')
        f.savefig(test_name + "_validation_for_paper.png")
    elif (case_study == "staticflat"):
        from setup_staticflat import *
        val_label = "val_t_" + f"{prb.tmax:.2f}"
        f, axs = plot_static_or_steady_tmax_1_panels(test_name, prb, nd_params, val_dir, plot_subdir, val_file = val_label, figure_title="Still water over horizontal bottom")
        axs[3].annotate("z" , (-5, 0.05) )
        axs[3].annotate("wse" , (-5, 0.9) )
        axs[3].legend(bbox_to_anchor=(0.99, 0.85), loc='upper right')
        f.savefig(test_name + "_validation_for_paper.png")
    elif (case_study == "staticparab"):
        from setup_staticparab import *
        val_label = "val_t_" + f"{prb.tmax:.2f}"
        f, axs = plot_static_or_steady_tmax_1_panels(test_name, prb, nd_params, val_dir, plot_subdir, val_file = val_label, figure_title="Still water over parabolic bottom", add_error_z=True)
        axs[3].annotate("z" , (-1.05, -0.25) )
        axs[3].annotate("wse" , (-0.2, 0.1) )
        axs[3].legend()
        axs[2].legend(loc="upper center")
        f.savefig(test_name + "_validation_for_paper.png")
    elif (case_study == "staticstep"):
        from setup_staticstep import *
        val_label = "val_t_" + f"{prb.tmax:.2f}"
        f, axs = plot_static_or_steady_tmax_1_panels(test_name, prb, nd_params, val_dir, plot_subdir, val_file = val_label, figure_title="Still water over bottom with step")
        axs[3].annotate("z" , (-5, 0.1) )
        axs[3].annotate("wse" , (-5, 1.8) )
        axs[3].legend(bbox_to_anchor=(0.99, 0.9), loc='upper right')
        f.savefig(test_name + "_validation_for_paper.png")
    elif (case_study == "thacker"):
        from setup_thacker import *
        nsteps_plot = 5
        tvals = ["0", "0.25T", "0.5T", "0.75T", "T"]
        val_labels = get_val_filenames(prb, nsteps_plot, val_root="val_t_")
        f, axs = plot_unsteady_multiple_times_panels(test_name, prb, nd_params, val_dir, plot_subdir, val_labels, "Oscillating planar surface over parabolic basin", wse_max=False, add_error_z=True)
        for i in range(nsteps_plot):
            axs[i,3].annotate("z" , (1.05, -0.25) )
            axs[i,3].annotate("wse" , (-0.2, 0.1) )
            for j in range(4):
                if (j==0 or j==1):
                    if (i==1 and j==1):
                        at = AnchoredText("t="+ tvals[i], prop=dict(size=11), frameon=False, loc='lower left')
                    else:
                        at = AnchoredText("t="+ tvals[i], prop=dict(size=11), frameon=False, loc='upper left')
                elif (j==2):
                    at = AnchoredText("t="+ tvals[i], prop=dict(size=11), frameon=False, loc='lower right')
                else:
                    at = AnchoredText("t="+ tvals[i], prop=dict(size=11), frameon=False, loc='lower left')
                axs[i,j].add_artist(at)
            axs[i,3].legend()
            axs[i,2].legend(loc="upper center")
        f.savefig(test_name + "_validation_for_paper.png")
    else:
        print("Name provided: ", case_study)
        print("Name not recognized")
        print("Available cases: bumpsteady, dambreak, dbstep, smallpert, staticbump, staticflat, staticparab, staticstep, thacker")