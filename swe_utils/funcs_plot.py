import numpy as np
import matplotlib.pyplot as plt

# FUNCTIONS FOR PLOTTING VALIDATOR DATA and/or INFERENCER DATA

def add_legend_labels(vartype="var"):
    plt.legend()
    plt.xlabel("x", fontsize=14)
    plt.ylabel(vartype, fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    return

def plot_inferencer_loop(test_name, rp, nd_params, dt_infer, infer_dir, plot_subdir, fig_id=1):
    # plot inferencers (at fixed time)
    plt.figure(fig_id, figsize=(9,5), dpi=150)
    plt.suptitle("Inferencers")
    plt.subplot(1,2,1)
    plt.subplot(1,2,2)
    t_infer = np.arange(rp.tstart, rp.tmax+dt_infer, dt_infer)
    for i in range(0, t_infer.size):
        filename = f"inf_t_{t_infer[i]:.2f}.npz" 
        data_loop = np.load(infer_dir + filename, allow_pickle=True)
        data_loop = np.atleast_1d(data_loop.f.arr_0)[0]
        # re-scale to dimensional values and plot
        plt.subplot(1,2,1)
        plt.plot(data_loop["x"][:, 0] * nd_params.L0 + rp.xdisc, data_loop["h"][:, 0] * nd_params.H0, "--", label=f"h_inf_t_{t_infer[i]:.2f}")
        plt.subplot(1,2,2)
        plt.plot(data_loop["x"][:, 0] * nd_params.L0 + rp.xdisc, data_loop["u"][:, 0] * nd_params.U0, "--", label=f"u_inf_t_{t_infer[i]:.2f}")

    plt.subplot(1,2,1)
    add_legend_labels("h")
    plt.subplot(1,2,2)
    add_legend_labels("u")
    plt.savefig(plot_subdir + "/" + test_name + "_image_infer_loop")
    return

def plot_validator_profile_1(test_name, rp, nd_params, val_dir, plot_subdir, othersol = "", val_file = "val_data", savefig=True, label_id="", fig_id=1, zplot=False):
    # plot validators
    data_2 = np.load(val_dir + val_file + ".npz", allow_pickle=True)
    data_2 = np.atleast_1d(data_2.f.arr_0)[0]
    # re-scale to dimensional values and plot
    plt.figure(fig_id)
    plt.plot(data_2["x"][:, 0] * nd_params.L0 + rp.xdisc, data_2["pred_h"][:, 0] * nd_params.H0, "--", label="h_pred"+label_id)
    plt.plot(data_2["x"][:, 0] * nd_params.L0 + rp.xdisc, data_2["true_h"][:, 0] * nd_params.H0, "--", label="h_true"+label_id)
    plt.figure(fig_id+1)
    plt.plot(data_2["x"][:, 0] * nd_params.L0 + rp.xdisc, data_2["pred_u"][:, 0] * nd_params.U0, "--", label="u_pred"+label_id)
    plt.plot(data_2["x"][:, 0] * nd_params.L0 + rp.xdisc, data_2["true_u"][:, 0] * nd_params.U0, "--", label="u_true"+label_id)
    if(zplot):
        plt.figure(fig_id+3)
        plt.plot(data_2["x"][:, 0] * nd_params.L0 + rp.xdisc, data_2["pred_z"][:, 0] * nd_params.H0, "--", label="z_pred"+label_id)
        plt.plot(data_2["x"][:, 0] * nd_params.L0 + rp.xdisc, data_2["true_z"][:, 0] * nd_params.H0, "--", label="z_true"+label_id)
        # compute errors in validation data and print rmse
        z_obs = data_2["true_z"][:, 0] * nd_params.H0
        z_pred = data_2["pred_z"][:, 0] * nd_params.H0
        z_mae = np.mean(np.abs(z_obs-z_pred))    
        z_rmse = np.sqrt(np.mean(np.square(z_obs-z_pred)))
        print(label_id, "z_mae = ", z_mae, "z_rmse = ", z_rmse)
        if (savefig):
            add_legend_labels("z")
            plt.title("z_rmse(m) = " + str(z_rmse), fontsize=18)
            plt.savefig(plot_subdir+ "/" + test_name + "_image_z_" + val_file + ".png")

    # compute errors in validation data and print rmse as plot.title (dimensional value!)
    h_obs = data_2["true_h"][:, 0] * nd_params.H0
    h_pred = data_2["pred_h"][:, 0] * nd_params.H0
    h_mae = np.mean(np.abs(h_obs-h_pred))
    h_rmse = np.sqrt(np.mean(np.square(h_obs-h_pred)))
    print(label_id, "h_mae = ", h_mae, "h_rmse = ", h_rmse)     
    u_obs = data_2["true_u"][:, 0] * nd_params.U0
    u_pred = data_2["pred_u"][:, 0] * nd_params.U0
    u_mae = np.mean(np.abs(u_obs-u_pred))    
    u_rmse = np.sqrt(np.mean(np.square(u_obs-u_pred)))
    print(label_id, "u_mae = ", u_mae, "u_rmse = ", u_rmse)   

    # if available, compare with solution provided by an external model
    other_label = ""
    if not (othersol == ""):    
        comparison_sol = np.loadtxt(othersol, delimiter=",")
        # data in the csv file are already dimensional
        plt.figure(fig_id)
        plt.plot(comparison_sol[:,0], comparison_sol[:,1], "--", label="h_comparison")
        plt.figure(fig_id+1)
        plt.plot(comparison_sol[:,0], comparison_sol[:,2], "--", label="u_comparison")
        other_label = "_rus"

    plt.figure(fig_id)    
    if (savefig):
        add_legend_labels("h")
        plt.title("h_rmse(m) = " + str(h_rmse), fontsize=18)
        plt.savefig(plot_subdir + "/" + test_name + "_image_h_" + val_file + other_label + ".png")
    plt.figure(fig_id+1)
    if (savefig):
        add_legend_labels("u")
        plt.title("u_rmse(m/s) = " + str(u_rmse), fontsize=18)
        plt.savefig(plot_subdir + "/" + test_name + "_image_u_" + val_file + other_label + ".png")

    # plot specific discharge q=u*h (just to check)
    q_pred = np.multiply(h_pred, u_pred)
    q_obs = np.multiply(h_obs, u_obs)
    plt.figure(fig_id+2)
    plt.plot(data_2["x"][:, 0] * nd_params.L0 + rp.xdisc, q_obs, "--", label="q_true"+label_id)
    plt.plot(data_2["x"][:, 0] * nd_params.L0 + rp.xdisc, q_pred, "--", label="q_pred"+label_id)
    if (savefig):
        add_legend_labels("q")
        plt.savefig(plot_subdir+ "/" + test_name + "_image_q_" + val_file + other_label + ".png")
    return

def plot_validator_profile_loop(test_name, rp, nd_params, val_dir, plot_subdir, othersol = "", val_root = "val_t_", nsteps=5, fig_id=1, zplot=False):
    step = (rp.tmax - rp.tstart) / (nsteps-1)
    t_val_loop = np.arange(rp.tstart, rp.tmax+step, step)
    for i in range(0, t_val_loop.size):
        ts_val = t_val_loop[i]       
        label = val_root + f"{t_val_loop[i]:.2f}"
        plot_validator_profile_1(test_name, rp, nd_params, val_dir, plot_subdir, othersol, val_file = label, savefig=False, label_id=f"{t_val_loop[i]:.2f}", fig_id=fig_id, zplot=zplot)
    plt.figure(fig_id)    
    add_legend_labels("h")
    plt.savefig(plot_subdir + "/" + test_name + "_image_h_valid_exact_loop"+ ".png")
    plt.figure(fig_id+1)
    add_legend_labels("u")
    plt.savefig(plot_subdir + "/" + test_name + "_image_u_valid_exact_loop"+ ".png") 
    plt.figure(fig_id+2)
    add_legend_labels("q")
    plt.savefig(plot_subdir + "/" + test_name + "_image_q_valid_exact_loop" + ".png") 
    if(zplot):      
        plt.figure(fig_id+3)
        add_legend_labels("z")
        plt.savefig(plot_subdir + "/" + test_name + "_image_z_valid_exact_loop" + ".png") 
    return

def plot_validator_profile_1_panels(test_name, rp, nd_params, val_dir, plot_subdir, othersol = "", val_file = "val_data", savefig=True, label_id="", fig_id=1, zplot=False):
    # plot validators
    data_2 = np.load(val_dir + val_file + ".npz", allow_pickle=True)
    data_2 = np.atleast_1d(data_2.f.arr_0)[0]
    if(zplot):
        Nplots=5
    else:
        Nplots=3
    # re-scale to dimensional values and plot
    plt.figure(fig_id,figsize=(3*Nplots,4), dpi=100)
    plt.suptitle("SWE problem: PINN vs true solution at time " + label_id)
    plt.subplot(1,Nplots,1)
    plt.title("Solution (h)")
    plt.plot(data_2["x"][:, 0] * nd_params.L0 + rp.xdisc, data_2["pred_h"][:, 0] * nd_params.H0, "--", label="h_pred")
    plt.plot(data_2["x"][:, 0] * nd_params.L0 + rp.xdisc, data_2["true_h"][:, 0] * nd_params.H0, "--", label="h_true")
    add_legend_labels("h")
    plt.subplot(1,Nplots,2)
    plt.title("Solution (u)")
    plt.plot(data_2["x"][:, 0] * nd_params.L0 + rp.xdisc, data_2["pred_u"][:, 0] * nd_params.U0, "--", label="u_pred")
    plt.plot(data_2["x"][:, 0] * nd_params.L0 + rp.xdisc, data_2["true_u"][:, 0] * nd_params.U0, "--", label="u_true")
    add_legend_labels("u")
    plt.subplot(1,Nplots,3)
    plt.title("Solution (q)")
    q_pred = np.multiply(data_2["pred_h"][:, 0] * nd_params.H0, data_2["pred_u"][:, 0] * nd_params.U0)
    q_true = np.multiply(data_2["true_h"][:, 0] * nd_params.H0, data_2["true_u"][:, 0] * nd_params.U0)
    plt.plot(data_2["x"][:, 0] * nd_params.L0 + rp.xdisc, q_pred, "--", label="q_pred")
    plt.plot(data_2["x"][:, 0] * nd_params.L0 + rp.xdisc, q_true, "--", label="q_true")
    add_legend_labels("q")
    if(zplot):
        plt.subplot(1,Nplots,4)
        plt.title("Solution (z)")
        plt.plot(data_2["x"][:, 0] * nd_params.L0 + rp.xdisc, data_2["pred_z"][:, 0] * nd_params.H0, "--", label="z_pred")
        plt.plot(data_2["x"][:, 0] * nd_params.L0 + rp.xdisc, data_2["true_z"][:, 0] * nd_params.H0, "--", label="z_true")
        add_legend_labels("z")
        plt.subplot(1,Nplots,5)
        plt.title("Solution (wse)")
        plt.plot(data_2["x"][:, 0] * nd_params.L0 + rp.xdisc,data_2["true_z"][:, 0] * nd_params.H0, "-", label="bottom_true")
        plt.plot(data_2["x"][:, 0] * nd_params.L0 + rp.xdisc,data_2["true_z"][:, 0] * nd_params.H0 + data_2["pred_h"][:, 0] * nd_params.H0, "--", label="wse_pred")
        plt.plot(data_2["x"][:, 0] * nd_params.L0 + rp.xdisc,data_2["true_z"][:, 0] * nd_params.H0 + data_2["true_h"][:, 0] * nd_params.H0, "--", label="wse_true") 
        add_legend_labels("wse")
    plt.tight_layout()
    #save figure
    if (savefig):
        plt.savefig(plot_subdir+ "/" + test_name + "_panels_" + val_file + ".png")
    return

def plot_validator_profile_loop_panels(test_name, rp, nd_params, val_dir, plot_subdir, othersol = "", val_root = "val_t_", nsteps=5, fig_id=1, zplot=False):
    step = (rp.tmax - rp.tstart) / (nsteps-1)
    t_val_loop = np.arange(rp.tstart, rp.tmax+step, step)
    for i in range(0, t_val_loop.size):
        ts_val = t_val_loop[i]       
        label = val_root + f"{t_val_loop[i]:.2f}"
        plot_validator_profile_1_panels(test_name, rp, nd_params, val_dir, plot_subdir, othersol, val_file = label, savefig=True, label_id=f"{t_val_loop[i]:.2f}", fig_id=fig_id, zplot=zplot)
        fig_id=fig_id+1
    return

def plot_inferencer_time_series(test_name, rp, nd_params, infer_dir, plot_subdir, fig_id=1):
    # plot inferencers (time series at fixed point) after scaling back to dimensional values
    plt.figure(fig_id)
    data_t = np.load(infer_dir + "inf_point1.npz", allow_pickle=True)
    data_t = np.atleast_1d(data_t.f.arr_0)[0]
    plt.plot(data_t["t"][:, 0] * nd_params.T0, data_t["h"][:, 0] * nd_params.H0, "--", label="h_xP1")
    plt.plot(data_t["t"][:, 0] * nd_params.T0, data_t["u"][:, 0] * nd_params.U0, "--", label="u_xP1")
    plt.legend()
    plt.savefig(plot_subdir+ "/" + test_name + "_image_infer_timeseries")
    return

