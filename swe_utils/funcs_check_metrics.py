import numpy as np
import matplotlib.pyplot as plt

def compute_metrics_u_h(nd_params, val_dir, val_file = "val_data"):
    # plot validators
    data_2 = np.load(val_dir + val_file + ".npz", allow_pickle=True)
    data_2 = np.atleast_1d(data_2.f.arr_0)[0]
    # compute errors in validation data  (dimensional value!)
    h_obs = data_2["true_h"][:, 0] * nd_params.H0
    h_pred = data_2["pred_h"][:, 0] * nd_params.H0
    h_mae = np.mean(np.abs(h_obs-h_pred))
    h_rmse = np.sqrt(np.mean(np.square(h_obs-h_pred)))
    u_obs = data_2["true_u"][:, 0] * nd_params.U0
    u_pred = data_2["pred_u"][:, 0] * nd_params.U0
    u_mae = np.mean(np.abs(u_obs-u_pred))    
    u_rmse = np.sqrt(np.mean(np.square(u_obs-u_pred)))
    # specific discharge q=u*h (just to check)
    q_pred = np.multiply(h_pred, u_pred)
    q_obs = np.multiply(h_obs, u_obs)

    return h_mae, h_rmse, u_mae, u_rmse

def compute_metrics_u_h_z(nd_params, val_dir, val_file = "val_data"):
    # plot validators
    data_2 = np.load(val_dir + val_file + ".npz", allow_pickle=True)
    data_2 = np.atleast_1d(data_2.f.arr_0)[0]
    # compute errors in validation data  (dimensional value!)
    h_obs = data_2["true_h"][:, 0] * nd_params.H0
    h_pred = data_2["pred_h"][:, 0] * nd_params.H0
    h_mae = np.mean(np.abs(h_obs-h_pred))
    h_rmse = np.sqrt(np.mean(np.square(h_obs-h_pred)))
    u_obs = data_2["true_u"][:, 0] * nd_params.U0
    u_pred = data_2["pred_u"][:, 0] * nd_params.U0
    u_mae = np.mean(np.abs(u_obs-u_pred))    
    u_rmse = np.sqrt(np.mean(np.square(u_obs-u_pred)))
    z_obs = data_2["true_z"][:, 0] * nd_params.H0
    z_pred = data_2["pred_z"][:, 0] * nd_params.H0
    z_mae = np.mean(np.abs(z_obs-z_pred))    
    z_rmse = np.sqrt(np.mean(np.square(z_obs-z_pred)))

    return h_mae, h_rmse, u_mae, u_rmse, z_mae, z_rmse

def compute_validator_metrics(results_dict, test_name, rp, nd_params, val_dir, val_root = "val_t_", nsteps=5):
    step = (rp.tmax - rp.tstart) / (nsteps-1)
    t_val_loop = np.arange(rp.tstart, rp.tmax+step, step)
    for i in range(0, t_val_loop.size):    
        label = val_root + f"{t_val_loop[i]:.2f}"
        h_mae, h_rmse, u_mae, u_rmse, z_mae, z_rmse = compute_metrics_u_h_z(nd_params, val_dir, val_file = label)
        results_dict[label] = {}
        results_dict[label]["val_time"] = t_val_loop[i]
        results_dict[label]["h_rmse"] = h_rmse
        results_dict[label]["h_mae"] = h_mae
        results_dict[label]["u_rmse"] = u_rmse
        results_dict[label]["u_mae"] = u_mae
        results_dict[label]["z_rmse"] = z_rmse
        results_dict[label]["z_mae"] = z_mae
    return

def compute_validator_metrics_loop(results_dict, test_name, rp, nd_params, grid_search_param, val_dir, val_root = "val_t_", nsteps=5):
    step = (rp.tmax - rp.tstart) / (nsteps-1)
    t_val_loop = np.arange(rp.tstart, rp.tmax+step, step)
    neu = grid_search_param[0]
    lay = grid_search_param[1]
    for i in range(0, t_val_loop.size):    
        label = val_root + f"{t_val_loop[i]:.2f}"
        h_mae, h_rmse, u_mae, u_rmse, z_mae, z_rmse = compute_metrics_u_h_z(nd_params, val_dir, val_file = label)
        results_dict[label] = {}
        results_dict[label]["val_time"] = t_val_loop[i]
        results_dict[label]["neurons"] = neu
        results_dict[label]["layers"] = lay
        results_dict[label]["h_rmse"] = h_rmse
        results_dict[label]["h_mae"] = h_mae
        results_dict[label]["u_rmse"] = u_rmse
        results_dict[label]["u_mae"] = u_mae
        results_dict[label]["z_rmse"] = z_rmse
        results_dict[label]["z_mae"] = z_mae
    return

def compute_validator_metrics_loop_pts(results_dict, test_name, rp, nd_params, grid_search_param, val_dir, val_root = "val_t_", nsteps=5):
    step = (rp.tmax - rp.tstart) / (nsteps-1)
    t_val_loop = np.arange(rp.tstart, rp.tmax+step, step)
    pts = grid_search_param[0]
    icr = grid_search_param[1]
    bcr = grid_search_param[2]
    for i in range(0, t_val_loop.size):    
        label = val_root + f"{t_val_loop[i]:.2f}"
        h_mae, h_rmse, u_mae, u_rmse, z_mae, z_rmse = compute_metrics_u_h_z(nd_params, val_dir, val_file = label)
        results_dict[label] = {}
        results_dict[label]["val_time"] = t_val_loop[i]
        results_dict[label]["int_pts"] = pts
        results_dict[label]["ic_ratio"] = icr
        results_dict[label]["bc_ratio"] = bcr
        results_dict[label]["h_rmse"] = h_rmse
        results_dict[label]["h_mae"] = h_mae
        results_dict[label]["u_rmse"] = u_rmse
        results_dict[label]["u_mae"] = u_mae
        results_dict[label]["z_rmse"] = z_rmse
        results_dict[label]["z_mae"] = z_mae
    return

def compute_validator_metrics_loop_weights(results_dict, test_name, rp, nd_params, grid_search_param, val_dir, val_root = "val_t_", nsteps=5):
    step = (rp.tmax - rp.tstart) / (nsteps-1)
    t_val_loop = np.arange(rp.tstart, rp.tmax+step, step)
    wpde = grid_search_param[0]
    wic = grid_search_param[1]
    wbc = grid_search_param[2]
    temp_loss = grid_search_param[3]
    for i in range(0, t_val_loop.size):    
        label = val_root + f"{t_val_loop[i]:.2f}"
        h_mae, h_rmse, u_mae, u_rmse, z_mae, z_rmse = compute_metrics_u_h_z(nd_params, val_dir, val_file = label)
        results_dict[label] = {}
        results_dict[label]["val_time"] = t_val_loop[i]
        results_dict[label]["wei_pde"] = wpde
        results_dict[label]["wei_ic"] = wic
        results_dict[label]["wei_bc"] = wbc
        results_dict[label]["temp_loss"] = temp_loss
        results_dict[label]["h_rmse"] = h_rmse
        results_dict[label]["h_mae"] = h_mae
        results_dict[label]["u_rmse"] = u_rmse
        results_dict[label]["u_mae"] = u_mae
        results_dict[label]["z_rmse"] = z_rmse
        results_dict[label]["z_mae"] = z_mae
    return
                 
import pandas as pd

# function copied from: https://stackoverflow.com/questions/13575090/construct-pandas-dataframe-from-items-in-nested-dictionary
def flatten_dict(nested_dict):
    res = {}
    if isinstance(nested_dict, dict):
        for k in nested_dict:
            flattened_dict = flatten_dict(nested_dict[k])
            for key, val in flattened_dict.items():
                key = list(key)
                key.insert(0, k)
                res[tuple(key)] = val
    else:
        res[()] = nested_dict
    return res

# function copied from: https://stackoverflow.com/questions/13575090/construct-pandas-dataframe-from-items-in-nested-dictionary
def nested_dict_to_df(values_dict):
    flat_dict = flatten_dict(values_dict)
    df = pd.DataFrame.from_dict(flat_dict, orient="index")
    df.index = pd.MultiIndex.from_tuples(df.index)
    df = df.unstack(level=-1)
    df.columns = df.columns.map("{0[1]}".format)
    return df

def add_legend_labels_scatter(vartypex="x", vartypey="var"):
    plt.legend()
    plt.xlabel(vartypex, fontsize=14)
    plt.ylabel(vartypey, fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    return

def plot_metrics_from_dict_1run(results_dict, test_name, metric_type, rp, plot_subdir, nsteps=5, fig_id=1):
    # turn dictionary into dataframe
    df = nested_dict_to_df(results_dict)
    df.to_csv(plot_subdir + "/" + test_name + '_metrics_results.csv', encoding='utf-8')
    if(fig_id>0):
        # extract data for each timestep of validation
        step = (rp.tmax - rp.tstart) / (nsteps-1)
        t_val_loop = np.arange(rp.tstart, rp.tmax+step, step)
        # create plot with rmse
        plt.figure(fig_id)        
        plt.scatter(df["val_time"], df[metric_type], label=metric_type)
        add_legend_labels_scatter("val_time",metric_type)
        plt.title(metric_type)
        plt.savefig(plot_subdir + "/" + test_name + "_" + metric_type + ".png" )
        plt.close()
    return

def plot_metrics_from_dict(results_dict, test_name, metric_type, neuron_range, layer_range, rp, plot_subdir, nsteps=5, fig_id=1):
    # turn dictionary into dataframe
    df = nested_dict_to_df(results_dict)
    df.to_csv(plot_subdir + "/" + test_name + '_grid_search_results.csv', encoding='utf-8')
    if(fig_id>0):
        # extract data for each timestep of validation
        step = (rp.tmax - rp.tstart) / (nsteps-1)
        t_val_loop = np.arange(rp.tstart, rp.tmax+step, step)
        #id=fig_id
        for i in range(0, t_val_loop.size):   
            aa = df[df["val_time"] == t_val_loop[i]]
            # create plot with rmse
            plt.figure()  #(id)        
            for lay in layer_range:
                bb = aa[aa["layers"] == lay]
                plt.scatter(bb["neurons"], bb[metric_type], label=str(lay)+"layers")
            add_legend_labels_scatter("neurons",metric_type)
            plt.title("val time = " + str(t_val_loop[i]))
            plt.savefig(plot_subdir + "/" + test_name + "_grid_search_" + metric_type + "_t_" + str(t_val_loop[i]) + ".png" )
            plt.close()
            #id=id+1
    return

def plot_metrics_from_dict_pts(results_dict, test_name, metric_type, icr_range, bcr_range, rp, plot_subdir, nsteps=5, fig_id=1):
    # turn dictionary into dataframe
    df = nested_dict_to_df(results_dict)
    df.to_csv(plot_subdir + "/" + test_name + '_grid_search_results.csv', encoding='utf-8')
    if(fig_id>0):
        # extract data for each timestep of validation
        step = (rp.tmax - rp.tstart) / (nsteps-1)
        t_val_loop = np.arange(rp.tstart, rp.tmax+step, step)
        #id=fig_id
        for i in range(0, t_val_loop.size):   
            aa = df[df["val_time"] == t_val_loop[i]]
            # create plot with rmse
            plt.figure() #(id)        
            for icr in icr_range:
                bb = aa[aa["ic_ratio"] == icr]
                for bcr in bcr_range:
                    cc = bb[bb["bc_ratio"] == bcr]
                    plt.scatter(cc["int_pts"], cc[metric_type], label=str(icr)+"icr_"+str(bcr)+"bcr")
            add_legend_labels_scatter("int_pts",metric_type)
            plt.title("val time = " + str(t_val_loop[i]))
            plt.savefig(plot_subdir + "/" + test_name + "_grid_search_" + metric_type + "_t_" + str(t_val_loop[i]) + ".png" )
            plt.close()
            #id=id+1
    return

def plot_metrics_from_dict_weights(results_dict, test_name, metric_type, wic_range, wbc_range, rp, plot_subdir, nsteps=5, fig_id=1):
    # turn dictionary into dataframe
    df = nested_dict_to_df(results_dict)
    df.to_csv(plot_subdir + "/" + test_name + '_grid_search_results.csv', encoding='utf-8')
    if(fig_id>0):
        # extract data for each timestep of validation
        step = (rp.tmax - rp.tstart) / (nsteps-1)
        t_val_loop = np.arange(rp.tstart, rp.tmax+step, step)
        #id=fig_id
        for i in range(0, t_val_loop.size):   
            aa = df[df["val_time"] == t_val_loop[i]]
            # create plot with rmse
            plt.figure()  #(id)        
            for wic in wic_range:
                bb = aa[aa["wei_ic"] == wic]
                for wbc in wbc_range:
                    cc = bb[bb["wei_bc"] == wbc]
                    plt.scatter(cc["wei_pde"], cc[metric_type], label=str(wic)+"wic_"+str(wbc)+"wbc")
            add_legend_labels_scatter("wei_pde",metric_type)
            plt.title("val time = " + str(t_val_loop[i]))
            plt.savefig(plot_subdir + "/" + test_name + "_grid_search_" + metric_type + "_t_" + str(t_val_loop[i]) + ".png" )
            plt.close()
            #id=id+1
    return

