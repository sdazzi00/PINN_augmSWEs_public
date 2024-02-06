import os

def create_paths(cwd,test_name,model_name):

    network_dir = "./outputs/" + model_name + "/" + test_name + "/"
    infer_dir = network_dir + "inferencers/"
    val_dir = network_dir + "validators/"
    constr_dir = network_dir + "constraints/"

    sol_dir = os.path.join(cwd, "ref_sol/")
    try:
        os.mkdir(sol_dir)
    except FileExistsError:
        pass
    plot_dir = os.path.join(cwd, "plots/")
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)
    plot_subdir = os.path.join(plot_dir, test_name)
    if not os.path.exists(plot_subdir):
        os.mkdir(plot_subdir)

    return sol_dir, plot_subdir, infer_dir, val_dir, constr_dir

