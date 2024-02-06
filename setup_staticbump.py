import yaml, os
from swe_utils.bump_problems import Bump
from swe_utils.scales import Scales, nondimensionalize_param
from swe_utils.func_paths import create_paths

# SETUP OF THE CONFIGURATION FILE

# configuration parameters can be specified directly in this function
def modify_config_file(basefile, name_of_test):
    # open the base configuration file
    with open(basefile) as f:
        doc = yaml.safe_load(f)
    # create a case-specific config file: set the network_dir param equal to the test_name
    doc["network_dir"] = name_of_test
    # set weights of loss function
    doc["custom"]["weights"]["ic"] = 10
    doc["custom"]["weights"]["bc_coef1"] = 0
    doc["custom"]["weights"]["bc_coef2"] = 1
    doc["custom"]["weights"]["mass_coef1"] = 0
    doc["custom"]["weights"]["mass_coef2"] = 1
    doc["custom"]["weights"]["mom_coef1"] = 0
    doc["custom"]["weights"]["mom_coef2"] = 1
    doc["custom"]["weights"]["dep_pos"] = 10
    doc["custom"]["weights"]["uh_zero"] = 10
    doc["custom"]["weights"]["zbed"] = 1000
    # set params for importance sampling
    doc["custom"]["sampling"]["par1"] = 10
    doc["custom"]["sampling"]["par2"] = 1
    # modify other configuration parameters compared to the "base" file, for example:  
    doc["training"]["max_steps"] = 30000     # example to modify value of existing param
    doc["training"]["rec_results_freq"] = 2000 
    doc["batch_size"]["IC"] = 200    
    doc["batch_size"]["BC"] = 100 
    doc["batch_size"]["interior"] = 1000
    doc["optimizer"]["lr"] = 0.001
    doc["scheduler"]["decay_steps"] = 200
    doc["arch"]["fully_connected"]["layer_size"] = 300
    doc["arch"]["fully_connected"]["nr_layers"] = 7
    doc["arch"]["fully_connected"]["activation_fn"] = "tanh"
    # possibly useful = change config to avoid training (only evaluation of pre-trained model)
    ## doc["run_mode"] = "eval"                 
    # new configuration parameters can be added, e.g.:
    ## doc["batch_size"]["new_batch"] = 400     # example to add new param with its value
    # add checkpoint source directory if available
    #doc["initialization_network_dir"] = "../../checkpoint_transfer/"
    # save the updated config file that will be provided to modulus 
    filename_config_new = "conf/config_" + name_of_test + ".yaml"
    with open(filename_config_new, 'w') as out:
        yaml.dump(doc, out, sort_keys=False, default_flow_style=False)

# SETUP OF THE PROBLEM

test_name = "staticbump"      # name of the test
model_name = "run_staticbump"    # name of the script with the solver
use_conserv_eqs = True          # flag for using the conservative form of SWEs (if False, the non-conservative form is used)
print_plots = True              # flag for printing validators/inferencers at the end of training 
print_collocation = False        # flag for plotting collocation points at the end of training
change_config_file = True       # flag for setting up the configuration file with the above function (otherwise the config file has to be prepared manually with the correct name!)
cwd = os.path.dirname(os.path.realpath(__file__))
# create paths to useful directories
sol_dir, plot_subdir, infer_dir, val_dir, constr_dir = create_paths(cwd,test_name,model_name)

# instantiate problem
prb = Bump("B0")                                              # with predefined values (B0-B1)
#prb = Bump(dim_vals=[0.2,0.05,2.,2.,4.42],L=25.,xdisc=10.,tmax=100.)    # alternatively: with custom values
prb.print_recap()
## check on depth positivity (if necessary)
#prb.check_drybed()

# params for validation, monitors, inferencers, etc.
# possible time and name of csv file (external) that can be read and used for validators --> put the files in the ref_sol folder!
# csv files must include three columns: x, h, u
# the first is for data to be read and used for validator, the second is only read in the final plots (i.e. it can be the solution from a different model shown for comparison)
dim_time_val = 1    # time at which we have validation data (profile along x axis)
filename_valdata = ""  #if empty, it is skipped in the final plots
filename_othersol = ""  #if empty, it is skipped in the final plots
# parameters for data obtained in functions here in swe_utils (analytic or finite volume)
dim_dt_infer = 0.2  # time interval to plot profiles as inferencers 
nsteps = 3          # number of profiles to be extracted for validation
npts = 1000          # number of points to discretize the profile of validators
x_p1 = 0.           # location for inferencing time series
# create paths to validation data
valdata_abspath = sol_dir + filename_valdata
# run fv solver to create the validation data at fixed time steps (dimensional) --> only for flat bottom!!
dx = prb.L / npts 

# define non_dimensional parameters for scaling (L0, H0, U0, T0)
nd_params = Scales([1.,1.,1.,1.], nd_par_hypothesis = False, g=9.81)   # if True: assuming Str=1 and nd_g=dim_g to compute U0 and T0
nd_params.print_adim_numbers()
# apply scaling to all values
prb.nondimensionalize_values(nd_params.H0, nd_params.U0, nd_params.L0, nd_params.T0)
prb.nondimensionalize_domain(nd_params.L0, nd_params.T0)
time_val = nondimensionalize_param(dim_time_val,nd_params.T0)
dt_infer = nondimensionalize_param(dim_dt_infer,nd_params.T0)
x_p1 = nondimensionalize_param(x_p1,nd_params.H0) - prb.xdisc

# configuration file
if (change_config_file):
    # start from a "base" config file containing mostly default values (add parameters to this file if necessary)
    filename_config_base = "conf/config_base2.yaml"
    # call the function to update the config file
    modify_config_file(filename_config_base, test_name)
else:
    # the file is already available: it will be read directly from the "run" script
    filename_config_base = "conf/config_" + test_name + ".yaml"
