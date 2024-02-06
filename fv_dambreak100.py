import yaml, os, time
import numpy as np
from swe_utils.riemann_problems import RiemannProblem
from swe_utils.scales import Scales, nondimensionalize_param
from swe_utils.func_paths import create_paths
from swe_utils.fv_solver import create_fv_sol_with_z, create_ics_rp
from swe_utils.riemann_problems import create_valdata_from_analytical_and_z

# SETUP OF THE PROBLEM

test_name = "dambreak100"      # name of the test
model_name = "run_dambreak"    # name of the script with the solver
use_conserv_eqs = True          # flag for using the conservative form of SWEs (if False, the non-conservative form is used)
print_plots = True              # flag for printing validators/inferencers at the end of training 
print_collocation = True        # flag for plotting collocation points at the end of training
change_config_file = True       # flag for setting up the configuration file with the above function (otherwise the config file has to be prepared manually with the correct name!)
cwd = os.path.dirname(os.path.realpath(__file__))
# create paths to useful directories
sol_dir, plot_subdir, infer_dir, val_dir, constr_dir = create_paths(cwd,test_name,model_name)

# instantiate problem
prb = RiemannProblem("RP1")                                              # with predefined values (RP0-RP1)
#prb = RiemannProblem(dim_vals=[1.,0.,0.5,0.],L=12.,xdisc=0.,tmax=0.5)    # alternatively: with custom values
prb.print_recap()
# check on depth positivity (if necessary)
prb.check_drybed()

# params for validation, monitors, inferencers, etc.
# possible time and name of csv file (external) that can be read and used for validators --> put the files in the ref_sol folder!
# csv files must include three columns: x, h, u
# the first is for data to be read and used for validator, the second is only read in the final plots (i.e. it can be the solution from a different model shown for comparison)
dim_time_val = 1    # time at which we have validation data (profile along x axis)
filename_valdata = ""  #if empty, it is skipped in the final plots
filename_othersol = ""  #if empty, it is skipped in the final plots
# parameters for data obtained in functions here in swe_utils (analytic or finite volume)
dim_dt_infer = 0.2  # time interval to plot profiles as inferencers (optional)
nsteps = 5          # number of profiles to be extracted for validation
npts = 100          # number of points to discretize the profile of validators
x_p1 = 0.           # location for inferencing time series (optional)
# create paths to validation data
valdata_abspath = sol_dir + filename_valdata
# run fv solver to create the validation data at fixed time steps (dimensional) --> only for flat bottom!!
dx = prb.L / npts
start =time.time()
x_in, h_in, u_in, q_in, z_in = create_ics_rp(prb, dx)
x0, h_fv1, q_fv1, z0 = create_fv_sol_with_z(prb, dx, (nsteps-1), sol_dir + test_name, x_in, z_in)
end =time.time()

#write txt file with analytical solution with the same resolution of the FV one
xa, ta, ha, ua, qa, za = create_valdata_from_analytical_and_z(prb, dim_time_val, npts)
filename_analyt = sol_dir + test_name + "_analyt_" + f"{dim_time_val:.5f}" + ".txt"
A = np.stack((xa[:,0],ha[:,0],ua[:,0],qa[:,0]), axis=1)
np.savetxt(filename_analyt, A, fmt='%1.5f')

#compute errors
h_mae = np.mean(np.abs(ha-h_fv1))
h_rmse = np.sqrt(np.mean(np.square(ha-h_fv1)))
u_fv1 = np.divide(q_fv1,h_fv1)   #division by zero not considered
u_mae = np.mean(np.abs(ua-u_fv1))    
u_rmse = np.sqrt(np.mean(np.square(ua-u_fv1)))
print("RMSE h =", h_rmse, "- RMSE u =", u_rmse)

total_time = end - start
print("\n  TIME: "+ str(total_time))