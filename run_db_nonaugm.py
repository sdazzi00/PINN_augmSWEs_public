import numpy as np
from sympy import Symbol, Eq, Piecewise
import time
import torch

import modulus
from modulus.hydra import ModulusConfig, instantiate_arch
from modulus.solver import Solver
from modulus.domain import Domain
from modulus.geometry.primitives_1d import Line1D
from modulus.geometry import Parameterization
from modulus.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,    
)
from modulus.domain.validator import PointwiseValidator
from modulus.domain.inferencer import PointwiseInferencer
from modulus.domain.monitor import PointwiseMonitor
from swe_utils.plotters import CustomValidatorPlotter

from modulus.key import Key
from modulus.node import Node

from swe_utils.swe_pde import ShallowWater1D_adim
from swe_utils.riemann_problems import read_val_profile_from_csv_and_nondim, create_val_profile_bottom_flat, create_valdata_from_analytical_and_z

# import script with function to setup the Riemann problem
from setup_db_nonaugm import *

@modulus.main(config_path="conf", config_name="config_" + test_name)
def run(cfg: ModulusConfig) -> None:

    # nodes
    sw=ShallowWater1D_adim(Sr=nd_params.Str, Fr0_2=nd_params.Fr0_sq, conserv=use_conserv_eqs)
    sw_net = instantiate_arch(
        input_keys=[Key("x"), Key("t")],
        output_keys=[Key("h"), Key("u")],
        cfg=cfg.arch.fully_connected,
    )
    nodes = sw.make_nodes() + [
        sw_net.make_node(name="swe_network", jit=cfg.jit)
    ]

    # params for domain
    geom = Line1D(prb.x_start, prb.x_end)

    t_symbol = Symbol("t")
    x = Symbol("x")
    time_range = {t_symbol: (prb.tstart, prb.tmax)}

    # make domain
    domain = Domain()

    # pre-define values/functions for IC and BC
    bed_elev = 0
    hin = Piecewise((prb.values[0], x < prb.xdisc),(prb.values[2], True))
    uin = Piecewise((prb.values[1], x < prb.xdisc),(prb.values[3], True))
    z_bound_L = 0     
    z_bound_R = 0
    h_bound_L = prb.values[0]
    h_bound_R = prb.values[2]
    u_bound_L = prb.values[1]
    u_bound_R = prb.values[3]

    # add constraints
    # initial conditions
    IC = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geom,
        outvar={"h": hin, 
                "u": uin 
                },
        batch_size=cfg.batch_size.IC,
        parameterization=Parameterization({t_symbol: prb.tstart}),
        lambda_weighting={"h": cfg.custom.weights.ic, 
                          "u": cfg.custom.weights.ic
                          },
    )
    domain.add_constraint(IC, name="IC")

    # boundary conditions
    # left
    BC_L = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geom,
        outvar={"h": h_bound_L, "u": u_bound_L },
        batch_size=cfg.batch_size.BC,
        criteria=Eq(x, prb.x_start),
        parameterization=time_range,
        lambda_weighting={"h": cfg.custom.weights.bc_coef1*(1-t_symbol/prb.tmax) + cfg.custom.weights.bc_coef2, 
                          "u": cfg.custom.weights.bc_coef1*(1-t_symbol/prb.tmax) + cfg.custom.weights.bc_coef2
                          },
    )
    domain.add_constraint(BC_L, "BC_L")
    # right
    BC_R = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geom,
        outvar={"h": h_bound_R, "u": u_bound_R},
        batch_size=cfg.batch_size.BC,
        criteria=Eq(x, prb.x_end),
        parameterization=time_range,
        lambda_weighting={"h": cfg.custom.weights.bc_coef1*(1-t_symbol/prb.tmax) + cfg.custom.weights.bc_coef2, 
                          "u": cfg.custom.weights.bc_coef1*(1-t_symbol/prb.tmax) + cfg.custom.weights.bc_coef2
                          },
    )
    domain.add_constraint(BC_R, "BC_R")

    # interior
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geom,
        outvar={"mass": 0, "momentum": 0, "depthpos": 0, "uzero_if_hzero": 0, "uhzero_if_hzero": 0},
        batch_size=cfg.batch_size.interior,
        parameterization=time_range,
        lambda_weighting={
            "mass": cfg.custom.weights.mass_coef1*(1-t_symbol/prb.tmax) + cfg.custom.weights.mass_coef2, 
            "momentum": cfg.custom.weights.mom_coef1*(1-t_symbol/prb.tmax) + cfg.custom.weights.mom_coef2,            
            "depthpos": cfg.custom.weights.dep_pos,
            "uzero_if_hzero": cfg.custom.weights.uh_zero,
	        "uhzero_if_hzero": cfg.custom.weights.uh_zero 
            },
    )
    domain.add_constraint(interior, "interior")
    
    # validator (analytical solution sampled at fixed times)
    plotter = CustomValidatorPlotter()

    # validator loop using the in-house functions for exact solution of RPs
    step = (prb.tmax - prb.tstart) / (nsteps-1)
    t_val_loop = np.arange(prb.tstart, prb.tmax+step, step)
    for i in range(0, t_val_loop.size):
        ts_val = t_val_loop[i] 
        xa, ta, ha, ua, qa, za = create_valdata_from_analytical_and_z(prb, ts_val, npts)
        label = f"val_t_{t_val_loop[i]:.2f}"       
        val_loop = PointwiseValidator(
            nodes=nodes,
            invar={"x": xa, "t": ta},
            true_outvar={"h": ha, "u": ua},
            batch_size=1000,
            plotter = plotter,
        )
        domain.add_validator(val_loop, label)

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":

    start = time.time()
    run()
    end = time.time()

    # write metrics to dictionary
    from swe_utils.funcs_check_metrics import compute_validator_metrics, plot_metrics_from_dict_1run
    results_dict = {}
    #compute_validator_metrics(results_dict, test_name, prb, nd_params, val_dir, val_root = "val_t_", nsteps=nsteps)
    # make plots
    if print_plots:
        from swe_utils.funcs_plot import *
        plot_validator_profile_loop_panels(test_name, prb, nd_params, val_dir, plot_subdir, othersol="", val_root = "val_t_", nsteps=nsteps, fig_id=9, zplot=False)
        plot_validator_profile_loop(test_name, prb, nd_params, val_dir, plot_subdir, othersol="", val_root = "val_t_", nsteps=nsteps, fig_id=9+nsteps, zplot=False)
        if not (filename_othersol == ""):    
            othersol = sol_dir + filename_othersol
        else:
            othersol = filename_othersol
        #plot_validator_profile_1(test_name, prb, nd_params, val_dir, plot_subdir, othersol, val_file = "val_data_csv", savefig=True, label_id="", fig_id=2)
        #plot_inferencer_loop(test_name, prb, nd_params, dt_infer, infer_dir, plot_subdir, fig_id=1)
        #plot_inferencer_time_series(test_name, prb, nd_params, infer_dir, plot_subdir, fig_id=20)
    if print_collocation:
        from swe_utils.plot_colloc_points import plot_colloc_pts
        plot_colloc_pts(test_name, constr_dir, plot_subdir)

    # read and plot result metrics from dictionary + write to csv 
    #plot_metrics_from_dict_1run(results_dict, test_name, "h_rmse", prb, plot_subdir, nsteps=nsteps, fig_id=1000)
    #plot_metrics_from_dict_1run(results_dict, test_name, "u_rmse", prb, plot_subdir, nsteps=nsteps, fig_id=1000+nsteps+2)
    #plot_metrics_from_dict_1run(results_dict, test_name, "z_rmse", prb, plot_subdir, nsteps=nsteps, fig_id=1000+2*nsteps+2)
    
    total_time = end - start
    print("\n  TRAINING TIME: "+ str(total_time))