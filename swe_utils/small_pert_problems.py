from math import sqrt 
import numpy as np

# CLASS WITH SMALL_PERTURBATION PROBLEM DATA

class SmallPert:
    #name = "SmallPert"

    def __init__(self, code="custom",dim_vals=[0.8,0.2,0.4,0.],L=12.,xdisc=0.,tmax=1.):
        self.code = code
        self.tstart = 0.
        self.type = "smallpert"
        
        if(self.code=="custom" or self.code==""):  # user-defined values for RP            
            self.values = dim_vals      # initial values hL, uL, hR, uR (h=depth, u=velocity)
            self.L = L                  # size of the spatial domain 
            self.xdisc = xdisc          # position of the initial discontinuity
            self.tmax = tmax            # size of the temporal domain  
        else:    # values from pre-defined SP (1)
            self.values = [1.,0.,1.,0.]   
            self.tmax = 1.
            if(self.code=="SP0"):       # static case
                self.L = 12.
                self.xdisc = 0.
            elif(self.code=="SP1"):     # h_init = h_base + wave_height * exp(-x**2 / param)     and     u_init = u_base
                self.L = 12.
                self.xdisc = 0.
                self.values[0] = 0.8    #h_base
                self.values[1] = 0.2    #wave_height
                self.values[2] = 0.4    #param in exponential 
                self.values[3] = 0.0    #u_base              
            else:
                raise ValueError ("Code not recognized: only \"SP0\" to \"SP1\" available. Provide \"custom\" and user-defined values as arguments")
       
        self.x_start = self.xdisc - self.L/2
        self.x_end = self.xdisc + self.L/2
        self.hpos_flag = False 
        
    def print_recap(self):
        print("Small perturbation problem:", self.code, " with ICs: h =", self.values[0], "+", self.values[1], "* exp(-x**2 /", self.values[2], ");   u =", self.values[3])
    
    def nondimensionalize_values(self, scale_H = float, scale_U = float, scale_L=float, scale_T=float):
        self.values[0] = self.values[0] / scale_H
        self.values[1] = self.values[1] / scale_H
        self.values[2] = self.values[2] / (scale_L**2)
        self.values[3] = self.values[3] / scale_U
    
    def nondimensionalize_domain(self, scale_L = float, scale_T = float):
        self.L = self.L / scale_L 
        self.xdisc = self.xdisc / scale_L
        self.x_start = self.x_start / scale_L
        self.x_end = self.x_end / scale_L
        self.tmax = self.tmax / scale_T

    def check_drybed(self):
        # checks if the problem starts with dry bed somewhere
        if (self.values[0]<1e-10):
            self.hpos_flag = True
            print("Dry bed on the bottom")        
        else:
            print("Wet bed everywhere")


# -----------------

# USEFUL FUNCTIONS FOR VALIDATION

def create_valdata_ic_sp(sp, n_pts=100):
    # create array of exact solution at t=0 for validator using a pre-defined number of points
    x0 = np.linspace(sp.x_start, sp.x_end, n_pts)
    h0 = np.full(np.shape(x0), sp.values[0])
    h0 = sp.values[0] + sp.values[1] * np.exp(-(np.multiply(x0,x0) / sp.values[2]))
    u0 = np.full(np.shape(x0), sp.values[3])
    x0 = np.expand_dims(x0, axis=-1)
    t0 = np.full(np.shape(x0), sp.tstart)
    h0 = np.expand_dims(h0, axis=-1)
    u0 = np.expand_dims(u0, axis=-1)
    q0 = np.multiply(h0,u0)
    return x0, t0, h0, u0, q0

def read_val_profile_from_csv_and_nondim(filename, nd_params, time_val=1., xdisc=0):
    # reads array of exact solution at t=time_val for validator and performs non-dimensionalization
    sol = np.loadtxt(filename, delimiter=",")
    x1 = np.expand_dims(sol[:,0], axis=-1) / nd_params.L0 - xdisc
    t1 = np.full(np.shape(x1), time_val)
    h1 = np.expand_dims(sol[:,1], axis=-1) / nd_params.H0
    u1 = np.expand_dims(sol[:,2], axis=-1) / nd_params.U0
    q1 = np.multiply(h1,u1)
    return x1, t1, h1, u1, q1

def create_val_profile_bottom_flat(prb, x0, nd_params):
    z1 = np.full(np.shape(x0), 0) / nd_params.H0
    return z1

    