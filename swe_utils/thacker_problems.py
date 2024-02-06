from math import sqrt, cos, sin, pi
import numpy as np

# CLASS WITH THACKER PROBLEM DATA

class Thacker:
    #name = "Thacker"

    def __init__(self, code="custom",dim_vals=[1.,0.5,0.],L=4.,xdisc=0.,tmax=2.006066681):
        self.code = code
        self.tstart = 0.
        self.type = "thacker"

        if(self.code=="custom" or self.code==""):  # user-defined values for RP            
            self.values = dim_vals      # initial values of parameters [a, h0, 0]
            self.values[2] = sqrt(2*9.81*self.values[1]) / self.values[0]  # parameter omega
            self.L = L                  # size of the spatial domain 
            self.xdisc = xdisc          # position of the initial discontinuity (useless)
            self.tmax = tmax            # size of the temporal domain  
        else:    # values from pre-defined T (1)
            self.values = [1., 0.5, 3.132091953]   
            self.L = 4.
            self.xdisc = 0. 
            if(self.code=="T0"):  #static
                self.values[2] = 0.   
                self.tmax = 1.
            elif(self.code=="T1"):           
                self.values[0] = 1.    #a
                self.values[1] = 0.5   #h0
                self.values[2] = sqrt(2*9.81*self.values[1]) / self.values[0]  # parameter omega    
                self.tmax = 2.*pi/self.values[2]  # one period      
            else: 
                raise ValueError ("Code not recognized: only \"T0\" and \"T1\" available. Provide \"custom\" and user-defined values as arguments")            

        self.x_start = self.xdisc - 0.5*self.L
        self.x_end = self.x_start + self.L
        self.hpos_flag = True 
        
    def print_recap(self):
        print("Thacker problem with planar surface:", self.code)
        print("   with bottom: z = -", self.values[1], "*( x /", self.values[0], ")**2 ) - 1. ) ;  u0 = 0")
        print("   with domain size L =", self.L, "centered in x =", self.xdisc, "and with temporal size tmax =", self.tmax)
    
    def nondimensionalize_values(self, scale_H = float, scale_U = float, scale_L = float, scale_T = float):
        self.values[0] = self.values[0] / scale_L
        self.values[1] = self.values[1] / scale_H
        self.values[2] = self.values[2] * scale_T 
    
    def nondimensionalize_domain(self, scale_L = float, scale_T = float):
        self.L = self.L / scale_L 
        self.xdisc = self.xdisc / scale_L
        self.x_start = self.x_start / scale_L
        self.x_end = self.x_end / scale_L
        self.tmax = self.tmax / scale_T

    def check_drybed(self):
        # checks if the problem starts with dry bed somewhere 
        # method kept for compatibility with riemann problems
        self.hpos_flag = True
        print("Thacker is a wet-dry test")        


# -----------------

g = 9.81 

# USEFUL FUNCTIONS FOR VALIDATION

def temporal_profile_funct_thacker_planar(tp, x, t_):
    if (tp.code=="T0"):
        ht = -tp.values[1] * ( (1./tp.values[0] * (x))**2 - 1. )
        ut = 0.
    else:
        ht = -tp.values[1] * ( (1./tp.values[0] * (x + 0.5*cos(tp.values[2]*t_)))**2 - 1. )
        ut = 0.5*tp.values[2]* sin(tp.values[2]*t_)
    ht[ht<0] = 0
    mask = np.where(ht>0, 1, 0)
    ut = mask * ut
    return ht, ut

def create_valdata_ic_thacker_planar(tp, n_pts=100):
    # create array of exact solution at t=0 for validator using a pre-defined number of points
    x0 = np.linspace(tp.x_start, tp.x_end, n_pts)
    z0 = tp.values[1] * ( (x0/tp.values[0])**2 - 1. )    
    h0, u0 = temporal_profile_funct_thacker_planar(tp, x0, t_=0.)
    x0 = np.expand_dims(x0, axis=-1)
    t0 = np.full(np.shape(x0), tp.tstart)
    z0 = np.expand_dims(z0, axis=-1)
    h0 = np.expand_dims(h0, axis=-1)
    u0 = np.expand_dims(u0, axis=-1)
    q0 = np.multiply(u0, h0)
    return x0, t0, h0, u0, q0, z0

def analytical_solution_thacker_planar(tp, t_, n_pts=101):
    xa = np.linspace(tp.x_start, tp.x_end, n_pts)
    za = tp.values[1] * ( (xa/tp.values[0])**2 - 1. )    
    ha, ua = temporal_profile_funct_thacker_planar(tp, xa, t_)
    return xa, ha, ua, za

def create_thacker_planar_sol_from_analytical(tp, time_val=1., n_pts=101):
    if (time_val>0):
        xa, ha, ua, za = analytical_solution_thacker_planar(tp, time_val, n_pts)
        xa = np.expand_dims(xa, axis=-1)
        ta = np.full(np.shape(xa), time_val)
        ha = np.expand_dims(ha, axis=-1)
        za = np.expand_dims(za, axis=-1)
        ua = np.expand_dims(ua, axis=-1)
        qa = np.multiply(ha,ua)
    else:
        xa, ta, ha, ua, qa, za = create_valdata_ic_thacker_planar(tp, n_pts)
    return xa, ta, ha, ua, qa, za

def read_val_profile_from_csv_and_nondim(filename, nd_params, time_val=1., xdisc=0):
    # reads array of exact solution at t=time_val for validator and performs non-dimensionalization
    sol = np.loadtxt(filename, delimiter=",")
    x1 = np.expand_dims(sol[:,0], axis=-1) / nd_params.L0 - xdisc
    t1 = np.full(np.shape(x1), time_val)
    h1 = np.expand_dims(sol[:,1], axis=-1) / nd_params.H0
    u1 = np.expand_dims(sol[:,2], axis=-1) / nd_params.U0
    q1 = np.multiply(h1,u1)
    return x1, t1, h1, u1, q1

def create_val_profile_bottom_thacker(prb, x0, nd_params):
    z1 = prb.values[1] * ( (x0/prb.values[0])**2 - 1. )  
    z1 = z1 / nd_params.H0
    return z1
