from math import sqrt 
import numpy as np

# CLASS WITH BUMP PROBLEM DATA

class Bump:
    #name = "Bump"

    def __init__(self, code="custom",dim_vals=[0.2,0.05,2.,2.,4.42],L=20.,xdisc=10.,tmax=100.):
        self.code = code
        self.tstart = 0.
        self.type = "bump"
        
        if(self.code=="custom" or self.code==""):  # user-defined values for RP            
            self.values = dim_vals      # initial values hL, uL, hR, uR (h=depth, u=velocity)
            self.L = L                  # size of the spatial domain 
            self.xdisc = xdisc          # position of the initial discontinuity
            self.tmax = tmax            # size of the temporal domain  
        else:    # values from pre-defined BP (1)
            self.values = [0.2,0.05,2.,0.5,0.]   
            self.L = 20.
            self.xdisc = 0.          
            self.tmax = 100.
            if(self.code=="B0"):       # static case
                self.tmax = 10.
                self.L = 20.
                self.xdisc = 0.      
            elif(self.code=="B1"):     # z_bump = z_base - param * (x-xdisc)**2   if xdisc-b<x<xdisc+b   else z=0   and     h_init = h_base   and q_init = q_base
                self.values[0] = 0.2    #z_base
                self.values[1] = 0.05   #param
                self.values[2] = 2.     #b (half width of the bump) 
                self.values[3] = 2.     #h_base  (and h_downstream)
                self.values[4] = 4.42   #q_base       
            else:
                raise ValueError ("Code not recognized: only \"B0\" to \"B1\" available. Provide \"custom\" and user-defined values as arguments")
       
        self.x_start = self.xdisc - 0.5*self.L
        self.x_end = self.x_start + self.L
        self.hpos_flag = False 
        
    def print_recap(self):
        bump_extent = (self.xdisc - self.values[2], self.xdisc + self.values[2])
        print("Bump problem:", self.code, " with bottom: z =", self.values[0], "-", self.values[1], "*(x-", self.xdisc, ")**2", "if x in ", bump_extent, " and BCs: wse_downstr =", self.values[3], ";   q_upstr =", self.values[4])
    
    def nondimensionalize_values(self, scale_H = float, scale_U = float, scale_L=float, scale_T = float):
        self.values[0] = self.values[0] / scale_H
        self.values[1] = self.values[1] / (scale_H / scale_L**2)
        self.values[2] = self.values[2] / scale_L
        self.values[3] = self.values[3] / scale_H
        self.values[4] = self.values[4] / (scale_U*scale_H)
    
    def nondimensionalize_domain(self, scale_L = float, scale_T = float):
        self.L = self.L / scale_L 
        self.xdisc = self.xdisc / scale_L
        self.x_start = self.x_start / scale_L
        self.x_end = self.x_end / scale_L
        self.tmax = self.tmax / scale_T

    def check_drybed(self):
        # checks if the problem starts with dry bed somewhere
        if (self.values[3]-self.values[0] < 1e-10):
            self.hpos_flag = True
            print("Dry bed : emerged bump")        
        else:
            print("Wet bed everywhere")


# -----------------

# USEFUL FUNCTIONS FOR VALIDATION


def create_valdata_ic_bump(sp, n_pts=100):
    # create array of exact solution at t=0 for validator using a pre-defined number of points
    x0 = np.linspace(sp.x_start, sp.x_end, n_pts)
    z0 = np.full(np.shape(x0), 0)
    z0 = sp.values[0] - sp.values[1] * (x0-sp.xdisc)**2
    z0[z0<0] = 0.
    h0 = np.full(np.shape(x0), sp.values[3])
    h0 = h0 - z0
    q0 = np.full(np.shape(x0), sp.values[4])
    x0 = np.expand_dims(x0, axis=-1)
    t0 = np.full(np.shape(x0), sp.tstart)
    z0 = np.expand_dims(z0, axis=-1)
    h0 = np.expand_dims(h0, axis=-1)
    q0 = np.expand_dims(q0, axis=-1)
    u0 = np.divide(q0, h0, out=np.zeros_like(q0), where=h0!=0)
    return x0, t0, h0, u0, q0, z0

def read_val_profile_from_csv_and_nondim(filename, nd_params, time_val=1., xdisc=0):
    # reads array of exact solution at t=time_val for validator and performs non-dimensionalization
    sol = np.loadtxt(filename, delimiter=",")
    x1 = np.expand_dims(sol[:,0], axis=-1) / nd_params.L0 - xdisc
    t1 = np.full(np.shape(x1), time_val)
    h1 = np.expand_dims(sol[:,1], axis=-1) / nd_params.H0
    u1 = np.expand_dims(sol[:,2], axis=-1) / nd_params.U0
    q1 = np.multiply(h1,u1)
    return x1, t1, h1, u1, q1

def create_val_profile_bottom_bump(prb, x0, nd_params):
    z1 = np.full(np.shape(x0), 0)
    z1 = prb.values[0] - prb.values[1] * (x0-prb.xdisc)**2
    z1[z1<0] = 0.
    z1 = z1 / nd_params.H0
    return z1

g = 9.81 
FrsqUnit = 1/g

def find_h_roots(coeff):
    htemp = np.roots(coeff)
    #print(htemp, htemp.size)
    #real_sol = np.real(htemp[np.isreal(htemp)])
    real_sol = htemp.real[abs(htemp.imag)<1e-6]
    return real_sol

def check_transcritical(values, Fr_sq=0.1019):
    # compute critical height
    hk = (Fr_sq * values[4]**2)**(1./3.)
    # compute h at x(z=zmax) for subcritical case and check
    cost1 = values[3] + 0.5*Fr_sq * (values[4]/values[3])**2
    cost2 = 0.5* Fr_sq *values[4]**2 
    h_sol = find_h_roots([1., values[0]-cost1, 0., cost2])
    if(h_sol[0] < hk):
        transcr = True
        cost3 = values[0] + hk + 0.5*Fr_sq * (values[4]/hk)**2   
    else:
        transcr = False
        cost3 = 0  # value not used
    return cost1, cost2, cost3, transcr

def total_force(h, q, Fr_sq=0.1019):
    return 0.5*h**2 / Fr_sq + q**2/h

def check_submergence(values, cost2, cost3, Fr_sq=0.1019):
    # compute hydraulic force for downstream depth
    Fdown = total_force(values[3], values[4], Fr_sq)
    # compute hydraulic force at the toe of the bump
    htoe = find_h_roots([1., values[0]-cost3, 0., cost2])
    #print(htoe, htoe.size)
    Fup = total_force(htoe[1], values[4], Fr_sq)
    if(Fup < Fdown):
        shock = True
    else:
        shock = False
    return shock

def subcritical_profile_bump(za, cost1, cost2):
    ha = np.full(np.shape(za), 0.)
    for i in range(za.size):
        hi = find_h_roots([1., za[i]-cost1, 0., cost2])
        ha[i]=hi[0]
    return ha

def transcritical_profile_bump(za, xa, cost3, cost2, xdisc):
    ha = np.full(np.shape(za), 0.)
    for i in range(za.size):
        hi = find_h_roots([1., za[i]-cost3, 0., cost2])
        if (xa[i]<xdisc):
            ha[i]=hi[0]
        else:
            ha[i]=hi[1]
    return ha

def transcr_shock_profile_bump(za, xa, values, cost1, cost2, cost3, xdisc, Fr_sq=0.1019):
    ha = np.full(np.shape(xa), values[3])
    qa = np.full(np.shape(xa), values[4])
    h_subcr = subcritical_profile_bump(za, cost1, cost2)
    h_transcr = transcritical_profile_bump(za, xa, cost3, cost2, xdisc)
    F_subcr = total_force(h_subcr, qa, Fr_sq)
    F_transcr = total_force(h_transcr, qa, Fr_sq)
    for i in range(xa.size):        
        if (xa[i]<xdisc):
            ha[i]=h_transcr[i]
        else:
            if (F_transcr[i]>F_subcr[i]):
                ha[i]=h_transcr[i]
            else:
                ha[i]=h_subcr[i]
    return ha

def analytical_solution_bump(bp, Fr_sq=0.1019, n_pts=101):
    # currently this is valid for subcritical flow only 
    xa = np.linspace(bp.x_start, bp.x_end, n_pts)
    za = np.full(np.shape(xa), 0)
    za = bp.values[0] - bp.values[1] * (xa-bp.xdisc)**2
    za[za<0] = 0.    
    ha = np.full(np.shape(xa), bp.values[3])
    cost1, cost2, cost3, transcr = check_transcritical(bp.values, Fr_sq) 
    #print("costanti", cost1, cost2, cost3)
    if (not transcr):   # subcritical flow
        ha = subcritical_profile_bump(za, cost1, cost2)
    else:             # transcritical flow
        shock = check_submergence(bp.values, cost2, cost3, Fr_sq)
        if (not shock):
            ha = transcritical_profile_bump(za, xa, cost3, cost2, bp.xdisc)
        else:       # with shock
            ha = transcr_shock_profile_bump(za, xa, bp.values, cost1, cost2, cost3, bp.xdisc, Fr_sq)
    ua = bp.values[4] / ha
    return xa, ha, ua, za

def create_bump_sol_from_analytical(bp, Fr_sq=FrsqUnit, time_val=1., n_pts=101):
    if (time_val>0):
        xa, ha, ua, za = analytical_solution_bump(bp, Fr_sq, n_pts)
        xa = np.expand_dims(xa, axis=-1)
        ta = np.full(np.shape(xa), time_val)
        ha = np.expand_dims(ha, axis=-1)
        za = np.expand_dims(za, axis=-1)
        ua = np.expand_dims(ua, axis=-1)
        qa = np.multiply(ha,ua)
    else:
        xa, ta, ha, ua, qa, za = create_valdata_ic_bump(bp, n_pts)
    return xa, ta, ha, ua, qa, za

    