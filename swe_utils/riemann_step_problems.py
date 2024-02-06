from math import sqrt 
import numpy as np

# CLASS WITH RIEMANN PROBLEM WITH STEP

class RiemannProblemStep:
    #name = "RiemannProblemStep"

    def __init__(self, code="custom",dim_vals=[4.,0.,1.,0.,0.,1.],L=12.,xdisc=0.,tmax=1.):
        self.code = code
        self.tstart = 0.
        self.type = "riemannstep"

        if(self.code=="custom" or self.code==""):  # user-defined values for RP            
            self.values = dim_vals      # initial values hL, uL, hR, uR (h=depth, u=velocity)
            self.L = L                  # size of the spatial domain 
            self.xdisc = xdisc          # position of the initial discontinuity
            self.tmax = tmax            # size of the temporal domain  
        else:    # values from pre-defined RP (0-8)
            self.values = [2.,0.,1.,0.,0.,1.]   
            self.L = 12.
            self.xdisc = 0.          
            self.tmax = 1.
            if(self.code=="RS0"):       # static case, no discontinuity
                pass
            elif(self.code=="RS1"):
                self.values[0] = 3.     #hL
                self.values[1] = 0.     #uL
                self.values[2] = 1.     #hR
                self.values[3] = 0      #uR
                self.values[4] = 0.     #zL
                self.values[5] = 1.     #zR
            else:
                raise ValueError ("Code not recognized: only \"RS0\" to \"RS1\" available. Provide \"custom\" and user-defined values as arguments")
       
        self.x_start = self.xdisc - self.L/2
        self.x_end = self.xdisc + self.L/2
        self.hpos_flag = False 
        
    def print_recap(self):
        print("Riemann problem with step:", self.code, "left state:", self.values[0], self.values[1], "right state", self.values[2], self.values[3], "z", self.values[4], self.values[5])
    
    def nondimensionalize_values(self, scale_H = float, scale_U = float, scale_L = float, scale_T = float):
        self.values[0] = self.values[0] / scale_H
        self.values[1] = self.values[1] / scale_U
        self.values[2] = self.values[2] / scale_H
        self.values[3] = self.values[3] / scale_U
        self.values[4] = self.values[4] / scale_H
        self.values[5] = self.values[5] / scale_H
    
    def nondimensionalize_domain(self, scale_L = float, scale_T = float):
        self.L = self.L / scale_L 
        self.xdisc = self.xdisc / scale_L
        self.x_start = self.x_start / scale_L
        self.x_end = self.x_end / scale_L
        self.tmax = self.tmax / scale_T

    def check_drybed(self):
        # checks if the RP includes dry bed         
        if (self.values[0]<1e-10):
            self.hpos_flag = True
            print("Dry bed on the left")
        elif (self.values[2]<1e-10):
            self.hpos_flag = True
            print("Dry bed on the right")            
        else:
            print("Wet bed everywhere (but dry states in the star region are not checked)")

# -----------------


# ANALYTICAL SOLUTION OF RP WITH STEP (see Addition to SWASHES 2022)
# !!! ONLY WORKS FOR SUBCRITICAL CASE WITH LEFT RAREFACTION AND RIGHT SHOCK !!!

from scipy.optimize import root
g=9.81

def get_values_from_rp_step(rp=RiemannProblemStep()):
    hL = rp.values[0]
    uL = rp.values[1]
    hR = rp.values[2]
    uR = rp.values[3]
    zL = rp.values[4]
    zR = rp.values[5]
    return hL, uL, hR, uR, zL, zR

def fun(x, riL, hR, uR, FR, dz):
    # reminder of array elemnts: x[0]=h1, x[1]=u1, x[2]=h2, x[3]=u2, x[4]=s2
    return [riL - x[1]  -2.*sqrt(g*x[0]),  
           uR*hR - x[2]*x[3] - x[4]*(hR - x[2]),
           FR - x[3]*x[3]*x[2] - 0.5*g*x[2]*x[2] - x[4]*(uR*hR - x[3]*x[2]),
           x[1]*x[0] - x[3]*x[2],
           (x[3]*x[3]*x[2] + 0.5*g*x[2]*x[2]) - (x[1]*x[1]*x[0] + 0.5*g*x[0]*x[0]) + g*(x[0]-0.5*dz)*dz ]

def star_region_solution_step_case1(hL,hR,uL,uR, dz):
    riL = uL + 2.*sqrt(g*hL)
    FR = uR*uR*hR + 0.5*g*hR*hR
    x0 = [dz+0.7*(hL-dz), riL-10., 0.3*(hL-dz), 2.*riL-20., 2.*sqrt(g*hR)]   # initial estimate of x
    sol = root(fun, x0, args=(riL, hR, uR, FR, dz, ))
    h1, u1 = sol.x[0], sol.x[1]
    h2, u2 = sol.x[2], sol.x[3]
    return h1, u1, h2, u2

def sol_sampling_step(csi,hL,hR,uL,uR,zL,zR,h1,u1,h2,u2):
    if (csi<=0): #left of bottom step
        #left
        z = zL
    else: #right of bottom step
        z = zR 
    
    if (csi<=0): #left of bottom step
        if(h1>hL): # left shock
            print("not supported")
        else: # left rarefaction
            head = uL - sqrt(g*hL) 
            tail = u1 - sqrt(g*h1)
            if(csi<=head):    # left
                h=hL 
                u=uL 
            elif(csi>=tail):  # right
                h=h1
                u=u1 
            else:              # inside rarefaction
                h = ( (uL+2.*sqrt(g*hL)-csi)/3. )**2 / g 
                u = csi + sqrt(g*h)   
    else: #csi>0      
        if(h2>hR):  # right shock
            s = uR + sqrt(0.5*g*h2/hR*(h2+hR))
            if(csi<=s):
                h = h2
                u = u2
            else:
                h = hR
                u = uR 
        else:  # right rarefaction
            print("not supported")  
        
    return h, u, z

def analytical_solution_riemann_step(rp=RiemannProblemStep(), ts=1., n_pts=101):
    rp.check_drybed()
    if(rp.hpos_flag):  #dry_solution
        raise ValueError("analytical solution of dambreak with step and dry states not supported")
    else:  #wet solution
        hL, uL, hR, uR, zL, zR = get_values_from_rp_step(rp)
        dz = zR - zL
        h1,u1,h2,u2 = star_region_solution_step_case1(hL,hR,uL,uR, dz)
        xe=np.linspace(rp.x_start,rp.x_end,n_pts)
        he, ue, ze = np.zeros(np.size(xe)), np.zeros(np.size(xe)), np.zeros(np.size(xe))
        for i in range(np.size(xe)):
            csi=(xe[i]-rp.xdisc)/ts
            he[i], ue[i], ze[i] =sol_sampling_step(csi,hL,hR,uL,uR,zL,zR,h1,u1,h2,u2)
    return xe, he, ue, ze

# -----------------

# USEFUL FUNCTIONS FOR VALIDATION

def create_valdata_ic_step(rp, n_pts=100):
    # create array of exact solution at t=0 for validator using a pre-defined number of points
    x0 = np.linspace(rp.x_start, rp.x_end, n_pts)
    h0 = np.full(np.shape(x0), rp.values[0])
    h0[int(np.size(h0)/2):] = rp.values[2]
    u0 = np.full(np.shape(x0), rp.values[1])
    u0[int(np.size(u0)/2):] = rp.values[3]
    z0 = np.full(np.shape(x0), rp.values[4])
    z0[int(np.size(z0)/2):] = rp.values[5]
    x0 = np.expand_dims(x0, axis=-1)
    t0 = np.full(np.shape(x0), rp.tstart)
    h0 = np.expand_dims(h0, axis=-1)
    u0 = np.expand_dims(u0, axis=-1)
    z0 = np.expand_dims(z0, axis=-1)
    q0 = np.multiply(h0,u0)
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

def create_val_profile_bottom_step(prb, x1, nd_params):
    z1 = np.full(np.shape(x1), prb.values[4])
    z1[int(np.size(z1)/2):] = prb.values[5]
    z1 = z1 / nd_params.H0
    return z1

def create_valdata_from_analytical_step(rp, time_val=1., n_pts=101):
    if (time_val>0):
        xa, ha, ua, za = analytical_solution_riemann_step(rp, time_val, n_pts)
        xa = np.expand_dims(xa, axis=-1)
        ta = np.full(np.shape(xa), time_val)
        ha = np.expand_dims(ha, axis=-1)
        ua = np.expand_dims(ua, axis=-1)
        za = np.expand_dims(za, axis=-1)
        qa = np.multiply(ha,ua)
    else:
        xa, ta, ha, ua, qa, za = create_valdata_ic_step(rp, n_pts)
    return xa, ta, ha, ua, qa, za
