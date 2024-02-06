from math import sqrt 
import numpy as np

# FINITE-VOLUME SOLVER WITH HLL 
# this is not an optimized version, it is only meant to provide validation data for the PINN model

g=9.81

def create_ics_rp(rp, dx):
    n_pts = int(rp.L / dx)
    # create array of exact solution at t=0 for validator using a pre-defined number of points
    x0 = np.linspace(rp.x_start+dx/2, rp.x_end-dx/2, n_pts)
    h0 = np.full(np.shape(x0), rp.values[0])
    h0[int(np.size(h0)/2):] = rp.values[2]
    u0 = np.full(np.shape(x0), rp.values[1])
    u0[int(np.size(u0)/2):] = rp.values[3]
    q0 = np.multiply(h0,u0)
    z0 = np.zeros(x0.size)
    return x0, h0, u0, q0, z0

def create_ics_sp(sp, dx):
    n_pts = int(sp.L / dx)
    # create array of exact solution at t=0 for validator using a pre-defined number of points
    x0 = np.linspace(sp.x_start+dx/2, sp.x_end-dx/2, n_pts)
    h0 = sp.values[0] + sp.values[1] * np.exp(-(np.multiply(x0,x0) / sp.values[2]))
    u0 = np.full(np.shape(x0), sp.values[3])
    q0 = np.multiply(h0,u0)
    z0 = np.zeros(x0.size)
    return x0, h0, u0, q0, z0

def read_file(filename1="output_h_bl.txt", filename2="output_u_bl.txt"):
    hf = np.loadtxt(filename1)    
    uf = np.loadtxt(filename2)
    return hf, uf

def read_file2(filename1="prova.csv"):
    sol = np.loadtxt(filename1, delimiter=",")
    x1 = sol[:,0]
    h1 = sol[:,1]
    u1 = sol[:,2]
    q1 = np.multiply(h1,u1)
    return x1, h1, u1, q1

def read_val_profile_from_python_sol_and_nondim(filename, nd_params, time_val=1., xdisc=0):
    # reads array of fv solution at t=time_val for validator and performs non-dimensionalization
    sol = np.loadtxt(filename)
    x1 = np.expand_dims(sol[:,0], axis=-1) / nd_params.L0 - xdisc
    t1 = np.full(np.shape(x1), time_val)
    h1 = np.expand_dims(sol[:,1], axis=-1) / nd_params.H0
    u1 = np.expand_dims(sol[:,2], axis=-1) / nd_params.U0
    #q1 = np.expand_dims(sol[:,3], axis=-1) / (nd_params.U0 * nd_params.H0)
    q1 = np.multiply(h1,u1)
    return x1, t1, h1, u1, q1

def read_val_profile_from_python_sol_and_nondim_flat_bottom(filename, nd_params, time_val=1., xdisc=0):
    x1, t1, h1, u1, q1 = read_val_profile_from_python_sol_and_nondim(filename, nd_params, time_val=time_val, xdisc=xdisc)
    z1 = np.full(np.shape(x1), 0)
    return x1, t1, h1, u1, q1, z1

# -----------------

def compute_fake_muscl_extrap(h, q):

    Hextr_xE=h
    Hextr_xW=h
    Qextr_xE=q
    Qextr_xW=q

    return Hextr_xE, Qextr_xE, Hextr_xW, Qextr_xW

def solverHLL(hl,hr,ql,qr): 

    ul=ql/hl if(hl>0) else 0.
    ur=qr/hr if (hr>0) else 0.
    cl=sqrt(g*hl)
    cr=sqrt(g*hr)
    cstar=0.5*(cl + cr) + 0.25*(ul - ur)
    hstar = cstar*cstar/g
    hmin=min(hl,hr)
    ustar=0
    
    if (hstar<=hmin):
        ustar=0.5*(ul+ur)+cl-cr
    else:
        gel = sqrt(0.5*g*(hstar+hl)/(hstar*hl))
        ger = sqrt(0.5*g*(hstar+hr)/(hstar*hr))
        hstar  = (gel*hl + ger*hr + ul -ur)/(gel + ger)
        ustar=0.5*(ul+ur)+0.5*((hstar-hr)*ger-(hstar-hl)*gel)
    
    f1l=hl*ul
    f1r=hr*ur
    f2l=ul*ul*hl+0.5*g*hl*hl
    f2r=ur*ur*hr+0.5*g*hr*hr

    if ((hstar-hl)<=0):
        sl=ul-cl
    else:
        sl=ul-cl*sqrt(0.5*hstar*(hstar+hl))/hl

    if ((hstar-hr)<=0):
        sr=ur+cr
    else:
        sr=ur+cr*sqrt(0.5*hstar*(hstar+hr))/hr

    # calcolo del flusso intercella (HLL)
    if (sl*sr<=0):
        if ((sr-sl)*(sr-sl)>0):
            flux1=(sr*f1l-sl*f1r+sr*sl*(hr-hl))/(sr-sl)
            flux2=(sr*f2l-sl*f2r+sr*sl*(hr*ur-hl*ul))/(sr-sl)
        else:
            flux1=0
            flux2=0
    else:
        if (sl>=0):
            flux1=f1l
            flux2=f2l
        else:
            flux1=f1r
            flux2=f2r

    return flux1, flux2

def compute_fluxes(Hextr_xE, Qextr_xE, Hextr_xW, Qextr_xW, bctype_up="transmissive", bctype_down="transmissive"): 

    F_WEST1=np.zeros(Hextr_xE.size)
    F_WEST2=np.zeros(Hextr_xE.size)
    F_EAST1=np.zeros(Hextr_xE.size)
    F_EAST2=np.zeros(Hextr_xE.size)

    # interior cells
    for i in range(1,Hextr_xE.size-1):
        F_WEST1[i], F_WEST2[i] = solverHLL(Hextr_xE[i-1],Hextr_xW[i],Qextr_xE[i-1],Qextr_xW[i])
        F_EAST1[i], F_EAST2[i] = solverHLL(Hextr_xE[i],Hextr_xW[i+1],Qextr_xE[i],Qextr_xW[i+1])

    # upstream cell
    i = 0
    F_EAST1[i], F_EAST2[i] = solverHLL(Hextr_xE[i],Hextr_xW[i+1],Qextr_xE[i],Qextr_xW[i+1])
    if(bctype_up=="transmissive" or bctype_up=="inflow"): 
        F_WEST1[i], F_WEST2[i] = solverHLL(Hextr_xW[i],Hextr_xW[i],Qextr_xW[i],Qextr_xW[i])
    elif(bctype_up=="wall"):   
        F_WEST1[i], F_WEST2[i] = solverHLL(Hextr_xW[i],Hextr_xW[i],-Qextr_xW[i],Qextr_xW[i])
        
    # downstream cell
    i = Hextr_xE.size-1
    F_WEST1[i], F_WEST2[i] = solverHLL(Hextr_xE[i-1],Hextr_xW[i],Qextr_xE[i-1],Qextr_xW[i])
    if(bctype_down=="transmissive" or bctype_down=="level"): 
        F_EAST1[i], F_EAST2[i] = solverHLL(Hextr_xE[i],Hextr_xE[i],Qextr_xE[i],Qextr_xE[i])
    elif(bctype_down=="wall"): 
        F_EAST1[i], F_EAST2[i] = solverHLL(Hextr_xE[i],Hextr_xE[i],Qextr_xE[i],-Qextr_xE[i])

    return F_WEST1, F_WEST2, F_EAST1, F_EAST2

def write_results_fv(x0, h, q, time, testname):
    filename = testname + "_python_fv_sol_" + f"{time:.5f}" + ".txt"
    u_ = np.divide(q, h, out=np.zeros_like(q), where=h!=0)
    A = np.stack((x0,h,u_,q), axis=1)
    np.savetxt(filename, A, fmt='%1.5f')
    return 

def run_fv_solver_z(x0, h0, u0, q0, z0, tmax, dx, Nprints=1, bctype_up="transmissive", bctype_down="transmissive", bcvalues=[0., 0.], testname="test", cfl=0.8):

    time = 0.
    h = h0
    u = u0
    q = q0
    write_results_fv(x0, h, q, time, testname)  # note: initial condition is printed here
    print_step = tmax / Nprints         
    t_print = np.arange(print_step, tmax+print_step, print_step)
    print_count = 0
    print_flag=False 
    Qin = bcvalues[0]
    hout = bcvalues[1]

    while(time<tmax):

        # compute dt
        lambd = np.add(np.sqrt(g*h), np.abs(u))     
        dts = cfl*dx * np.divide(np.ones_like(lambd), lambd, out=1E10*np.ones_like(lambd), where=lambd!=0)
        dt = np.amin(dts)
        if (time+dt > t_print[print_count]):
            dt = t_print[print_count] - time
            print_flag=True 

        # first order = "fake" muscl reconstruction! Just to ease future extension to second order...
        Hextr_xE, Qextr_xE, Hextr_xW, Qextr_xW = compute_fake_muscl_extrap(h, q)
        # overwrite bc values if required
        if (bctype_up=="inflow"):
            Qextr_xW[0] = Qin
        if (bctype_down=="level"):
            Hextr_xE[h.size-1] = hout

        # flux computation
        F_WEST1, F_WEST2, F_EAST1, F_EAST2 = compute_fluxes(Hextr_xE, Qextr_xE, Hextr_xW, Qextr_xW, bctype_up, bctype_down)

        # update variables
        h = h - (dt/dx) * (F_EAST1 - F_WEST1) 
        q = q - (dt/dx) * (F_EAST2 - F_WEST2) 

        time = time + dt
        #print("time", time, "dt", dt)
        if (print_flag):
            print("save results function at time", time)
            write_results_fv(x0, h, q, time, testname)
            print_flag=False
            print_count = print_count + 1

    return h, q

def create_fv_sol_with_z(prb, dx, Nprints, testname, x1, z1):
    bctype_up = "transmissive"
    bctype_down = "transmissive"    
    if(prb.type == "riemann"):
        x0, h0, u0, q0, z0 = create_ics_rp(prb, dx)
        bcvalues = [prb.values[0]*prb.values[1], prb.values[2]]
    elif(prb.type == "smallpert"):
        x0, h0, u0, q0, z0 = create_ics_sp(prb, dx)
        bcvalues = [prb.values[0]*prb.values[3], prb.values[0]]
    else:
        raise ValueError("FV solver: Please add function to create initial conditions for new test. Solver not available for non-horizontal bottom")
    h_fv1, q_fv1 = run_fv_solver_z(x0, h0, u0, q0, z0, prb.tmax, dx, Nprints=Nprints, bctype_up=bctype_up, bctype_down=bctype_down, bcvalues=bcvalues, testname=testname)
    return x0, h_fv1, q_fv1, z0

