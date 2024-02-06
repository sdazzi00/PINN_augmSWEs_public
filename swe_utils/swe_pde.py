from sympy import Symbol, Function, Number, Heaviside
from modulus.eq.pde import PDE

class ShallowWater1D_adim(PDE):
    name = "ShallowWater1D_adim"
    # contains both versions of the equations (conservative and non conservative), which can be set using the conserv parameter
    def __init__(self, Sr=1, Fr0_2=0.319, conserv = False):
            
            self.conserv = conserv

            # coordinates
            x = Symbol("x")

            # time
            t = Symbol("t")

            # make input variables
            input_variables = {"x": x, "t": t}

            # make u function
            h = Function("h")(*input_variables)
            u = Function("u")(*input_variables)

            # set equations
            self.equations = {}
            if self.conserv:
                self.equations["mass"] = Sr * h.diff(t) + (u * h).diff(x)
                self.equations["momentum"] = Sr * (u * h).diff(t) + ( u*u*h +0.5*h*h/Fr0_2 ).diff(x)
            else:
                self.equations["mass"] = Sr * h.diff(t) + u * h.diff(x) + h * u.diff(x)
                self.equations["momentum"] = Sr* u.diff(t) + u * u.diff(x) + h.diff(x) / Fr0_2 

            self.equations["depthpos"] = h - Heaviside(h + 1E-12) * h
            self.equations["uhzero_if_hzero"] = (u*h) - Heaviside(h + 1E-12) * (u*h)
            self.equations["uzero_if_hzero"] = u - Heaviside(h + 1E-12) * u


class ShallowWater1D_adim_fixed_bed(PDE):
    name = "ShallowWater1D_adim_fixed_bed"
    # contains both versions of the equations (conservative and non conservative), which can be set using the conserv parameter
    def __init__(self, Sr=1, Fr0_2=0.1019, conserv = False, time = True):
            
            self.conserv = conserv
            self.time = time
            # coordinates
            x = Symbol("x")

            # time
            t = Symbol("t")

            # make input variables
            input_variables = {"x": x, "t": t}
            if not self.time:
                input_variables.pop("t")

            # make u function
            h = Function("h")(*input_variables)
            u = Function("u")(*input_variables)
            z = Function("z")(*input_variables)

            # set equations
            self.equations = {}
            if self.conserv:
                self.equations["mass"] = Sr * h.diff(t) + (u * h).diff(x)
                self.equations["momentum"] = Sr * (u * h).diff(t) + ( u*u*h +0.5*h*h/Fr0_2 ).diff(x) + h/Fr0_2 * z.diff(x)
            else:
                self.equations["mass"] = Sr * h.diff(t) + u * h.diff(x) + h * u.diff(x)
                self.equations["momentum"] = Sr* u.diff(t) + u * u.diff(x) + h.diff(x) / Fr0_2  + z.diff(x) / Fr0_2  

            if self.time:
                self.equations["bottom"] = z.diff(t)

            if not self.time:
                self.equations["discharge"] = u * h
            
            self.equations["depthpos"] = h - Heaviside(h + 1E-12) * h
            self.equations["uhzero_if_hzero"] = (u*h) - Heaviside(h + 1E-12) * (u*h)
            self.equations["uzero_if_hzero"] = u - Heaviside(h + 1E-12) * u

class Swe1d_flux_uh(PDE):
    name = "Swe1d_flux_uh"
    # variables x,t and h,u are non-dimensional (tilde is omitted for the sake of clarity)
    def __init__(self, time=False):
            self.time = time
            # coordinates
            x = Symbol("x")

            # time
            t = Symbol("t")

            # make input variables
            input_variables = {"x": x, "t": t}
            if not self.time:
                input_variables.pop("t")

            # make u function
            h = Function("h")(*input_variables)
            u = Function("u")(*input_variables)

            # set equations
            self.equations = {}
            self.equations["flux_uh"] = u*h 
            self.equations["uh_der"] = (u*h).diff(x)
            self.equations["h_pos"] = h - Heaviside(h + 1E-12) * h


