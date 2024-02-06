from math import sqrt

class Scales():
    name = "Scales"

    def __init__(self, scale_ls=[1.,1.,1.,1.], nd_par_hypothesis = False, nd_par_type = 1, g=9.81):
        self.L0 = scale_ls[0]
        self.H0 = scale_ls[1]
        self.U0 = scale_ls[2]
        self.T0 = scale_ls[3]
        self.nd_par_hypothesis = nd_par_hypothesis
        self.nd_par_type = nd_par_type

        if(self.nd_par_hypothesis):     
            if(self.nd_par_type==1):   #imposing Str=1 and Fr0_sq = 1/g, giving L0 and H0
                self.U0 = sqrt(self.H0)
                self.T0 = self.L0 / self.U0
            elif(self.nd_par_type==2): #imposing Str=1 and Fr0_sq = 1, giving L0 and T0
                self.U0 = self.L0 / self.T0
                self.H0 = self.U0**2 / g
            elif(self.nd_par_type==3): #imposing Str=1 and Fr0_sq = 1/g, giving L0 and T0
                self.U0 = self.L0 / self.T0
                self.H0 = self.U0**2

        self.Str = self.L0 / (self.U0 * self.T0)
        self.Fr0_sq = self.U0**2 / (g * self.H0)
    
    def print_adim_numbers(self):
        print("Str=", self.Str, "Fr0**2=", self.Fr0_sq)

def nondimensionalize_param(param, scale):
    return param / scale

def dimensionalize_param(param, scale):
    return param * scale

