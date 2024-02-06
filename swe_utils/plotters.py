import numpy as np
import matplotlib.pyplot as plt

from modulus.utils.io.plotter import ValidatorPlotter
from swe_utils.funcs_plot import add_legend_labels

# define custom class
class CustomValidatorPlotter(ValidatorPlotter):

    def __call__(self, invar, true_outvar, pred_outvar):
        "Custom plotting function for validator"

        # get input variables
        x = invar["x"][:,0]

        # get and interpolate output variable
        try:
            # case with outputs h, u
            h_true, h_pred = true_outvar["h"][:,0], pred_outvar["h"][:,0]
            u_true, u_pred = true_outvar["u"][:,0], pred_outvar["u"][:,0]
            q_pred = np.multiply(h_pred, u_pred)
            q_true = np.multiply(h_true, u_true)
        except:
            # case with outputs h, q
            h_true, h_pred = true_outvar["h"][:,0], pred_outvar["h"][:,0]
            q_true, q_pred = true_outvar["q"][:,0], pred_outvar["q"][:,0]
            u_pred = np.divide(q_pred, h_pred, out=np.zeros_like(q_pred), where=h_pred!=0)
            u_true = np.divide(q_true, h_true, out=np.zeros_like(q_true), where=h_true!=0)

        try:
            z_true, z_pred = true_outvar["z"][:,0], pred_outvar["z"][:,0]
            Nplots=5
        except:
            Nplots=3    

        # make plot
        f = plt.figure(figsize=(3*Nplots,4), dpi=100)
        plt.suptitle("Riemann problem: PINN vs true solution")
        plt.subplot(1,Nplots,1)
        plt.title("Solution (h)")
        plt.plot(x,h_pred, "--", label="h_pred")
        plt.plot(x,h_true, "--", label="h_true")        
        add_legend_labels("h")
        plt.subplot(1,Nplots,2)
        plt.title("Solution (u)")
        plt.plot(x,u_pred, "--", label="u_pred")
        plt.plot(x,u_true, "--", label="u_true") 
        add_legend_labels("u")
        plt.subplot(1,Nplots,3)
        plt.title("Solution (q)")
        plt.plot(x,q_pred, "--", label="q_pred")
        plt.plot(x,q_true, "--", label="q_true") 
        add_legend_labels("q")
        if (Nplots==5):
            plt.subplot(1,Nplots,4)
            plt.title("Solution (z)")
            plt.plot(x,z_pred, "--", label="z_pred")
            plt.plot(x,z_true, "--", label="z_true") 
            add_legend_labels("z")
            plt.subplot(1,Nplots,5)
            plt.title("Solution (wse)")
            plt.plot(x,z_true, "-", label="bottom_true")
            plt.plot(x,z_pred+h_pred, "--", label="wse_pred")
            plt.plot(x,z_true+h_true, "--", label="wse_true") 
            add_legend_labels("wse")
        plt.tight_layout()

        return [(f, "custom_plot"),]
