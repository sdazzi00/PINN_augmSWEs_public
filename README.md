# PINNs for the 1D augmented system of SWEs

This Python code is used to solve the 1D Shallow Water Equations (SWE) with Physics-Informed Neural Networks (PINN) using NVIDIA Modulus.

It was developed for the test cases presented in Dazzi (2024).

**Reference:** Dazzi S. (2024). Physics-Informed Neural Networks for the Augmented System of Shallow Water Equations with Topography. *Water Resources Research*.

## Requirements

[NVIDIA Modulus](https://docs.nvidia.com/deeplearning/modulus/user_guide/getting_started/installation.html) and its dependencies (including Linux OS).

## How to run (validation tests)

- Check the case setup in the `setup_XX.py` script.
- Launch the `run_XX.py` script:
``` 
python run_XX.py  		# when running on a local machine 
srun python run_XX.py  	# when running on a HPC cluster with SLURM, after requesting the appropriate computational resources
```

## Scripts content (validation tests)

Two scripts are provided for each test case presented in the paper.

`run_XX.py` is the main file which contains the instructions to run the model training (and plot results).

`setup_XX.py` contains the *test_name* and the case setup (domain size, left/right states, weights, config, etc.) and the preliminary operations of non-dimensionalization of variables. It is imported at the beginning of the "run" script.

The following correspondence is used for the tests listed in the paper:

- Test SW1 --> XX = staticflat
- Test SW2 --> XX = staticbump
- Test SW3 --> XX = staticstep
- Test SW4 --> XX = staticparab
- Test SF1 --> XX = bumpsteady
- Test UF1 --> XX = smallpert
- Test UF2 --> XX = dambreak
- Test UF3 --> XX = dbstep
- Test UF4 --> XX = thacker

- Test UF2 with non-augmented SWEs --> XX = db_nonaugm
- Test UF1 config. A (as above) --> XX = smallpert
- Test UF1 config. B --> XX = NoOBS_B
- Test UF1 with data assimilation, config. A --> XX = OBS_A
- Test UF1 with data assimilation, config. B --> XX = OBS_B

The *swe_utils* directory includes all the useful functions/classes to create the test case, get exact solutions, and print results. Some unused functions might be included.

The *conf* directory contains the base configuration file (`config_base2.yaml`), with the model hyperparameters.

## Outputs (validation tests)

The configuration file (`config_XX.yaml`) for each test case will be saved in the *conf* directory.

Training results will be automatically saved in a directory named *output/* and sub-folder named *run_XX*. Please refer to the Modulus documentation for additional info.

Custom results (validators) will also be plotted as .png files, which will be saved in the directory *plots/* and subdirectory named as the *test_name*. The plots compare the PINN results and the reference solution in terms of profiles of depth, velocity and bottom elevation at fixed time steps. These images are automatically saved if the flag *print_plots* is set to True in the "setup" script. Error metrics are also computed and printed.

Moreover, a scatter plot of collocation points in the x-t plane is created if the flag *print_collocation* is set to True in the "setup" script.

Finally, a *ref_sol/* directory will appear when running Test UF1, in which .txt files with the solution obtained from the finite-volume solver will be saved.

## Other scripts: comparison with finite-volume method

Two additional scripts (`fv_dambreak100.py` and `fv_dambreak1000.py`) can be used to compute the finite-volume solution of the dam-break case (UF2) with 100 or 1000 computational cells.

Results are written in the *ref_sol/* directory as text files (name: `dambreak100_python_fv_sol_TIME.txt`, where TIME indicates the time at which profiles x, h, u, uh are saved). 

Please notice that the FV solver is not implemented in a computationally efficient way, because it is only used to perform very small simulations and provide reference solutions for flat-bottom test cases. 

## Other scripts: plots

The plots obtainable from the above scripts are not the figures shown in the paper, because the latter were additionally customized to improve clarity.

To create the figures reported in the paper, the following scripts should be used (after training the PINN models):

- `make_plots_for_paper300dpi.py` can be used to create Figures 2 to 10. The script can be launched as follows by substituting XX with the values above (from staticflat to thacker):
``` 
python make_plots_for_paper300dpi.py "XX"
```
- `make_plot_with_nonaugm300dpi.py` can be used to create Figure 11. No arguments are required in the command line.
- `make_plot_with_FV300dpi.py` can be used to create Figure 12. No arguments are required in the command line.
- `make_plot_with_datafusion300dpi` can be used to create Figure 13. No arguments are required in the command line.

## Additional remarks

Please notice that training is non-repeatable, therefore running these scripts can provide slightly different results compared to what reported in the paper.

Be aware that the code performance for different test cases or configurations was not tested, despite the presence of some additional options in the code.

## Contacts

Dr. Susanna Dazzi - Dept. of Engineering and Architecture, University of Parma, Italy (email: susanna.dazzi@unipr.it)
