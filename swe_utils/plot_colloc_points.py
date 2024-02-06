import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
import matplotlib.pyplot as plt

def plot_colloc_pts(test_name, constr_dir, plot_subdir):

    filenames = (
        "interior",
        "BC_L",
        "BC_R",
        "IC",
    )

    count=0
    reader = vtk.vtkXMLPolyDataReader()
    plt.figure()

    for str in filenames:

        reader.SetFileName(constr_dir + str + ".vtp")
        reader.Update()

        polydata = reader.GetOutput()
        points = polydata.GetPoints()
        array = points.GetData()
        point_coordinates = vtk_to_numpy(array)
        #point_coordinates contiene x,y,z (nel nostro caso x,0,0 e serve solo la prima colonna)

        pointData = polydata.GetPointData()
        #pointData contiene gli "attributi" dei punti, tra cui il tempo t

        t = pointData.GetArray('t')
        t_array = vtk_to_numpy(t)
        #in questo modo ho estratto un array np con i valori t

        #li affianco
        union = np.column_stack((point_coordinates[:,0] , t_array ))
        #np.savetxt("stampa"+str+".txt", union, fmt='%1.5f')
        plt.scatter(point_coordinates[:,0], t_array, label=str)
        plt.show()
        count= count+1

    plt.xlabel("x", fontsize=14)
    plt.ylabel("t", fontsize=14)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=5)
    plt.savefig(plot_subdir + "/" + test_name + "_collocation_points"+ ".png")

    return