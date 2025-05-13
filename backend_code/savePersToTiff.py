from vtkmodules.all import (
    vtkTIFFReader,
    vtkTIFFWriter
)
import gc
from vtkmodules.util.numpy_support import vtk_to_numpy
import numpy as np
import json
import cv2

from topologytoolkit import (
    ttkFTMTree,
    ttkTopologicalSimplificationByPersistence,
    ttkScalarFieldSmoother
)


if __name__ == '__main__':

    pers = [0.1, 0.08, 0.06, 0.04, 0.02, 0.015, 0.01]
    # regions = [1, 2, 3, 5, 6, 7, 8, 9]
    regions = [1]
    for j in regions:
        pread = vtkTIFFReader()
        pread.SetFileName(f"./elevation{j}.tiff")

        # smoother = ttkScalarFieldSmoother()
        # smoother.SetInputConnection(0, pread.GetOutputPort())
        # smoother.SetInputArrayToProcess(0, 0, 0, 0, "Tiff Scalars")
        # smoother.SetNumberOfIterations(5)
        # smoother.Update()

        simplify = ttkTopologicalSimplificationByPersistence()
        simplify.SetInputConnection(0, pread.GetOutputPort())
        simplify.SetInputArrayToProcess(0, 0, 0, 0, "Tiff Scalars")
        simplify.SetThresholdIsAbsolute(False)

        # tree = ttkFTMTree()
        # tree.SetInputConnection(0, simplify.GetOutputPort())
        # tree.SetInputArrayToProcess(0, 0, 0, 0, "Tiff Scalars")
        # tree.SetTreeType(2)
        # tree.SetWithSegmentation(1)
        for i in pers:
            simplify.SetPersistenceThreshold(i)
            simplify.Update()
            # tree.Update()
            test = vtk_to_numpy(simplify.GetOutput().GetPointData().GetArray(0))
            elevation_map = test.reshape([1856, 4104])
            cv2.imwrite(f"test{i}.tiff", elevation_map)
            # test = vtk_to_numpy(simplify.GetOutput(2).GetPointData().GetArray(2))
            # print(j, i, np.max(test))
            # test.astype('int16').tofile(f"segmentation_region{j}_pers{i}.data")
            # test.tofile(f"simplified_region{j}_pers{i}.data")
            # response = {"array": test.astype('int16').tolist(), "max": int(np.max(test))}
            # json_object = json.dumps(response)
            # with open(f"segmentation_region{j}_pers{i}.json", "w") as outfile:
            #     outfile.write(json_object)