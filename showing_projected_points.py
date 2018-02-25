import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import vtk
from numpy import random

def display_matplotlib():
    # transformation to pointcloud form
    xs = vecs_p[0, :]
    ys = vecs_p[1, :]
    zs = vecs_p[2, :]

    # visualization
    fig = plt.figure()

    ax = Axes3D(fig)
    #ax = fig.add_subplot(111, projection='3d')
    #for i in range(vecs_p.shape[1]):
    #    ax.scatter(xs[i], ys[i], zs[i], c=colors[i], marker='o')
    ax.scatter(xs, ys, zs, c='b', marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

def display_vtk():
    class VtkPointCloud:

        def __init__(self, zMin=-10.0, zMax=10.0, maxNumPoints=1e6):
            self.maxNumPoints = maxNumPoints
            self.vtkPolyData = vtk.vtkPolyData()
            self.clearPoints()
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(self.vtkPolyData)
            mapper.SetColorModeToDefault()
            mapper.SetScalarRange(zMin, zMax)
            mapper.SetScalarVisibility(1)
            self.vtkActor = vtk.vtkActor()
            self.vtkActor.SetMapper(mapper)

        def addPoint(self, point):
            if self.vtkPoints.GetNumberOfPoints() < self.maxNumPoints:
                pointId = self.vtkPoints.InsertNextPoint(point[:])
                self.vtkDepth.InsertNextValue(point[2])
                self.vtkCells.InsertNextCell(1)
                self.vtkCells.InsertCellPoint(pointId)
            else:
                raise("maximum points reached")
            self.vtkCells.Modified()
            self.vtkPoints.Modified()
            self.vtkDepth.Modified()

        def clearPoints(self):
            self.vtkPoints = vtk.vtkPoints()
            self.vtkCells = vtk.vtkCellArray()
            self.vtkDepth = vtk.vtkDoubleArray()
            self.vtkDepth.SetName('DepthArray')
            self.vtkPolyData.SetPoints(self.vtkPoints)
            self.vtkPolyData.SetVerts(self.vtkCells)
            self.vtkPolyData.GetPointData().SetScalars(self.vtkDepth)
            self.vtkPolyData.GetPointData().SetActiveScalars('DepthArray')

    pointCloud = VtkPointCloud()
    for point in vecs_p.T:
        pointCloud.addPoint(point[0:3])

    # Renderer
    renderer = vtk.vtkRenderer()
    renderer.AddActor(pointCloud.vtkActor)
    renderer.SetBackground(.0, .0, .0)
    renderer.ResetCamera()

    # Render Window
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)

    transform = vtk.vtkTransform()
    transform.Translate(1.0, 0.0, 0.0)

    axes = vtk.vtkAxesActor()
    #  The axes are positioned with a user transform
    axes.SetUserTransform(transform)

    # Interactor
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)

    # Begin Interaction
    renderWindow.Render()
    renderWindowInteractor.Start()


if __name__ == '__main__':
    with open('points.rick', mode='rb') as file:
        struct = pickle.load(file)

    points = struct['points']
    vecs_p = struct['vecs_p']
    colors = struct['colors']
    name = struct['name']

    # display_matplotlib()
    display_vtk()