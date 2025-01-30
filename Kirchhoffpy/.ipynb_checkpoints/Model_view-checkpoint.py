import pyvista as pv
import numpy as np


def view(p):
    # Define 2D profile points (e.g., a semi-circle)
    theta = np.linspace(0, np.pi/3, 20)  # Semi-circle
    x = np.linspace(0,5,20)
    z = np.cos(theta)  # Z-coordinates
    y = np.zeros_like(x)  # Y-coordinates (axis of rotation)
    
    # Create PolyData from points
    points = np.column_stack((x,y,z))
    profile = pv.PolyData(points)

    # Create a surface from the points (connect them)
    profile = profile.delaunay_2d()
    
    # Revolve the profile around the Y-axis (extrude it rotationally)
    body = profile.extrude_rotate(resolution=60)  # 60 steps in rotation
    
    # Plot the result
    #p = pv.Plotter()
    p.add_mesh(body, color="lightblue", opacity=0.8,name = 'face1')
    p.show()
p = pv.Plotter()
view(p)
print(p.actors.keys())
