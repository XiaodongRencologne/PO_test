import pyvista as pv
import numpy as np

def view(p):
    # Define 2D profile points (e.g., a semi-circle)
    theta = np.linspace(0, np.pi, 20)  # Semi-circle
    x = np.linspace(0,5,20)
    y = np.zeros_like(theta)  # Y-coordinates (axis of rotation)
    z = np.cos(theta)  # Z-coordinates
    
    # Create PolyData from points
    points = np.column_stack((x, y, z))
    profile = pv.PolyData(points)
    
    # Create a surface from the points (connect them)
    profile = profile.delaunay_2d()
    
    # Revolve the profile around the Y-axis (extrude it rotationally)
    body = profile.extrude_rotate(resolution=60)  # 60 steps in rotation
    
    # Plot the result
    #p = pv.Plotter()
    p.add_mesh(body, color="lightblue", opacity=0.8)
    p.show()