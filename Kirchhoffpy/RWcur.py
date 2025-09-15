import h5py
from .coordinate_operations import Coord
from .Vopy import vector,abs_v,scalarproduct, CO
## save the lens surface information into H5 files, includeing surface, normal, field E and H
def saveh5_surf(file_h5,face,face_n, E, H, T, R,name = 'face'):
    group = file_h5.create_group(name)
    group.create_dataset('x', data = face.x)
    group.create_dataset('y', data = face.y)
    group.create_dataset('z', data = face.z)
    group.create_dataset('w', data = face.w)
    group.create_dataset('nx', data = face_n.x)
    group.create_dataset('ny', data = face_n.y)
    group.create_dataset('nz', data = face_n.z)
    group.create_dataset('N', data = face_n.N)
    group.create_dataset('Ex', data = E.x)
    group.create_dataset('Ey', data = E.y)
    group.create_dataset('Ez', data = E.z)
    group.create_dataset('Hx', data = H.x)
    group.create_dataset('Hy', data = H.y)
    group.create_dataset('Hz', data = H.z)
    group.create_dataset('T', data = T)
    group.create_dataset('R', data = R)
    #group.create_dataset('poynting', data = poynting)
## save the lens surface information into H5 files, includeing surface, normal, field E and H

def read_cur(filename):
    face = Coord()
    face_n = Coord()
    H = vector()
    E = vector()
    print(filename)
    with h5py.File(filename,'r') as f:
        face.x = f['f2/x'][:].ravel()
        face.y = f['f2/y'][:].ravel()
        face.z = f['f2/z'][:].ravel()
        face.w = f['f2/w'][:].ravel()
        face_n.x = f['f2/nx'][:].ravel()
        face_n.y = f['f2/ny'][:].ravel()
        face_n.z = f['f2/nz'][:].ravel()
        face_n.N = f['f2/N'][:].ravel()
        H.x = f['f2/Hx'][:].ravel()
        H.y = f['f2/Hy'][:].ravel()
        H.z = f['f2/Hz'][:].ravel()
        E.x = f['f2/Ex'][:].ravel()
        E.y = f['f2/Ey'][:].ravel()
        E.z = f['f2/Ez'][:].ravel()
    return face, face_n, H, E