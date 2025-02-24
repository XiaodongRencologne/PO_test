import numpy as np

'''
1. read cut files
'''
def read_cut(filename):
    cuts = {}
    with open(filename,'r') as f:
        while True:
            line = f.readline()  # Read a line
            if not line:  # Stop if end of file is reached
                break
            elif line == 'Tabulated feed data\n':
                    inf = f.readline().split()
                    phi = str(int(float(inf[3])))
                    cuts[phi] = {}
                    theta = np.linspace(float(inf[0]), float(inf[0]) + float(inf[1]) * float(inf[2]), int(inf[2]))
                    cuts[phi]['theta'] = theta
                    data = np.genfromtxt(f,max_rows = int(inf[2]))
                    cuts[phi]['E_co'] = data[:,0]
                    cuts[phi]['E_cx'] = data[:,1]
                    '''
                    for _ in range(int(inf[2])):
                         if not f.readline():
                              break
                    '''
    return cuts

def read_grd(filename):
    with open(filename,'r') as f:
        while True:
            line = f.readline()  # Read a line
            if not line:  # Stop if end of file is reached
                break
            elif line == '++++\n':
                Type = int(f.readline().split()[0])
                if Type == 1:
                    pass
                    #print('Ticra file tyep!\n')
                for _ in range(2):
                    f.readline()
                line = f.readline().split()
                x0 = float(line[0])
                x1 = float(line[2])
                y0 = float(line[1])
                y1 = float(line[3])
                line = f.readline().split()
                Nx = int(line[0])
                Ny = int(line[1])
                x = np.linspace(x0,x1,Nx)
                y = np.linspace(y0,y1,Ny)
                data = np.genfromtxt(f)
                E_co = (data[:,0] + 1j*data[:,1]).reshape(Ny,Nx)
                E_cx = (data[:,2] + 1j*data[:,3]).reshape(Ny,Nx)
                E_z =  (data[:,4] + 1j*data[:,5]).reshape(Ny,Nx)

    return x, y, E_co, E_cx, E_z


                
