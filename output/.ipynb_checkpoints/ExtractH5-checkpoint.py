from Read_GRASP_file import read_grd
import os
import h5py as h5

def saveh5(fname,outputfile):
    X,Y,F_co,F_cx = read_grd(fname)
    print(outputfile)
    with h5.File(outputfile,'w') as f:
        f.create_dataset("co-polar",data = F_co)
        f.create_dataset("cx-polar",data = F_cx)
        f.create_dataset("Az",data = Y)
        f.create_dataset("El",data = X)
        
def readh5(fname):
    name = fname.split('.')
    with h5.File(name[0]+'.h5','r') as f:
        co = f['co-polar'][:,:]
        cx = f['cx-polar'][:,:]
        Az = f['Az'][:]
        El = f['El'][:]
    return Az, El, co      
    
def toH5(folder):
    if not os.path.exists(folder):
        print('Check the input folder!')
    else:
        n=0
        outputfolder = folder +'_H5'
        if os.path.exists(outputfolder):
            pass
        else:
            os.makedirs(outputfolder)
        for Dir in os.listdir(folder):
            Folder = os.path.join(folder,Dir)
            if os.path.isdir(Folder):
                for file in os.listdir(Folder):
                    file = os.path.join(Folder,file)
                    if file.endswith(".grd"):
                        print(n)
                        saveh5(file,os.path.join(outputfolder,
                                                 Dir+'GHz.h5'))
                        n+=1

if __name__ == "__main__":
    toH5('./')
