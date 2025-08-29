#from .POpyGPU import PO_far_GPU2 as PO_far_GPU
#from .POpyGPU import epsilon,mu
#from .POpyGPU import PO_GPU_2 as PO_GPU

import time 
import sys

def update_screen(error, po=10,po_ref = 10,
                  test = 'po1',
                  status = False):
    string = (error and f"{error:.3f}") or "" 
    if status:
        if test == 'po1':
            sys.stdout.write(f"\rpo1:{po1},    po2:{po2}\n")
        else:
            sys.stdout.write(f"\rpo1:{po_ref},    po2:{po}\n")
        sys.stdout.flush()
    else:
        if test == 'po1':
            sys.stdout.write(f"\rpo1:{po},    po2:{po_ref},  " + string )
        else:
            sys.stdout.write(f"\rpo1:{po_ref},    po2:{po},  " + string)
        sys.stdout.flush()
        
def try_convergence(method,
                    po_test,
                    Accuracy_ref,
                    max_loops = 11,
                    po_fix = 10,
                    test = 'po1'):
    Ref = Accuracy_ref
    F_ref=method(po_test)
    loop_count =0
    po_ref = po_test
    Error = None
    while loop_count < max_loops:
        N = 2**(loop_count+1)
        po = po_test*N
        ### print the test points
        update_screen('', po,po_fix)
        F_E = method(po)
        Error = np.abs(F_E - F_ref)/np.abs(F_ref)
        update_screen(Error, po,po_fix,test = test)
        if Error.max() < Ref:
            loop_count+=1
            F_ref = F_E * 1.0
            sub_loop = 0
            while (loop_count + sub_loop) < max_loops:
                po_near = int(po_ref/2)
                po = po_ref - int((po_ref - po_near)/2)
                ### print the test points
                update_screen('', po,po_fix,test = test)
                F_E = method(po)
                Error = np.abs(F_E - F_ref)/np.abs(F_ref)
                update_screen(20*np.log10(Error.max()), po,po_fix,test = test)
                sub_loop += 1
                if Error.max() <= Ref:
                    po_ref = po * 1
                    if sub_loop > 2:
                        return po_ref, loop_count+sub_loop, True
                else:
                    return po_ref,loop_count+sub_loop,True    
        else:
            F_ref = F_E * 1.0
            po_ref = po * 1
            loop_count+=1
    update_screen(20*np.log10(Error.max()), po,po_fix,status = True)
    print('Auto-convergence failed, the iteration loop number exceeds the maximum ' + str(max_loops)+'!!!!')
    return po_ref, loop_count, False
