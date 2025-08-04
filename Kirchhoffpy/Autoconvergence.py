#from .POpyGPU import PO_far_GPU2 as PO_far_GPU
#from .POpyGPU import epsilon,mu
#from .POpyGPU import PO_GPU_2 as PO_GPU

import time 
import sys

def update_screen(error, po=10,po_ref = 10,
                  test = 'po1'
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
    F_ref=method(po_test, po_fix)
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
        update_screen(Error, po,po_fix)
        if Error.max() < Ref:
            loop_count+=1
            F_ref = F_E * 1.0
            sub_loop = 0
            while (loop_count + sub_loop) < max_loops:
                po_near = int(po_ref/2)
                po = po_ref - int((po_ref - po_near)/2)
                ### print the test points
                update_screen('', po,po_fix)
                F_E = method(po)
                Error = np.abs(F_E - F_ref)/np.abs(F_ref)
                update_screen(20*np.log10(Error.max()), po,po_fix)
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


def autoconvergence(method,
                    po1=10,po2=10,
                    target1=11,target2=11,
                    fuction,
                    field_accuracy = -80.
                    Maxpoints = 10000):
    # points in target surface is fixed or defined by the user. It is better to use uniform sampling.
    Ref = 10**(field_accuracy/20)

    #1. auto convergence test for po1
    


    #2. auto convergence test for po2
    for item in [10,20,30,40]:
        update_screen(item,po2)
        time.sleep(1)
    update_screen(po1,po2,True)
    PO1 = 10
    for item in [10,20,30,40]:
        sys.stdout.write(f"\rpo1:{PO1},    po2:{item}")
        time.sleep(1)
        sys.stdout.flush()
    



