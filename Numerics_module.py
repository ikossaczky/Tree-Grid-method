import numpy as np 

def neqMinus(A,B):
    # function for computing the difference of two (1D or 2D) arrays by taking only each nth element of the larger
    # in each direction.
    if A.ndim==2:
        if A.shape[0]>=B.shape[0]:
            step_x=int((A.shape[0]-1)/(B.shape[0]-1))
            step_y=int((A.shape[1]-1)/(B.shape[1]-1))
            return  A[0::step_x,0::step_y]-B
        else:
            step_x=int((B.shape[0]-1)/(A.shape[0]-1))
            step_y=int((B.shape[1]-1)/(A.shape[1]-1))
            return  A-B[0::step_x,0::step_y]       
    else:
        if A.shape[0]>=B.shape[0]:
            step=int((A.shape[0]-1)/(B.shape[0]-1))
            return  A[0::step]-B
        else:
            step=int((B.shape[0]-1)/(A.shape[0]-1))
            return  A-B[0::step]    
    
def GridVolume1D(X):
    # function for computing volumes implied by the nodes of an 1D Grid
    X0=np.concatenate([[X[0]],X])
    X1=np.concatenate([X,[X[-1]]])
    return 0.5*(X0[1:]-X0[:-1]+X1[1:]-X1[:-1])

def GridVolume2D(X,Y):
    # function for computing volumes implied by the nodes of an 2D Grid
    Xv=GridVolume1D(X)
    Yv=GridVolume1D(Y)
    Yvv, Xvv = np.meshgrid(Yv, Xv)
    return Xvv*Yvv

def NormError(Sol,RefSol, Volumes, Norm, normalized=0):
    # function for computing normed error of solution for given refererence solution
    PointErrs=abs(neqMinus(RefSol, Sol))
    if Norm=='Inf':
        return np.max(PointErrs)
    elif normalized==0:
        return (np.sum(Volumes*PointErrs**Norm))**(1/Norm)
    else:
        return (np.sum(Volumes*PointErrs**Norm)/np.sum(Volumes))**(1/Norm)  
    
def PointError1D(X, Sol, RefSol, Point, absval=1):
    # function for computing error (difference) in an specific point for 1D solution
    # with some numerical reference solution
    if absval==1:
        return np.asscalar(abs(neqMinus(Sol, RefSol))[X==Point])
    else:
        return np.asscalar(neqMinus(Sol, RefSol)[X==Point])

def PointErrorExact1D(X, Sol, ExactSol, Point, absval=1):
    # function for computing error (difference) in an specific point for 1D solution
    # with known exact value in the point
    if absval==1:
        return abs(np.asscalar(Sol[X==Point])-ExactSol)
    else:
        return np.asscalar(Sol[X==Point])-ExactSol

def PointError2D(X, Y, Sol, RefSol, Point, absval=1):
    # function for computing error (difference) in an speciffic point for 2D solution 
    # with some numerical reference solution
    Yy, Xx = np.meshgrid(Y, X)
    if absval==1:
        return np.asscalar(abs(neqMinus(Sol, RefSol))[(Xx==Point[0])*(Yy==Point[1])])
    else:
        return np.asscalar(neqMinus(Sol, RefSol)[(Xx==Point[0])*(Yy==Point[1])])

def PointErrorExact2D(X, Y, Sol, ExactSol, Point, absval=1):
    # function for computing error (difference) in an speciffic point for 2D solution
    # with known exact value in the point
    Yy, Xx = np.meshgrid(Y, X)
    if absval==1:
        return abs(np.asscalar(Sol[(Xx==Point[0])*(Yy==Point[1])])-ExactSol)
    else:
        return np.asscalar(Sol[(Xx==Point[0])*(Yy==Point[1])])-ExactSol
