import numpy as np 

##########################################################################################
######          2D Tree-Grid                                                         #####
##########################################################################################

def TreeGrid(X,T, model, TCBC):
    # INPUT: X -sorted state-space, T -sorted time-axis, model -object including functions for drift, volatility,
    # reward, and discount, TCBC -object including functions for terminal and boundary conditions
    # feasible BCs: dependent on both x (space) and t (time)
    M=T.shape[0]
    N=X.shape[0]
    Controls=model.controls
    QN=Controls.shape[0]
    ext=model.extrem
    dt=T[1]-T[0]
    # reserving place for computed transition probabilities, transition indices and states after transition:
    [PV_left, PV_o, PV_right,  PbeyondBC_left, PbeyondBC_right,\
     index_left, index_o, index_right, X_left, X_o, X_right, Discount]\
    =[[0]*QN, [0]*QN, [0]*QN, [0]*QN, [0]*QN, [0]*QN, [0]*QN, [0]*QN, [0]*QN, [0]*QN, [0]*QN, [0]*QN]
    # computing transition probabilites, indices and states for all possible controls:
    for k in range(QN):
        [PV_left[k], PV_o[k], PV_right[k], PbeyondBC_left[k], PbeyondBC_right[k],index_left[k],\
         index_o[k], index_right[k], X_left[k], X_o[k], X_right[k]]=approximate_flow(X,dt,Controls[k],model)
        Discount[k]=1-dt*model.discount(X,Controls[k])
    # creating "probability-vectors" for adding boundary conditions to boundary nodes:
    PBC_left=np.zeros(N)
    PBC_left[0]=1
    PBC_right=np.zeros(N)
    PBC_right[N-1]=1
    # starting with terminal condition:
    V=TCBC.TC(X)
    # reserving place for new-time-layer-candidates:
    W=np.zeros([QN,N])
    # consecutively computing V in all time layers: 
    for t in range(M-1):
        # computing new time-layer candidate for all possible controls:
        for k in range(QN):
            # 1. row:flows from inner nodes to inner or boundary nodes
            # 2. row: flows from inner nodes beyond the boundaries 
            # 3. row values at the boundary (first and last) nodes
            W[k]=Discount[k]*(PV_left[k]*V[index_left[k]]+PV_o[k]*V[index_o[k]]+PV_right[k]*V[index_right[k]])\
            +PbeyondBC_left[k]*TCBC.BCL(X,T[M-1-t])+PbeyondBC_right[k]*TCBC.BCR(X,T[M-1-t])\
            +PBC_left*TCBC.BCL(X[0],T[M-2-t])+PBC_right*TCBC.BCR(X[N-1],T[M-2-t])
        # choosing (elementwise) maximum/minimum from candidates => constructing V in new time layer:
        if ext=='min':
            V=np.amin(W, axis=0)
        else:
            V=np.amax(W,axis=0) 
    # returning solution V in the last (here initial) time layer:
    return V
 
def TreeGridFull(X,T, model, TCBC):
    # INPUT: X -sorted state-space, T -sorted time-axis, model -object including functions for drift, volatility,
    # reward, and discount, TCBC -object including functions for terminal and boundary conditions
    # feasible BCs: dependent on both x (space) and t (time)
    # (this implementation is (up to more feasible BCs) equivalent to my matlab-function treegrid4_artdif_hjb(S,T,obj) )
    M=T.shape[0]
    N=X.shape[0]
    Controls=model.controls
    QN=Controls.shape[0]
    ext=model.extrem
    dt=T[1]-T[0]
    # reserving place for computed transition probabilities, transition indices and states after transition:
    [PV_left, PV_o, PV_right,  PbeyondBC_left, PbeyondBC_right,\
     index_left, index_o, index_right, X_left, X_o, X_right, Discount]\
    =[[0]*QN, [0]*QN, [0]*QN, [0]*QN, [0]*QN, [0]*QN, [0]*QN, [0]*QN, [0]*QN, [0]*QN, [0]*QN, [0]*QN]
    # computing transition probabilites, indices and states for all possible controls:
    for k in range(QN):
        [PV_left[k], PV_o[k], PV_right[k], PbeyondBC_left[k], PbeyondBC_right[k],index_left[k],\
         index_o[k], index_right[k], X_left[k], X_o[k], X_right[k]]=approximate_flow(X,dt,Controls[k],model)
        Discount[k]=1-dt*model.discount(X,Controls[k])
    # creating "probability-vectors" for adding boundary conditions to boundary nodes:
    PBC_left=np.zeros(N)
    PBC_left[0]=1
    PBC_right=np.zeros(N)
    PBC_right[N-1]=1
    # reserving place for solution:
    VV=np.zeros([M,N])
    # starting with terminal condition:
    VV[0,:]=TCBC.TC(X)
    # reserving place for new-time-layer-candidates:
    W=np.zeros([QN,N])
    # consecutively computing V in all time layers: 
    for t in range(M-1):
        # computing new time-layer candidate for all possible controls:
        for k in range(QN):
            # 1. row:flows from inner nodes to inner or boundary nodes
            # 2. row: flows from inner nodes beyond the boundaries 
            # 3. row values at the boundary (first and last) nodes
            W[k]=Discount[k]*(PV_left[k]*VV[t,index_left[k]]+PV_o[k]*VV[t,index_o[k]]+PV_right[k]*VV[t,index_right[k]])\
            +PbeyondBC_left[k]*TCBC.BCL(X,T[M-1-t])+PbeyondBC_right[k]*TCBC.BCR(X,T[M-1-t])\
            +PBC_left*TCBC.BCL(X[0],T[M-2-t])+PBC_right*TCBC.BCR(X[N-1],T[M-2-t])
        # choosing (elementwise) maximum/minimum from candidates => constructing V in new time layer:
        if ext=='min':
            VV[t+1,:]=np.amin(W, axis=0)
        else:
            VV[t+1,:]=np.amax(W,axis=0) 
    # returning solution VV in the last (here initial) time layer:
    return VV

def approximate_flow(X,dt,q,model):
    # INPUT: X -sorted state-space, dt -current time-step, q -current control, 
    # model-object including functions for drift, volatility, reward, and discount
    N=X.shape[0]
    X_inner=X[1:N-1]
    dx=np.max(X[1:N]-X[0:N-1])
    # computing volatility, drift and variance, expected value of new state-old state:
    vol=model.volatility(X_inner,q)
    drf=model.drift(X_inner,q)
    artVar=0.25*(abs(drf)*dt+np.sqrt((abs(drf)*dt)**2+4*dx*dt*abs(drf)))**2-(abs(drf)*dt)**2
    Var=np.maximum(vol**2*dt,artVar)
    E=drf*dt
    # computing exact left and right flow from each non-boundary (inner) state:
    X_inner_left_flow=X_inner-np.sqrt((dt*drf)**2+Var)
    X_inner_right_flow=X_inner+np.sqrt((dt*drf)**2+Var)
    # searching indices of the nearest left grid-point  to left flow and nearest right grid point to right flow:
    # (specially treated are cells with zero flow, to not divide by zero later when computing trans. probabilities)
    # (if the left flow lands beyond left boundary index=-1, if right lands beyond right boundary index=N)
    index_inner_left=np.searchsorted(X,X_inner_left_flow,side='right') -1 - (X_inner_left_flow==X_inner)
    index_inner_right=np.searchsorted(X,X_inner_right_flow,side='left') + (X_inner_right_flow==X_inner)
    
    # computing states from X-grid corresponding to left and right flow indices: 
    # (special treatment needed for unvalid indices -1, N; here state corresponds to exact flow, not to some grid state)
    X_inner_left=X[np.maximum(index_inner_left,0)]
    X_inner_left[index_inner_left==-1]=X_inner_left_flow[index_inner_left==-1]
    X_inner_right=X[np.minimum(index_inner_right,N-1)]
    X_inner_right[index_inner_right==N]=X_inner_right_flow[index_inner_right==N]
    # computing Deltas -distances between left/right state and initial (or middle) state:
    Delta_right_X=X_inner_right-X_inner
    Delta_left_X=X_inner-X_inner_left
    # computing transition probabilities for non-boundary (inner) states:
    P_inner_left=(-E*(Delta_right_X-E)+Var)/(Delta_left_X*(Delta_left_X+Delta_right_X))
    P_inner_o=((Delta_right_X-E)*(-Delta_left_X-E)+Var)/(-Delta_left_X*Delta_right_X)
    P_inner_right=(-E*(-Delta_left_X-E)+Var)/(Delta_right_X*(Delta_left_X+Delta_right_X))
    # extracting probabilites of transition to grid states (boundary states included),
    # if the transition is beyond the boundary for some state, probability=0:
    # (note: firstand last, (boundary) nodes are treated like being beyond the boundary)
    PV_left=np.concatenate([[0],P_inner_left*(index_inner_left!=-1),[0]])
    PV_o=np.concatenate([[0],P_inner_o,[0]])
    PV_right=np.concatenate([[0],P_inner_right*(index_inner_left!=N),[0]])
    # extracting probabilites of transition beyond left/right boundary (boundary states not included),
    # if the transition is to a grid state for some state, probability=0:
    PbeyondBC_left=np.concatenate([[0],P_inner_left*(index_inner_left==-1),[0]])
    PbeyondBC_right=np.concatenate([[0],P_inner_right*(index_inner_left==N),[0]])
    # rewriting unvalid indices (-1,N) by (0,N-1) to avoid out-of-scope-error:
    # (zero PV-probabilites for the states with indices (-1,N) will take care that
    # these states are not used as grid-states anyway)
    index_left=np.concatenate([[1],np.maximum(index_inner_left,0),[N-1]])
    index_o=np.concatenate([[1],np.arange(1,N-1),[N-1]])
    index_right=np.concatenate([[1],np.minimum(index_inner_right,N-1),[N-1]])
    # collecting all left, right, middle states for the case that BC's are state dependent:
    # (the grid-states are for this purpose redundant but doesn't matter.)
    X_left=np.concatenate([[X[0]],X_inner_left,[X[N-1]]])
    X_o=X
    X_right=np.concatenate([[X[0]],X_inner_right,[X[N-1]]])
    #returning transition probabilities, (for grid-states and beyond-boundary states separatelly) indices and states:
    return (PV_left,PV_o,PV_right,PbeyondBC_left,PbeyondBC_right,index_left,index_o,index_right,X_left,X_o,X_right)

##########################################################################################
######          2D Tree-Grid                                                         #####
##########################################################################################

def TreeGrid2D(X,Y,T,model,TCBC,BC_args, SpaceStepCoef):
    # INPUT: X,Y -sorted state-spaces, T -sorted time-axis, model -object including functions for drift, volatility,
    # reward, and discount, TCBC -object including functions for terminal and boundary conditions, BC_args -data needed 
    # for constructions of boundary conditions. SpaceStepCoef - multiplicator of max(cx,dy) in definition of h.
    # feasible BCs: time-dependent, but constant in direction tagential to the boundary
    M=T.shape[0]
    Nx=X.shape[0]    
    Ny=Y.shape[0]
    Controls=model.controls
    QN=Controls.shape[0]
    ext=model.extrem
    dt=T[1]-T[0]
    # reserving place for computed transition probabilities, transition indices and discount factor:
    [Po,PXl,PXr,PYl,PYr,PXYplus,PXYminus,IndXo,IndXl,IndXr,IndYo,IndYl,IndYr, Discount]\
    =[[0]*QN, [0]*QN, [0]*QN, [0]*QN, [0]*QN, [0]*QN, [0]*QN, [0]*QN, [0]*QN, [0]*QN, [0]*QN, [0]*QN, [0]*QN, [0]*QN]
    # computing transition probabilites, indices and discount factors for all possible controls:
    for k in range(QN):
        # computation with larger stencil -h dependent on dt,dx,dy:
        [Po[k],PXl[k],PXr[k],PYl[k],PYr[k],PXYplus[k],PXYminus[k],IndXo[k],IndXl[k],IndXr[k],IndYo[k],IndYl[k],IndYr[k],\
        CvEx]=approximate_flow_2D(X,Y,dt,Controls[k],model,SpaceStepCoef)
        # computing discount factor:
        Discount[k]=1-dt*model.discount(X,Y,Controls[k])     
    # starting with terminal condition:
    V=TCBC.TC(X,Y)
    # reserving place for new-time-layer-candidates:
    W=np.zeros([QN,Nx,Ny])
    #creating lambda function for minimization/maximization (dependent on extrem):
    if ext=='min':
        optimize= lambda W:np.amin(W, axis=0)
    else:
        optimize= lambda W:np.amax(W, axis=0) 
    # consecutively computing V in all time layers: 
    for t in range(M-1):
        # computing new time-layer candidate for all possible controls:
        for k in range(QN):
            # 1.-4. rows:flows from inner nodes to inner nodes, boundary nodes, or beyond the boundary, where the solution
            # is supposed the same value as in the nearest boundary node
            # 5. row: solution in boundary nodes
            W[k]=Discount[k]*(Po[k]*V[IndXo[k],IndYo[k]] + PXl[k]*V[IndXl[k], IndYo[k]] + PXr[k]*V[IndXr[k], IndYo[k]]\
            + PYl[k]*V[IndXo[k], IndYl[k]] + PYr[k]*V[IndXo[k], IndYr[k]]\
            + PXYplus[k]*(V[IndXl[k],IndYl[k]]+V[IndXr[k],IndYr[k]])\
            + PXYminus[k]*(V[IndXl[k],IndYr[k]]+V[IndXr[k],IndYl[k]]))\
            + TCBC.BC(t+1,BC_args)
        # searching optimum in new time layer:
        V=optimize(W)
    # saving Flags -informations about solution (e.g.: for debbuging)
    Flags=0
    # returning solution V in the last (here initial) time layer and Flags:
    return V, Flags

def TreeGrid2DDoubleFlow(X,Y,T,model,TCBC,BC_args, SpaceStepCoefs):
    # INPUT: X,Y -sorted state-spaces, T -sorted time-axis, model -object including functions for drift, volatility,
    # reward, and discount, TCBC -object including functions for terminal and boundary conditions, BC_args -data needed 
    # for constructions of boundary conditions. SpaceStepCoefs - two multiplicators of max(cx,dy) in definition of h
    # (first should be larger and is used only if second is not good enough to reproduce the correlation exactly)
    # feasible BCs: time-dependent, but constant in direction tagential to the boundary
    M=T.shape[0]
    Nx=X.shape[0]    
    Ny=Y.shape[0]
    Controls=model.controls
    QN=Controls.shape[0]
    ext=model.extrem
    dt=T[1]-T[0]
    # reserving place for computed transition probabilities, transition indices and discount factor:
    [Po,PXl,PXr,PYl,PYr,PXYplus,PXYminus,IndXo,IndXl,IndXr,IndYo,IndYl,IndYr, Discount]\
    =[[0]*QN, [0]*QN, [0]*QN, [0]*QN, [0]*QN, [0]*QN, [0]*QN, [0]*QN, [0]*QN, [0]*QN, [0]*QN, [0]*QN, [0]*QN, [0]*QN]
    # computing transition probabilites, indices and discount factors for all possible controls:
    for k in range(QN):
        # computation with larger stencil -h dependent on dt,dx,dy:
        [Po[k],PXl[k],PXr[k],PYl[k],PYr[k],PXYplus[k],PXYminus[k],IndXo[k],IndXl[k],IndXr[k],IndYo[k],IndYl[k],IndYr[k],\
        CvEx]=approximate_flow_2D(X,Y,dt,Controls[k],model,SpaceStepCoefs[0])
        # computation with smaller stencil -h dependent only on dt:
        [Po0,PXl0,PXr0,PYl0,PYr0,PXYplus0,PXYminus0,IndXo0,IndXl0,IndXr0,IndYo0,IndYl0,IndYr0,CovExact]\
        =approximate_flow_2D(X,Y,dt,Controls[k],model,SpaceStepCoefs[1])
        # taking results computed with smaller stencil in nodes wher this was large to reproduce correlation exactly
        Po[k][CovExact]=Po0[CovExact]
        PXYplus[k][CovExact]=PXYplus0[CovExact]
        PXYminus[k][CovExact]=PXYminus0[CovExact]
        PXl[k][CovExact]=PXl0[CovExact]
        PXr[k][CovExact]=PXr0[CovExact]
        IndXo[k][CovExact]=IndXo0[CovExact]
        IndXl[k][CovExact]=IndXl0[CovExact]
        IndXr[k][CovExact]=IndXr0[CovExact]
        PYl[k][CovExact]=PYl0[CovExact]
        PYr[k][CovExact]=PYr0[CovExact]
        IndYo[k][CovExact]=IndYo0[CovExact]
        IndYl[k][CovExact]=IndYl0[CovExact]
        IndYr[k][CovExact]=IndYr0[CovExact]
        # computing discount factor:
        Discount[k]=1-dt*model.discount(X,Y,Controls[k])     
    # starting with terminal condition:
    V=TCBC.TC(X,Y)
    # reserving place for new-time-layer-candidates:
    W=np.zeros([QN,Nx,Ny])
    # consecutively computing V in all time layers: 
    for t in range(M-1):
        # computing new time-layer candidate for all possible controls:
        for k in range(QN):
            # 1.-4. rows:flows from inner nodes to inner nodes, boundary nodes, or beyond the boundary, where the solution
            # is supposed the same value as in the nearest boundary node
            # 5. row: solution in boundary nodes
            W[k]=Discount[k]*(Po[k]*V[IndXo[k],IndYo[k]] + PXl[k]*V[IndXl[k], IndYo[k]] + PXr[k]*V[IndXr[k], IndYo[k]]\
            + PYl[k]*V[IndXo[k], IndYl[k]] + PYr[k]*V[IndXo[k], IndYr[k]]\
            + PXYplus[k]*(V[IndXl[k],IndYl[k]]+V[IndXr[k],IndYr[k]])\
            + PXYminus[k]*(V[IndXl[k],IndYr[k]]+V[IndXr[k],IndYl[k]]))\
            + TCBC.BC(t+1,BC_args)
        if ext=='min':
            V=np.amin(W, axis=0)
        else:
            V=np.amax(W, axis=0) 
    # saving Flags -informations about solution (e.g.: for debbuging)
    Flags=CovExact
    # returning solution V in the last (here initial) time layer and Flags:
    return V, Flags

def approximate_flow_2D(X,Y,dt,q,model,SpaceStepCoef):    
    # INPUT: X,Y -sorted state-spaces, dt -current time-step, q -current control, 
    # model-object including functions for drift, volatility, reward, and discount
    # SpaceStepCoef multiplicator of max(cx,dy) in definition of h (~square of stencil size parameter)
    Nx=X.shape[0]
    Ny=Y.shape[0]
    Y_inner, X_inner=np.meshgrid(Y[1:Ny-1], X[1:Nx-1])

    # computing volatility, drift and correlataion:
    vol_x=model.volatility_x(X_inner,Y_inner,q)
    vol_y=model.volatility_y(X_inner,Y_inner,q)
    drf_x=model.drift_x(X_inner,Y_inner,q)
    drf_y=model.drift_y(X_inner,Y_inner,q)
    corr=model.correlation(X_inner,Y_inner,q)
    dx=np.max(X[1:Nx]-X[0:Nx-1])
    dy=np.max(Y[1:Ny]-Y[0:Ny-1])

    # computing parameter h (step) which will determinate stencil size
    h=np.maximum(dt,SpaceStepCoef*np.maximum(dx,dy))
    b=h/dt
    
    # computing variance, expected value of new state-old state:
    E_x=drf_x*dt
    E_y=drf_y*dt
    artVar_x=0.5*(abs(E_x)*np.sqrt(4*b**2*E_x**2+16*b*dx*abs(E_x)) - (2*b-2)*E_x**2 + 4*dx*abs(E_x))
    artVar_y=0.5*(abs(E_y)*np.sqrt(4*b**2*E_y**2+16*b*dy*abs(E_y)) - (2*b-2)*E_y**2 + 4*dy*abs(E_y))
    Var_x=np.maximum(np.maximum(vol_x**2*dt,artVar_x),E_x**2)
    Var_y=np.maximum(np.maximum(vol_y**2*dt,artVar_y),E_y**2)
    W_x=Var_x+E_x**2
    W_y=Var_y+E_y**2

    # Case of non-negative expectation E_x
    # computing exact left and right flow from each non-boundary (inner) state (X direction):
    X_inner_left_flow1=X_inner-np.sqrt(2*b*W_x)
    X_inner_right_flow1=X_inner+np.sqrt(2*b*W_x)
    
    # searching indices of the nearest left grid-point  to left flow and nearest right grid point to right flow:
    # (specially treated are cells with zero flow, to not divide by zero later when computing trans. probabilities)
    # (if the left flow lands beyond left boundary index=-1, if right lands beyond right boundary index=Nx)
    Xindex_inner_left1=np.searchsorted(X,X_inner_left_flow1,side='right') -1 - (X_inner_left_flow1==X_inner)
    Xindex_inner_right1=np.searchsorted(X,X_inner_right_flow1,side='left') + (X_inner_right_flow1==X_inner)
    
    # computing states from X-grid corresponding to left and right flow indices: 
    # (special treatment needed for unvalid indices -1, N; here state corresponds to exact flow, not to some grid state)
    X_inner_left1=X[np.maximum(Xindex_inner_left1,0)]
    X_inner_left1[Xindex_inner_left1==-1]=X_inner_left_flow1[Xindex_inner_left1==-1]
    X_inner_right1=X[np.minimum(Xindex_inner_right1,Nx-1)]
    X_inner_right1[Xindex_inner_right1==Nx]=X_inner_right_flow1[Xindex_inner_right1==Nx]
    
    # Case of positive expectation E_x
    X_inner_left_flow2=np.minimum(X_inner_left_flow1,2*X_inner-X_inner_right1)
    X_inner_right_flow2=np.maximum(X_inner_right_flow1,2*X_inner-X_inner_left1)

    Xindex_inner_left2=np.searchsorted(X,X_inner_left_flow2,side='right') -1 - (X_inner_left_flow2==X_inner)
    Xindex_inner_right2=np.searchsorted(X,X_inner_right_flow2,side='left') + (X_inner_right_flow2==X_inner)
    X_inner_left2=X[np.maximum(Xindex_inner_left2,0)]
    X_inner_left2[Xindex_inner_left2==-1]=X_inner_left_flow2[Xindex_inner_left2==-1]
    X_inner_right2=X[np.minimum(Xindex_inner_right2,Nx-1)]
    X_inner_right2[Xindex_inner_right2==Nx]=X_inner_right_flow2[Xindex_inner_right2==Nx]
    
    # Combining case of non-negative E_x and of negative E_x
    Xindex_inner_left=np.array(Xindex_inner_left1)
    Xindex_inner_left[E_x<0]=Xindex_inner_left2[E_x<0]
    X_inner_left=np.array(X_inner_left1)
    X_inner_left[E_x<0]=X_inner_left2[E_x<0]
    Xindex_inner_right=np.array(Xindex_inner_right1)
    Xindex_inner_right[E_x<0]=Xindex_inner_right2[E_x<0]
    X_inner_right=np.array(X_inner_right1)
    X_inner_right[E_x<0]=X_inner_right2[E_x<0]

    # Repeating everything in the Y-direction
    Y_inner_left_flow1=Y_inner-np.sqrt(2*b*W_y)
    Y_inner_right_flow1=Y_inner+np.sqrt(2*b*W_y)

    Yindex_inner_left1=np.searchsorted(Y,Y_inner_left_flow1,side='right') -1 - (Y_inner_left_flow1==Y_inner)
    Yindex_inner_right1=np.searchsorted(Y,Y_inner_right_flow1,side='left') + (Y_inner_right_flow1==Y_inner)
    Y_inner_left1=Y[np.maximum(Yindex_inner_left1,0)]
    Y_inner_left1[Yindex_inner_left1==-1]=Y_inner_left_flow1[Yindex_inner_left1==-1]
    Y_inner_right1=Y[np.minimum(Yindex_inner_right1,Ny-1)]
    Y_inner_right1[Yindex_inner_right1==Ny]=Y_inner_right_flow1[Yindex_inner_right1==Ny]

    Y_inner_left_flow2=np.minimum(Y_inner_left_flow1,2*Y_inner-Y_inner_right1)
    Y_inner_right_flow2=np.maximum(Y_inner_right_flow1,2*Y_inner-Y_inner_left1)

    Yindex_inner_left2=np.searchsorted(Y,Y_inner_left_flow2,side='right') -1 - (Y_inner_left_flow2==Y_inner)
    Yindex_inner_right2=np.searchsorted(Y,Y_inner_right_flow2,side='left') + (Y_inner_right_flow2==Y_inner)
    Y_inner_left2=Y[np.maximum(Yindex_inner_left2,0)]
    Y_inner_left2[Yindex_inner_left2==-1]=Y_inner_left_flow2[Yindex_inner_left2==-1]
    Y_inner_right2=Y[np.minimum(Yindex_inner_right2,Ny-1)]
    Y_inner_right2[Yindex_inner_right2==Ny]=Y_inner_right_flow2[Yindex_inner_right2==Ny]

    Yindex_inner_left=np.array(Yindex_inner_left1)
    Yindex_inner_left[E_y<0]=Yindex_inner_left2[E_y<0]
    Y_inner_left=np.array(Y_inner_left1)
    Y_inner_left[E_y<0]=Y_inner_left2[E_y<0]
    Yindex_inner_right=np.array(Yindex_inner_right1)
    Yindex_inner_right[E_y<0]=Yindex_inner_right2[E_y<0]
    Y_inner_right=np.array(Y_inner_right1)
    Y_inner_right[E_y<0]=Y_inner_right2[E_y<0]

    # computing Deltas -distances between left/right state and initial (or middle) state:
    Delta_right_X=X_inner_right-X_inner
    Delta_left_X=X_inner-X_inner_left
    Delta_right_Y=Y_inner_right-Y_inner
    Delta_left_Y=Y_inner-Y_inner_left
    
    # computing values that the |W_xy|/Delta_c should not exceed
    ppx_right=(W_x+E_x*Delta_left_X)/(Delta_right_X**2+Delta_left_X*Delta_right_X)
    ppx_left=(W_x-E_x*Delta_right_X)/(Delta_left_X**2+Delta_left_X*Delta_right_X)
    ppy_right=(W_y+E_y*Delta_left_Y)/(Delta_right_Y**2+Delta_left_Y*Delta_right_Y)
    ppy_left=(W_y-E_y*Delta_right_Y)/(Delta_left_Y**2+Delta_left_Y*Delta_right_Y)

    # computing new covariance in such manner, that the correlation coefficient remains same 
    # also ofter redefining Variance (through possible addition of artifitial diffusion)
    cov=np.sqrt(Var_x*Var_y)*corr/dt
    # computing Delta_c
    Delta_c=Delta_right_X*Delta_right_Y + Delta_left_X*Delta_left_Y
    Delta_c[cov*dt+E_x*E_y<0]=(Delta_right_X*Delta_left_Y + Delta_left_X*Delta_right_Y)[cov*dt+E_x*E_y<0]
    
    # Computing new covariance in such manner that PXl,PXr,PYl,PYr will be non-negative
    C_xy=np.minimum(np.minimum(np.minimum(ppx_right,ppx_left),np.minimum(ppy_right,ppy_left))*Delta_c,abs(cov*dt+E_x*E_y))
    Cov=C_xy-E_x*E_y
    Cov[cov*dt+E_x*E_y<0]=(-C_xy-E_x*E_y)[cov*dt+E_x*E_y<0]
    W_xy=Cov+E_x*E_y
    
    # Computing transition probabilites. 
    # Probabilities on the boundary are zero, the boundary condition is added separately
    Po=np.zeros([Nx,Ny])
    Po[1:Nx-1,1:Ny-1]=1 - W_y/(Delta_left_Y*Delta_right_Y) - W_x/(Delta_left_X*Delta_right_X)\
    +E_y*(Delta_right_Y-Delta_left_Y)/(Delta_left_Y*Delta_right_Y)\
    +E_x*(Delta_right_X-Delta_left_X)/(Delta_left_X*Delta_right_X) + 2*abs(W_xy)/Delta_c

    PXr=np.zeros([Nx,Ny])
    PXr[1:Nx-1,1:Ny-1]=ppx_right-abs(W_xy)/Delta_c
    PXl=np.zeros([Nx,Ny])
    PXl[1:Nx-1,1:Ny-1]=ppx_left-abs(W_xy)/Delta_c
    PYr=np.zeros([Nx,Ny])
    PYr[1:Nx-1,1:Ny-1]=ppy_right-abs(W_xy)/Delta_c
    PYl=np.zeros([Nx,Ny])
    PYl[1:Nx-1,1:Ny-1]=ppy_left-abs(W_xy)/Delta_c

    pxyplus=np.zeros([Nx-2,Ny-2])
    pxyplus[W_xy>=0]=(abs(W_xy)/Delta_c)[W_xy>=0]
    PXYplus=np.zeros([Nx,Ny])
    PXYplus[1:Nx-1,1:Ny-1]=pxyplus
    pxyminus=np.zeros([Nx-2,Ny-2])
    pxyminus[W_xy<0]=(abs(W_xy)/Delta_c)[W_xy<0]
    PXYminus=np.zeros([Nx,Ny])
    PXYminus[1:Nx-1,1:Ny-1]=pxyminus
    
    # rewriting unvalid indices (-1,Nx) by (0,Nx-1) to avoid out-of-scope-error:
    # these rewrited indices represent flows boeyond the boundaries, and rewriting them with boundary indices 0, Nx-1 is
    # valid if the solution behind the boundary is approximately constant in the direction tangential to the boundary.
    IndXl=np.zeros([Nx,Ny]).astype(int)
    IndXl[1:Nx-1,1:Ny-1]=np.maximum(Xindex_inner_left,0)
    IndXr=np.zeros([Nx,Ny]).astype(int)
    IndXr[1:Nx-1,1:Ny-1]=np.minimum(Xindex_inner_right,Nx-1)
    
    # Doing the same in Y-direction
    IndYl=np.zeros([Nx,Ny]).astype(int)
    IndYl[1:Nx-1,1:Ny-1]=np.maximum(Yindex_inner_left,0)
    IndYr=np.zeros([Nx,Ny]).astype(int)
    IndYr[1:Nx-1,1:Ny-1]=np.minimum(Yindex_inner_right,Ny-1)
    IndYo,IndXo=np.meshgrid(np.arange(0,Ny), np.arange(0,Nx))
    
    # saving information in which nodes the covariance was modelled exactly. (if it was not the case in some node,
    # and we used h dependent only on dt, we have to use h dependent also on dx dy.)
    CovExact=np.zeros([Nx,Ny]).astype(bool)
    CovExact[1:Nx-1,1:Ny-1]=(Cov==cov*dt)
    #returning transition probabilities indices and information of covariance approximation:    
    return (Po,PXl,PXr,PYl,PYr,PXYplus,PXYminus, IndXo, IndXl, IndXr, IndYo, IndYl, IndYr, CovExact)

##########################################################################################
######          Sample stochastic control problem classes                            #####
##########################################################################################

class SCP_uncertain_volatility:
    def drift(self,x,q):
        return self.interest*x
    def volatility(self,x,q):
        return x*q
    def reward(self,x,q):
        return 0
    def discount(self,x,q):
        return self.interest
    def __init__(self, r=0.05, sigma_min=0.3, sigma_max=0.5, extrem='max'):
        self.interest=r
        self.controls=np.array([sigma_min, sigma_max])
        self.extrem=extrem

class SCP_BS:
    def drift(self,x,q):
        return self.interest*x
    def volatility(self,x,q):
        return x*q
    def reward(self,x,q):
        return 0
    def discount(self,x,q):
        return self.interest
    def __init__(self, r=0.05, sigma=0.3, extrem='max'):
        self.interest=r
        self.controls=np.array([sigma])
        self.extrem=extrem

class TCBC_butterfly:
    def TC(self,X):
        V0=np.zeros(X.shape[0])
        for k in range(X.shape[0]):
                V0[k]=np.maximum(X[k]-self.K1,0)+np.maximum(X[k]-self.K2,0)-2*np.maximum(X[k]-(self.K1+self.K2)/2,0)
        return V0
    def BCL(self,x,t):
        return 0
    def BCR(self,x,t):
        return 0
    def __init__(self,K1=34,K2=46):
        self.K1=K1
        self.K2=K2

class SCP_uncertain_volatility_2D:
    def drift_x(self,x,y,q):
        return self.interest*x
    def drift_y(self,x,y,q):
        return self.interest*y
    def volatility_x(self,x,y,q):
        return x*q[0]
    def volatility_y(self,x,y,q):
        return y*q[1]
    def covariance(self,x,y,q):
        return x*y*q[0]*q[1]*q[2]
    def correlation(self,x,y,q):
        return q[2]
    def reward(self,x,y,q):
        return 0
    def discount(self,x,y,q):
        return self.interest
    def __init__(self,r=0.05,sigma_x_min=0.3,sigma_x_max=0.5,sigma_y_min=0.3,sigma_y_max=0.5,ro_min=0.3,ro_max=0.5,\
                 no_controls_x=3, no_controls_y=3, extrem='max'):
        self.interest=r
        Q1=np.linspace(sigma_x_min,sigma_x_max,no_controls_x)
        Q2=np.linspace(sigma_y_min,sigma_y_max,no_controls_y)
        Q3=np.array([ro_min,ro_max])
        Contr=[]
        for l in Q3:
            for j in Q1:
                Contr.append([j,Q2[0],l])
                Contr.append([j,Q2[-1],l])
            for k in Q2[1:-1]:
                Contr.append([Q1[0],k,l])
                Contr.append([Q1[-1],k,l])
        self.controls=np.array(Contr)
        self.extrem=extrem    

class SCP_BS_2D:
    extrem='max'
    interest=0.05
    controls=np.array([[0.3, 0.3, 0.3]])
    def drift_x(self,x,y,q):
        return 1*self.interest*x
    def drift_y(self,x,y,q):
        return 1*self.interest*y
    def volatility_x(self,x,y,q):
        return x*q[0]
    def volatility_y(self,x,y,q):
        return y*q[1]
    def covariance(self,x,y,q):
        return x*y*q[0]*q[1]*q[2]
    def correlation(self,x,y,q):
        return q[2]
    def reward(self,x,y,q):
        return 0
    def discount(self,x,y,q):
        return self.interest
    def __init__(self, r=0.05, sigma_x=0.3, sigma_y=0.3, ro=0.3, extrem='max'):
        self.interest=r
        self.controls=np.array([[sigma_x, sigma_y, ro]])
        self.extrem=extrem

class TCBC_butterfly_2D:
    def TC(self,X,Y):
        V0=np.zeros([X.shape[0],Y.shape[0]])
        for j in range(X.shape[0]):
            for k in range(Y.shape[0]):
                if self.typ=='max':
                    Sm=np.maximum(X[j],Y[k])
                else:
                    Sm=np.minimum(X[j],Y[k])
                V0[j,k]=np.maximum(Sm-self.K1,0)+np.maximum(Sm-self.K2,0)-2*np.maximum(Sm-(self.K1+self.K2)/2,0)
        return V0
    def BC(self,t_index, BC_args):
        if self.typ=='max':
            BC_down=BC_args[0]
            BC_left=BC_args[1]
            boundary=np.zeros([BC_down.shape[1],BC_left.shape[1]])            
            boundary[0,:]=BC_left[t_index,:]
            boundary[:,0]=BC_down[t_index,:]
        else:
            boundary=0
        return boundary
    def __init__(self,K1=34,K2=46,typ='max'):
        self.K1=K1
        self.K2=K2
        self.typ=typ
