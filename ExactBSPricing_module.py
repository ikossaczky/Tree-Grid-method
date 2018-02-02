import numpy as np
import rpy2

from rpy2.robjects.packages import importr
exo = importr("fExoticOptions")
bas = importr("base")

def ExactBSButterflyPrice(X,Y, r=0.05, sigma_x=0.3, sigma_y=0.3, ro=0.3, extrem='max', T=0.25, K1=34, K2=46):
    V=np.zeros([X.shape[0],Y.shape[0]])
    for j in range(X.shape[0]):
        for k in range(Y.shape[0]):
            s1=float(X[j])
            s2=float(Y[k])
            ext='c'+extrem
            KK=[K1,(K1+K2)/2,K2] #Strikes
            OO=[0,0,0]
            for K in range(len(KK)):
                calloption=exo.TwoRiskyAssetsOption('cmax', S1=s1, S2=s2, X=KK[K], Time=T, r=r,\
                                            b1=r, b2=r, sigma1=sigma_x, sigma2=sigma_y, rho=ro)
                OO[K]=bas.attr(calloption,"price")[0]
            V[j,k]=OO[0]-2*OO[1]+OO[2]
    return V
