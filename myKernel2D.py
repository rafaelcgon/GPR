
# GPy kernel
# implement a new kernel
#   1) implement the new covariance as a GPy.kern.src.kern.Kern object
#   2) update the GPy.kern.src file

import GPy
from GPy.kern import Kern
from GPy.core import Param
import numpy as np

############################################################################################
class divFreeK(Kern):
    def __init__(self,input_dim,active_dims=[0,1],var=1.,ly=1.,lx=1.):
        super(divFreeK, self).__init__(input_dim,active_dims, 'divFreeK')
        assert input_dim == 2, "For this kernel we assume input_dim=2"
        self.var = Param('var', var)
        self.var.constrain_positive()

        self.ly = Param('ly', ly)
        self.ly.constrain_positive()

        self.lx = Param('lx', lx)
        self.lx.constrain_positive()

        self.link_parameters(self.var,self.ly,self.lx)

    def parameters_changed(self):
        # nothing todo here
        pass

    def K(self,X,X2):
        if X2 is None: X2 = X
        p = 2 # number of dimensions   
        dy = X[:,0][:,None] - X2[:,0]    
        dx = X[:,1][:,None] - X2[:,1]    

        ly2 = np.square(self.ly) 
        lx2 = np.square(self.lx) 

        Byy = dy*dy/ly2
        Bxx = dx*dx/lx2

        expo = ( Byy + Bxx)/(-2.)
        C = self.var*np.exp(expo)
        # divergence free (df)
        By = (1 - Bxx)/lx2
        Bx = (1 - Byy)/ly2
        Byx = dy*dx/(ly2*lx2)

        A = np.concatenate([np.concatenate([By , Byx],axis=1),
                            np.concatenate([Byx , Bx],axis=1)],axis=0)
        C = np.concatenate([np.concatenate([C,C],axis=1),
                         np.concatenate([C,C],axis=1)],axis=0)
        return C*A 

    def Kdiag(self,X):
        return np.ones(X.shape[0]*2)*self.var

    def update_gradients_full(self, dL_dK, X, X2): # edit this###########3
        if X2 is None: X2 = X
        # variance gradient
        self.var.gradient = np.sum(self.K(X, X2)* dL_dK)/self.var
        # ly and lx terms
        ly2 = np.square(self.ly) 
        ly3 = self.ly * ly2
        lx2 = np.square(self.lx) 
        lx3 = self.lx * lx2
        dy = X[:,0][:,None] - X2[:,0]    
        dx = X[:,1][:,None] - X2[:,1]    
        Byy = (dy*dy)/ly2
        By = np.concatenate([np.concatenate([Byy,Byy],axis=1),
                         np.concatenate([Byy,Byy],axis=1)],axis=0)
        Bxx = (dx*dx)/lx2
        Bx = np.concatenate([np.concatenate([Bxx,Bxx],axis=1),
                         np.concatenate([Bxx,Bxx],axis=1)],axis=0)
        Byx = (dy*dx)/(lx2*ly2)
        expo = ( Byy + Bxx)/(-2.)
        C = self.var*np.exp(expo)
        C = np.concatenate([np.concatenate([C,C],axis=1),
                         np.concatenate([C,C],axis=1)],axis=0)
        # ly.gradient
        dA1  = Bxx*0
        dA12 = -2*Byx/(self.ly)
        dA2  = (4*Byy - 2)/ly3
        dA = np.concatenate([np.concatenate([dA1,dA12],axis=1),
                             np.concatenate([dA12,dA2],axis=1)],axis=0)
        self.ly.gradient = np.sum(((By/self.ly) * self.K(X,X2) + C*dA)*dL_dK)
        # lx.gradient
        dA1  = (4*Bxx - 2)/lx3
        dA12 = -2*Byx/(self.lx)
        dA2  = Bxx*0
        dA = np.concatenate([np.concatenate([dA1,dA12],axis=1),
                             np.concatenate([dA12,dA2],axis=1)],axis=0)
        self.lx.gradient = np.sum(((Bx/self.lx) * self.K(X,X2) + C*dA)*dL_dK)


    def update_gradients_diag(self, dL_dKdiag, X):
        pass


    def gradients_X(self,dL_dK,X,X2):
        pass
    def gradients_X_diag(self,dL_dKdiag,X):
        # no diagonal gradients
        pass
######################################################################################
class curlFreeK(Kern):
    def __init__(self,input_dim,active_dims=[0,1],var=1.,ly=1.,lx=1.):
        super(curlFreeK, self).__init__(input_dim,active_dims, 'curlFreeK')
        assert input_dim == 2, "For this kernel we assume input_dim=2"
        self.var = Param('var', var)
        self.var.constrain_positive()
     #   self.var.constrain_bounded(1e-06,1)

        self.ly = Param('ly', ly)
        self.ly.constrain_positive()

        self.lx = Param('lx', lx)
        self.lx.constrain_positive()

        self.link_parameters(self.var,self.ly,self.lx)

    def parameters_changed(self):
        # nothing todo here
        pass

    def K(self,X,X2):
        if X2 is None: X2 = X
        dy = X[:,0][:,None] - X2[:,0]
        dx = X[:,1][:,None] - X2[:,1]

        ly2 = np.square(self.ly)
        lx2 = np.square(self.lx)

        Byy = dy*dy/ly2
        Bxx = dx*dx/lx2

        expo = (Byy + Bxx)/(-2.)
        C = self.var*np.exp(expo)
        # curl-free 
        By = (1 - Byy)/ly2
        Bx = (1 - Bxx)/lx2
        Byx = (-1)*dy*dx/(ly2*lx2)

        A = np.concatenate([np.concatenate([By , Byx],axis=1),
                            np.concatenate([Byx , Bx],axis=1)],axis=0)
        C = np.concatenate([np.concatenate([C,C],axis=1),
                         np.concatenate([C,C],axis=1)],axis=0)
        return C*A


 
    def Kdiag(self,X):
        return np.ones(X.shape[0]*2)*self.var

    def update_gradients_full(self, dL_dK, X, X2): # edit this###########3
        if X2 is None: X2 = X
        # variance gradient
        self.var.gradient = np.sum(self.K(X, X2)* dL_dK)/self.var
        # ly and lx terms
        ly2 = np.square(self.ly)
        ly3 = self.ly * ly2
        lx2 = np.square(self.lx)
        lx3 = self.lx * lx2
        dy = X[:,0][:,None] - X2[:,0]
        dx = X[:,1][:,None] - X2[:,1]
        Byy = (dy*dy)/ly2
        By = np.concatenate([np.concatenate([Byy,Byy],axis=1),
                         np.concatenate([Byy,Byy],axis=1)],axis=0)
        Bxx = (dx*dx)/lx2
        Bx = np.concatenate([np.concatenate([Bxx,Bxx],axis=1),
                         np.concatenate([Bxx,Bxx],axis=1)],axis=0)
        Byx = (dy*dx)/(lx2*ly2)
        expo = (Byy + Bxx)/(-2.)
        C = self.var*np.exp(expo)
        C = np.concatenate([np.concatenate([C,C],axis=1),
                         np.concatenate([C,C],axis=1)],axis=0)
        # ly.gradient
        dA1  = (4*Byy - 2)/ly3
        dA2  = Bxx*0
        dA12 = 2*Byx/(self.ly)
        dA = np.concatenate([np.concatenate([dA1,dA12],axis=1),
                             np.concatenate([dA12,dA2],axis=1)],axis=0)
        self.ly.gradient = np.sum(((By/self.ly) * self.K(X,X2) + C*dA)*dL_dK)
        # lx.gradient
        dA1  = Bxx*0
        dA2  = (4*Bxx - 2)/lx3
        dA12 = 2*Byx/(self.lx)
        dA = np.concatenate([np.concatenate([dA1,dA12],axis=1),
                             np.concatenate([dA12,dA2],axis=1)],axis=0)
        self.lx.gradient = np.sum(((Bx/self.lx) * self.K(X,X2) + C*dA)*dL_dK)


    def update_gradients_diag(self, dL_dKdiag, X):
        pass


    def gradients_X(self,dL_dK,X,X2):
        pass

    def gradients_X_diag(self,dL_dKdiag,X):
        # no diagonal gradients
        pass


