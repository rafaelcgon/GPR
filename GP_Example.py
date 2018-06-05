import myKernel2D
import GPy
import numpy as np
from matplotlib import pylab as pl
import matplotlib.pyplot as plt               #
from matplotlib import rc,rcParams
import matplotlib as mpl 

def generate_spiral(Lx=5,Ly=5,dx=0.1,dy=0.1,lx=3., ly=3.,ratio=0.8):
	'''
	Generate the velocity field of a "spiral eddy".
	The velocity field is a sum of a non-divergent velocity (Und) to a non-rotational velocity (Unr):

		U = ratio*Und  + (1-ratio)*Unr

	lx and ly are decaying scales. 
	'''
	x = np.arange(-1*Lx,Lx+dx,dx)
	y = np.arange(-1*Ly,Ly+dy,dy)
	A = 1.5
	X,Y = np.meshgrid(x,y)
	phi = A*np.exp(-(X**2)/lx - (Y**2)/ly)
	dpdy,dpdx = np.gradient(phi,dx,axis=[0,1])

 # non-divergent velocity components
	u1 = dpdy
	v1 = -dpdx
#non- rotational velocity components
	u2 = dpdx
	v2 = dpdy

	u = u1*ratio + (1-ratio)*u2
	v = v1*ratio + (1-ratio)*v2

	return x,y,phi,u,v

########

def main(nsamples = 15,ratio=0.8):
	'''
	nsamples: number of observations, which are randomly picked from the velocity field.
	'''

# get velocity field:
	x,y,phi,u,v = generate_spiral(ratio=ratio)

	X,Y = np.meshgrid(x,y)
        X = X.reshape([X.size,1])
        Y = Y.reshape([Y.size,1])
        GridPoints = np.concatenate([Y,X],axis=1)

	U = u.reshape([u.size,1])
	V = v.reshape([v.size,1])

        # get observations 
	ii = np.random.randint(0,u.size,nsamples)
	obs = np.concatenate([V[ii,0][:,None],U[ii,0][:,None]],axis=1)
	Xo = np.concatenate([Y[ii,0][:,None],X[ii,0][:,None]],axis=1)


# Scalar analysis: treating the velocity components u and v as independent scalars

        # get covariance function for GPR
        # here I chose the squared exponential (RBF), but there are other options 
	k = GPy.kern.RBF(input_dim = 2,ARD=True) 

	# model for velocity component u: 
	model_u = GPy.models.GPRegression(Xo,obs[:,1][:,None],k.copy())
        # optimize hyper-parameters
	model_u.optimize_restarts(num_restarts=10)
        # regression
	Up,Kpu = model_u.predict(GridPoints)
#       Up is the posterior mean and Kpu is the posterior covariance
	up = np.reshape(Up,[y.size,x.size])

        # model for velocity component v
	model_v = GPy.models.GPRegression(Xo,obs[:,0][:,None],k.copy())
	model_v.optimize_restarts(num_restarts=10)
	Vp,Kpv = model_v.predict(GridPoints)
	vp = np.reshape(Vp,[y.size,x.size])

# vector analysis: consider cross -correlations between u and v

	knd = myKernel2D.divFreeK(input_dim=2, active_dims=[0,1]) # impose non-divergence constraint
	knr = myKernel2D.curlFreeK(input_dim=2, active_dims=[0,1])# impose non-rotation constraint

        obs2 =  np.concatenate([V[ii,0][:,None],U[ii,0][:,None]],axis=0) # observations of u and v must be in the same "observation vector".
     
        model = GPy.models.GPRegression(Xo,obs2,knd.copy()+knr.copy())
        # just one model for both components. The covariance function is composed by a 
        # non-divergent part (knd) and a non-rotating part (knr).          
	model.optimize_restarts(num_restarts=10)
	VU,Kvu = model.predict(GridPoints)
	vp2 = np.reshape(VU[:VU.size/2,0],[y.size,x.size])
	up2 = np.reshape(VU[VU.size/2:,0],[y.size,x.size])

	ds = 2
	figW = 12.
	figH = 4.
	fig = pl.figure(figsize=(figW,figH))

	plot = fig.add_subplot(1,3,1,aspect = 'equal') 
	plot.quiver(x[::ds],y[::ds],u[::ds,::ds],v[::ds,::ds],scale=10)
	plot.streamplot(x,y,u,v)
	plot.plot(Xo[:,1],Xo[:,0],'og')
	plot.set_title('Original velocity')

	plot = fig.add_subplot(1,3,2,aspect = 'equal') 
	plot.quiver(x[::ds],y[::ds],up[::ds,::ds],vp[::ds,::ds],scale=10)
	plot.streamplot(x,y,up,vp)
	plot.plot(Xo[:,1],Xo[:,0],'og')
	plot.set_title('Scalar Regression')

	plot = fig.add_subplot(1,3,3,aspect = 'equal') 
	plot.quiver(x[::ds],y[::ds],up2[::ds,::ds],vp2[::ds,::ds],scale=10)
	plot.streamplot(x,y,up2,vp2)
	plot.plot(Xo[:,1],Xo[:,0],'og')
	plot.set_title('Scalar Regression')

        return model_u,model_v,model

