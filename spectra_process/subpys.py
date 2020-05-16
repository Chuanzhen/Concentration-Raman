# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 10:06:11 2019

@author: Chuanzhen Hu
"""
import sys
import struct
import scipy as sp
import scipy.linalg as splin
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import splu
import scipy.optimize as optimize
# In[] define function for polynomial backgrounds
def myploy(pow_val, num_of_pixels):
    S = np.arange(1., 2., 1./(num_of_pixels-1)).reshape((1, num_of_pixels-1))
    S = np.append(S, 2*np.ones((S.shape[0], 1)))
    S = np.tile(S, [pow_val+1, 1])
    for ij in range(pow_val+1):
        S[ij, :] = np.power(S[ij, :], ij)
        
    return S

# In[] define function for spectra smoothing
def speyediff(N, d, format='csc'):
    """
    (utility function)
    Construct a d-th order sparse difference matrix based on 
    an initial N x N identity matrix
    
    Final matrix (N-d) x N
    """
    
    assert not (d < 0), "d must be non negative"
    shape     = (N-d, N)
    diagonals = np.zeros(2*d + 1)
    diagonals[d] = 1.
    for i in range(d):
        diff = diagonals[:-1] - diagonals[1:]
        diagonals = diff
    offsets = np.arange(d+1)
    spmat = sparse.diags(diagonals, offsets, shape, format=format)
    return spmat


def whittaker_smooth(spectra, lmbda, d):
    """
    Implementation of the Whittaker smoothing algorithm,
    based on the work by Eilers [1].
    [1] P. H. C. Eilers, "A perfect smoother", Anal. Chem. 2003, (75), 3631-3636
    
    The larger 'lmbd', the smoother the data.
    For smoothing of a complete data series, sampled at equal intervals
    This implementation uses sparse matrices enabling high-speed processing
    of large input vectors
    
    ---------
    
    Arguments :
    
    spectra  : vector containing raw data, column vector
    lmbda    : parameter for the smoothing algorithm (roughness penalty)
    d        : order of the smoothing 
    
    ---------
    Returns :
    
    z       : vector of the smoothed data.
    """
    if spectra.shape[0] == 1 and spectra.shape[1] != 1:
        spectra = np.transpose(spectra)
        flag = 1
    else:
        flag = 0
    
    z = np.zeros(spectra.shape)
    for ij in range(spectra.shape[1]):
        y = spectra[:, ij]
        m = len(y)
        E = sparse.eye(m, format='csc')
        D = speyediff(m, d, format='csc')
        coefmat = E + lmbda * D.conj().T.dot(D)
        z[:, ij] = splu(coefmat).solve(y)
    
    if flag == 1:
        z = np.transpose(z)
    return z    

# In[]  Python implementation of NNLS algorithm, C:one column, d: one row
#def lsqnonneg(C, d):
#    '''Linear least squares with nonnegativity constraints.
#    (x, resnorm, residual) = lsqnonneg(C,d) returns the vector x that minimizes norm(d-C*x)
#    subject to x >= 0, C and d must be real
#    C: column vectore, nxm, d: n, list
#    '''
#    eps = 2.22e-16    # from matlab
#
#    tol = 10*eps*np.linalg.norm(C,1)*(max(C.shape)+1)
#
#    C = np.asarray(C)
#
#    (m,n) = C.shape
#    P = []
#    R = [x for x in range(0, n)]
#
#    x = np.zeros(n)
#
#    resid = d - np.dot(C, x)
#    w = np.dot(C.T, resid)
#
#    # outer loop to put variables into set to hold positive coefficients
#    while np.any(R) and np.max(w) > tol:
#
#        j = np.argmax(w)
#        P.append(j)
#        R.remove(j)
#
#        AP = np.zeros(C.shape)
#        AP[:,P] = C[:,P]
#
#        s=np.dot(np.linalg.pinv(AP), d)
#
#        s[R] = 0
#     
#        while np.min(s) < 0:
#
#            i = [i for i in P if s[i] <= 0]
#
#            alpha = min(x[i]/(x[i] - s[i]))
#            x = x + alpha*(s-x)
#
#            j = [j for j in P if x[j] == 0]
#            if j:
#                R.append(*j)
#                P.remove(j)
#            
#            AP = np.zeros(C.shape)
#            AP[:,P] = C[:,P]
#            s=np.dot(np.linalg.pinv(AP), d)
#            s[R] = 0
#     
#        x = s
#        resid = d - np.dot(C, x)
#
#        w = np.dot(C.T, resid)
#
#    return (x, sum(resid * resid), resid)
# In[] Asymmetric least square
# function [ x ] = asls( A, b, p )
# Asymmetric least squares.  Algorithm finds coefficients x of a linear
# model, A*x = b, where b is the measurement vector and A is a matrix of
# pure components.  Starting from an initial, least-squares estimate of x,
# the program finds a residual vector r = b-A*x. A merit function f is
# constructed, with f = sum(w.*r.^2), where w is a weighting vector that
# has values of p for positive residuals and 1-p for negative residuals.
# This asymmetric penalty function helps to ensure positive-going
# residuals.  p is a critical parameter and must be set by the user.  See,
# for example, Journal of Chromatography A, 1057 (2004) 21-30.
#
# INPUTS:   A   =   matrix of pure components.
#                   [m x n]
#           b   =   measurement vector.
#                   [1xn]
#           p   =   penalty term (0 < p < 1)
#                   [1 x 1]
#           x0  =   initial guess of x0 (like from a standard LS fit
#                   [n x 1]
#           nn  =   use nonnegative ls for initial guess
#                   [1 x 1] boolean
#           bounds= limits to estimates of x (to prevent ridiculous values)
#                   [n x 2] (in the style of [lower, upper]
#
# OUTPUTS:  x   =   coefficient vector
#                   [n x 1]    


#################
def asls(A, b, p):
    [x0, residual] = optimize.nnls(A.T, b[0, :])
    bounds = [-1]
#    bounds = [0, 1e3]
    def aslsres(x):
        r = b - np.matmul(A.T, x)
        w = np.zeros(r.shape)
        w[r >= 0] = p
        w[r < 0] = 1-p
        # calculate residual
        f = np.sum(w*r**2)
        if len(bounds) == 2:
            if np.any(x < bounds[0]) or np.any(x > bounds[1]):
                f = 9e99
                print('9e99\n')
        return f
    opt =  optimize.minimize(aslsres, x0, options={'maxiter': 100000})
    return opt.x   

# In[] VCA
#############################################
# Internal functions
#############################################

def estimate_snr(Y,r_m,x):

  [L, N] = Y.shape           # L number of bands (channels), N number of pixels
  [p, N] = x.shape           # p number of endmembers (reduced dimension)
  
  P_y     = sp.sum(Y**2)/float(N)
  P_x     = sp.sum(x**2)/float(N) + sp.sum(r_m**2)
  snr_est = 10*sp.log10( (P_x - p/L*P_y)/(P_y - P_x) )

  return snr_est



def vca(Y,R,verbose = True,snr_input = 0):
# Vertex Component Analysis
#
# Ae, indice, Yp = vca(Y,R,verbose = True,snr_input = 0)
#
# ------- Input variables -------------
#  Y - matrix with dimensions L(channels) x N(pixels)
#      each pixel is a linear mixture of R endmembers
#      signatures Y = M x s, where s = gamma x alfa
#      gamma is a illumination perturbation factor and
#      alfa are the abundance fractions of each endmember.
#  R - positive integer number of endmembers in the scene
#
# ------- Output variables -----------
# Ae     - estimated mixing matrix (endmembers signatures)
# indice - pixels that were chosen to be the most pure
# Yp     - Data matrix Y projected.   
#
# ------- Optional parameters---------
# snr_input - (float) signal to noise ratio (dB)
# v         - [True | False]
# ------------------------------------
#
# Author: Adrien Lagrange (adrien.lagrange@enseeiht.fr)
# This code is a translation of a matlab code provided by 
# Jose Nascimento (zen@isel.pt) and Jose Bioucas Dias (bioucas@lx.it.pt)
# available at http://www.lx.it.pt/~bioucas/code.htm under a non-specified Copyright (c)
# Translation of last version at 22-February-2018 (Matlab version 2.1 (7-May-2004))
#
# more details on:
# Jose M. P. Nascimento and Jose M. B. Dias 
# "Vertex Component Analysis: A Fast Algorithm to Unmix Hyperspectral Data"
# submited to IEEE Trans. Geosci. Remote Sensing, vol. .., no. .., pp. .-., 2004
# 
# 

  #############################################
  # Initializations
  #############################################
  if len(Y.shape)!=2:
    sys.exit('Input data must be of size L (number of bands i.e. channels) by N (number of pixels)')

  [L, N]=Y.shape   # L number of bands (channels), N number of pixels
       
  R = int(R)
  if (R<0 or R>L):  
    sys.exit('ENDMEMBER parameter must be integer between 1 and L')
        
  #############################################
  # SNR Estimates
  #############################################

  if snr_input==0:
    y_m = sp.mean(Y,axis=1,keepdims=True)
    Y_o = Y - y_m           # data with zero-mean
    Ud  = splin.svd(sp.dot(Y_o,Y_o.T)/float(N))[0][:,:R]  # computes the R-projection matrix 
    x_p = sp.dot(Ud.T, Y_o)                 # project the zero-mean data onto p-subspace

    SNR = estimate_snr(Y,y_m,x_p);
    
    if verbose:
      print("SNR estimated = {}[dB]".format(SNR))
  else:
    SNR = snr_input
    if verbose:
      print("input SNR = {}[dB]\n".format(SNR))

  SNR_th = 15 + 10*sp.log10(R)
         
  #############################################
  # Choosing Projective Projection or 
  #          projection to p-1 subspace
  #############################################

  if SNR < SNR_th:
    if verbose:
      print("... Select proj. to R-1")
                
      d = R-1
      if snr_input==0: # it means that the projection is already computed
        Ud = Ud[:,:d]
      else:
        y_m = sp.mean(Y,axis=1,keepdims=True)
        Y_o = Y - y_m  # data with zero-mean 
         
        Ud  = splin.svd(sp.dot(Y_o,Y_o.T)/float(N))[0][:,:d]  # computes the p-projection matrix 
        x_p =  sp.dot(Ud.T,Y_o)                 # project thezeros mean data onto p-subspace
                
      Yp =  sp.dot(Ud,x_p[:d,:]) + y_m      # again in dimension L
                
      x = x_p[:d,:] #  x_p =  Ud.T * Y_o is on a R-dim subspace
      c = sp.amax(sp.sum(x**2,axis=0))**0.5
      y = sp.vstack(( x, c*sp.ones((1,N)) ))
  else:
    if verbose:
      print("... Select the projective proj.")
             
    d = R
    Ud  = splin.svd(sp.dot(Y,Y.T)/float(N))[0][:,:d] # computes the p-projection matrix 
                
    x_p = sp.dot(Ud.T,Y)
    Yp =  sp.dot(Ud,x_p[:d,:])      # again in dimension L (note that x_p has no null mean)
                
    x =  sp.dot(Ud.T,Y)
    u = sp.mean(x,axis=1,keepdims=True)        #equivalent to  u = Ud.T * r_m
    y =  x / sp.dot(u.T,x)

 
  #############################################
  # VCA algorithm
  #############################################

  indice = sp.zeros((R),dtype=int)
  A = sp.zeros((R,R))
  A[-1,0] = 1

  for i in range(R):
    w = sp.random.rand(R,1);   
    f = w - sp.dot(A,sp.dot(splin.pinv(A),w))
    f = f / splin.norm(f)
      
    v = sp.dot(f.T,y)

    indice[i] = sp.argmax(sp.absolute(v))
    A[:,i] = y[:,indice[i]]        # same as x(:,indice(i))

  Ae = Yp[:,indice]

  return Ae,indice,Yp

# In[] Read the SPE file and load it's content into RAM
def spe_reader(path):
	# Read the SPE file and load it's content into RAM
	f=open(path,"rb")
	var={"filename":path}
	t={"exp":('f',10,4),"nx":('H',42,2),"ny":('H',656,2),"nframes":('I',1446,4),"datatype":('H',108,2),"date":('10s',20,10),"time":('6s',172,6),"usercomment1":('80s',200,80),"usercomment2":('80s',280,80),"usercomment3":('80s',360,80),"usercomment4":('80s',440,80),"usercomment5":('80s',520,80),'cw':('f',72,4)}
	for x in t:
		f.seek(t[x][1])
		var[x]=struct.unpack(t[x][0],f.read(t[x][2]))[0]
#	print(x,var[x])
	length=var['nx']*var['ny']*var['nframes']
	f.seek(4100)
	if var['datatype']==0:
		rawdata=np.array(struct.unpack("%if"%(length),f.read(length*4)))
	elif var['datatype']==1:
		rawdata=np.array(struct.unpack("%ii"%(length),f.read(length*4)))
	elif var['datatype']==2:
		rawdata=np.array(struct.unpack("%iH"%(length),f.read(length*4)))
	elif var['datatype']==3:
		rawdata=np.array(struct.unpack("%ih"%(length),f.read(length*4)))
        
	rawdata=np.reshape(rawdata,(var['nframes'],var['ny'],var['nx']))
    
	x0 = var['nx']/2+0.5
	x=np.arange(var['nx'])
	if var['cw']==600.0:
		a2=-1.54334e-5
		a1=0.382039
		wavelength=var['cw']+a1*(x-x0)+a2*(x-x0)**2
	elif var['cw']==700.0:
		a2=-1.58478e-5
		a1=0.378275
		wavelength=var['cw']+a1*(x-x0)+a2*(x-x0)**2
	elif var['cw']==800.0:
		a2=-1.62586e-5
		a1=0.374399
		wavelength=var['cw']+a1*(x-x0)+a2*(x-x0)**2
	elif var['cw']==900.0:
		a2=-1.66658e-5
		a1=0.37041
		wavelength=var['cw']+a1*(x-x0)+a2*(x-x0)**2
	else:
		print("pixel -> wavelength function unknown! Saving pixel value for x")
		wavelength=x
	
	return {"var":var,"wavelength":wavelength,"rawdata":rawdata}
# In[]