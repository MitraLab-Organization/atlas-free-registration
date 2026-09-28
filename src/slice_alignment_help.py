import matplotlib
matplotlib.use("Agg")

import torch
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from os.path import join,basename,splitext
from scipy.interpolate import interpn

try:
    from IPython.display import display
except ImportError:
    class _NullDisplay:
        def update(self, *args, **kwargs):
            pass

    def display(*args, **kwargs):
        return _NullDisplay()

def atlas_from_aligned_slices_and_weights(xI,I,dtype,device,phiiRiJ,W,asquare,niter=10,draw=0,fig=None,hfig=None,anisotropy_factor=4**2,return_K=False,return_fwhm=False):
    """
    TODO - Function Description
    
    Parameters:
    ===========
    xI : list[torch.Tensor[int]]
        List of 3 orthogonal axes used to define voxel locations
    I : torch.Tensor
        TODO
    dtype : str
        TODO
    device : str
        TODO
    phiiRiJ : torch.Tensor
        TODO
    W : torch.Tensor
        TODO
    asquare : float
        TODO
    niter : int
        TODO
    draw : bool
        TODO
    fig : matplotlib.pyplot.figure
        TODO
    hfig :  matplotlib.pyplot.figure
        TODO
    anisotropy_factor : float
        TODO
    return_K : bool
        TODO
    return_fwhm : bool
        TODO


    Returns:
    ========
    I : torch.Tensor
        TODO
    E : float
        TODO
    ER : float
        TODO
    """
    
    if draw:
        if fig is None:
            fig = plt.figure()
        if hfig is None:
            hfig = display(fig,display_id=True)
    
    # we don't want to propagate gradients here
    phiiRiJ = phiiRiJ.clone().detach()
    I = I.clone().detach()    
    W = W.clone().detach()
    
    
    
    # normalize W
    Wm = torch.max(W)
    Wn = W/Wm

     # make a Fourier domain
    n = len(xI[0])
    dI = torch.stack([x[1] - x[0] for x in xI])
    DI = torch.prod(dI)
    #d = dI[0]
    # we use the convention that slices are one unit apart
    d=1
    #f = torch.arange(n,dtype=dtype,device=device)/n/d
    
    # try here to blur in 3D
    f = [torch.arange(ni,dtype=dtype,device=device)/ni/di for ni,di in zip(I.shape[1:],dI)]
    F = torch.stack(torch.meshgrid(*f,indexing='ij'),-1)
    
    
    # define the operator in the fourier domain
    # LL plus identity
    #L = (2.0*asquare* (1-torch.cos(2.0*np.pi*f*d))/d**2)
    asquare3d = torch.tensor([asquare,asquare*anisotropy_factor,asquare*anisotropy_factor],device=device,dtype=dtype)
    L = 2.0* torch.sum( asquare3d*(1-torch.cos(2.0*np.pi*F*dI))/dI**2,-1)
    LL = L**2
    LLnorm=LL/Wm
    oooperator = 1.0 / (1.0 + LLnorm)   
    # for FWHM computation, let's take the inverse fourier transform
    if return_fwhm or return_K:
        K = torch.fft.ifftn(oooperator).real

    Esave = []

            
    ERsave = []
    EMsave = []
    Esave = []
    for it in range(niter):
        # value of the loss
        #ER = torch.sum( torch.fft.ifftn( torch.fft.fftn(I,dim=1)*L[...,None,None] , dim=1).real**2 )*DI
        ER = torch.sum( torch.fft.ifftn( torch.fft.fftn(I,dim=(1,2,3))*L , dim=(1,2,3)).real**2 )*DI
        #print('Ishape is',I.shape)
        #print('phiiRiJ shape is',phiiRiJ.shape)
        #print('W shape is',W.shape)
        #EMM = torch.sum(  (I - phiiRiJ)**2 * W )/2*DI*Wm
        #EM = torch.sum(   ((I-phiiRiJ)**2/(c + (I-phiiRiJ)**2))   )*DI # middle term of 2.12
        EM = torch.sum( (I-phiiRiJ)**2*W)*DI # daniel changed this
        E = ER + EM
        Esave.append(E.item())
        ERsave.append(ER.item())
        EMsave.append(EM.item())
        
        # update
        toblur = ( phiiRiJ*Wn + I*(1-Wn) )
        blurred = torch.fft.ifftn(torch.fft.fftn(toblur,dim=(1,2,3))*oooperator,dim=(1,2,3)).real
        I = blurred
        
        if draw and (not it%draw or it == niter-1):
            fig.clf()
            ax = fig.add_subplot(2,3,1)
            ax.imshow(I[:,I.shape[1]//2].permute(1,2,0))
            ax = fig.add_subplot(2,3,2)
            ax.imshow(I[:,:,I.shape[2]//2].permute(1,2,0),interpolation='none',aspect='auto')
            ax = fig.add_subplot(2,3,3)
            ax.imshow(I[:,:,:,I.shape[3]//2].permute(1,2,0),interpolation='none',aspect='auto')
            
            ax = fig.add_subplot(2,3,4)
            ax.imshow(phiiRiJ[:,I.shape[1]//2].permute(1,2,0))
            ax = fig.add_subplot(2,3,5)
            ax.imshow(phiiRiJ[:,:,I.shape[2]//2].permute(1,2,0),interpolation='none',aspect='auto')
            ax = fig.add_subplot(2,3,6)
            ax.imshow(phiiRiJ[:,:,:,I.shape[3]//2].permute(1,2,0),interpolation='none',aspect='auto')
            # hfig.update(fig)
        
    if return_K:
        return K
    elif return_fwhm:

        half_width_half_max = np.where(K[:,0,0].numpy()<0.5*K[0,0,0].numpy())[0][0]
        full_width_half_max_0 = 2*half_width_half_max+1
        half_width_half_max = np.where(K[0,:,0].numpy()<0.5*K[0,0,0].numpy())[0][0]
        full_width_half_max_1 = 2*half_width_half_max+1
        half_width_half_max = np.where(K[0,0,:].numpy()<0.5*K[0,0,0].numpy())[0][0]
        full_width_half_max_2 = 2*half_width_half_max+1
    
        return full_width_half_max_0,full_width_half_max_1,full_width_half_max_2
    else: # normal returns
        return I,E.item(),ER.item()

def _atlas_from_aligned_slices_and_weights(xI,I,phiiRiJ,W,asquare,niter=10,draw=0,fig=None,hfig=None):
    """
    TODO - Function Description
    
    Parameters:
    ===========
    xI : list[torch.Tensor[int]]
        List of 3 orthogonal axes used to define voxel locations
    I : torch.Tensor
        TODO
    phiiRiJ : torch.Tensor
        TODO
    W : torch.Tensor
        TODO
    asquare : float
        TODO
    niter : int
        TODO
    draw : bool
        TODO
    fig : matplotlib.pyplot.figure
        TODO
    hfig :  matplotlib.pyplot.figure
        TODO

    Returns:
    ========
    I : torch.Tensor
        TODO
    E : float
        TODO
    ER : float
        TODO
    """
    
    if draw:
        if fig is None:
            fig = plt.figure()
        if hfig is None:
            hfig = display(fig,display_id=True)
    
    # we don't want to propagate gradients here
    phiiRiJ = phiiRiJ.clone().detach()
    I = I.clone().detach()    
    W = W.clone().detach()
    
    # normalize W
    Wm = torch.max(W)
    Wn = W/Wm

     # make a Fourier domain
    n = len(xI[0])
    dI = torch.stack([x[1] - x[0] for x in xI])
    DI = torch.prod(dI)
    #d = dI[0]
    # we use the convention that slices are one unit apart
    d=1
    f = torch.arange(n,dtype=dtype,device=device)/n/d
  
    # define the operator in the fourier domain
    # LL plus identity
    L = (2.0*asquare* (1-torch.cos(2.0*np.pi*f*d))/d**2)
    LL = L**2
    LLnorm=LL/Wm
    oooperator = 1.0 / (1.0 + LLnorm)   
    Esave = []

            
    ERsave = []
    EMsave = []
    Esave = []
    for it in range(niter):
        # value of the loss
        ER = torch.sum( torch.fft.ifftn( torch.fft.fftn(I,dim=1)*L[...,None,None] , dim=1).real**2 )*DI
        #print('Ishape is',I.shape)
        #print('phiiRiJ shape is',phiiRiJ.shape)
        #print('W shape is',W.shape)
        #EMM = torch.sum(  (I - phiiRiJ)**2 * W )/2*DI*Wm
        #EM = torch.sum(   ((I-phiiRiJ)**2/(c + (I-phiiRiJ)**2))   )*DI # middle term of 2.12
        EM = torch.sum( (I-phiiRiJ)**2*W)*DI # daniel changed this
        E = ER + EM
        Esave.append(E.item())
        ERsave.append(ER.item())
        EMsave.append(EM.item())
        
        # update
        toblur = ( phiiRiJ*Wn + I*(1-Wn) )
        blurred = torch.fft.ifftn(torch.fft.fftn(toblur,dim=1)*oooperator[...,None,None],dim=1).real
        I = blurred
        
        if draw and (not it%draw or it == niter-1):
            fig.clf()
            ax = fig.add_subplot(2,3,1)
            ax.imshow(I[:,I.shape[1]//2].permute(1,2,0))
            ax = fig.add_subplot(2,3,2)
            ax.imshow(I[:,:,I.shape[2]//2].permute(1,2,0),interpolation='none',aspect='auto')
            ax = fig.add_subplot(2,3,3)
            ax.imshow(I[:,:,:,I.shape[3]//2].permute(1,2,0),interpolation='none',aspect='auto')
            
            ax = fig.add_subplot(2,3,4)
            ax.imshow(phiiRiJ[:,I.shape[1]//2].permute(1,2,0))
            ax = fig.add_subplot(2,3,5)
            ax.imshow(phiiRiJ[:,:,I.shape[2]//2].permute(1,2,0),interpolation='none',aspect='auto')
            ax = fig.add_subplot(2,3,6)
            ax.imshow(phiiRiJ[:,:,:,I.shape[3]//2].permute(1,2,0),interpolation='none',aspect='auto')
            hfig.update(fig)
        
    
    return I,E.item(),ER.item()

def AX(Ai,XJ):    
    """
    TODO - Function description

    Parameters:
    ===========
    Ai : torch.Tensor[torch.float32]
        TODO
    XJ : torch.Tensor[torch.float32]
        TODO

    Returns:
    ========
    Xs : torch.Tensor[torch.float32]
        TODO
    """
    Xs = (Ai[:,None,None,:3,:3]@XJ[...,None])[...,0] + Ai[:,None,None,:3,-1]
    return Xs


def A2DtoA3D(A):
    """
    TODO - Function description

    Parameters:
    ===========
    A : torch.Tensor[torch.float32]
        TODO

    Returns:
    ========
    out : torch.Tensor[torch.float32]
        TODO
    """
    row = torch.concatenate( (torch.ones_like(A[:,0,0,None,None]) , torch.zeros_like(A[:,0,None])  ),-1)
    col = torch.zeros_like(A[:,:,0,None])
    colA = torch.concatenate([col,A],-1)
    out = torch.concatenate([row,colA],-2)
    
    return out

def detjac(xv,v):
    """
    TODO - Function description

    Parameters:
    ===========
    xv : list[torch.Tensor[torch.float32]]
        TODO
    v : torch.Tensor[torch.float32]
        TODO

    Returns:
    ========
    detjac : torch.Tensor[torch.float32]
        TODO
    """
    dv = [(x[1] - x[0]).item() for x in xv]    
    detjac = torch.linalg.det( torch.stack(torch.gradient( exp(xv,v2DToV3D(v))[...,1:] , dim=(1,2),spacing=(dv[1],dv[2])),-1) )
    return detjac

def down2ax(I,ax):
    """
    TODO - Function description

    Parameters:
    ===========
    I : nd-array
        TODO
    ax : matplotlib.pyplot.axis
        TODO

    Returns:
    ========
    Id : nd-array
        TODO
    """
    ndims = I.ndim
    s0 = [slice(None) for i in range(ndims)]
    s1 = [slice(None) for i in range(ndims)]
    n = I.shape[ax]
    nd = n//2
    
    s0[ax] = slice(0,nd*2,2)
    s1[ax] = slice(1,nd*2,2)
    Id = I[tuple(s0)]*0.5 + I[tuple(s1)]*0.5
    return Id

def exp(x,v,N=5):
    '''
    Group exponential by scaling and squaring
    v should have xyz components at the end and be 3d
    
    Take a small displacement, and compose it with itself 
    many times, to give a big displacement.
    
    If N = 1, the output is id + v
    
    If N = 2, the output is (id + v/2)\circ (id + v/2)
    
    If N = 3, the output is (id + v/8)\circ ... \circ (id + v/8)

    Parameters:
    ===========
    x : list[torch.Tensor[torch.float32]]
        TODO
    v : torch.Tensor[torch.float32]
        TODO
    N : int
        TODO

    Returns:
    ========
    phi : torch.Tensor[torch.float32]
        TODO
    '''
    X = torch.stack(torch.meshgrid(*x,indexing='ij'),-1)
    phi = X.clone() + v / 2**N
    for i in range(N-1):
        # when interpolating, move xyz component to the beginning, then back
        # 0 boundary conditions are probably ok
        phi = interp(x,(phi-X).permute(-1,0,1,2),phi).permute(1,2,3,0) + phi
    
    return phi

def extent_from_x(x):
    """
    This function gets a coordinate system and output the extent argument for matplotlib imshow

    Parameters:
    ===========
    x : arr[int]
        An N-dimensional coordinate system

    Returns:
    ========
    out : arr
        The extent of x to be used as an input for matplotlib.pyplot.imshow()
    """
    d = [xi[1]-xi[0] for xi in x]
    return (x[1][0]-d[1]/2, x[1][-1]+d[1]/2, x[0][-1]+d[0]/2, x[0][0]-d[0]/2)


def interp(x,I,phii,**kwargs):
    """
    Interpolate a signal with specified voxel spacing, in torch. Note, the input data should be 3D images, with first dimension a channel. phii has xyz on the last dimension.

    Parameters:
    ===========
    x : list[torch.Tensor[torch.float32]]
        TODO
    I : torch.Tensor[torch.float32]
        TODO
    phii : torch.Tensor[torch.float32]
        TODO
    kwargs : dict
        TODO

    Returns:
    ========
    phiI : torch.Tensor[torch.float32]
        TODO
    """
    # first we center phii based on x
    x0 = torch.stack([xi[0] for xi in x])
    d = torch.stack([xi[-1] - xi[0] for xi in x]) # this will fail if there's only one slice
    phii = (phii - x0)/d*2-1
    
    if 'align_corners' not in kwargs:
        kwargs['align_corners'] = True
    
    phiI = torch.nn.functional.grid_sample(I[None],phii.flip(-1)[None],**kwargs)[0]
    return phiI

def inverse_transform_image(xI,I,xv,v,A,xJ,**kwargs):
    """
    Note this is redundant with the below, but is a nice helper function
    
    Parameters:
    ===========
    xI : list[torch.Tensor[torch.float32]]
        TODO
    I : torch.Tensor[torch.float32]
        TODO
    xv : list[torch.Tensor[torch.float32]]
        TODO
    v : torch.Tensor[torch.float32]
        TODO
    A : torch.Tensor[torch.float32]
        TODO
    xJ : list[torch.Tensor[torch.float32]]
        TODO
    kwargs : dict
        TODO

    Returns:
    ========
    phiiAiI : torch.Tensor[torch.float32]
        TODO
    """
    XJ = torch.stack(torch.meshgrid(*xJ,indexing='ij'),-1)
    XV = torch.stack(torch.meshgrid(*xv,indexing='ij'),-1)
    
    
    A = A2DtoA3D(A)
    

    # convert v to phii
    phi = exp(xv,v2DToV3D(v))
    
    # sample on xj
    phis = interp(xv,(phi-XV).permute(-1,0,1,2),XJ).permute(1,2,3,0) + XJ
    
    A = A.to(dtype=phis.dtype, device=phis.device)
    Xs = AX(A,phis)
        
    # transform the image
    if 'padding_mode' not in kwargs:
        kwargs['padding_mode'] = 'border'
    phiiAiI = interp(xI,I,Xs,**kwargs)

    return phiiAiI

def L_from_xv_a_p(xv,a,p):
    """
    TODO - Function description

    Parameters:
    ===========
    xv : list[torch.Tensor[torch.float32]]
        TODO
    a : float
        TODO
    p : float
        TODO

    Returns:
    ========
    L : TODO
        TODO
    """
    dv = xv[-1][1] - xv[-1][0]
    fv = [torch.arange(n)/n/dv for n in (len(xv[-2]), len(xv[-1]))]
    FV = torch.stack(torch.meshgrid(fv,indexing='ij'),-1)
    L = (1.0 - torch.sum(2.0*a**2*(torch.cos(2.0*np.pi*FV*dv) - 1)/dv,-1))**p
    return L

def robust_loss(RphiI,xJ,J,W,c=0.5, return_weights=False):
    """ We input the deformed atlas I, the target J, and the voxel coordinates of J, and the robust constant c
    note there is a W here which should be binary because we're not going to sum over all the pixels
    
    Parameters:
    ===========
    RphiI : torch.Tensor[torch.float32]
        TOOD
    xJ : list[torch.Tensor[torch.float32]]
        TODO
    J : torch.Tensor[torch.float32]
        TODO
    W : torch.Tensor[torch.float32]
        TODO
    c : float32
        Default - 0.5; TODO - Parameter description
    return_weights - bool
        Default - False; If true, return the weights too

    Returns:
    ========
    E : TODO
        TODO
    W : TODO
        TODO
    """
    
    dxJ = torch.stack([(x[1]-x[0]) for x in xJ])
    DJ = torch.prod(dxJ)
    
    err2 =  torch.sum( (RphiI - J)**2  , 0)*W # make sure the error does not go on padding
    E = c*torch.sum(err2 / (  err2 + c )  )*DJ
    
    if not return_weights:
        return E
    else:
        W = c**2 / ( (c + err2.clone().detach())**2 )*W # make sure weight is 0
        # note: we do not include pixel size d here because it will get multiplied by d later
        # no gradient calculations here
        # note, if c is really big, this goes to 0
        # and if c is really small, this looks like c/err2**2, and so also goes to 0
        return E, W

def transform_image(xI,I,xv,v,A,xJ,**kwargs):
    """Note this is redundant with the below, but is a nice helper function
    
    Parameters:
    ===========
    xI : list[torch.Tensor[torch.float32]]
        TODO
    I : torch.Tensor[torch.float32]
        TODO
    xv : list[torch.Tensor[torch.float32]]
        TODO
    v : torch.Tensor[torch.float32]
        TODO
    A : torch.Tensor[torch.float32]
        TODO
    xJ : list[torch.Tensor[torch.float32]]
        TODO
    kwargs : dict
        TODO

    Returns:
    ========
    AphiI : torch.Tensor[torch.float32]
        TODO
    """
    XJ = torch.stack(torch.meshgrid(*xJ,indexing='ij'),-1)
    XV = torch.stack(torch.meshgrid(*xv,indexing='ij'),-1)
    
    Ai = torch.linalg.inv(A)
    Ai = A2DtoA3D(Ai)
    Ai = Ai.to(dtype=XJ.dtype, device=XJ.device) # Switch to torch.float32
    Xs = AX(Ai,XJ)

    # convert v to phii
    phii = exp(xv,v2DToV3D(-v))

    # sample at Xs
    Xs = interp(xv,(phii - XV).permute(-1,0,1,2), Xs).permute(1,2,3,0) + Xs

    # transform the image
    if 'padding_mode' not in kwargs:
        kwargs['padding_mode'] = 'border'
    AphiI = interp(xI,I,Xs,**kwargs)

    return AphiI

def v2DToV3D(v2d):
    """
    TOOD - Function description

    Parameters:
    ===========
    v2d : torch.Tensor[torch.float32]
        TODO

    Returns:
    ========
    out : torch.Tensor[torch.float32]
        TODO
    """
    return torch.concatenate( ( torch.zeros_like(v2d[...,0,None]), v2d ) , -1)

def weighted_see_registration(xI,I,xJ,J,W,xv,v,A,a,p,sigmaM,sigmaR,niter,epT,epL,epv,draw=0,fig=None,hfig=None):
    """
    TODO - Function description

    Parameters:
    ===========
    xI : list[torch.Tensor[torch.float32]]
        TODO
    I : torch.Tensor[torch.float32]
        TODO
    xJ : list[torch.Tensor[torch.float32]]
        TODO
    J : torch.Tensor[torch.float32]
        TODO
    W : torch.Tensor[torch.float32]
        TODO
    xv : list[torch.Tensor[torch.float32]]
        TODO
    v : torch.Tensor[torch.float32]
        TODO
    A : torch.Tensor[torch.float32]
        TODO
    a : float
        TODO
    p : float
        TODO
    sigmaM : float
        TODO
    sigmaR : float
        TODO
    niter : int
        TODO
    epT : float
        TODO
    epL : float
        TODO
    epv : float
        TODO
    draw : bool
        TODO
    fig : matplotlib.pyplot.figure
        TODO
    hfig : matplotlib.pyplot.figure
        TODO

    Returns:
    ========
    A : torch.Tensor[torch.float32]
        TODO
    v : torch.Tensor[torch.float32]
        TODO
    E : float
        TODO
    ER : float
        TODO
    """

    A.requires_grad = True
    v.requires_grad = True

    XJ = torch.stack(torch.meshgrid(*xJ,indexing='ij'),-1)
    XV = torch.stack(torch.meshgrid(*xv,indexing='ij'),-1)
    dv = torch.stack([x[1] - x[0] for x in xv])
    Dv = torch.prod(dv)
    dJ = torch.stack([x[1] - x[0] for x in xJ])
    DJ = torch.prod(dJ)
    L = L_from_xv_a_p(xv,a,p)
    LL = L**2
    K = 1.0/LL

    Esave = []
    Tsave = []
    Lsave = []
    maxvsave = []
    if draw:
        if fig is None:
            fig,ax = plt.subplots(2,3)
            ax = ax.ravel()
        else:
            fig.clf()
            ax = []
            for i in range(2):
                for j in range(3):
                    ax.append(fig.add_subplot(2,3,i*3+j+1))
        if hfig is None:
            hfig = display(fig,display_id=True)
    for it in range(niter):
        if v.grad is not None:
            v.grad.zero_()
        if A.grad is not None:
            A.grad.zero_()
        # act on XJ
        Ai = torch.linalg.inv(A)
        Ai = A2DtoA3D(Ai)
        Ai = Ai.to(dtype=XJ.dtype, device=XJ.device) # Switch to torch.float32
        Xs = AX(Ai,XJ)

        # convert v to phii
        phii = exp(xv,v2DToV3D(-v))

        # sample at Xs
        Xs = interp(xv,(phii - XV).permute(-1,0,1,2), Xs).permute(1,2,3,0) + Xs

        # transform the image
        AphiI = interp(xI,I,Xs,padding_mode='border') # border boundary condition is important when we have a white background

        # get error
        EM = torch.sum((AphiI - J)**2*W)*DJ/sigmaM**2/2.0
        ER = torch.sum(torch.sum( torch.abs( torch.fft.fftn(v,dim=(1,2)) )**2 , -1)*LL)/sigmaR**2/2.0/v[0,...,0].numel()*Dv
        E = EM + ER
        Esave.append([E.item(),EM.item(),ER.item()])
        Tsave.append(    A[:,:2,-1].clone().detach().cpu().ravel().numpy()  )
        Lsave.append( A[:,:2,:2].clone().detach().cpu().ravel().numpy()    )
        maxvsave.append(  (torch.amax(torch.sum(v.clone().detach()**2,-1))**0.5).cpu().numpy().item()  )
        # backprop
        E.backward()

        # update
        with torch.no_grad():
            # update T
            A[:,:2,-1] -= A.grad[:,:2,-1]*epT
            # update L
            A[:,:2,:2] -= A.grad[:,:2,:2]*epL
            # rigid
            u,s,vh = torch.linalg.svd(A[:,:2,:2])
            A[:,:2,:2] = u@vh


            # update v
            v[:] = v[:] - torch.fft.ifftn(torch.fft.fftn(v.grad,dim=(1,2))*K[...,None],dim=(1,2)).real*epv

        # draw
        with torch.no_grad():
            if draw and (not it%draw or it == niter-1):
                ax[0].cla()
                ax[0].plot(Esave)
                ax[0].set_title('energy')
                ax[1].cla()
                ax[1].plot(Tsave)
                ax[1].set_title('Translation')
                ax[2].cla()
                ax[2].plot(Lsave)
                ax[2].set_title('Linear')
                ax[3].cla()
                ax[3].plot(maxvsave)
                ax[3].set_title('max |v|')
                
                
                ax[4].cla()
                ax[4].imshow(  ( (AphiI[:,:,J.shape[2]//2]-J[:,:,J.shape[2]//2])*W[:,J.shape[2]//2]  ).permute(1,2,0).cpu()*0.5+0.5  ,aspect='auto', interpolation='none')
                #ax[5].cla()
                #ax[5].imshow(  (  (AphiI[:,:,:,J.shape[3]//2]-J[:,:,:,J.shape[3]//2])*W[:,:,J.shape[3]//2] ).permute(1,2,0).cpu()*0.5+0.5  ,aspect='auto', interpolation='none')
                
                ax[5].cla()
                ax[5].imshow(  (  (AphiI[:,:,:,J.shape[3]//2]-J[:,:,:,J.shape[3]//2])*W[:,:,J.shape[3]//2] ).permute(1,2,0).cpu()*0.5+0.5  ,aspect='auto', interpolation='none')

                # hfig.update(fig)

    A.requires_grad = False
    v.requires_grad = False
    
    return A,v,E.item(),ER.item()

def x_from_n_d(n,d):
    """
    This function takes in an image size and pixel size, and outputs a zero centered coordinate system

    Parameters:
    ===========
    n : int or arr[int]
        The image size along each axis
    d : int or arr[int]
        The pixel size along each axis

    Returns:
    ========
    x : arr
        The zero-centered coordinate system for an image of size 'n' with pixel size 'd'
    """
    x = [torch.arange(ni)*di - (ni-1)*di/2 for ni,di in zip(n,d)]
    return x

