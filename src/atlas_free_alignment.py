import matplotlib
matplotlib.use("Agg")

import numpy as np
from glob import glob
from os.path import join
import matplotlib.pyplot as plt

try:
    from IPython.display import display
except ImportError:
    class _NullDisplay:
        def update(self, *args, **kwargs):
            pass

    def display(*args, **kwargs):
        return _NullDisplay()

import tifffile
import sys

from scipy.interpolate import interpn
from scipy.ndimage import gaussian_filter

import torch


# this extent was copied from emlddmm
def extent_from_x(xJ):
    ''' Given a set of pixel locations, returns an extent 4-tuple for use with imshow.
    
    Note
    ----
    Note inputs are locations of pixels along each axis, i.e. row column not xy.
    
    Parameters
    ----------
    xJ : list of torch tensors
        Location of pixels along each axis
    
    Returns
    -------
    extent : tuple
        (xmin, xmax, ymin, ymax) tuple
    
    Example
    -------
    Draw a 2D image stored in J, with pixel locations of rows stored in xJ[0] and pixel locations
    of columns stored in xJ[1].
    
    >>> import matplotlib.pyplot as plt
    >>> extent_from_x(xJ)
    >>> fig,ax = plt.subplots()
    >>> ax.imshow(J,extent=extentJ)
    
    '''
    dJ = [x[1]-x[0] for x in xJ]
    extentJ = ( (xJ[1][0] - dJ[1]/2.0).item(),
               (xJ[1][-1] + dJ[1]/2.0).item(),
               (xJ[0][-1] + dJ[0]/2.0).item(),
               (xJ[0][0] - dJ[0]/2.0).item())
    return extentJ



    
# this draw function was copied from emlddmm to avoid the dependency
def draw(J,xJ=None,fig=None,n_slices=5,vmin=None,vmax=None,disp=True,cbar=False,slices_start_end=[None,None,None],**kwargs):    
    """ Draw 3D imaging data.
    
    Images are shown by sampling slices along 3 orthogonal axes.
    Color or grayscale data can be shown.
    
    Parameters
    ----------
    J : array like (torch tensor or numpy array)
        A 3D image with C channels should be size (C x nslice x nrow x ncol)
        Note grayscale images should have C=1, but still be a 4D array.
    xJ : list
        A list of 3 numpy arrays.  xJ[i] contains the positions of voxels
        along axis i.  Note these are assumed to be uniformly spaced. The default
        is voxels of size 1.0.
    fig : matplotlib figure
        A figure in which to draw pictures. Contents of the figure will be cleared.
        Default is None, which creates a new figure.
    n_slices : int
        An integer denoting how many slices to draw along each axis. Default 5.
    vmin
        A minimum value for windowing imaging data. Can also be a list of size C for
        windowing each channel separately. Defaults to None, which corresponds 
        to tha 0.001 quantile on each channel.
    vmax
        A maximum value for windowing imaging data. Can also be a list of size C for
        windowing each channel separately. Defaults to None, which corresponds 
        to tha 0.999 quantile on each channel.
    disp : bool
        Figure display toggle
    kwargs : dict
        Other keywords will be passed on to the matplotlib imshow function. For example
        include cmap='gray' for a gray colormap

    Returns
    -------
    fig : matplotlib figure
        The matplotlib figure variable with data.
    axs : array of matplotlib axes
        An array of matplotlib subplot axes containing each image.


    Example
    -------
    Here is an example::

       >>> example test
   
    TODO
    ----
    Put interpolation='none' in keywords


    """
    if type(J) == torch.Tensor:
        J = J.detach().clone().cpu()
    J = np.array(J)
    if xJ is None:
        nJ = J.shape[-3:]
        xJ = [np.arange(n) - (n-1)/2.0 for n in nJ] 
    if type(xJ[0]) == torch.Tensor:
        xJ = [np.array(x.detach().clone().cpu()) for x in xJ]
    xJ = [np.array(x) for x in xJ]
    
    if fig is None:
        fig = plt.figure()
    fig.clf()    
    if vmin is None:
        vmin = np.quantile(J,0.001,axis=(-1,-2,-3))
    if vmax is None:
        vmax = np.quantile(J,0.999,axis=(-1,-2,-3))
    vmin = np.array(vmin)
    vmax = np.array(vmax)    
    # I will normalize data with vmin, and display in 0,1
    if vmin.ndim == 0:
        vmin = np.repeat(vmin,J.shape[0])
    if vmax.ndim == 0:
        vmax = np.repeat(vmax,J.shape[0])
    if len(vmax) >= 2 and len(vmin) >= 2:
        # for rgb I'll scale it, otherwise I won't, so I can use colorbars
        J -= vmin[:,None,None,None]
        J /= (vmax[:,None,None,None] - vmin[:,None,None,None])
        J[J<0] = 0
        J[J>1] = 1
        vmin = 0.0
        vmax = 1.0
    # I will only show the first 3 channels
    if J.shape[0]>3:
        J = J[:3]
    if J.shape[0]==2:
        J = np.stack((J[0],J[1],J[0]))
    
    
    axs = []
    axsi = []
    # ax0
    slices = np.round(np.linspace(0,J.shape[1]-1,n_slices+2)[1:-1]).astype(int)     
    if slices_start_end[0] is not None:
        slices = np.round(np.linspace(slices_start_end[0][0],slices_start_end[0][1],n_slices+2)[1:-1]).astype(int)     
        
    # for origin upper (default), extent is x (small to big), then y reversed (big to small)
    extent = (xJ[2][0],xJ[2][-1],xJ[1][-1],xJ[1][0])
    for i in range(n_slices):
        ax = fig.add_subplot(3,n_slices,i+1)
        toshow = J[:,slices[i]].transpose(1,2,0)
        if toshow.shape[-1] == 1:
            toshow = toshow.squeeze(-1)
        ax.imshow(toshow,vmin=vmin,vmax=vmax,aspect='equal',extent=extent,**kwargs)
        if i>0: ax.set_yticks([])
        axsi.append(ax)
    axs.append(axsi)
    axsi = []
    # ax1
    slices = np.round(np.linspace(0,J.shape[2]-1,n_slices+2)[1:-1]).astype(int)    
    if slices_start_end[1] is not None:
        slices = np.round(np.linspace(slices_start_end[1][0],slices_start_end[1][1],n_slices+2)[1:-1]).astype(int)         
    extent = (xJ[2][0],xJ[2][-1],xJ[0][-1],xJ[0][0])
    for i in range(n_slices):
        ax = fig.add_subplot(3,n_slices,i+1+n_slices)      
        toshow = J[:,:,slices[i]].transpose(1,2,0)
        if toshow.shape[-1] == 1:
            toshow = toshow.squeeze(-1)
        ax.imshow(toshow,vmin=vmin,vmax=vmax,aspect='equal',extent=extent,**kwargs)
        if i>0: ax.set_yticks([])
        axsi.append(ax)
    axs.append(axsi)
    axsi = []
    # ax2
    slices = np.round(np.linspace(0,J.shape[3]-1,n_slices+2)[1:-1]).astype(int)        
    if slices_start_end[2] is not None:
        slices = np.round(np.linspace(slices_start_end[2][0],slices_start_end[2][1],n_slices+2)[1:-1]).astype(int)     
    
    extent = (xJ[1][0],xJ[1][-1],xJ[0][-1],xJ[0][0])
    for i in range(n_slices):        
        ax = fig.add_subplot(3,n_slices,i+1+n_slices*2)
        toshow = J[:,:,:,slices[i]].transpose(1,2,0)
        if toshow.shape[-1] == 1:
            toshow = toshow.squeeze(-1)
        ax.imshow(toshow,vmin=vmin,vmax=vmax,aspect='equal',extent=extent,**kwargs)
        if i>0: ax.set_yticks([])
        axsi.append(ax)
    axs.append(axsi)
    
    fig.subplots_adjust(wspace=0,hspace=0)
    if not disp:
        plt.close(fig)
    axs = np.array(axs)
    
    if cbar and disp:
        plt.colorbar(mappable=[h for h in axs[0][0].get_children() if 'Image' in str(h)][0],ax=np.array(axs).ravel())
    return fig,axs

# also from emlddmm

def downsample_ax(I,down,ax,W=None):
    '''
    Downsample imaging data along one of the first 5 axes.
    
    Imaging data is downsampled by averaging nearest pixels.
    Note that data will be lost from the end of images instead of padding.
    This function is generally called repeatedly on each axis.
    
    Parameters
    ----------
    I : array like (numpy or torch)
        Image to be downsampled on one axis.
    down : int
        Downsampling factor.  2 means average pairs of nearest pixels 
        into one new downsampled pixel
    ax : int
        Which axis to downsample along.
    W : np array
        A mask the same size as I, but without a "channel" dimension
    
    Returns
    -------
    Id : array like
        The downsampled image.
    
    Raises
    ------
    Exception
        If a mask (W) is included and ax == 0. 
    '''
    nd = list(I.shape)        
    nd[ax] = nd[ax]//down
    if type(I) == torch.Tensor:
        Id = torch.zeros(nd,device=I.device,dtype=I.dtype)
    else:
        Id = np.zeros(nd,dtype=I.dtype)
    if W is not None:
        if type(W) == torch.Tensor:
            Wd = torch.zeros(nd[1:],device=W.device,dtype=W.dtype)
        else:
            Wd = np.zeros(nd[1:],dtype=W.dtype)            
    if W is None:
        for d in range(down):
            if ax==0:        
                Id += I[d:down*nd[ax]:down]
            elif ax==1:        
                Id += I[:,d:down*nd[ax]:down]
            elif ax==2:
                Id += I[:,:,d:down*nd[ax]:down]
            elif ax==3:
                Id += I[:,:,:,d:down*nd[ax]:down]
            elif ax==4:
                Id += I[:,:,:,:,d:down*nd[ax]:down]
            elif ax==5:
                Id += I[:,:,:,:,:,d:down*nd[ax]:down]
            # ... should be enough but there really has to be a better way to do this        
            # note I could use "take"
        Id = Id/down
        return Id
    else:
        # if W is not none
        for d in range(down):
            if ax==0:        
                Id += I[d:down*nd[ax]:down]*W[d:down*nd[ax]:down]
                raise Exception('W not supported with ax=0')
                
            elif ax==1:        
                Id += I[:,d:down*nd[ax]:down]*W[d:down*nd[ax]:down]
                Wd += W[d:down*nd[ax]:down]
            elif ax==2:
                Id += I[:,:,d:down*nd[ax]:down]*W[:,d:down*nd[ax]:down]
                Wd += W[:,d:down*nd[ax]:down]
            elif ax==3:
                Id += I[:,:,:,d:down*nd[ax]:down]*W[:,:,d:down*nd[ax]:down]
                Wd += W[:,:,d:down*nd[ax]:down]
            elif ax==4:
                Id += I[:,:,:,:,d:down*nd[ax]:down]*W[:,:,:,d:down*nd[ax]:down]
                Wd += W[:,:,:,d:down*nd[ax]:down]
            elif ax==5:
                Id += I[:,:,:,:,:,d:down*nd[ax]:down]*W[:,:,:,:,d:down*nd[ax]:down]
                Wd += W[:,:,:,:,d:down*nd[ax]:down]
        Id = Id / (Wd + Wd.max()*1e-6)
        
        
        Wd = Wd / down
        return Id,Wd
# also from emlddmm
def downsample(I,down,W=None):
    '''
    Downsample an image by an integer factor along each axis. Note extra data at 
    the end will be truncated if necessary.
    
    If the first axis is for image channels, downsampling factor should be 1 on this.
    
    Parameters
    ----------
    I : array (numpy or torch)
        Imaging data to downsample
    down : list of int
        List of downsampling factors for each axis.
    W : array (numpy or torch)
        A weight of the same size as I but without the "channel" dimension
    
    Returns
    -------
    Id : array (numpy or torch as input)
        Downsampled imaging data.
    '''
    down = list(down)
    while len(down) < len(I.shape):
        down.insert(0,1)    
    if type(I) == torch.Tensor:
        Id = torch.clone(I)
    else:
        Id = np.copy(I)
    if W is not None:
        if type(W) == torch.Tensor:
            Wd = torch.clone(W)
        else:
            Wd = np.copy(W)
    for i,d in enumerate(down):
        if d==1:
            continue
        if W is None:
            Id = downsample_ax(Id,d,i)
        else:
            Id,Wd = downsample_ax(Id,d,i,W=Wd)
    if W is None:
        return Id
    else:
        return Id,Wd
# again from emlddmm
def downsample_image_domain(xI,I,down,W=None): 
    '''
    Downsample an image as well as pixel locations
    
    Parameters
    ----------
    xI : list of numpy arrays
        xI[i] is a numpy array storing the locations of each voxel
        along the i-th axis.
    I : array like
        Image to be downsampled
    down : list of ints
        Factor by which to downsample along each dimension
    W : array like
        Weights the same size as I, but without a "channel" dimension
        
    Returns
    -------
    xId : list of numpy arrays
        New voxel locations in the same format as xI
    Id : numpy array
        Downsampled image.
    
    Raises
    ------
    Exception
        If the length of down and xI are not equal.
    '''
    if len(xI) != len(down):
        raise Exception('Length of down and xI must be equal')
    if W is None:
        Id = downsample(I,down)    
    else:
        Id,Wd = downsample(I,down,W=W)
    xId = []
    for i,d in enumerate(down):
        xId.append(downsample_ax(xI[i],d,0))
    if W is None:
        return xId,Id
    else:
        return xId,Id,Wd
        
    
def get_weights(I,J,c,W0=1.0):
    return c**2/(torch.sum((I-J)**2,0) + c)**2 * W0



def get_frequency_operators(nI,dI,a,p=2.0,k=1.0,Wmax=1, device=None, dtype=None):
    dI = torch.as_tensor(dI, device=device, dtype=dtype)
    a = torch.as_tensor(a, device=device, dtype=dtype)
    fI = [torch.arange(n, device=device, dtype=dtype)/n/d for n,d in zip(nI,dI)]
    FI = torch.stack(torch.meshgrid(*fI,indexing='ij'),-1)
    Lhat = ( 2*(a**2*( 1.0 -  torch.cos(2.0*np.pi*FI*dI))/dI**2).sum(-1) )**p
    
    LLhat = Lhat**2
    Kblurhat = 1.0 / (1.0 + k/Wmax/2 * LLhat)
    return Lhat, LLhat, Kblurhat


def interp_slow(xI,I,Xs,**kwargs):
    x0 = torch.stack([x[0] for x in xI])
    l = torch.stack([x[-1] for x in xI]) - x0
    Xs = (Xs - x0)/l
    Xs = Xs * 2 - 1
    return torch.nn.functional.grid_sample(I[None],Xs[None].flip(-1), align_corners=True, **kwargs)[0]


# we want slices to be the batch dimension
def interp(xI,I,Xs,**kwargs):
    xI = xI[1:]
    Xs = Xs[...,1:]
    x0 = torch.stack([x[0] for x in xI])
    l = torch.stack([x[-1] for x in xI]) - x0
    Xs = (Xs - x0)/l
    Xs = Xs * 2 - 1
    
    return torch.nn.functional.grid_sample(I.permute(1,0,2,3),Xs.flip(-1), align_corners=True, **kwargs).permute(1,0,2,3)    

def Xs_from_R(R,XI,xI):
    RXI = (R[:,None,None,:2,:2]@XI[None,...,None])[...,0] + R[:,None,None,:2,-1]
    # first indexx should be just xI0
    O = torch.ones_like(RXI[...,0,None])
    RXI = torch.concatenate( ((O*xI[0][...,None,None,None]), RXI ), -1)
    return RXI
    

def sinc_upsample(Idout0,nI):
    I_ = Idout0.clone()
    for i in [-1,-2,-3]:
        I_ = torch.fft.irfft( torch.fft.rfft(I_,dim=i) , dim=i, n=nI[i])
    I_ = I_ * (I_.shape[1]*I_.shape[2]*I_.shape[3])/(Idout0.shape[1]*Idout0.shape[2]*Idout0.shape[3])    
    return I_


def _should_draw(it, ndraw):
    return bool(ndraw) and ((it + 1) % ndraw == 0)


def atlas_free_alignment(xI,I,xJ,J,W0,Kblurhat,
                         R,
                         niter, niter_atlas, eL, eT, 
                         c, ndraw=5, figprefix='it_00'):
    if ndraw:
        figI = plt.figure()
        hfigI = display(figI,display_id=True)
        figJ = plt.figure()
        hfigJ = display(figJ,display_id=True)
        figErr = plt.figure()
        hfigErr = display(figErr,display_id=True)
        figW = plt.figure()
        hfigW = display(figW,display_id=True)
        figE,axE = plt.subplot_mosaic([['matching_loss', 'T','theta']])
        hfigE = display(figE,display_id=True)
    else:
        figI = figJ = figErr = figW = figE = axE = None
        hfigI = hfigJ = hfigErr = hfigW = hfigE = None
    matching_loss_save = []
    Tsave = []
    thetasave = []

    XI = torch.stack(torch.meshgrid(*xI[1:],indexing='ij'),-1)
    XJ = torch.stack(torch.meshgrid(*xJ[1:],indexing='ij'),-1)

    
    for it in range(niter):
        print(f'starting it {it}', flush=True)
        if _should_draw(it, ndraw):
            draw(I,xI,fig=figI,vmin=0,vmax=1,interpolation='none')
            hfigI.update(figI)
            figI.savefig(f'{figprefix}_{it:06d}_atlas.png')
        
        # step 1, take a step of gradient descent
        R.requires_grad = True
        
        Ri = torch.linalg.inv(R)
        Xs = Xs_from_R(Ri,XJ,xJ)
        
        RI = interp(xI,I,Xs,padding_mode='border')
    
        E = torch.sum( (RI-J)**2 , 0)
    
        pixel_loss =  c*E/(c + E) 
        loss = torch.sum(pixel_loss*W0)
        if R.grad is not None: R.grad.zero_()
            
        loss.backward()
        matching_loss_save.append(loss.item())
        if _should_draw(it, ndraw):
            axE['matching_loss'].cla()
            axE['matching_loss'].plot(matching_loss_save)
            axE['matching_loss'].set_yscale('log')
            
        
        # update registration parameters
        with torch.no_grad():
            # translation
            R[:,:2,-1] = R[:,:2,-1] - R.grad[:,:2,-1]*eT
            
            # rotation
            R[:,:2,:2] = R[:,:2,:2] - R.grad[:,:2,:2]*eL
            # project
            U,S,Vh = torch.linalg.svd( R[:,:2,:2] )
            R[:,:2,:2] = U@Vh
            
            
            
            
            R.requires_grad = False
    
            thetasave.append(torch.atan2(R[:,1,0],R[:,0,0]).cpu().numpy()*180/np.pi)
            Tsave.append(R[:,:2,-1].cpu().numpy().copy())
            if _should_draw(it, ndraw):
                axE['T'].cla()
                axE['T'].plot(np.stack(Tsave).reshape(len(Tsave),-1))
                axE['theta'].cla()
                axE['theta'].plot(thetasave)
                
            
            # step 2, given the current parameters, update the template
            W = c**2/(E + c)**2 * W0 # or call the function get_weight?
            Wmax = torch.amax(W)
            Wmax = 1.0 # this needs to be true for the way I defined my frequency operator
            Xs = Xs_from_R(R,XI,xI)
            # in previous work we found that using nearest for atlas construction worked best (assuming pixel sizes are appropriate)
            mode = 'nearest'
            #mode = 'bilinear'
            RiJ = interp(xJ,J,Xs,mode=mode,padding_mode='zeros')
            RiW = interp(xJ,W[None],Xs,mode=mode,padding_mode='zeros')[0]
            RiW_Wmax = RiW / Wmax
            if _should_draw(it, ndraw):
                draw(RiJ,xI,fig=figJ,vmin=0,vmax=1,interpolation='none')
                hfigJ.update(figJ)
                figJ.savefig(f'{figprefix}_{it:06d}_recon.png')                
                #draw((RiJ - I)*RiW_Wmax[None]*0.5+0.5,xI,fig=figErr,vmin=0,vmax=1,interpolation='none')
                draw((J - RI)*W*0.5+0.5,xI,fig=figErr,vmin=0,vmax=1,interpolation='none')
                hfigErr.update(figErr)
                figErr.savefig(f'{figprefix}_{it:06d}_error.png')
                draw(RiW_Wmax[None],xI,fig=figW,vmin=0,vmax=1,interpolation='none')
                hfigW.update(figW)
                figW.savefig(f'{figprefix}_{it:06d}_weight.png')
                
            
            RiJRiW_Wmax = RiJ*RiW_Wmax
            omRiW_Wmax = 1.0 - RiW_Wmax
            # todo reflection padding
            # TODO: kblurhat should be updated as Wmax changes.  Wmax is surely less than 1 and it could be faster to account for this
            # it is correct either way though
            for it_atlas in range(niter_atlas):        
                #I = torch.fft.ifftn( torch.fft.fftn( RiJRiW_Wmax + I*omRiW_Wmax , dim=(1,2,3))*Kblurhat , dim=(1,2,3)).real
                # this way is slightly faster
                # swapping the order of dimensions is significantly slower
                Ihat = torch.fft.rfftn( RiJRiW_Wmax + I*omRiW_Wmax , dim=(1,2,3))                
                I = torch.fft.irfftn( Ihat*Kblurhat[...,:Ihat.shape[-1]] , dim=(1,2,3), s=I.shape[1:])

        if _should_draw(it, ndraw):
            hfigE.update(figE)

    plt.close('all')
    return R, I, RiJ, RiW, RI.clone().detach()

        
        