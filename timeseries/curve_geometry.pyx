# cython: language_level=3, boundscheck=True, linetrace=True, embedsignature=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1
r""" 
This Cython module contains the core algorithms for the geometry of curves.

If a function requires its own explicit iterator in a loop, probably it should
go here.  Anything that is just a vectorized operation in NumPy or Pandas can
go in plain python elsewhere. 

See http://docs.cython.org/src/tutorial/np.html for ideas on writing efficient
code in Cython.

This module is compiled by either of these commands
 - :code:`python setup.py install`  (as called by :code:`pip` for standard installation and use)
 - :code:`python setup.py build_ext --inplace` (as run by developers for code testing)

Notes
-----
Automatically-generated online documentation will never see the C-only functions that
are defined with :code:`cdef` in Cython, as shown in 
http://cython.readthedocs.io/en/latest/src/reference/extension_types.html
Because of the limitations of Sphinx, you'll have to simply view the sourcecode
for further information. For this module, these C-only functions include 

 - :func:`homology.dim0.root`
 - :func:`homology.dim0.connected`
 - :func:`homology.dim0.merge`    

Copyright
---------
- This file is part of https://github.com/geomdata/gda-public/ 
- 2015, 2016, 2017 by Geometric Data Analytics, Inc. (http://geomdata.com)
- AGPL license. See `LICENSE` or https://github.com/geomdata/gda-public/blob/master/LICENSE
"""

"""
The following stanza is for docstring discovery in Cython.
>>> import numpy
>>> from timeseries.curve_geometry import *
>>> print("test")
test

"""


### Allow us to use NumPy indexing and datatypes
import numpy as np
cimport numpy as np
cimport cython
ctypedef np.int64_t NINT_t
ctypedef np.float64_t NDBL_t
ctypedef np.uint8_t NBIT_t

# For convenient indexing
import itertools
import collections

import scipy.linalg as la
from scipy.spatial.distance import euclidean, cdist
from scipy import integrate
from scipy import signal


cpdef np.ndarray[NDBL_t, ndim=2] cleanup(
      np.ndarray[NDBL_t, ndim=1] time,
      np.ndarray[NDBL_t, ndim=1] position,
          NDBL_t max_speed=500.0):
    """ Throw out bad points by speed comparison.

    Points whose speed exceed max_speed will be removed and replaced by a
    a speed less than max_speed.

    Parameters
    ----------
    time : `numpy.ndarray`
        N by 1 time array.
    position : `numpy.ndarray`
        N by 3 position array.
    max_speed : `numpy.float64`
        Max speed allowable.

    Returns
    -------
    [time, corrected] : list
        List of *time* N by 1 and the corrected array *corrected* N by 3.
    """

    cdef NINT_t N = time.shape[0]
    assert N == position.shape[0]
    cdef NINT_t i, j
    
    cdef NDBL_t speed, delta_t

    cdef np.ndarray[NDBL_t, ndim=1] corrected = position.copy()


    # If speed between x1 and x2 is too big, replace x2 with (x1+x3)/2.
    # But, if new speed between x1 and x2 is too big, replace x2 with (x1+x4)/3.
    # But, if new speed between x1 and x2 is too big, replace x2 with (x1+x5)/3.
    # etc

    for i in range(N-2):
        delta_t = time[i+1]-time[i]
 
        speed = np.abs((corrected[i+1]-corrected[i])/delta_t)

        j = i+1
        while speed > max_speed and j < N-1:
            j+=1
            delta_T = time[j]-time[i]
            corrected[i+1] = corrected[i] + (delta_t)*(corrected[j]-corrected[i])/delta_T
            speed = np.abs((corrected[i+1]-corrected[i])/delta_t)

    return np.asarray([time, corrected])

def mollifier_uneven(time, position, refinement=1, width=50.0):
    """Smooth function in a way that retains shape by adding new graph values
    via the linear interpolate and smoothing via convolution with the
    Friedrichs' mollifier. This version is an attempt at mollifying
    on general time griding, but encounters issues with such. It
    seems that one will need to investigate more closely convolution and
    potential re-implement.

    Parameters
    ----------
    time : ndarray
        N by 1 time array.
    position : ndarray
        N by 3 position array.
    refinement : int
        Factor to refine the time discretization.
    width : double
        One half support width for Friedrichs' mollifier.

    Notes
    -----
    Following Gilbarg and Trudinger [1]_ a *mollifier* is a non-negative
    :math:`C^\infty(\mathbb{R}^n)` function vanishing outside the unit ball
    whose integral is one. The standard (Friedrichs') mollifier is given by

    .. math::
       \\rho(x) = \\left \\{
       \\begin{aligned}
       &c \\exp \\left ( \\frac{1}{|x|^2 - 1} \\right ) &\\text{for }
       &|x| \\leq 1 \\\\
       &0 &\\text{for } &|x| \\geq 1.
       \\end{aligned}
       \\right .

    For :math:`u \\in L_{\\text{loc}}^1(\\Omega)` and :math:`h > 0`, the
    *regularization* of :math:`u` is given by

    .. math::
       u_h(x) = h^{-n} \\int_\\Omega \\rho\\left ( \\frac{x - y}{h} \\right )
       \\
       u(y) dy, \\quad h < \\text{dist}(x, \\partial \\Omega).

    This regularization has the following nice properties.

    - If :math:`u \\in L^1(\\Omega)`, then
      :math:`u_h \\in C^{\\infty}_0(\\Omega')` for any :math:`\Omega'` compactly
      contained in :math:`\\Omega` and for arbitrary :math:`h > 0`.
    - The function :math:`y \\mapsto h^{-n} \\rho((x-y)h^{-1})` tends a Dirac
      delta distribution as :math:`h \\to 0`.
    - If :math:`u \\in C^0(\\Omega)`, then :math:`u_h \\to u` uniformly on
      compact subsets of :math:`\\Omega`.

    The overall algorithm is taken from Knowles--Renka [2]_ and
    is summarized as follows.

    - Specify a mollifier width. Greater width yields greater smoothness.
    - Add points in via linear interpolation to increase resolution if desired.
    - Mollify

      - Form discrete versions of the proper dimensions of :math:`\\rho`,
        :math:`y \\mapsto h^{-1} \\rho((x-y)h^{-1})`.
      - Compute the discrete convolution analogous to

        .. math::
           h^{-n} \\int_\\Omega \\rho\\left ( \\frac{x - y}{h} \\right ) u(y) dy

    References
    ----------
    .. [1] D. Gilbarg and N. S. Trudinger, Elliptic Partial Differential
        Equations of Second Order. Springer Science & Business Media, 2001.

    .. [2] I. Knowles and R. J. Renka, "Methods for numerical differentiation of
        noisy data," Electronic Journal of Differential Equations, pp. 235--246,
        2014.

    Returns
    -------
    [s, smooth] : list
        List of the augmented N*refinement by 1 time array *s* and the
        N*refinement by 3 smooth array *smooth*.
    """

    N = time.shape[0]
    assert N == position.shape[0]

    ### Pad the original arrays representing time and position by
    ### extending linearly in time by average time step and
    ### and by a constant for the function values of position.
    ### That is extend [time[0],time[-1]] to [time[0]-width,time[-1]+width]
    ### and extend position to match

    delta = (time[-1]-time[0])/(N-1)

    ### compute boundary time padding
    left_pad = np.arange(time[0],time[0]-(width+delta),step=-delta)
    left_pad = np.flipud(left_pad)[:-1]
    left_pad_num = left_pad.shape[0]
    right_pad = np.arange(time[-1],time[-1]+(width+delta),step=delta)[1:]
    right_pad_num = right_pad.shape[0]
    time_pad = np.concatenate((left_pad,time,right_pad))

    ### compute boundary position padding
    position_pad = np.pad(position,(left_pad_num,right_pad_num),'edge')

    ### fill in the difference of size greater than width
    ### since this will behave exactly like a boundary

    ### Correct bad interior times
    #orig_time_pad = time_pad
    #offset = 0
    #for index, diff in enumerate(np.diff(time_pad)):
    #    if diff > width:
    #        add_pts = np.arange(time_pad[index+offset],
    #                              time_pad[index+offset+1],delta)[1:]
    #        time_pad = np.insert(time_pad,(offset+index+1),add_pts)
    #        offset = offset + add_pts.shape[0]

    ### Correct bad interior positions by linear interpolation
    #position_pad = np.interp(time_pad, orig_time_pad, position_pad)

    ### Define a new smaller time scale s (we use the spacings at the end)
    s = time_pad
    ### Find midpoints of time and insert them
    for step in range(refinement):
        midpoints = (0.5)*(s[(np.arange(s.shape[0])+1)[:-1]]+s[(np.arange(s.shape[0]))[:-1]])
        s = np.insert(s,(np.arange(s.shape[0])+1)[:-1],midpoints)
        left_pad_num = 2*left_pad_num
        right_pad_num = 2*right_pad_num

    position_interp = np.interp(s,time_pad,position_pad)

    ### Compute the mollifier kernel
    norm_const,err = integrate.quad(lambda x: np.exp(1.0/(x**2-1.0)),-1.0,1.0)
    norm_const = 1.0/norm_const

    ### Compute the mollifier rho
    p = np.abs((s - (s[0]+s[-1])/2.0)/width)
    r = np.zeros_like(s)
    q = p[p<1.0]
    r[p<1.0] = np.exp(1.0/(q**2-1.0))
    rho = (norm_const/width)*r

    ### weight to make integral
    weighted = position_interp[:-1]*np.diff(s)

    ### Perform convolution to make smooth reconstruction
    if weighted.shape[0] > 500:
        #print('Convolving with FFT.')
        smooth = signal.fftconvolve(weighted,rho,mode='same')

        ### remove padding for FFT
        s = s[left_pad_num:-right_pad_num]
        smooth = smooth[left_pad_num:-(right_pad_num - 1)]
    else:
        #print('Convolving without FFT.')
        smooth = np.convolve(weighted,rho,mode='same')

        ### remove padding for non-FFT
        s = s[left_pad_num:-right_pad_num]
        smooth = smooth[left_pad_num:-(right_pad_num)]

    ### testing block
    ### print('time_pad.shape[0]',time_pad.shape[0])
    ### print('left_pad_num',left_pad_num)
    ### print('right_pad_num',right_pad_num)
    ### print('s.shape[0]',s.shape[0])
    ### print('weighted.shape[0]',weighted.shape[0])
    ### print('smooth.shape[0]',smooth.shape[0])

    ### Return the new time scale and the reconstruction
    return np.asarray([s, smooth])

def mollifier(time, position, refinement=1, width=20.0):
    """Smooth function in a way that retains shape by adding new graph values
    via the linear interpolate and smoothing via convolution with the
    Friedrichs' mollifier.

    Parameters
    ----------
    time : ndarray
        N by 1 time array.
    position : ndarray
        N by 3 position array.
    refinement : int
        Integer factor to refine the time discretization. Position
        information is added by linear interpolation
    width : double
        One half support width for Friedrichs' mollifier.

    Returns
    -------
    [s, smooth] : list
        List of the augmented N*(refinement) by 1 time array *s* and the
        N*(refinement) by 1 smooth array *smooth*.

    Notes
    -----
    Following Gilbarg and Trudinger [1]_ a *mollifier* is a non-negative
    :math:`C^\infty(\mathbb{R}^n)` function vanishing outside the unit ball
    whose integral is one. The standard (Friedrichs') mollifier is given by

    .. math::
       \\rho(x) = \\left \\{
       \\begin{aligned}
       &c \\exp \\left ( \\frac{1}{|x|^2 - 1} \\right ) &\\text{for }
       &|x| \\leq 1 \\\\
       &0 &\\text{for } &|x| \\geq 1.
       \\end{aligned}
       \\right .

    For :math:`u \\in L_{\\text{loc}}^1(\\Omega)` and :math:`h > 0`, the
    *regularization* of :math:`u` is given by

    .. math::
       u_h(x) = h^{-n} \\int_\\Omega \\rho\\left ( \\frac{x - y}{h} \\right )
       \\
       u(y) dy, \\quad h < \\text{dist}(x, \\partial \\Omega).

    This regularization has the following nice properties.

    - If :math:`u \\in L^1(\\Omega)`, then
      :math:`u_h \\in C^{\\infty}_0(\\Omega')` for any :math:`\Omega'` compactly
      contained in :math:`\\Omega` and for arbitrary :math:`h > 0`.
    - The function :math:`y \\mapsto h^{-n} \\rho((x-y)h^{-1})` tends a Dirac
      delta distribution as :math:`h \\to 0`.
    - If :math:`u \\in C^0(\\Omega)`, then :math:`u_h \\to u` uniformly on
      compact subsets of :math:`\\Omega`.

    The overall algorithm is taken from Knowles--Renka [2]_ and
    is summarized as follows.

    - Specify a mollifier width. Greater width yields greater smoothness.
    - Add points in via linear interpolation to increase resolution if desired.
    - Mollify

      - Form discrete versions of the proper dimensions of :math:`\\rho`,
        :math:`y \\mapsto h^{-1} \\rho((x-y)h^{-1})`.
      - Compute the discrete convolution analogous to

        .. math::
           h^{-n} \\int_\\Omega \\rho\\left ( \\frac{x - y}{h} \\right ) u(y) dy

    References
    ----------
    .. [1] D. Gilbarg and N. S. Trudinger, Elliptic Partial Differential
        Equations of Second Order. Springer Science & Business Media, 2001.

    .. [2] I. Knowles and R. J. Renka, "Methods for numerical differentiation of
        noisy data," Electronic Journal of Differential Equations, pp. 235--246,
        2014.
    """

    N = time.shape[0]
    assert N == position.shape[0]

    ### Pad the original arrays representing time and position by
    ### extending linearly in time by average time step and
    ### and by a constant for the function values of position.
    ### That is extend [time[0],time[-1]] to [time[0]-width,time[-1]+width]
    ### and extend position to match

    delta = (time[-1]-time[0])/(N-1)

    ### compute boundary time padding
    left_pad = np.arange(time[0],time[0]-(width+delta),step=-delta)
    left_pad = np.flipud(left_pad)[:-1]
    left_pad_num = left_pad.shape[0]
    right_pad = np.arange(time[-1],time[-1]+(width+delta),step=delta)[1:]
    right_pad_num = right_pad.shape[0]
    time_pad = np.concatenate((left_pad,time,right_pad))

    ### compute boundary position padding
    position_pad = np.pad(position,(left_pad_num,right_pad_num),'edge')

    ### Define a new smaller time scale s, ds (here we a evenly spaced)
    s, ds = np.linspace(time_pad[0],time_pad[-1],
                           (refinement)*time_pad.shape[0],
                           retstep=True)
    right_pad_num = (refinement)*right_pad_num
    left_pad_num = (refinement)*left_pad_num
    position_interp = np.interp(s,time_pad,position_pad)

    ### Compute the mollifier kernel
    norm_const,err = integrate.quad(lambda x: np.exp(1.0/(x**2-1.0)),-1.0,1.0)
    norm_const = 1.0/norm_const

    ### Compute the mollifier rho
    p = np.abs((s - (s[0]+s[-1])/2.0)/width)
    r = np.zeros_like(s)
    q = p[p<1.0]
    r[p<1.0] = np.exp(1.0/(q**2-1.0))
    rho = (norm_const/width)*r

    ### Perform convolution to make smooth reconstruction
    if s.shape[0] > 500:
        #print('Convolving with FFT.')
        smooth = signal.fftconvolve(ds*position_interp,rho,mode='same')
    else:
        #print('Convolving without FFT.')
        smooth = np.convolve(ds*position_interp,rho,mode='same')

    ### remove padding
    s = s[left_pad_num:-right_pad_num]
    smooth = smooth[left_pad_num:-(right_pad_num)]

    ### testing block
    #print('time.shape[0]',time.shape[0])
    #print('left_pad.shape[0]',left_pad.shape[0])
    #print('right_pad.shape[0]',right_pad.shape[0])
    #print('time_pad.shape[0]',time_pad.shape[0])
    #print('left_pad_num',left_pad_num)
    #print('right_pad_num',right_pad_num)
    #print('s.shape[0]',s.shape[0])
    #print('smooth.shape[0]',smooth.shape[0])

    return np.asarray([s, smooth])

            
cpdef np.ndarray[NDBL_t, ndim=2] secant_derivative(
      np.ndarray[NDBL_t, ndim=1] time,
      np.ndarray[NDBL_t, ndim=2] position,
      NINT_t secant_radius=1):
    """ 
    Compute the secant derivatives using points equally spaced BY INDEX.

    Parameters
    ----------
    time : `numpy.ndarray`
        N by 1 time array.
    position : `numpy.ndarray`
        N by 3 position array.
    secant_radius : `numpy.int64`
        The spacing is set by secant_radius (default 1).

    Returns
    -------
    sec_vel : ndarray
        Array of same dimensions as position, computed with second-order
        differences.

    Notes
    -----
    This is not ideal, numerically, but shouldn't be terrible for our data,
    as dt ~ 1 second.  As long as we don't sample by nanoseconds, it's OK.
    """
    
    cdef np.ndarray[NDBL_t, ndim=2] sec_vel = np.ndarray(
        shape=(position.shape[0], position.shape[1]))
    cdef NINT_t i, w = secant_radius
    cdef NINT_t N = time.shape[0]

    # middle is two-sided
    for i in range(w, sec_vel.shape[0] - w):
        delta_p = position[i+w,:] - position[i-w,:]
        delta_t = time[i+w] - time[i-w]
        sec_vel[i,:] = delta_p/delta_t

    # ends are one-sided.
    for i in range(w):
        delta_p = position[i+w,:] - position[i,:]
        delta_t = time[i+w] - time[i]
        sec_vel[i,:] = delta_p/delta_t

        delta_p = position[-i-1,:] - position[-i-1-w,:]
        delta_t = time[-i-1] - time[-i-1-w]
        sec_vel[i-1,:] = delta_p/delta_t

    return sec_vel

cpdef np.ndarray[NDBL_t, ndim=1] tangent_arclength(
      np.ndarray[NDBL_t, ndim=1] time,
      np.ndarray[NDBL_t, ndim=2] velocity):
    r"""Compute the cumulative arc-length by integrating speed.

    Parameters
    ----------
    time : ndarray
        N by 1 time array.
    position : ndarray
        N by 3 position array.

    Returns
    -------
    jump.cumsum() : double
        computes the sum :math:`\sum \|v_i\| dt_i`
    """

    cdef np.ndarray[NDBL_t, ndim=1] jump = np.ndarray(
        shape=(time.shape[0],) ) 
    jump[0] = 0.0
    jump[1:] = la.norm(velocity[:-1,:],axis=1)*np.diff(time)
    return jump.cumsum()


cpdef np.ndarray[NDBL_t, ndim=1] secant_arclength(
      np.ndarray[NDBL_t, ndim=2] position):
    """ Compute the cumulative arc-length sample-by-sample."""
    return secant_jump(position).cumsum()

cpdef np.ndarray[NDBL_t, ndim=1] secant_jump(
      np.ndarray[NDBL_t, ndim=2] position):
    """Compute the secant displacement sample-by-sample."""

    cdef np.ndarray[NDBL_t, ndim=1] jump = np.ndarray(
        shape=(position.shape[0],) ) 
    jump[0] = 0.0
    jump[1:] = la.norm(np.diff(position,axis=0),axis=1)
    return jump

cpdef np.ndarray[NDBL_t, ndim=2] frenet_kappa_tau(
      np.ndarray[NDBL_t, ndim=2] velocity,
      np.ndarray[NDBL_t, ndim=2] acceleration,
      np.ndarray[NDBL_t, ndim=2] jerk):
    r"""Compute KAPPA curvature and TAU torsion for a curve

    Parameters
    ----------
    velocity : ndarray
        N by 3 velocity array (time derivative of position).
    acceleration : ndarray
        N by 3 acceleration array (time derivative of velocity).
    jerk : ndarray
        N by 3 jerk array (time derivative acceleration).

    Returns
    -------
    kappa_tau : list
        List of :math:`\kappa` and :math:`\tau`

    Notes
    -----
    For a regular curve :math:`\alpha:I \to \mathbb{R}^3` we have

    .. math:: \kappa = \frac{|\alpha' \times \alpha''|}{|\alpha'|^3}

    .. math::
        \tau =
        \frac{(\alpha' \times \alpha'')\cdot \alpha'''}
        {|\alpha' \times \alpha''|^3}

    Here we identify :math:`\alpha'` with velocity, :math:`\alpha''` with
    acceleration, and :math:`\alpha'''` with jerk. See for example
    Oprea [1]_ page 29.

    References
    ----------
    .. [1] J. Oprea, Differential Geometry and Its Applications. MAA, 2007.

    See Also
    --------
    frenet_frame : a function for calculating T and N of a Frenet frame.
    """
    if not velocity.shape[1] == 3:
        raise ValueError("Frenet Frame requires 3-dimensional vectors.")

    cdef np.ndarray[NDBL_t, ndim=2] crossterm
    cdef np.ndarray[NDBL_t, ndim=1] crossnorm
    cdef np.ndarray[NDBL_t, ndim=2] kappa_tau = np.ndarray(shape=(velocity.shape[0],2))
    
    crossterm = np.cross(velocity, acceleration)
    crossnorm = la.norm(crossterm, axis=1)

    kappa_tau[:, 0] = crossnorm/(la.norm(velocity,axis=1)**3)
    kappa_tau[:, 1] = (crossterm*jerk).sum(axis=1)/(crossnorm**2)
    return kappa_tau 

cpdef np.ndarray[NDBL_t, ndim=3] frenet_frame(velocity, acceleration):
    r"""Find the Frenet frame using QR-decomposition at each time step.

    Parameters
    ----------
    velocity : ndarray
        N by 3 velocity array (time derivative of position).
    acceleration : ndarray
        N by 3 acceleration array (time derivative of velocity).

    Returns
    -------
    Q : ndarray
        Q from a reduced QR-decomposition, that is an orthonormalization of
        of velocity and acceleration resulting T and N of the Frenet frame.
        Here T[i,:] is Q[i,:,0], N[i,:] is Q[i,:,1].

    Notes
    -----
    The Frenet frame [1]_ provides local kinematics of a space curve and is
    subject to following fundamental relationship with curvature :math:`\kappa`
    and torsion :math:`\tau`: the Frenet formalas

    .. math::
        \begin{bmatrix}
        T'\\N'\\B'
        \end{bmatrix}=
        \begin{bmatrix}
        & \kappa & \\
        -\kappa & & \tau\\
        & -\tau
        \end{bmatrix}
        \begin{bmatrix}
        T\\N\\B
        \end{bmatrix}


    See Also
    --------
    frenet_kappa_tau : a function to calculate curvature and torsion

    References
    ----------
    .. [1] J. Oprea, Differential Geometry and Its Applications. MAA, 2007.

    """
    if not velocity.shape[1] == 3:
        raise ValueError("Frenet Frame requires 3-dimensional vectors.")

    cdef NINT_t i 
    cdef np.ndarray[NDBL_t, ndim=2] tmpX = np.ndarray(shape = (3, 2))
    cdef np.ndarray[NDBL_t, ndim=2] tmpQ = np.ndarray(shape = (3, 3))
    cdef np.ndarray[NDBL_t, ndim=2] tmpR = np.ndarray(shape = (3, 2))

    cdef np.ndarray[NDBL_t, ndim=3] Q = np.ndarray(
        shape = (velocity.shape[0], 3, 3) )
    
    cdef np.ndarray[NDBL_t, ndim=3] R = np.ndarray(
        shape = (velocity.shape[0], 3, 2) )

    for i in range(velocity.shape[0]):
        tmpX[:,0] = velocity[i,:]
        tmpX[:,1] = acceleration[i,:]
        tmpQ,tmpR = la.qr(tmpX)
        Q[i,:,:] = tmpQ[:,:]
        #R[i,:,:] = tmpR[:,:]

    return Q


cpdef similarity_matrix(a, b, func=euclidean):

    r"""similarity_matrix computes the similarity matrix of two 
    iterables given a specified metric, defaulting to a euclidean metric.

    Parameters
    ----------
    a : iterable
        some finite iterable, of length n
    b : iterable
        some finite iterable, of length n

    Returns
    -------
    out : `numpy.ndarray`
        an nxn numpy.ndarray where n is the length of a or b.  The i,jth 
        element of the matrix is the distance between the ith and jth element 
        of a,b respectively.

    Other Parameters
    ----------------
    func 
        the metric passed to compute distance.  defaults to `scipy.spatial.distance.euclidean`
    """
    
    assert len(a) == len(b), "Arrays must have same length"
    cdef NINT_t n = len(a)
    
    cdef NINT_t i, j
    cdef NDBL_t tmp
    cdef np.ndarray[NDBL_t, ndim=2] out = np.zeros((n,n), dtype=np.float64)
    for i in range(n):
        for j in range(i,n):
            tmp = func(a[i],b[j])
            out[i,j]=tmp
            out[j,i]=tmp
    return out

cpdef np.ndarray[NINT_t] contour_crossings(x, contour=0):
    """Take a one dimensional array and find the positions where the array crosses
    (or touches) a contour.  For the number of crossings, use
    `len(contour_crossings(x))` or `contour_crossings(x).shape[0]`
 
    Parameters
    ----------

    x : a 1-dimensional np array
    contour: contour to check crossings of.  Default 0

    Returns
    -------
    numpy array of int64 index locations where a contour crossing has occured.

    Examples
    --------
    >>> tmp = numpy.array([0.,1.,2.,3.,4.,-1.,-2.,-3.,4.,5.])
    >>> contour_crossings(tmp)
    array([0, 4, 7])
    >>> len(contour_crossings(tmp))
    3
    >>> contour_crossings(numpy.arange(42))
    array([0])
    >>> tmp = numpy.array([0.,1.,2.,3.,4.,-1.,-2.,0.,0.,-3.,4.,5.])
    >>> contour_crossings(tmp)
    array([0, 4, 9])
    
    """
   
    cdef np.ndarray[NBIT_t, cast=True] pos = (x > contour)
    cdef np.ndarray[NBIT_t, cast=True] npos = ~pos
    return ((pos[:-1] & npos[1:]) | (npos[:-1] & pos[1:])).nonzero()[0]

