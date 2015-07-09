import numpy
from scipy.special import ellipj, ellipk

#
# The scipy implementation of ellipj only works on reals;
# this gives cn(z,k) for complex z. 
# It works with array or scalar input.
#
def cplx_cn( z, k):
    z = numpy.asarray(z)
    if z.dtype != complex:
        return ellipj(z,k)[1]

    # note, 'dn' result of ellipj is buggy, as of Mar 2015
    # see https://github.com/scipy/scipy/issues/3904
    ss,cc = ellipj( z.real, k )[:2]
    dd = numpy.sqrt(1-k*ss**2)   # make our own dn
    s1,c1 = ellipj( z.imag, 1-k )[:2]
    d1 = numpy.sqrt(1-k*s1**2)

    ds1 = dd*s1
    den = (1-ds1**2)
    rx = cc*c1/den
    ry = ss*ds1*d1/den
    return rx - 1j*ry
#
# Kval is the first solution to cn(x,1/2) = 0
# This is K(k) (where 4*K(k) is the period of the function).
Kval = ellipk(0.5) # 1.8540746773013719

#######################################################
# map a complex point in unit square to unit circle
# The following points are the corners of the square (and map to themselves):
#     1    -1     j    -j
#  The origin also maps to itself.
# Points which are in : abs( re(z)) <=1, abs(im(z)) <=1, but outside the square, will map to
# points outside the unit circle, but are still consistent with mapping a full-sphere
# peirce projection to a full-sphere stereographic projection; however that means that
# the corners 1+j, 1-j, -1+j -1-j all map to the 'south pole' at infinity. You will get
# a divide-by-zero, or near to it, at or near those points.
# It works with array or scalar input.
#
def peirce_map( z ):
    return cplx_cn( Kval*(1-z), 0.5 )