import numpy as np
import scipy as sci
# import numba
# from numba import jit

class shapeFunctions(object):
    """
    Class implementing the fixed part of a hodographic trajectory shape
    The coefficients for the base functions here are computed in the
    hodographicShaping class from the boundary conditions of the
    corresponding interplanetary transfer
    I recommend using Matlab Symbolic or a similar tool to compute the analytical
    integrals and derivatives of new shaping functions as these derivations
    can become rather lengthy..
    The _scaled versions scale the time variable with respect to the total time
    of flight
    """

    def __init__(self, N, shorthand='CPowPow2', tMax=1):

        self.shape = shorthand
        self.N = N
        self.scale = 1/tMax
        a = self.scale

        if self.shape == 'CosR5P3CosR5P3SinR5':
            self.v1 = lambda t: np.cos((N + 0.5)*2*np.pi*t)
            self.v2 = lambda t: t**3 * np.cos((N + 0.5) * 2 * np.pi * t)
            self.v3 = lambda t: t**3 * np.sin((N + 0.5) * 2 * np.pi * t)
            self.Iv1 = lambda t: 1/((N + 0.5)*2*np.pi) * np.sin((N + 0.5)*2*np.pi*t)
            intPowSineCos =   lambda t, i, expo: (-1)**((i+1)/2)*sci.special.factorial(expo)\
                    /sci.special.factorial(expo+1-i)*t**(expo+1-i)\
                    *(1/((N + 0.5) * 2 * np.pi))**i*np.cos((N + 0.5) * 2 * np.pi * t)
            intPowSineSin =   lambda t, j, expo: (-1)**((j+2)/2)*sci.special.factorial(expo)\
                    /sci.special.factorial(expo+1-j)*t**(expo+1-j)\
                    *(1/((N + 0.5) * 2 * np.pi))**j*np.sin((N + 0.5) * 2 * np.pi * t)
            intPowCosineCos = lambda t, j, expo: (-1)**((j+2)/2)*sci.special.factorial(expo)\
                    /sci.special.factorial(expo+1-j)*t**(expo+1-j)\
                    *(1/((N + 0.5) * 2 * np.pi))**j*np.cos((N + 0.5) * 2 * np.pi * t)
            intPowCosineSin = lambda t, i, expo: (-1)**((i-1)/2)*sci.special.factorial(expo)\
                    /sci.special.factorial(expo+1-i)*t**(expo+1-i)\
                    *(1/((N + 0.5) * 2 * np.pi))**i*np.sin((N + 0.5) * 2 * np.pi * t)
            self.Iv2 = lambda t: intPowCosineCos(t, 2, 3) + intPowCosineCos(t, 4, 3)\
                    + intPowCosineSin(t, 1, 3) + intPowCosineSin(t, 3, 3)
            self.Iv3 = lambda t: intPowSineCos(t, 1, 3)   + intPowSineCos(t, 3, 3)\
                    + intPowSineSin(t, 2, 3)   + intPowSineSin(t, 4, 3)
            self.Dv1 = lambda t: - np.sin((N + 0.5)*2*np.pi*t) * (N + 0.5)*2*np.pi
            self.Dv2 = lambda t: 3*t**2 * np.cos((N + 0.5) * 2 * np.pi * t)\
                    - t**3 * (N + 0.5) * 2 * np.pi * np.sin((N + 0.5) * 2 * np.pi * t)
            self.Dv3 = lambda t: 3*t**2 * np.sin((N + 0.5) * 2 * np.pi * t)\
                    + t**3 * (N + 0.5) * 2 * np.pi * np.cos((N + 0.5) * 2 * np.pi * t)

        elif self.shape == 'CPowPow2':
            # Vr = c[0] + c[1]*t + c[2]*t**2
            self.v1 =  lambda t: 1
            self.v2 =  lambda t: t
            self.v3 =  lambda t: t**2
            self.Iv1 = lambda t: t
            self.Iv2 = lambda t: 0.5*t**2
            self.Iv3 = lambda t: 1/3*t**3
            self.Dv1 = lambda t: 0
            self.Dv2 = lambda t: 1
            self.Dv3 = lambda t: 2*t

        elif self.shape == 'CPow2CosR5':
            # Vr = c[0] + c[1]*t + c[2]*t**2
            self.v1 =  lambda t: 1
            self.v2 =  lambda t: t**2
            self.v3 =  lambda t: np.cos((N + 0.5)*2*np.pi*t*a)
            self.Iv1 = lambda t: t
            self.Iv2 = lambda t: 1/3*t**3
            self.Iv3 = lambda t: 1/((N + 0.5)*2*np.pi*a)\
                        * np.sin((N + 0.5)*2*np.pi*t*a)
            self.Dv1 = lambda t: 0
            self.Dv2 = lambda t: 2*t
            self.Dv3 = lambda t: - np.sin((N + 0.5)*2*np.pi*t*a)\
                        * (N + 0.5)*2*np.pi*a

        elif self.shape == 'CosR5P3CosR5P3SinR5_scaled':
            self.v1 =	lambda t: np.cos(a*t*np.pi*(2*N + 1))
            self.v2 =	lambda t: a**3*t**3*np.cos(a*t*np.pi*(2*N + 1))
            self.v3 =	lambda t: a**3*t**3*np.sin(a*t*np.pi*(2*N + 1))
            self.Dv1 =	lambda t: -a*np.pi*np.sin(a*t*np.pi*(2*N + 1))*(2*N + 1)
            self.Dv2 =	lambda t: 3*a**3*t**2*np.cos(a*t*np.pi*(2*N + 1))\
                        - a**4*t**3*np.pi*np.sin(a*t*np.pi*(2*N + 1))*(2*N + 1)
            self.Dv3 =	lambda t: 3*a**3*t**2*np.sin(a*t*np.pi*(2*N + 1))\
                        + a**4*t**3*np.pi*np.cos(a*t*np.pi*(2*N + 1))*(2*N + 1)
            self.Iv1 =	lambda t: np.sin(a*t*np.pi*(2*N + 1))/(a*np.pi*(2*N + 1))
            self.Iv2 =	lambda t: (6*a**3)/(a*np.pi + 2*N*a*np.pi)**4\
                        - (6*a**3*np.cos(a*t*np.pi*(2*N + 1)))\
                        /(a*np.pi + 2*N*a*np.pi)**4\
                        + (3*a**3*t**2*np.cos(a*t*np.pi*(2*N + 1)))\
                        /(a*np.pi + 2*N*a*np.pi)**2\
                        - (6*a**3*t*np.sin(a*t*np.pi*(2*N + 1)))\
                        /(a*np.pi + 2*N*a*np.pi)**3\
                        + (a**2*t**3*np.sin(a*t*np.pi*(2*N + 1)))\
                        /(np.pi*(2*N + 1))
            self.Iv3 =	lambda t: - a**3*np.cos(a*t*np.pi*(2*N + 1))\
                        *(t**3/(a*np.pi*(2*N + 1))\
                        - (6*t)/(a**3*np.pi**3*(2*N + 1)**3))\
                        - a**3*np.sin(a*t*np.pi*(2*N + 1))\
                        *(6/(a**4*np.pi**4*(2*N + 1)**4)\
                        - (3*t**2)/(a**2*np.pi**2*(2*N + 1)**2))

        elif self.shape == 'CPowPow2_scaled':
            # as above with an explicit scaling
            self.v1 =  lambda t: 1
            self.v2 =  lambda t: t * a
            self.v3 =  lambda t: (t*a)**2
            self.Iv1 = lambda t: t
            self.Iv2 = lambda t: 0.5*a * t**2
            self.Iv3 = lambda t: 1/3*a**2 * t**3
            self.Dv1 = lambda t: 0
            self.Dv2 = lambda t: a
            self.Dv3 = lambda t: 2*a**2 * t

        else:
            print('ERROR: This shape function is not defined ->', shorthand)

class shapeFunctionsFree(object):
    """
    Class implementing the free part of a hodographic trajectory shape
    The coefficients for the base functions here are set by an external loop,
    usually an optimizer
    """

    def __init__(self, N, coefficients, shorthand, tMax=1):

        self.shape = shorthand
        self.N = N
        self.scale = 1/tMax
        self.coeff = coefficients
        a = self.scale

        if self.shape == 'PSin05PCos05_scaled':
            self.v1 =	lambda t: a*t*np.sin((a*t*np.pi)/2)
            self.v2 =	lambda t: a*t*np.cos((a*t*np.pi)/2)
            self.Dv1 =	lambda t: a*np.sin((a*t*np.pi)/2)\
                        + (a**2*t*np.pi*np.cos((a*t*np.pi)/2))/2
            self.Dv2 =	lambda t: a*np.cos((a*t*np.pi)/2)\
                        - (a**2*t*np.pi*np.sin((a*t*np.pi)/2))/2
            self.Iv1 =	lambda t: (4*np.sin((a*t*np.pi)/2))/(a*np.pi**2)\
                        - (2*t*np.cos((a*t*np.pi)/2))/np.pi
            self.Iv2 =	lambda t: (2*t*np.sin((a*t*np.pi)/2))/np.pi\
                        - (8*np.sin((a*t*np.pi)/4)**2)/(a*np.pi**2)
            self.funNum = 2

        elif self.shape == 'P4CosR5P4SinR5_scaled':
            self.v1 =	lambda t: a**4*t**4*np.cos(a*t*np.pi*(2*N + 1))
            self.v2 =	lambda t: a**4*t**4*np.sin(a*t*np.pi*(2*N + 1))
            self.Dv1 =	lambda t: 4*a**4*t**3*np.cos(a*t*np.pi*(2*N + 1))\
                        - a**5*t**4*np.pi*np.sin(a*t*np.pi*(2*N + 1))*(2*N + 1)
            self.Dv2 =	lambda t: 4*a**4*t**3*np.sin(a*t*np.pi*(2*N + 1))\
                        + a**5*t**4*np.pi*np.cos(a*t*np.pi*(2*N + 1))*(2*N + 1)
            self.Iv1 =	lambda t: (24*a**4*np.sin(a*t*np.pi*(2*N + 1)))\
                        /(a*np.pi + 2*N*a*np.pi)**5\
                        + (4*a**4*t**3*np.cos(a*t*np.pi*(2*N + 1)))\
                        /(a*np.pi + 2*N*a*np.pi)**2\
                        - (12*a**4*t**2*np.sin(a*t*np.pi*(2*N + 1)))\
                        /(a*np.pi + 2*N*a*np.pi)**3\
                        - (24*a**4*t*np.cos(a*t*np.pi*(2*N + 1)))\
                        /(a*np.pi + 2*N*a*np.pi)**4 + \
                        (a**3*t**4*np.sin(a*t*np.pi*(2*N + 1)))/(np.pi*(2*N + 1))
            self.Iv2 =	lambda t: 24/(a*np.pi**5*(2*N + 1)**5)\
                        + a**4*np.sin(a*t*np.pi*(2*N + 1))\
                        *((4*t**3)/(a**2*np.pi**2*(2*N + 1)**2)\
                        - (24*t)/(a**4*np.pi**4*(2*N + 1)**4))\
                        - a**4*np.cos(a*t*np.pi*(2*N + 1))\
                        *(24/(a**5*np.pi**5*(2*N + 1)**5)\
                        + t**4/(a*np.pi*(2*N + 1))\
                        - (12*t**2)/(a**3*np.pi**3*(2*N + 1)**3))
            self.funNum = 2

        elif self.shape == 'Pow3Pow4':
            # Vr = c[0] + c[1]*t + c[2]*t**2
            self.v1 =  lambda t: a**3*t**3
            self.v2 =  lambda t: a**4*t**4
            self.Iv1 = lambda t: 1/4*a**3*t**4
            self.Iv2 = lambda t: 1/5*a**4*t**5
            self.Dv1 = lambda t: 3*a**3*t**2
            self.Dv2 = lambda t: 4*a**4*t**3
            self.funNum = 2

        else:
            print('ERROR: This shape function is not defined ->', shorthand)

        # check if the number of parameters match the ones expected for the shape
        if len(self.coeff) != self.funNum:
            print('ERROR: The number of coefficients does not match the chosen shape(', shorthand, '):')
            print('Given: ', len(self.coeff), ' Expected: ', self.funNum, '\n')

        # provide one function of time for position, velocity and acceleration
        self.v =  lambda t: self.coeff[0] * self.v1(t)  + self.coeff[1] * self.v2(t)
        self.Dv = lambda t: self.coeff[0] * self.Dv1(t) + self.coeff[1] * self.Dv2(t)
        self.Iv = lambda t: self.coeff[0] * self.Iv1(t) + self.coeff[1] * self.Iv2(t)
