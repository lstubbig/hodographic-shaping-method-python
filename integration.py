import time

import scipy as sci
import numpy as np

def integrate(function, x0, x1, method='quad', nSteps=25):
    """
    Different integration methods
    """

    if method == 'quad':
        int, err = sci.integrate.quad(function, x0, x1)
        return int

    if method == 'trapz':
        xSamples = np.linspace(x0, x1, nSteps+1)
        funcSamples = function(xSamples)
        int = sci.integrate.trapz(funcSamples, xSamples)
        return int

    else:
        print('ERROR: Integration method not found: ', str(method))
