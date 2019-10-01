import os
import time
import numpy as np
import scipy as sci
import scipy.misc
import pykep as pk

from hodographicShaping_SI import hodographicShaping
from integration import integrate
from shapingFunctions import shapeFunctions
from shapingFunctions import shapeFunctionsFree
from patchedTrajectoryUtils import *
from conversions import *

def fullTest1(rtol = 1e-3, atol = 1e-2):

    depMjd = 8002
    tof = 1500
    N = 3
    arrMjd = depMjd + tof
    depBody = 'earth'
    arrBody = 'mars'

    scStateDep, __, __ = ephemeris(depBody, depMjd)
    scStateArr, __, __ = ephemeris(arrBody, arrMjd)

    transfer = hodographicShaping(scStateDep, scStateArr,
                            departureBody='earth', arrivalBody='mars',
                            departureDate=8002, tof=1500, N=3,
                            rShape =         'CPowPow2_scaled',
                            thetaShape =     'CPowPow2_scaled',
                            zShape =         'CosR5P3CosR5P3SinR5_scaled',
                            rShapeFree =     'PSin05PCos05_scaled',
                            thetaShapeFree = 'PSin05PCos05_scaled',
                            zShapeFree =     'P4CosR5P4SinR5_scaled',
                            rFreeC =         [137, 178],
                            thetaFreeC =     [100, 1364],
                            zFreeC =         [32, 283],
                            )
    transfer.shapingRadial()
    transfer.shapingVertical()
    transfer.shapingTransverse()
    transfer.assembleThrust()
    transfer.checkBoundaryConditions()
    transfer.evaluate(evalThrust='Grid', printTime=False, nEvalPoints = 1000)

    testPoints = [transfer.psiTransfer, transfer.deltaV, transfer.maxThrust]
    # print(testPoints)
    knownPoints = [3.858953628300317, 273559.833321966, 0.003146874363948356]

    test = np.allclose(testPoints, knownPoints, rtol, atol)

    if test == True:
        print('OK\tFull test 1 (earth-mars, departureDate=8002, tof=1500, N=3)')
    else:
        print('ERROR\tFull test 1 (earth-mars, departureDate=8002,',
                'tof=1500, N=3)')
        print('\tComputed: ', testPoints)
        print('\tExpected: ', knownPoints)

def fullTest2(rtol = 1e-3, atol = 1e-2):

    # trajectory settings
    depMjd = 10025
    tof = 1050
    N = 2
    arrMjd = depMjd + tof
    depBody = 'earth'
    arrBody = 'mars'

    # departure and arrival states: rendezvous
    scStateDep, __, __ = ephemeris(depBody, depMjd)
    scStateArr, __, __ = ephemeris(arrBody, arrMjd)

    transfer = hodographicShaping(scStateDep, scStateArr,
                            departureDate=depMjd, tof=tof, N=N,
                            departureBody = depBody,
                            arrivalBody = arrBody,
                            rShape =         'CPowPow2_scaled',
                            thetaShape =     'CPowPow2_scaled',
                            zShape =         'CosR5P3CosR5P3SinR5_scaled',
                            rShapeFree =     'PSin05PCos05_scaled',
                            thetaShapeFree = 'PSin05PCos05_scaled',
                            zShapeFree =     'P4CosR5P4SinR5_scaled',
                            rFreeC =         [0, 0],
                            thetaFreeC =     [0, 0],
                            zFreeC =         [0, 0],
                            )
    transfer.shapingRadial()
    transfer.shapingVertical()
    transfer.shapingTransverse()
    transfer.assembleThrust()
    transfer.checkBoundaryConditions()
    transfer.evaluate(evalThrust='Grid', printTime=False, nEvalPoints = 1000)

    testPoints = [transfer.psiTransfer, transfer.deltaV, transfer.maxThrust]
    # print(testPoints)
    knownPoints = [2.573118029750549, 6338.852298544599, 0.00015143346642256032]

    test = np.allclose(testPoints, knownPoints, rtol, atol)

    if test == True:
        print('OK\tFull test 2 (earth-mars, departureDate=10025,',
                'tof=1050, N=2)')
    else:
        print('ERROR\tFull test 2 (earth-mars, departureDate=10025,',
                'tof=1050, N=2)')
        print('\tComputed: ', testPoints)
        print('\tExpected: ', knownPoints)

def fullTest3(rtol = 1e-3, atol = 1e-2):

    # trajectory settings
    depMjd = 12000
    tof = 300
    N = 1
    arrMjd = depMjd + tof
    depBody = 'earth'
    arrBody = 'venus'

    # departure and arrival states: rendezvous
    scStateDep, __, __ = ephemeris(depBody, depMjd)
    scStateArr, __, __ = ephemeris(arrBody, arrMjd)

    transfer = hodographicShaping(scStateDep, scStateArr,
                            departureDate=depMjd, tof=tof, N=N,
                            departureBody = depBody,
                            arrivalBody = arrBody,
                            rShape =         'CPowPow2_scaled',
                            thetaShape =     'CPowPow2_scaled',
                            zShape =         'CPowPow2_scaled',
                            rShapeFree =     'PSin05PCos05_scaled',
                            thetaShapeFree = 'PSin05PCos05_scaled',
                            zShapeFree =     'P4CosR5P4SinR5_scaled',
                            rFreeC =         [2, 20000],
                            thetaFreeC =     [173, 3460],
                            zFreeC =         [1990, 3333],
                            )
    transfer.shapingRadial()
    transfer.shapingVertical()
    transfer.shapingTransverse()
    transfer.assembleThrust()
    transfer.checkBoundaryConditions()
    transfer.evaluate(evalThrust='Grid', printTime=False, nEvalPoints = 1000)

    testPoints = [transfer.psiTransfer, transfer.deltaV, transfer.maxThrust]
    # print(testPoints)
    knownPoints = [0.71804214972971, 22535.748458173402, 0.0012091070214979359]

    test = np.allclose(testPoints, knownPoints, rtol, atol)

    if test == True:
        print('OK\tFull test 3 (earth-venus, departureDate=12000,',
                'tof=300, N=1)')
    else:
        print('ERROR\tFull test 3 (earth-venus, departureDate=12000,',
                'tof=300, N=1)')
        print('\tComputed: ', testPoints)
        print('\tExpected: ', knownPoints)

def fullTest4(rtol = 1e-3, atol = 1e-2):

    # trajectory settings
    depMjd = 9453
    tof = 844
    N = 1
    arrMjd = depMjd + tof
    depBody = '3'
    arrBody = '4'
    ephems = 'spice'

    # departure and arrival states: rendezvous
    scStateDep, __, __ = ephemeris(depBody, depMjd, mode=ephems)
    scStateArr, __, __ = ephemeris(arrBody, arrMjd, mode=ephems)

    transfer = hodographicShaping(scStateDep, scStateArr,
                            departureDate=depMjd, tof=tof, N=N,
                            departureBody = depBody,
                            arrivalBody = arrBody,
                            rShape =         'CPowPow2_scaled',
                            thetaShape =     'CPowPow2_scaled',
                            zShape =         'CosR5P3CosR5P3SinR5_scaled',
                            rShapeFree =     'PSin05PCos05_scaled',
                            thetaShapeFree = 'PSin05PCos05_scaled',
                            zShapeFree =     'P4CosR5P4SinR5_scaled',
                            rFreeC =         [137, 178],
                            thetaFreeC =     [100, 1364],
                            zFreeC =         [32, 283],
                            )
    transfer.shapingRadial()
    transfer.shapingVertical()
    transfer.shapingTransverse()
    transfer.assembleThrust()
    transfer.checkBoundaryConditions()
    transfer.evaluate(evalThrust='Grid', printTime=False, nEvalPoints = 1000)

    testPoints = [transfer.psiTransfer, transfer.deltaV, transfer.maxThrust]
    knownPoints = [5.213985160376169, 45508.713465061395, 0.0010283457257095393]

    test = np.allclose(testPoints, knownPoints, rtol, atol)

    if test == True:
        print('OK\tFull test 4 (3-4, departureDate=9453, tof=844, N=1)')
    else:
        print('ERROR\tFull test 4 (3-4, departureDate=9453, tof=844, N=1)')
        print('\tComputed: ', testPoints)
        print('\tExpected: ', knownPoints)

def boundaryConditionTestVel(rtol = 1e-3, atol = 1e-2):

    # trajectory settings
    depMjd = 10025
    tof = 1050
    N = 2
    arrMjd = depMjd + tof
    depBody = 'earth'
    arrBody = 'mars'
    ephems = 'jpl'

    # departure and arrival states: rendezvous
    scStateDep, __, __ = ephemeris(depBody, depMjd, mode=ephems)
    scStateArr, __, __ = ephemeris(arrBody, arrMjd, mode=ephems)

    transfer = hodographicShaping(scStateDep, scStateArr,
                            departureDate=depMjd, tof=tof, N=N,
                            departureBody = depBody,
                            arrivalBody = arrBody,
                            rShape =         'CPowPow2_scaled',
                            thetaShape =     'CPowPow2_scaled',
                            zShape =         'CosR5P3CosR5P3SinR5_scaled',
                            rShapeFree =     'PSin05PCos05_scaled',
                            thetaShapeFree = 'PSin05PCos05_scaled',
                            zShapeFree =     'P4CosR5P4SinR5_scaled',
                            rFreeC =         [0, 0],
                            thetaFreeC =     [0, 0],
                            zFreeC =         [0, 0],
                            )
    transfer.shapingRadial()
    transfer.shapingVertical()
    transfer.shapingTransverse()

    testPoints = [transfer.rDot(transfer.tofSec),
                transfer.tDot(transfer.tofSec),
                transfer.zDot(transfer.tofSec)]
    # print(testPoints)
    knownPoints = [2161.3061456572896, 24897.147647368587, 802.1571912246434]
    test = np.allclose(testPoints, knownPoints, rtol, atol)

    if test == True:
        print('OK\tBoundary conditions velocity test')
    else:
        print('ERROR\tBoundary conditions velocity test')
        print('\tComputed: ', testPoints)
        print('\tExpected: ', knownPoints)

def boundaryConditionTestPos(rtol = 1e-3, atol = 1e-2):

    # trajectory settings
    depMjd = 10025
    tof = 1050
    N = 2
    arrMjd = depMjd + tof
    depBody = 'earth'
    arrBody = 'mars'
    ephems = 'jpl'

    # departure and arrival states: rendezvous
    scStateDep, __, __ = ephemeris(depBody, depMjd, mode=ephems)
    scStateArr, __, __ = ephemeris(arrBody, arrMjd, mode=ephems)

    transfer = hodographicShaping(scStateDep, scStateArr,
                            departureDate=depMjd, tof=tof, N=N,
                            departureBody = depBody,
                            arrivalBody = arrBody,
                            rShape =         'CPowPow2_scaled',
                            thetaShape =     'CPowPow2_scaled',
                            zShape =         'CosR5P3CosR5P3SinR5_scaled',
                            rShapeFree =     'PSin05PCos05_scaled',
                            thetaShapeFree = 'PSin05PCos05_scaled',
                            zShapeFree =     'P4CosR5P4SinR5_scaled',
                            rFreeC =         [0, 0],
                            thetaFreeC =     [0, 0],
                            zFreeC =         [0, 0],
                            )
    transfer.shapingRadial()
    transfer.shapingVertical()
    transfer.shapingTransverse()

    testPoints = [transfer.r(transfer.tofSec),
                transfer.t(transfer.tofSec),
                # 0,
                transfer.z(transfer.tofSec)]
    # print(testPoints)
    knownPoints = [219832470724.13513, 13.418874318624699, -77456667.88140798]
    test = np.allclose(testPoints, knownPoints, rtol, atol)

    if test == True:
        print('OK\tBoundary conditions position test')
    else:
        print('ERROR\tBoundary conditions position test')
        print('\tComputed: ', testPoints)
        print('\tExpected: ', knownPoints)

def boundaryConditionComparison():

    # trajectory settings
    depMjd = 3421
    tof = 349
    N = 1
    arrMjd = depMjd + tof
    depBody = '3'
    arrBody = '4'
    ephems = 'spice'

    # departure and arrival states: rendezvous
    scStateDep, __, __ = ephemeris(depBody, depMjd, mode=ephems)
    scStateArr, __, __ = ephemeris(arrBody, arrMjd, mode=ephems)

    transfer = hodographicShaping(scStateDep, scStateArr)
    transfer.shapingRadial()
    transfer.shapingVertical()
    transfer.shapingTransverse()
    transfer.assembleThrust()
    transfer.checkBoundaryConditions()
    if transfer.velCompare == True and transfer.posCompare == True:
        print('OK\tBoundary conditions comparison (built-in)')
    else:
        print('ERROR\tBoundary conditions comparison (built-in)')

def integrationTest1(rtol = 1e-3, atol = 1e-2):

    func1 = lambda x: x**2 * np.cos(x)
    x0 = 0
    x1 = 100
    nSteps = int(1e5)

    # quad is reference solution
    intResult1 = integrate(func1, x0, x1, method='quad')

    # compute same result using other methods
    intResult2 = integrate(func1, x0, x1, method='trapz', nSteps=nSteps)

    test = np.allclose(intResult1, intResult2, rtol, atol)

    if test == True:
        print('OK\tIntegration test')
    else:
        print('ERROR\tIntegration test')
        print('\tComputed quad:\t', intResult1)
        print('\tComputed others:', intResult2)

def checkShapes(rtol = 1e-3, atol = 1e-2):

    def aNum(function, t):
        '''
        Numerical derivative
        '''
        a = sci.misc.derivative(function, t, 1e-5)
        return a

    def sNum(function, t):
        '''
        Numerical integral
        '''
        s, err = sci.integrate.quad(function, 0, t)
        return s

    def sampleShapes(shape, tEval, nShapes=3):
        '''
        Samples a set of shape functions (integral and derivative)
            at the points in tEval
        Also computes numerical approximations of integrals and derivatives
        '''

        # sample position
        Iv1 = np.zeros(np.shape(tEval))
        Iv2 = np.zeros(np.shape(tEval))
        Iv1num = np.zeros(np.shape(tEval))
        Iv2num = np.zeros(np.shape(tEval))
        Dv1 = np.zeros(np.shape(tEval))
        Dv2 = np.zeros(np.shape(tEval))
        Dv1num = np.zeros(np.shape(tEval))
        Dv2num = np.zeros(np.shape(tEval))
        if nShapes == 3:
            Iv3 = np.zeros(np.shape(tEval))
            Iv3num = np.zeros(np.shape(tEval))
            Dv3 = np.zeros(np.shape(tEval))
            Dv3num = np.zeros(np.shape(tEval))

        for i in np.arange(0, len(tEval)):
            Iv1[i] = shape.Iv1(tEval[i])
            Iv2[i] = shape.Iv2(tEval[i])
            Iv1num[i] = sNum(shape.v1, tEval[i])
            Iv2num[i] = sNum(shape.v2, tEval[i])
            Dv1[i] = shape.Dv1(tEval[i])
            Dv2[i] = shape.Dv2(tEval[i])
            Dv1num[i] = aNum(shape.v1, tEval[i])
            Dv2num[i] = aNum(shape.v2, tEval[i])
            if nShapes == 3:
                Iv3[i] = shape.Iv3(tEval[i])
                Iv3num[i] = sNum(shape.v3, tEval[i])
                Dv3[i] = shape.Dv3(tEval[i])
                Dv3num[i] = aNum(shape.v3, tEval[i])

        if nShapes == 3:
            samplesAnalytical = np.vstack([Dv1, Dv2, Dv3, Iv1, Iv2, Iv3])
            samplesNumerical = np.vstack([Dv1num, Dv2num, Dv3num,
                                        Iv1num, Iv2num, Iv3num])
        elif nShapes == 2:
            samplesAnalytical = np.vstack([Dv1, Dv2, Iv1, Iv2])
            samplesNumerical = np.vstack([Dv1num, Dv2num, Iv1num, Iv2num])

        return samplesAnalytical, samplesNumerical

    # number of revolutions
    N = 2

    # time interval and sampling steps
    tMax = 500*24*60*60
    tMax = 11
    nSamples = 1001
    tEval = np.linspace(0, tMax, nSamples)

    # test shaping functions
    shorthands = [  'CPowPow2',
                    'CPowPow2_scaled',
                    'CPow2CosR5',
                    'CosR5P3CosR5P3SinR5',
                    'CosR5P3CosR5P3SinR5_scaled',
                    ]

    # return error if one of the specified functions is not close to the
    # numerical computation
    test = True
    errShapes = ''
    for shorthand in shorthands:
        shape = shapeFunctions(N=N, shorthand=shorthand, tMax=tMax)
        samplesAnalytical, samplesNumerical = sampleShapes(shape,
                                                        tEval, nShapes=3)
        comp = np.allclose(samplesAnalytical, samplesNumerical, rtol, atol)
        if comp == False:
            errShapes = errShapes + shorthand + ' '
        test = test and comp

    if test == True:
        print('OK\tShaping base functions test')
    else:
        print('ERROR\tShaping base functions test')
        print('\tError occured in shape(s)', errShapes)

    # test shaping functions for free parameters
    shorthands = [  'PSin05PCos05_scaled',
                    'P4CosR5P4SinR5_scaled',
                    'Pow3Pow4',
                    ]

    test = True
    errShapes = ''
    for shorthand in shorthands:
        shape = shapeFunctionsFree(N=N, coefficients=[0, 0],
                                shorthand=shorthand, tMax=tMax)
        samplesAnalytical, samplesNumerical = sampleShapes(shape,
                                                        tEval, nShapes=2)
        comp = np.allclose(samplesAnalytical, samplesNumerical, rtol, atol)
        if comp == False:
            errShapes = errShapes + shorthand + ' '
        test = test and comp

    if test == True:
        print('OK\tShaping free base functions test')
    else:
        print('ERROR\tShaping free base functions test')
        print('\tError occured in shape(s)', errShapes)

def testDeltaV():
    '''
    Test the application of an instantanious velocity change (impulsive shot)
    to a state vector with a simple example
    '''

    initialState = np.array([1, 1, 0, 0, 1, 0])
    deltaV = np.array([1, 0, 0])
    computedState = applyDeltaV(initialState, deltaV)
    knownState = np.array([1, 1, 0, 1, 1, 0])

    comp = np.allclose(computedState, knownState)
    if comp == True:
        print('OK\tApply impulsive shot test')
    else:
        print('ERROR\tApply impulsive shot test')
        print('\tComputed', computedState)
        print('\tExpected', knownState)

def testFlybys(rtol = 1e-3, atol = 1e-2):
    '''
    Test the functions that compute an unpowered planetary swingby (2D and 3D)
    Both are not comparable because the 2D version internally projects the
        planet's velocity into the ecliptic plane
    '''

    np.set_printoptions(precision=3)

    planetName = 'mars'
    mjd = 3204.4180595730954
    Bmult = 2
    makePlot = False
    printStatus = False

    # arrival state vector
    stateVector = [2.341e11, -2.3698, 6.9953e8, -2.6145e3, 2.2085e4, 0]
    stateVectorArr = stateCyl2cart(stateVector)

    # 2D flyby function (old)
    newState1cart, DeltaV1 = flyby2D(stateVectorArr, planetName, mjd=mjd,
                        Bmult=Bmult, alphaSignFlip=False, save=False, show=False,
                        makePlot=makePlot, printStatus=printStatus)
    newState1 = stateCart2cyl(newState1cart)

    # known state vector (from an old simulation)
    knownState1 = [2.341e11, -2.37, 6.995e8, -9.453e2, 2.416e4, 0]

    # compute the flyby distance for the given Bmult
    statePlaCyl, statePlaCart, planet = ephemeris(planetName, mjd, mode='jpl')
    R = planet.radius
    mu = planet.mu_self
    B = R*Bmult
    V2 = stateVectorArr[3:6]
    Vplanet = statePlaCart[3:6]
    Vplanet[2] = 0
    Vinfty = np.linalg.norm(V2 - Vplanet)
    r3 = mu/Vinfty**2 * (np.sqrt(1 + B**2 * Vinfty**4 / mu**2) - 1)

    # 3D flyby function (new)
    newState2cart, DeltaV2 = flyby(stateVectorArr, planetName, r3, np.pi/2,
                                mjd=mjd, save=False, show=True,
                                makePlot=makePlot, printStatus=printStatus)
    newState2 = stateCart2cyl(newState2cart)

    # known state vector (from an old simulation)
    knownState2 = [2.341e11, -2.37, 6.995e8, -1.122e3, 2.382e4, -1.926e3]

    comp = np.allclose(newState1, knownState1, rtol, atol)
    if comp == True:
        print('OK\tFlyby test: 2D Results did not change')
    else:
        print('ERROR\tFlyby test: 2D Change of results')
        print('\tComputed:', np.array(newState1))
        print('\tExpected:', np.array(knownState1))

    comp = np.allclose(newState2, knownState2, rtol, atol)
    if comp == True:
        print('OK\tFlyby test: 3D Results did not change')
    else:
        print('ERROR\tFlyby test: 3D Change of results')
        print('\tComputed:', np.array(newState2))
        print('\tExpected:', np.array(knownState2))

    return None

if __name__ == "__main__":
    """
    Run the unit tests defined above
    Static reference values were created using the precise but slow 'quad'
    integration method
    Some test dynamically compute a reference value and compare to that
    Running this is useful to check if the local installation works or to check
    if changes break something
    """

    loadSpiceKernels()

    print('###################################################################')
    print('Running unit tests:')
    rtol = 1e-2
    atol = 0.1
    print('Absolute tolerance: ', atol)
    print('Relative tolerance: ', rtol, '\n')

    # test integration routines
    # compare to reference implementation
    integrationTest1(rtol, atol)

    # test the patched trajectory functions
    # compare to tabulated data from previous simulation
    testDeltaV()
    testFlybys(rtol, atol)

    # test analytical shaping functions
    # compare to numerical computed function
    checkShapes(rtol, atol)

    # boundary condition check is built in
    # compare planet to shape positions
    boundaryConditionComparison()

    # test full shaping method
    # compare final DeltaV and maxThrust to known values
    boundaryConditionTestVel(rtol, atol)
    boundaryConditionTestPos(rtol, atol)
    fullTest1(rtol, atol)
    fullTest2(rtol, atol)
    fullTest3(rtol, atol)
    fullTest4(rtol, atol)
    print('###################################################################')
