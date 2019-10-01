# Hodographic shaping in SI units
# Leon Stubbig, 2019

# Based on
# Paper: [Gondelach 2015]
# Thesis: [Gondelach 2012]

import time

import numpy as np
import matplotlib as mlt
import matplotlib.pyplot as plt
import pykep as pk
import scipy as sci

from conversions import *
from utils import *
from shapingFunctions import shapeFunctions
from shapingFunctions import shapeFunctionsFree
from integration import integrate
from patchedTrajectoryUtils import ephemeris

# main computation
class hodographicShaping(object):
    """
    Class implementing the hodographic shaping method
    Creating an object creates a trajectory, whose parameters can be computed
    by the methods
    Call methods in the order they are defined, transverseShaping needs the
    result from radialShaping
    Call status last to show results of computation
    Example:    transfer = hodographicShaping()
                transfer.shapingRadial()
                transfer.shapingVertical()
                transfer.shapingTransverse()
                transfer.assembleThrust()
                transfer.checkBoundaryConditions()
                transfer.evaluate()
                transfer.status()
    """

    def __init__(self,
                departureState,
                arrivalState,
                departureDate = 7400,        # only used in status and plotting
                tof = 500,
                N = 0,
                departureBody = 'earth',     # only used in status and plotting
                arrivalBody = 'mars',        # only used in status and plotting
                rShape = 'CPowPow2_scaled',
                thetaShape = 'CPowPow2_scaled',
                zShape = 'CosR5P3CosR5P3SinR5_scaled',
                rShapeFree = 'PSin05PCos05_scaled',
                thetaShapeFree = 'PSin05PCos05_scaled',
                zShapeFree = 'P4CosR5P4SinR5_scaled',
                rFreeC = [0, 0],
                thetaFreeC = [0, 0],
                zFreeC = [0, 0]
                ):
        '''
        Initializes the trajectory computation object
        '''

        # store time at start of computation
        self.timeStart = time.process_time()

        # number of orbital revolutions
        self.N = N
        # catch instances where N is not an integer
        if (N%1 != 0):
            print('ERROR: N has to be an integer. N =', N)

        # time of flight in days
        self.tof = tof

        # pick departure date and ToF/arrival date
        # dates in mjd2000
        self.jdDep = departureDate
        self.jdArr = self.jdDep + self.tof

        # chose shaping functions
        self.rShape = rShape
        self.thetaShape = thetaShape
        self.zShape = zShape
        self.rShapeFree = rShapeFree
        self.thetaShapeFree = thetaShapeFree
        self.zShapeFree = zShapeFree

        # free coefficients, set externally (by optimizer)
        self.rFreeC = rFreeC
        self.thetaFreeC = thetaFreeC
        self.zFreeC = zFreeC

        # get state vectors (for trajectory computation)
        self.xDep = departureState
        self.xArr = arrivalState
        self.rDepCyl = self.xDep[0:3]
        self.rArrCyl = self.xArr[0:3]
        self.vDepCyl = self.xDep[3:6]
        self.vArrCyl = self.xArr[3:6]

        # planets (for plotting and convenience)
        self.departureBody = departureBody
        self.arrivalBody = arrivalBody

        # polar angle at departure, transfer angle and arrival
        self.psiTransfer = self.rArrCyl[1] - self.rDepCyl[1]
        if self.psiTransfer < 0:
            self.psiTransfer = self.psiTransfer + 2*np.pi
        self.thetaArr = self.psiTransfer + N*2*np.pi
        # time of flights in seconds
        self.tofSec = self.tof * 24 * 60 * 60

    def status(self, printBC=False, precision=2):
        '''
        Retrieve and print the status of the computation (settings, results,...)
        The boundary conditions can be toggled on and off
        Call in the end of the computation when all results are available
        '''

        np.set_printoptions(precision=precision)

        print('###############################################################')
        print(f'Hodographic Shaping Problem: {self.departureBody} to {self.arrivalBody}')
        print('\nSettings')
        print('Departure state: ', np.array(self.xDep))
        print('Arrival state:\t ', np.array(self.xArr))
        print('Departure date:\t', pk.epoch(self.jdDep, 'mjd2000'))
        print('Departure date:\t', np.array(self.jdDep), 'mjd2000')
        print('Arrival date:\t', pk.epoch(self.jdArr, 'mjd2000'))
        print('Time of Flight:\t', np.array(self.tof), ' days')
        print('Revolutions:\t', self.N)
        print('Transfer angle: ', round(self.psiTransfer*180/np.pi, 2), ' deg')
        print('Radial velocity:\t', self.rShape)
        print('Traverse velocity:\t', self.thetaShape)
        print('Axial velocity:\t\t', self.zShape)
        print('\nFree part of shape (input)')
        print('Radial velocity free:\t' + self.rShapeFree)
        print('Traverse velocity free:\t' + self.thetaShapeFree)
        print('Axial velocity free:\t' + self.zShapeFree)
        print('Radial coefficients free:\t', np.array(self.rFreeC))
        print('Transverse coefficients free:\t', np.array(self.thetaFreeC))
        print('Vertical coefficients free:\t', np.array(self.zFreeC))
        print('\nVelocity functions')
        print('Radial coefficients:\t\t', self.cRadial)
        print('Transverse coefficients:\t', self.cTheta)
        print('Vertical coefficients:\t\t', self.cVertical)
        print('Position offsets (r0, theta0, z0): ', np.array([self.rDepCyl[0],
                self.rDepCyl[1], self.rDepCyl[2]]))
        try:
            if self.velCompare:
                print('\nBoundary condition check:')
                print('Velocity boundary conditions are satisfied!',
                        ' Difference < ', self.velTolAbs, ' m/s')
            else:
                print('\nBoundary condition check:')
                print('ERROR: Velocity boundary conditions are not satisfied!',
                        ' Difference > ', self.velTolAbs, ' m/s')
            if self.posCompare:
               print('Position boundary conditions are satisfied!',
                    ' Difference < ', self.posTolAbs, 'm and rad')
            else:
               print('ERROR: Position boundary conditions are not satisfied!',
                    ' Difference > ', self.posTolAbs, ' m and rad')
        except AttributeError:
            print('\nFullfilment of boundary conditions was not',
                    'explicitly checked.')
        if printBC:
            print('\nBoundary conditions:')
            print('Pos planet dep (r, theta, z):\t',
                np.array( [self.rDepCyl[0], self.rDepCyl[1], self.rDepCyl[2]] ))
            print('Pos shape dep (r, theta, z):\t',
                np.array( [self.r(0), self.t(0), self.z(0)] ))
            print('Pos planet arr (r, theta, z):\t',
                np.array( [self.rArrCyl[0], self.rArrCyl[1], self.rArrCyl[2]] ))
            print('Pos shape arr (r, theta, z):\t',
                np.array( [self.r(self.tofSec),
                self.t(self.tofSec), self.z(self.tofSec)] ))
            print('Vel planet dep (rDot, vTheta, zDot):\t',
                np.array( [self.vDepCyl[0], self.vDepCyl[1], self.vDepCyl[2]] ))
            print('Vel shape dep (rDot, vTheta, zDot):\t',
                np.array( [self.rDot(0), self.tDot(0), self.zDot(0)] ))
            print('Vel planet arr (rDot, vTheta, zDot):\t',
                np.array( [self.vArrCyl[0], self.vArrCyl[1], self.vArrCyl[2]] ))
            print('Vel shape arr (rDot, vTheta, zDot):\t',
                np.array( [self.rDot(self.tofSec),
                self.tDot(self.tofSec), self.zDot(self.tofSec)] ))
        # computation time
        print('\nComputation time')
        timeEnd = time.process_time()
        print('Computing this trajectory took {:.3f} ms'\
            .format((timeEnd - self.timeStart)*1000.0))
        # print results
        print('\nResults')
        print('DeltaV:\t\t', round(self.deltaV/1e3, 5), ' km/s')
        print('Max thrust:\t', round(self.maxThrust, 7), ' m/s^2')
        print('###############################################################')

    def shapingRadial(self):
        '''
        Compute coefficients for the radial shape
        '''

        rFunc = shapeFunctions(self.N, shorthand=self.rShape, tMax=self.tofSec)
        rFuncFree = shapeFunctionsFree(self.N, self.rFreeC,
                                    shorthand=self.rShapeFree, tMax=self.tofSec)

        # compute parameters
        A = np.array([[rFunc.v1(0), rFunc.v2(0), rFunc.v3(0)],
                      [rFunc.v1(self.tofSec), rFunc.v2(self.tofSec),
                       rFunc.v3(self.tofSec)],
                      [rFunc.Iv1(self.tofSec) - rFunc.Iv1(0),
                       rFunc.Iv2(self.tofSec) - rFunc.Iv2(0),
                       rFunc.Iv3(self.tofSec)-rFunc.Iv3(0)]])
        b = np.array([self.vDepCyl[0] - rFuncFree.v(0),
                      self.vArrCyl[0] - rFuncFree.v(self.tofSec),
                      self.rArrCyl[0] - self.rDepCyl[0]
                      - (rFuncFree.Iv(self.tofSec) - rFuncFree.Iv(0))])
        self.cRadial = np.linalg.solve(A, b)

        # assemble shape
        self.r = lambda t: (self.rDepCyl[0] + self.cRadial[0] * rFunc.Iv1(t)
                            + self.cRadial[1] * rFunc.Iv2(t)
                            + self.cRadial[2] * rFunc.Iv3(t)
                            + rFuncFree.Iv(t))
        self.rDot = lambda t: (self.cRadial[0] * rFunc.v1(t)
                            + self.cRadial[1] * rFunc.v2(t)
                            + self.cRadial[2] * rFunc.v3(t)
                            + rFuncFree.v(t))
        self.rDDot = lambda t: (self.cRadial[0] * rFunc.Dv1(t)
                            + self.cRadial[1] * rFunc.Dv2(t)
                            + self.cRadial[2] * rFunc.Dv3(t)
                            + rFuncFree.Dv(t))

    def shapingVertical(self):
        '''
        Compute coefficients for the vertical (z) shape
        '''

        zFunc = shapeFunctions(self.N, shorthand=self.zShape, tMax=self.tofSec)
        zFuncFree = shapeFunctionsFree(self.N, self.zFreeC,
                                    shorthand=self.zShapeFree, tMax=self.tofSec)

        A = np.array([[zFunc.v1(0), zFunc.v2(0), zFunc.v3(0)],
                      [zFunc.v1(self.tofSec), zFunc.v2(self.tofSec),
                       zFunc.v3(self.tofSec)],
                      [zFunc.Iv1(self.tofSec)-zFunc.Iv1(0),
                       zFunc.Iv2(self.tofSec)-zFunc.Iv2(0),
                       zFunc.Iv3(self.tofSec)-zFunc.Iv3(0)]])
        b = np.array([self.vDepCyl[2] - zFuncFree.v(0),
                      self.vArrCyl[2] - zFuncFree.v(self.tofSec),
                      self.rArrCyl[2] - self.rDepCyl[2]
                      - (zFuncFree.Iv(self.tofSec) - zFuncFree.Iv(0))])
        self.cVertical = np.linalg.solve(A, b)

        # assemble shape
        self.z = lambda t: (self.rDepCyl[2] + self.cVertical[0] * zFunc.Iv1(t)
                        + self.cVertical[1] * zFunc.Iv2(t)
                        + self.cVertical[2] * zFunc.Iv3(t)
                        + zFuncFree.Iv(t))
        self.zDot = lambda t: (self.cVertical[0] * zFunc.v1(t)
                            + self.cVertical[1] * zFunc.v2(t)
                            + self.cVertical[2] * zFunc.v3(t)
                            + zFuncFree.v(t))
        self.zDDot = lambda t: (self.cVertical[0] * zFunc.Dv1(t)
                            + self.cVertical[1] * zFunc.Dv2(t)
                            + self.cVertical[2] * zFunc.Dv3(t)
                            + zFuncFree.Dv(t))

    def shapingTransverse(self):
        '''
        Compute coefficients for the transverse (theta) shape
        '''

        thetaFunc = shapeFunctions(self.N, shorthand=self.thetaShape,
                            tMax=self.tofSec)
        thetaFuncFree = shapeFunctionsFree(self.N, self.thetaFreeC,
                            shorthand=self.thetaShapeFree, tMax=self.tofSec)

        # intermediate values
        [K1, K2] = np.dot(np.linalg.inv([[thetaFunc.v1(0), thetaFunc.v2(0)],
                    [thetaFunc.v1(self.tofSec), thetaFunc.v2(self.tofSec)]]),
                    [-thetaFunc.v3(0), - thetaFunc.v3(self.tofSec)])
        [L1, L2] = np.dot(np.linalg.inv([[thetaFunc.v1(0), thetaFunc.v2(0)],
                    [thetaFunc.v1(self.tofSec), thetaFunc.v2(self.tofSec)]]),
                    [self.vDepCyl[1] - thetaFuncFree.v(0),
                     self.vArrCyl[1] - thetaFuncFree.v(self.tofSec)])

        # cTheta3
        integrand1 = lambda t: (L1*thetaFunc.v1(t) + L2*thetaFunc.v2(t)
                                + thetaFuncFree.v(t))/self.r(t)
        integrand2 = lambda t: (K1*thetaFunc.v1(t) + K2*thetaFunc.v2(t)
                                + thetaFunc.v3(t))/self.r(t)
        int1 = integrate(integrand1, 0, self.tofSec, method='trapz', nSteps=25)
        int2 = integrate(integrand2, 0, self.tofSec, method='trapz', nSteps=25)
        cTheta3 = (self.thetaArr - int1)/(int2)

        # cTheta1 and cTheta2
        cTheta12 = cTheta3 * np.array([K1, K2]) + np.array([L1, L2])
        self.cTheta = np.array([cTheta12[0], cTheta12[1], cTheta3])

        # assemble shape
        self.tDot = lambda t: (self.cTheta[0] * thetaFunc.v1(t)
                            + self.cTheta[1] * thetaFunc.v2(t)
                            + self.cTheta[2] * thetaFunc.v3(t)
                            + thetaFuncFree.v(t))
        self.thetaDot = lambda t: self.tDot(t)/self.r(t)
        self.tDDot = lambda t: (self.cTheta[0] * thetaFunc.Dv1(t)
                            + self.cTheta[1] * thetaFunc.Dv2(t)
                            + self.cTheta[2] * thetaFunc.Dv3(t)
                            + thetaFuncFree.Dv(t))

    def t(self, time):
        '''
        Convenience function to call the polar angle as a function of time
        Computationally inefficient due to numerical integration
        '''

        # compute theta value by integration of thetaDot
        thetaChange = integrate(self.thetaDot, 0, time, method='trapz',
                                nSteps=25)
        thetaFinal = thetaChange + self.rDepCyl[1]
        return thetaFinal

    def assembleThrust(self):
        '''
        Compute the thrust profile from the equations of motion
        See Equation 5.13-16 in [Gondelach, 2012]
        '''

        s = lambda t: np.sqrt(self.r(t)**2 + self.z(t)**2)
        self.fr = lambda t: (self.rDDot(t) - self.tDot(t)**2/self.r(t)
                            + pk.MU_SUN/(s(t)**3) * self.r(t))
        self.ft = lambda t: self.tDDot(t) + self.tDot(t)*self.rDot(t)/self.r(t)
        self.fz = lambda t: self.zDDot(t) + pk.MU_SUN/(s(t)**3) * self.z(t)
        self.fTotal = lambda t: (np.sqrt(self.fr(t)**2
                                + self.ft(t)**2
                                + self.fz(t)**2))


    def checkBoundaryConditions(self, velTolRel=0.001, velTolAbs=0.1,
                                posTolRel=0.001, posTolAbs=0.1):
        '''
        Check if the boundary conditions are satisfied
        Compare initial and final velocities to a given tolerance
        '''

        self.velTolRel = velTolRel
        self.velTolAbs = velTolAbs
        self.velCompare = np.allclose(
            [self.vDepCyl[0], self.vDepCyl[1], self.vDepCyl[2],
             self.vArrCyl[0], self.vArrCyl[1], self.vArrCyl[2]],
            [self.rDot(0), self.tDot(0), self.zDot(0),
             self.rDot(self.tofSec), self.tDot(self.tofSec),
             self.zDot(self.tofSec)],
            self.velTolRel, self.velTolAbs)
        # compare initial and final positions
        self.posTolRel = posTolRel
        self.posTolAbs = posTolAbs
        # theta may do several revolutions -> subtract these
        arrivalTheta = self.t(self.tofSec) - self.N * 2 * np.pi
        posCompare1 = np.allclose(
            [self.rDepCyl[0], self.rDepCyl[1], self.rDepCyl[2],
             self.rArrCyl[0], self.rArrCyl[1], self.rArrCyl[2]],
            [self.r(0), self.t(0), self.z(0),
             self.r(self.tofSec), arrivalTheta, self.z(self.tofSec)],
            self.posTolRel, self.posTolAbs)
        arrivalTheta2 = self.t(self.tofSec) - (self.N+1) * 2 * np.pi
        posCompare2 = np.allclose(
            [self.rDepCyl[0], self.rDepCyl[1], self.rDepCyl[2],
             self.rArrCyl[0], self.rArrCyl[1], self.rArrCyl[2]],
            [self.r(0), self.t(0), self.z(0),
             self.r(self.tofSec), arrivalTheta2, self.z(self.tofSec)],
            self.posTolRel, self.posTolAbs)
        self.posCompare = posCompare1 or posCompare2

    def evaluate(self, evalThrust=False, nEvalPoints=100, printTime=False):
        '''
        Compute DeltaV and maximum thrust
        By numerically integrating and sampling the thrust profile
        Number of sampling points has a serious impact on performance
        -> Activate thrust evaluation only when needed
        '''

        deltaVtemp = integrate(self.fTotal, 0, self.tofSec,
                            method='trapz', nSteps=25)
        self.deltaV = deltaVtemp

        if printTime==True:
            time1 = time.time()

        # perform grid search at equally spaced sample points
        if evalThrust=='Grid':
            self.maxThrust = np.max(self.fTotal(np.linspace(0, self.tofSec,
                                    nEvalPoints)))

        # call local optimizer from scipy (NOT RECOMMENED as not robust)
        elif evalThrust=='Optimize':
            maxThrustTime = sci.optimize.minimize_scalar(
                                            lambda t: -self.fTotal(t),
                                            bounds=[0,self.tofSec],
                                            method='bounded')
            self.maxThrust = self.fTotal(maxThrustTime.fun)

        # don't look for maximum thrust value
        else:
            self.maxThrust = -1

        # print the measured time spent in this method
        if printTime==True:
            time2 = time.time()
            print(f'Finding maximum of thrust profile took '
                  f'{(time2-time1)*1e3:.3f} ms')
