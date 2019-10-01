import time
import os

import numpy as np
import matplotlib as mlt
import matplotlib.pyplot as plt
import pygmo as pg
import pykep as pk

from plottingUtilsLaunchWindow import plottingGridSearch
from plottingUtilsIndividualTrajectory import plotting
from hodographicShaping_SI import hodographicShaping
from utils import *
from patchedTrajectoryUtils import *
from pygmoProblemsShaping import myProblemShapingSingle
from conversions import stateCyl2cart, stateCart2cyl
from plottingUtilsPatchedTrajectory import patchedPlots
from orbitFollowing import orbitFollowing

def getShapingResults(population, algorithm, time, initialGuess):
    '''
    Retrieve results from the final population and print a status report
    '''

    finalPop = population.get_x()
    log = algorithm.extract(pg.nlopt).get_log()
    bestTrajectory = np.array(population.get_x()[population.best_idx()])
    bestDeltaV = population.get_f()[population.best_idx()]
    nFevals = population.problem.get_fevals()
    print('\n#################################################################')
    print('Low-thrust transfer results')
    print('Initial guess:\t', initialGuess)
    print("Champion:\t", np.round(bestTrajectory[0], 2),
                         np.round(bestTrajectory[1], 2),
                         np.round(bestTrajectory[2], 2),
                         np.round(bestTrajectory[3], 2),
                         np.round(bestTrajectory[4], 2),
                         np.round(bestTrajectory[5], 2))
    print('Best DeltaV:\t', np.round(bestDeltaV[0], 2))
    print(f'Number of fitness evaluations:\t{nFevals}')
    print(f'Finished computation in\t\t{time:.2f} s')
    print(f'Time per fitness evaluation:\t{time/nFevals*1e3:.2f} ms')

    return bestDeltaV, bestTrajectory

def optimizationSetup(problem):
    '''
    Setup Pygmo for the given problem
    '''

    nl = pg.nlopt('neldermead')
    pop = pg.population(problem, 1)
    nl.xtol_rel = 1E-6
    nl.maxeval = 10_000
    algo = pg.algorithm(nl)
    algo.set_verbosity(100)
    pop = pg.population(problem, 1)
    pop.set_x(0, [0, 0, 0, 0, 0, 0])
    initialGuess = pop.get_x()[pop.best_idx()]

    return pop, algo, initialGuess

def main(folder = 'graveyard'):
    """
    Evaluate patched low thrust trajectories
    For explicit conditions resembling the Dawn mission:
        Launch from Earth -> impulsive shot -> low-thrust to Mars -> swingby
        -> low-thrust to Vesta -> rendezvous
    All state vectors at this level are in cylindrical coordinates:
        [radius, phi, z, v_r, vTheta, v_z]]
        the only exception is the flyby computation, see below
    """

    ############################################################################
    # Initialization

    # output folder: Contents are overwritten each run!
    outputFolder = os.path.join('output', folder)
    checkFolder(outputFolder)

    # reduce precision of printed numpy values
    np.set_printoptions(precision=3)

    # load spice kernels
    ephems = 'spice'
    loadSpiceKernels()

    ############################################################################
    # Free parameters

    # set up the problem: fixed parameters
    ephems = 'spice'

    # bodies
    depBody = '3'          # earth
    arrBody = '4'          # mars
    arrBody2 = '2000004'   # Vesta
    arrBody4 = '2000001'   # Ceres

    # number of revolutions
    N = 0
    N2 = 1
    N4 = 0

    # orbit insertion (tangential inpulsive shot)
    injectionDeltaV = [0, 2.42e3, 0]

    # flyby parameters
    swingbyPlanet = pk.planet.jpl_lp('mars')
    arrVelMars = [-2.21e3, 2.17e4, -6.37e1]
    flybyPlane = 2.71
    flybyAltitude = 471e3

    # dates
    depMjd = 2782.2
    tof = 407.6             # Earth - Mars
    tof2 = 1099.1           # Mars- Vesta
    tof3 = 559.2            # stay at Vesta
    tof4 = 912.7            # transfer Vesta - Ceres

    ############################################################################
    # Derived parameters

    # Earth - Mars
    arrMjd = depMjd + tof
    planetStateDep, __, __ = ephemeris(depBody, depMjd, mode=ephems)
    planetStateArr, plArCart, __ = ephemeris(arrBody, arrMjd, mode=ephems)
    print('Mars arrival date', arrMjd)
    print('Mars state cyl', planetStateArr)
    print('Mars state cart', plArCart)
    injectionDeltaVabs = np.linalg.norm(injectionDeltaV)
    scStateDep = applyDeltaV(planetStateDep, injectionDeltaV)
    scStateArr = np.array((planetStateArr[0:3], arrVelMars)).reshape(6,)

    # swingby
    scStateArrCart = stateCyl2cart(scStateArr)      # convert state to Cartesian

    # Mars - Vesta
    depMjd2 = arrMjd
    arrMjd2 = depMjd2 + tof2
    depBody2 = arrBody
    scStateArr2, __, __ = ephemeris(arrBody2, arrMjd2, mode=ephems)

    # stay at Vesta
    depMjd3 = arrMjd2
    arrMjd3 = depMjd3 + tof3
    body = arrBody2

    # Vesta - Ceres
    depMjd4 = arrMjd3
    arrMjd4 = depMjd4 + tof4
    depBody4 = body
    scStateDep4, __, __ = ephemeris(depBody4, depMjd4, mode=ephems)
    scStateArr4, __, __ = ephemeris(arrBody4, arrMjd4, mode=ephems)

    ############################################################################
    # Computations

    # low-thrust transfer to Mars
    prob = pg.problem(myProblemShapingSingle(scStateDep, scStateArr,
                            depDate=depMjd, tof=tof, N=N,
                            depBody=depBody, target=arrBody))
    pop, algo, initialGuess = optimizationSetup(prob)
    print('Run Nelder-Mead to find good shape coefficients.')
    start = time.process_time()
    pop = algo.evolve(pop)
    optiTime = time.process_time() - start
    DeltaV1, c1 = getShapingResults(pop, algo, optiTime, initialGuess)

    # flyby
    scStateNewCart, swingbyDeltaV = flyby(scStateArrCart, 'mars',
            flybyAltitude, flybyPlane, mode='jpl', mjd=arrMjd,
            folder=outputFolder, makePlot=True, save=True, show=True)
    scStateDep2 = stateCart2cyl(scStateNewCart)    # convert back to Cylindrical

    # low-thrust to Vesta
    prob = pg.problem(myProblemShapingSingle(scStateDep2, scStateArr2,
                            depDate=depMjd2, tof=tof2, N=N2,
                            depBody=depBody2, target=arrBody2))
    pop, algo, initialGuess = optimizationSetup(prob)
    start = time.process_time()
    print('Run Nelder-Mead to find good shape coefficients.')
    pop = algo.evolve(pop)
    optiTime = time.process_time() - start
    DeltaV2, c2 = getShapingResults(pop, algo, optiTime, initialGuess)

    # stay at Vesta
    transfer3 = orbitFollowing(startDate=depMjd3, tof=tof3, body=body,
                            ephemSource=ephems)

    # low-thrust to Ceres
    prob = pg.problem(myProblemShapingSingle(scStateDep4, scStateArr4,
                            depDate=depMjd4, tof=tof4, N=N4,
                            depBody=depBody4, target=arrBody4))

    pop, algo, initialGuess = optimizationSetup(prob)
    start = time.process_time()
    print('Run Nelder-Mead to find good shape coefficients.')
    pop = algo.evolve(pop)
    optiTime = time.process_time() - start
    DeltaV4, c4 = getShapingResults(pop, algo, optiTime, initialGuess)

    # compute the total DeltaV
    totalDeltaV = np.squeeze(DeltaV1 + DeltaV2 + DeltaV4)

    ############################################################################
    # Result outputs
    # compute the individual transfers again for detailed results and plotting
    details = False
    print('Detailed results for each leg')
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
                            rFreeC =         [c1[0], c1[1]],
                            thetaFreeC =     [c1[2], c1[3]],
                            zFreeC =         [c1[4], c1[5]],
                            )
    transfer.shapingRadial()
    transfer.shapingVertical()
    transfer.shapingTransverse()
    transfer.assembleThrust()
    transfer.checkBoundaryConditions()
    transfer.evaluate(evalThrust='Grid', printTime=False, nEvalPoints = 1000)
    if details:
        transfer.status(printBC=False)

    transfer2 = hodographicShaping(scStateDep2, scStateArr2,
                            departureDate=depMjd2, tof=tof2, N=N2,
                            departureBody = depBody2,
                            arrivalBody = arrBody2,
                            rShape =         'CPowPow2_scaled',
                            thetaShape =     'CPowPow2_scaled',
                            zShape =         'CosR5P3CosR5P3SinR5_scaled',
                            rShapeFree =     'PSin05PCos05_scaled',
                            thetaShapeFree = 'PSin05PCos05_scaled',
                            zShapeFree =     'P4CosR5P4SinR5_scaled',
                            rFreeC =         [c2[0], c2[1]],
                            thetaFreeC =     [c2[2], c2[3]],
                            zFreeC =         [c2[4], c2[5]],
                            )
    transfer2.shapingRadial()
    transfer2.shapingVertical()
    transfer2.shapingTransverse()
    transfer2.assembleThrust()
    transfer2.checkBoundaryConditions()
    transfer2.evaluate(evalThrust='Grid', printTime=False, nEvalPoints = 1000)
    if details:
        transfer2.status(printBC=True)

    transfer4 = hodographicShaping(scStateDep4, scStateArr4,
                            departureDate=depMjd4, tof=tof4, N=N4,
                            departureBody = depBody4,
                            arrivalBody = arrBody4,
                            rShape =         'CPowPow2_scaled',
                            thetaShape =     'CPowPow2_scaled',
                            zShape =         'CosR5P3CosR5P3SinR5_scaled',
                            rShapeFree =     'PSin05PCos05_scaled',
                            thetaShapeFree = 'PSin05PCos05_scaled',
                            zShapeFree =     'P4CosR5P4SinR5_scaled',
                            rFreeC =         [c4[0], c4[1]],
                            thetaFreeC =     [c4[2], c4[3]],
                            zFreeC =         [c4[4], c4[5]],
                            )
    transfer4.shapingRadial()
    transfer4.shapingVertical()
    transfer4.shapingTransverse()
    transfer4.assembleThrust()
    transfer4.checkBoundaryConditions()
    transfer4.evaluate(evalThrust='Grid', printTime=False, nEvalPoints = 1000)
    if details:
        transfer4.status(printBC=True)

    # print overal results
    print('###################################################################')
    print(f'Results of patched trajectory computation')
    print(f'Orbit injection impulse:\t{(injectionDeltaVabs)/1e3:>9.2f} km/s')
    print(f'DeltaV Earth -> Mars:\t\t{np.squeeze(DeltaV1)/1e3:>9.2f} km/s')
    print(f'Swingby DeltaV:\t\t\t{swingbyDeltaV/1e3:>9.2f} km/s')
    print(f'DeltaV Mars -> Vesta:\t\t{np.squeeze(DeltaV2)/1e3:>9.2f} km/s')
    print(f'DeltaV Vesta -> Ceres:\t\t{np.squeeze(DeltaV4)/1e3:>9.2f} km/s')
    print(f'Total low-thrust DeltaV:\t{totalDeltaV/1e3:>9.2f} km/s')
    print('###################################################################')

    # generate combined plots
    awesomePlotWonderland = patchedPlots(
                                [transfer, transfer2, transfer3, transfer4],
                                samples=1000,
                                folder=outputFolder,
                                save=True,
                                show=True,
                                addBody=[])
    # awesomePlotWonderland.planetSystem(plotSOI=True, scaling=False)
    awesomePlotWonderland.trajectory2D()
    awesomePlotWonderland.trajectory3D(scaling=False)
    # awesomePlotWonderland.hodograph()
    # awesomePlotWonderland.trajectory3Danimation(scaling=False, staticOrbits=True, save=False)
    # awesomePlotWonderland.trajectory2Danimation(staticOrbits=True, save=True)
    # awesomePlotWonderland.thrust()

if __name__ == '__main__':
    main(folder='dawn_fixed')
