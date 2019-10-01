import os
import time
import warnings

import pykep as pk
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from termcolor import colored
import pygmo as pg

from conversions import Pcart2cyl, Vcart2cyl
from utils import checkFolder, axisEqual3D, newFigure

def loadSpiceKernels(folder='ephemerides'):
    '''
    Loads all spice kernels in the given folder into memory
    '''

    start = time.time()
    print('Loading spice kernels:')
    for file in os.listdir(folder):
        if file.endswith('.bsp'):
            # print(file)
            pk.util.load_spice_kernel(os.path.join(folder, file))
            print('Loaded', file)
    end = time.time()
    print(f'Spice kernels succesfully loaded in {(end-start)*1e3:.2f} ms.')

def ephemeris(planet, modifiedJulianDay, mode='jpl', ref='sun'):
    '''
    Query the ephemeris for the specified planet and epoch
    Based on the Pykep package by ESA/ACT, developed with version 2.3
    Returns the state vector (pos, vel) in Cartesian and Cylindrical coordinates
    mode='jpl' uses low precision ephemerides of the planets ('earth', etc)
    mode='spice' uses high precision spice kernel. These need to be loaded
        into memory using loadSpiceKernels(). The  kernel (430.bsp)
        includes ephemerides for Earth, Moon and the planet system barycenters
        ('399', 301', '4', etc)
    mode='gtoc2' uses the asteroids from the GTOC2 competition (Keplerian)
    '''

    epoch = pk.epoch(modifiedJulianDay, 'mjd2000')

    if mode == 'jpl':
        planet = pk.planet.jpl_lp(planet)
    elif mode == 'spice':
        planet = pk.planet.spice(planet, ref, 'eclipj2000')
    elif mode == 'gtoc2':
        planet = pk.planet.gtoc2(planet)
    else:
        print('ERROR: This is not a valid source of ephemerides.')

    # retrieve Cartesian state vectors in SI units
    rCart, vCart = planet.eph(epoch)

    # conversion to cylindrical values to get boundary conditions
    rCyl = Pcart2cyl(rCart)
    vCyl = Vcart2cyl(vCart, rCart)

    # combine into (6,) vectors (3xposition, 3xvelocity)
    stateVectorCartesian = np.array((rCart, vCart)).reshape(6,)
    stateVectorCylindrical = np.array((rCyl, vCyl)).reshape(6,)

    return stateVectorCylindrical, stateVectorCartesian, planet

def applyDeltaV(stateVector, deltaV):
    '''
    Apply a delta V (impulsive shot) to the state vector
    Commonly used to add a V_infinity at departure
    Computation is valid for Cylindrical and Cartesian coordinates
    Both inputs need to be in the same orthogonal coordinates
    '''

    newVelocity = np.array(stateVector[3:6] + np.asarray(deltaV))
    newState = np.hstack((stateVector[0:3], newVelocity))

    return newState

def flyby2D(stateVector, planetName, mjd=0, Bmult=0, alphaSignFlip=False,
            mode='jpl', folder='graveyard', save=True, show=True,
            makePlot=True, printStatus=True, returnInEq=False):
    '''
    Perform a swing-by manoeuver [Wakker, 2015, Section 18.11]
    Approximate the swingby as an instantaneous velocity change in the ecliptic
    The position of the change is approximated as the planets position
    Input:  - stateVector   in Cartesian coordiantes [x, y, z, vx, vy, vz]
                            vz should be 0, in plane approximation
            - planet        flyby planet, string corresponding to ephemeris
            - mjd           modified julian date of the flyby
            - Bmult         distance planet to aiming point B over planet radius
            - alphaSignFlip positive or negative deflection angle
    Output: - newStateVector    also in Cartesian coordinates
    '''

    # gather parameters
    V2 = stateVector[3:6]
    pos = stateVector[0:3]
    statePlaCyl, statePlaCart, planet = ephemeris(planetName, mjd, mode)
    Vplanet = statePlaCart[3:6]
    mu = planet.mu_self
    R = planet.radius

    # project velocities in the x-y plane
    Vplanet[2] = 0

    # miss distance B
    B = R*Bmult

    # hyperbolic excess velocity relative to planet (vector)
    Vinfinityt = V2 - Vplanet
    Vinfty = np.linalg.norm(Vinfinityt)     # magnitude

    # alpha: asymptotic deflection angle, Eqn. 18.68
    alpha = 2*np.arcsin(1/(np.sqrt( 1 + B**2*Vinfty**4/(mu**2) )))
    if alphaSignFlip:
        alpha = - alpha

    # planetocentry hyperbolic excess velocity after flyby
    Vx, Vy = np.dot(
            [[np.cos(alpha),-np.sin(alpha)],
             [np.sin(alpha), np.cos(alpha)]],
            [Vinfinityt[0], Vinfinityt[1]] )
    VinfinitytStar = np.array([Vx, Vy, 0])

    # heliocentric velocity after flyby
    V4 = Vplanet + VinfinitytStar

    # compute Delta V
    DeltaV = np.linalg.norm(V4-V2)

    # patched conic approximation
    # sphere of influence is small compared to solar System
    # position does not change during flyby, simply replace velocity
    newStateVector = np.array((pos, V4)).reshape(6,)

    # check for impact, see [Wakker 2015, Eqn 18.70]
    lhs = (B/R)**2
    rhs = 1 + 2*mu/(R*Vinfty**2)

    if printStatus:
        print('###################################################################')
        print(f'Performing 2D flyby at {planetName} on {mjd} mjd2000.')
        # sanity checks
        # positions of planet and s/c
        if np.allclose(statePlaCart[0:3], stateVector[0:3], 1e-2, 1e3):
            print('OK\tSpacecraft and planet are at the same positions')
        else:
            print('ERROR\tSpacecraft and planet position are not the same!')
            print(f'\tSpacecraft position:\t{stateVector[0:3]}')
            print(f'\tPlanet position:\t{statePlaCart[0:3]}')
        # s/c velocities
        if V2[2] != 0:
            print('ERROR\tSpacecraft exhibits out-of-plane motion.')
        else:
            print('OK\tSpacecraft moves in the ecliptic.')
        if lhs < rhs:
            print('ERROR\tThis flyby is physically not possible (crash predicted)')
        else:
            print('OK\tCorrect flyby, no crash computed.')
        print(f'\tPlanet radius is {R/1e3:.0f} km.')
        print(f'\tImpact parameter is {B/1e3:.0f} km.')
        r3 = mu/Vinfty**2 * (np.sqrt(1+B**2*Vinfty**4/mu**2)-1)
        print(f'\tFlyby distance is {r3/1e3:.1f} km.')
        print(f'\tB/R is {Bmult:.3f}.')
        print(f'\tPlanet gravitational parameter is {mu:.3E} m^3/s^-2.')
        # alpha
        print(f'Deflection angle:\t{alpha*180/np.pi:.2f} degree')
        if not (0 < np.abs(alpha) < np.pi):
            print(f'ERROR: Alpha is out of range: {alpha*180/np.pi:.2f} deg')
        # results
        print(f'V_infinity:\t\t{Vinfty/1e3:.2f} km/s')
        print(f'Total Delta V:\t\t{DeltaV/1e3:.2f} km/s')
        if save:
            print('Saving vectordiagram of flyby geometry.')
        print('###################################################################')

    if makePlot:
        flybyVectors(V2, V4, Vinfinityt, VinfinitytStar, Vplanet, planetName,
                    mjd, folder=folder, save=save, show=show)

    if returnInEq:
        return newStateVector, DeltaV, (rhs - lhs)

    return newStateVector, DeltaV

def flyby(stateVector, planetName, r3, planeAngle, mjd=0,
            mode='jpl', folder='graveyard', save=True, show=True,
            makePlot=True, printStatus=True):
    '''
    Perform a swing-by manoeuver [Izzo, 2010] and [Tudat, gravityAssist.cpp]
    Approximate the swingby as an instantaneous velocity change
    The position of the change is approximated as the planets position
    Input:  - stateVector   Arrival state vector in Cartesian coordiantes
                            [x, y, z, vx, vy, vz]
            - planet        flyby planet, string corresponding to ephemeris
            - mjd           modified julian date of the flyby
            - r3            flyby distance
            - planeAngle    angle of the flyby plane wrt ecliptic
    Output: - newStateVector    also in Cartesian coordinates
    '''

    # gather parameters
    V2 = stateVector[3:6]
    pos = stateVector[0:3]
    statePlaCyl, statePlaCart, planet = ephemeris(planetName, mjd, mode)
    Vplanet = statePlaCart[3:6]
    # Vplanet[2] = 0        # project velocities in the x-y plane
    mu = planet.mu_self
    R = planet.radius

    # incoming hyperbolic excess velocity relative to planet
    Vinfinityt = V2 - Vplanet
    Vinfty = np.linalg.norm(Vinfinityt)     # magnitude

    # eccentricity of the flyby hyperbola
    e = 1 + r3/mu * Vinfty**2

    # angle between incoming and outgoing hyperbolic excess velocity
    # bending angle or asymptotic deflection angle
    bendingAngle = 2*np.arcsin(1/e)

    # unit vectors describing the plane of the flyby
    u1 = Vinfinityt/Vinfty
    u2long = np.cross(u1, Vplanet)
    u2 = u2long/np.linalg.norm(u2long)
    u3 = np.cross(u1, u2)

    # outgoing hyperbolic excess velocity relative to planet
    VinfinitytStar = Vinfty * (np.cos(bendingAngle) * u1
                            + np.sin(bendingAngle) * np.cos(planeAngle) * u2
                            + np.sin(bendingAngle) * np.sin(planeAngle) * u3)
    VinftyStar = np.linalg.norm(VinfinitytStar)

    # outgoing heliocentric velocity
    V4 = VinfinitytStar + Vplanet

    # compute DeltaV
    DeltaV = np.linalg.norm(V4 - V2)

    # assemble new state vector
    newStateVector = np.array((pos, V4)).reshape(6,)

    if printStatus:
        print('###############################################################')
        print(f'Performing 3D flyby at {planetName} on {mjd} mjd2000.')
        # sanity check positions of planet and s/c
        if np.allclose(statePlaCart[0:3], stateVector[0:3], 1e-2, 1e3):
            print('OK\tSpacecraft and planet are at the same positions')
        else:
            print('ERROR\tSpacecraft and planet position are not the same!')
            print(f'\tSpacecraft position:\t{stateVector[0:3]}')
            print(f'\tPlanet position:\t{statePlaCart[0:3]}')
        print(f'\tPlanet radius is {R/1e3:.0f} km.')
        print(f'\tFlyby distance is {r3/1e3:.1f} km.')
        print(f'\tPlanet gravitational parameter is {mu:.3E} m^3/s^-2.')
        print(f'Bending angle:\t\t{bendingAngle*180/np.pi:.2f} degree')
        print(f'Plane angle:\t\t{planeAngle*180/np.pi:.2f} degree')
        # results
        print(f'V_infinity:\t\t{Vinfty/1e3:.2f} km/s')
        print(f'Total Delta V:\t\t{DeltaV/1e3:.2f} km/s')
        if save:
            print('Saving vectordiagram of flyby geometry.')
        print('###############################################################')

    if makePlot:
        flybyVectors3D(V2, V4, Vinfinityt, VinfinitytStar, Vplanet, planetName,
                    mjd, folder=folder, save=save, show=show)

    return newStateVector, DeltaV


def flybyVectors(V2, V4, Vinfinityt, VinfinitytStar, Vplanet, planetName, mjd,
                folder='output/flybyExamples', save=True, show=True):

    DeltaV = V4 - V2

    fig = newFigure(height=6.2/2, half=True, target='thesis')
    ax = plt.gca()

    quivV2 = plt.quiver(0, 0, V2[0], V2[1],
       angles='xy', scale_units='xy', scale=1,
       color='C0', zorder=3)
    quivV4 = plt.quiver(0, 0, V4[0], V4[1],
       angles='xy', scale_units='xy', scale=1,
       color='C3', zorder=3)
    quivVplanet = plt.quiver(0, 0, Vplanet[0], Vplanet[1],
       angles='xy', scale_units='xy', scale=1,
       color='C5', zorder=3)
    quivDeltaV = plt.quiver(V2[0], V2[1], DeltaV[0], DeltaV[1],
       angles='xy', scale_units='xy', scale=1,
       color='C2', zorder=3)
    quivVinf = plt.quiver(0, 0, Vinfinityt[0], Vinfinityt[1],
       angles='xy', scale_units='xy', scale=1,
       color='C4', zorder=3)
    quivVinfStar = plt.quiver(0, 0, VinfinitytStar[0], VinfinitytStar[1],
       angles='xy', scale_units='xy', scale=1,
       color='C1', zorder=3)

    plt.quiverkey(quivV2, V2[0]/2, V2[1]/2, 2,
            r'$V_{\mathrm{in}}$', coordinates='data', labelpos='W', labelcolor='C0')
    plt.quiverkey(quivV4, V4[0]/3, V4[1]/3, 2,
            r'$V_{\mathrm{out}}$', coordinates='data', labelpos='E', labelcolor='C3')
    plt.quiverkey(quivDeltaV, V2[0]+DeltaV[0]/2, V2[1]+DeltaV[1]/2, 2,
            r'$\Delta V$', coordinates='data', labelpos='W', labelcolor='C2')
    plt.quiverkey(quivVplanet, Vplanet[0]/2, Vplanet[1]/2, 2,
            r'$V_\mathrm{p}$', coordinates='data', labelpos='E', labelcolor='C5')
    plt.quiverkey(quivVinf, Vinfinityt[0]/2, Vinfinityt[1]/2, 2,
            r'$\tilde{V}_{\mathrm{in}}$', coordinates='data', labelpos='W',
            labelcolor='C4')
    plt.quiverkey(quivVinfStar, VinfinitytStar[0]/3, VinfinitytStar[1]/3, 2,
            r'$\tilde{V}_{\mathrm{out}}$', coordinates='data', labelpos='E',
            labelcolor='C1')

    ax.arrow(Vinfinityt[0], Vinfinityt[1], DeltaV[0], DeltaV[1],
            length_includes_head=True, head_length=0,
            zorder=3)
    ax.arrow(0, 0, Vinfinityt[0]+DeltaV[0]/2, Vinfinityt[1]+DeltaV[1]/2,
            length_includes_head=True, head_length=0,
            zorder=3)

    factor = 1.1
    plt.xlim(factor*np.amin([0, V2[0], V4[0], Vinfinityt[0], VinfinitytStar[0],
                DeltaV[0], Vplanet[0]]),
            factor*np.amax([0, V2[0], V4[0], Vinfinityt[0], VinfinitytStar[0],
                DeltaV[0], Vplanet[0]]))
    plt.ylim(1000 + factor*np.amin([0, V2[1], V4[1], Vinfinityt[1], VinfinitytStar[1],
                DeltaV[1], Vplanet[1]]),
            1000 + factor*np.amax([0, V2[1], V4[1], Vinfinityt[1], VinfinitytStar[1],
                DeltaV[1], Vplanet[1]]))
    # constant value for the Verification section
    # plt.xlim([-25e3, 8e3])
    # plt.ylim([-35e3, 5e3])
    ax.set_aspect('equal', 'box')

    plt.ylabel(r'$v_y$ $[\mathrm{m} \mathrm{s}^{-1}]$')
    plt.xlabel(r'$v_x$ $[\mathrm{m} \mathrm{s}^{-1}]$')
    plt.title('Flyby @ ' + planetName + '\n' + str(mjd) + ' mjd')
    plt.grid(zorder=0)
    plt.tight_layout()

    if save==True:
        checkFolder(os.path.join(folder))
        plt.savefig(os.path.join(folder,
                    'flyby2D-'+planetName+str(mjd)+'.pdf'), dpi=300)
        plt.savefig(os.path.join(folder,
                    'flyby2D-'+planetName+str(mjd)+'.png'), dpi=300)
    if show:
        plt.show()
    else:
        plt.show(block=False)

def flybyVectors3D(V2, V4, Vinfinityt, VinfinitytStar, Vplanet, planetName, mjd,
                folder='output/flybyExamples', save=True, show=True):

    DeltaV = V4 - V2

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    quivV2 = plt.quiver(0, 0, 0, V2[0], V2[1], V2[2],
       # angles='xy', scale_units='xy',
       # scale=1,
       arrow_length_ratio=0.05,
       color='C0', zorder=3)
    quivV4 = plt.quiver(0, 0, 0, V4[0], V4[1], V4[2],
       # angles='xy', scale_units='xy', scale=1,
       arrow_length_ratio=0.05,
       color='C3', zorder=3)
    quivVplanet = plt.quiver(0, 0, 0, Vplanet[0], Vplanet[1], Vplanet[2],
       # angles='xy', scale_units='xy', scale=1,
       arrow_length_ratio=0.05,
       color='C5', zorder=3)
    quivDeltaV = plt.quiver(V2[0], V2[1], V2[2], DeltaV[0], DeltaV[1],
       DeltaV[2],
       # angles='xy', scale_units='xy', scale=1,
       arrow_length_ratio=0.5,
       color='C2', zorder=3)
    quivVinf = plt.quiver(0, 0, 0, Vinfinityt[0], Vinfinityt[1], Vinfinityt[2],
       # angles='xy', scale_units='xy', scale=1,
       arrow_length_ratio=0.5,
       color='C4', zorder=3)
    quivVinfStar = plt.quiver(0, 0, 0, VinfinitytStar[0], VinfinitytStar[1],
       VinfinitytStar[2],
       # angles='xy', scale_units='xy', scale=1,
       arrow_length_ratio=0.5,
       color='C1', zorder=3)

    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        ax.text(V2[0]/2, V2[1]/2, V2[2]/2, r'$V_2$', V2, color='C0')
        ax.text(V4[0]/2, V4[1]/2, V4[2]/2, r'$V_4$', V4, color='C3')
        ax.text(V2[0]+DeltaV[0]/2, V2[1]+DeltaV[1]/2, V2[2]+DeltaV[2]/2,
                r'$\Delta V$', V2, color='C2')
        ax.text(Vplanet[0]/2, Vplanet[1]/2, Vplanet[2]/2,
                r'$V_{planet}$', Vplanet, color='C5')
        ax.text(Vinfinityt[0]/2, Vinfinityt[1]/2, Vinfinityt[2]/2,
                r'$V_{\infty t}$', Vinfinityt, color='C4')
        ax.text(VinfinitytStar[0]/2, VinfinitytStar[1]/2, VinfinitytStar[2]/2,
                r'$V_{\infty t}^\star$', VinfinitytStar, color='C1')

    axisEqual3D(ax)

    ax.set_ylabel('v_y')
    ax.set_xlabel('v_x')
    ax.set_zlabel('v_z')
    plt.title('Flyby @ ' + planetName + '\n' + str(mjd) + ' mjd')
    plt.grid(zorder=0)

    if save==True:
        checkFolder(os.path.join(folder))
        plt.savefig(os.path.join(folder,
                    'flyby-'+planetName+str(mjd)+'.pdf'), dpi=300)
        plt.savefig(os.path.join(folder,
                    'flyby-'+planetName+str(mjd)+'.png'), dpi=300)
    if show:
        plt.show()
    else:
        plt.show(block=False)

def getResults(population, algorithm, duration, type='', printStatus=True):
    '''
    Print some quick overview of the results of a GA or nlopt algo from Pygmo
    '''

    if type == 'sga':
        log = algorithm.extract(pg.sga).get_log()
    elif type == 'nlopt':
        log = algorithm.extract(pg.nlopt).get_log()
    else:
        print('ERROR: Algorithm type not defined.')
    if printStatus:
        bestParameters = population.get_x()[population.best_idx()]
        bestDeltaV = population.get_f()[population.best_idx()]
        nfevals = population.problem.get_fevals()
        print('\nResults')
        print("Champion:\t", bestParameters)
        print(f'Departure:\t{bestParameters[0]:>12.3f} mjd')
        print(f'ToF 1:\t\t{bestParameters[1]:>12.3f} days')
        print(f'ToF 2:\t\t{bestParameters[2]:>12.3f} days')
        print(f'ToF 3:\t\t{bestParameters[3]:>12.3f} days')
        print(f'ToF 4:\t\t{bestParameters[4]:>12.3f} days')
        print(f'Launch DeltaV_t:{bestParameters[5]:>12.3f} m/s')
        print(f'Flyby vel. rad.:{bestParameters[6]:>12.3f} m/s')
        print(f'Flyby vel. tan.:{bestParameters[7]:>12.3f} m/s')
        print(f'Flyby vel. ver.:{bestParameters[8]:>12.3f} m/s')
        print(f'Flyby distance:\t{bestParameters[9]/1e3:>12.3f} km')
        print(f'Flyby plane angle:{bestParameters[10]*180/np.pi:>12.3f} deg')
        print(f'Best DeltaV:\t{bestDeltaV[0]/1e3:>9.3f} km/s')
        print(f'Number of fitness evaluations:\t {nfevals}')
        print(f'Random seed:\t{population.get_seed()}')
        print(f'Optimization took {duration//60:.0f} min {duration%60:.0f} s')
        print(f'Time per trajectory: {duration/nfevals*1e3:.2f} ms')

    return log, bestParameters, bestDeltaV
