# coordiante transformations

# not vectoriezed yet!
# inputs and outputs are single vectors

import numpy as np

def Pcart2cyl(r):
    """
    Converts a position vector from Cartesian to Cylindrical coordinates
    """
    x = r[0]
    y = r[1]
    z = r[2]
    radius = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return [radius, phi, z]

def Pcyl2cart(r):
    """
    Converts a position vector from Cylindrical to Cartesian coordinates
    """
    rho = r[0]
    phi = r[1]
    z = r[2]
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return [x, y, z]

def Vcart2cyl(V_Cart, R_Cart):
    """
    Converts a velocity vector from Cartesian to Cylindrical coordinates
    See p.150 of [Gondelach, 2012]
    Returns Vtheta, the velocity in the direction of the theta axis [m/s]
        and not thetaDot, the derivative of the theta angle [rad/s], as might
        be expected!
    """
    vx = V_Cart[0]
    vy = V_Cart[1]
    vz = V_Cart[2]
    x  = R_Cart[0]
    y  = R_Cart[1]
    z  = R_Cart[2]
    rDot = 1/np.sqrt(x**2 + y**2) * (x*vx + y*vy)
    thetaDot = 1/(x**2 + y**2) * (x*vy - vx*y)
    vTheta = thetaDot * np.sqrt(x**2 + y**2)
    zDot = vz
    return [rDot, vTheta, zDot]

def Vcyl2cart(vCyl, rCyl):
    """
    Converts a velocity vector from Cylindrical to Cartesian coordinates
    """
    vr = vCyl[0]
    vtheta = vCyl[1]
    vz = vCyl[2]
    r = rCyl[0]
    theta = rCyl[1]
    z = rCyl[2]
    xDot = vr*np.cos(theta) - vtheta*np.sin(theta)
    yDot = vr*np.sin(theta) + vtheta*np.cos(theta)
    zDot = vz
    return [xDot, yDot, zDot]

def rad2deg(angleInRadian):
    """
    Converts an angle in radian to degree
    """
    angleDeg = angleInRadian * 180 / np.pi
    return angleDeg

def stateCyl2cart(stateCyl):
    '''
    Convenience function to convert full state vectors [pos, vel]
    Cylindrical state to cartesian state
    '''
    rCyl = stateCyl[0:3]
    vCyl = stateCyl[3:6]
    rCart = Pcyl2cart(rCyl)
    vCart = Vcyl2cart(vCyl, rCyl)

    return np.array((rCart, vCart)).reshape(6,)

def stateCart2cyl(stateCart):
    '''
    Convenience function to convert full state vectors [pos, vel]
    Cartesian state to cylindrical state
    '''
    rCart = stateCart[0:3]
    vCart = stateCart[3:6]
    rCyl = Pcart2cyl(rCart)
    vCyl = Vcart2cyl(vCart, rCart)

    return np.array((rCyl, vCyl)).reshape(6,)
