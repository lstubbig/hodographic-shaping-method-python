import time

import numpy as np
import matplotlib as mlt
import matplotlib.pyplot as plt
import pykep as pk
import scipy as sci

from conversions import *
from utils import *
from patchedTrajectoryUtils import ephemeris

class orbitFollowing(object):
    '''
    Follow the orbit of a celestial body
    Consistent with the hodographic shaping class to build and plot patched
        trajectories where a celestial rendezvous lasts a while
    Create functions returning the state of the body (and the spacecraft
        following it)
    Same units (SI) and time (seconds stating at zero) as hodographicShaping
    '''

    def __init__(self,
                startDate = 7400,
                tof = 100,
                body = 'mars',
                ephemSource = 'jpl'
                ):

        # settings
        self.jdDep = startDate
        self.jdArr = startDate + tof
        self.tof = tof
        self.tofSec = self.tof * 24 * 60 * 60
        self.ephemSource = ephemSource
        self.departureBody = body           # for consistency
        self.arrivalBody = body             # for consistency
        self.body = body                    # used here

        # assume no thrust during this period
        self.fr = lambda t: 0
        self.ft = lambda t: 0
        self.fz = lambda t: 0
        self.fTotal = lambda t: 0
        self.deltaV = 0
        self.maxThrust = 0

    def r(self, t):
        # convert time from [s since launch] to [mjd2000]
        day = t/24/60/60
        mjd = day + self.jdDep
        stateCyl, __, __ = ephemeris(self.body, mjd, mode=self.ephemSource)
        return stateCyl[0]

    def t(self, t):
        day = t/24/60/60
        mjd = day + self.jdDep
        stateCyl, __, __ = ephemeris(self.body, mjd, mode=self.ephemSource)
        return stateCyl[1]

    def z(self, t):
        day = t/24/60/60
        mjd = day + self.jdDep
        stateCyl, __, __ = ephemeris(self.body, mjd, mode=self.ephemSource)
        return stateCyl[2]

    def rDot(self, t):
        day = t/24/60/60
        mjd = day + self.jdDep
        stateCyl, __, __ = ephemeris(self.body, mjd, mode=self.ephemSource)
        return stateCyl[3]

    def tDot(self, t):
        day = t/24/60/60
        mjd = day + self.jdDep
        stateCyl, __, __ = ephemeris(self.body, mjd, mode=self.ephemSource)
        return stateCyl[4]

    def zDot(self, t):
        day = t/24/60/60
        mjd = day + self.jdDep
        stateCyl, __, __ = ephemeris(self.body, mjd, mode=self.ephemSource)
        return stateCyl[5]
