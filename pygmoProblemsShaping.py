from hodographicShaping_SI import hodographicShaping

class myProblemShaping:
    '''
    User defined problem to be used with Pygmo
    Hodographic shaping problem
    Optimize for minimum DeltaV in the launch window
    '''

    def fitness(self, x):
        obj = 0
        transfer = hodographicShaping('earth', 'mars', departureDate=x[0],
                                        tof=x[1], N=x[2])
        transfer.shapingRadial()
        transfer.shapingVertical()
        transfer.shapingTransverse()
        transfer.assembleThrust()
        transfer.evaluate(evalThrust=False)
        obj = transfer.deltaV
        return [obj,]

    def get_bounds(self):
        # box bounds
        Nmin = 0
        Nmax = 4
        depDateMin = 7000
        depDateMax = 11000
        tofMin = 500
        tofMax = 2000
        return ([depDateMin, tofMin, Nmin], [depDateMax, tofMax, Nmax])

    # Integer Dimension
    def get_nix(self):
        return 1

    def get_name(self):
         return "Hodographic Shaping"


class myProblemShapingMulti:
    '''
    User defined problem to be used with Pygmo
    Multi-objective implementation of the hodographic shaping problems
    Optimize for DeltaV and maxThrust
    '''

    def fitness(self, x):
        obj1, obj2 = 0, 0
        transfer = hodographicShaping('earth', 'mars', departureDate=x[0],
                                        tof=x[1], N=x[2])
        transfer.shapingRadial()
        transfer.shapingVertical()
        transfer.shapingTransverse()
        transfer.assembleThrust()
        # transfer.checkBoundaryConditions()
        transfer.evaluate(evalThrust=True)
        obj1 = transfer.deltaV
        obj2 = transfer.maxThrust
        return [obj1, obj2]

    def get_nobj(self):
        return 2

    def get_bounds(self):
        # box bounds
        Nmin = 0
        Nmax = 4
        depDateMin = 7000
        depDateMax = 11000
        tofMin = 500
        tofMax = 2000
        return ([depDateMin, tofMin, Nmin], [depDateMax, tofMax, Nmax])

    # Integer Dimension
    def get_nix(self):
        return 1

class myProblemShapingSingle:
    '''
    User defined problem to be used with Pygmo
    Optimize the free parameters of a single hodographic shaping problem
    Two free parameters per shape
    '''

    def __init__(self,
                scStateDep,
                scStateArr,
                depDate = 000,
                tof = 1100,
                N = 2,
                depBody = 'parameter_not_set',
                target = 'parameter_not_set'):

        self.scStateDep = scStateDep
        self.scStateArr = scStateArr
        self.depDate = depDate
        self.tof = tof
        self.N = N
        self.target = target
        self.depBody = depBody

    def fitness(self, x):
        obj1 = 0
        transfer = hodographicShaping(self.scStateDep, self.scStateArr,
                                departureDate = self.depDate,
                                tof = self.tof,
                                N = self.N,
                                departureBody = self.depBody,
                                arrivalBody = self.target,
                                rShape =         'CPowPow2_scaled',
                                thetaShape =     'CPowPow2_scaled',
                                zShape =         'CosR5P3CosR5P3SinR5_scaled',
                                rShapeFree =     'PSin05PCos05_scaled',
                                thetaShapeFree = 'PSin05PCos05_scaled',
                                zShapeFree =     'P4CosR5P4SinR5_scaled',
                                rFreeC =         [x[0], x[1]],
                                thetaFreeC =     [x[2], x[3]],
                                zFreeC =         [x[4], x[5]],
                                )
        transfer.shapingRadial()
        transfer.shapingVertical()
        transfer.shapingTransverse()
        transfer.assembleThrust()
        transfer.evaluate(evalThrust=False)
        obj1 = transfer.deltaV

        return [obj1, ]

    def get_nobj(self):
        return 1

    def get_bounds(self):

        # box bounds
        parameterBoundsMin = [-1e6, -1e6, -1e6, -1e6, -1e6, -1e6]
        parameterBoundsMax = [ 1e6,  1e6,  1e6,  1e6,  1e6,  1e6]

        return (parameterBoundsMin, parameterBoundsMax)

    # Integer Dimension
    def get_nix(self):
        return 0
