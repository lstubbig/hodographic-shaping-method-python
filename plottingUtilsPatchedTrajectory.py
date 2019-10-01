import matplotlib as mlt
mlt.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import OrderedDict
import numpy as np
import pykep as pk

from utils import *
from conversions import *
from patchedTrajectoryUtils import ephemeris

class patchedPlots(object):
    '''
    Provide functionality to plot patched trajectories
    '''

    def __init__(self, transfers, samples=100, folder='graveyard',
                    save=False, show=True, addBody=[], ephems='spice'):
        '''
        Input:  transfers   list of hodographic shaping objects
                samples     number of sample points along trajectory and orbits
                folder      output folder
                save        determine if plots are saved to disk
                show        determine if plots are opened in individual windows
        '''

        print('\nBegin plotting.')
        print('Sampling at', samples, 'points.')

        if save:
            checkFolder(folder)
            print(f'Saving plots to folder: {folder}')

        self.samples = samples
        self.transfers = transfers
        self.save = save
        self.folder = folder
        self.show = show
        self.ephems = ephems
        self.epochs, self.tSamples = self.retrieveEpochs()
        self.bodyEphemeris, self.bodyNames = self.samplePlanets(add=addBody)
        self.scEphemeris, self.idxEpo = self.sampleTrajectory()
        self.scThrust = self.sampleTrajectoryThrust()

        self.colors = ['C0', 'C1', 'C2', 'C3', 'C4',
                       'C5', 'C6', 'C7', 'C8', 'C9']

    def plotSphere(self, posCart, radius):
        '''
        Draw a sphere of specified size at a specified location in the current
        figure
        Input:  posCart     Cartesian position vector [x, y, z]
                radius      radius of the sphere [r]
        '''

        ax = plt.gca()
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = posCart[0] + radius*np.cos(u)*np.sin(v)
        y = posCart[1] + radius*np.sin(u)*np.sin(v)
        z = posCart[2] + radius*np.cos(v)
        ax.plot_wireframe(x, y, z, color="r")

    def planetSystem(self, save=None, folder=None, plotSOI=False,
                        scaling=False):
        '''
        Plots the orbits of the visited planets
        Options:    save    save the plot (True/False) in png and pdf
                    folder  specify differing folder for the saved plot
                    plotSOI plots the sphere of influence of the planets
                    scaling adjusts the axis limits to the same, summetric
                            range, pass a number in [AU]
        '''

        print('Plot the planets in 3D.')

        ephem = self.bodyEphemeris

        # start figure
        fig = newFigure(height=6, target='paper')
        ax = fig.gca(projection='3d')

        # Sun
        ax.scatter([0], [0], [0], s=100, color='yellow', label='Sun',
                    marker='o', edgecolor='orange')

        # celestial bodies: orbits and position at launch
        for body in self.bodyNames:
            ax.plot(ephem[body]['rCart'][0, :]/pk.AU,
                    ephem[body]['rCart'][1, :]/pk.AU,
                    ephem[body]['rCart'][2, :]/pk.AU,
                    label=body)
            ax.scatter(ephem[body]['rCart'][0, 0]/pk.AU,
                       ephem[body]['rCart'][1, 0]/pk.AU,
                       ephem[body]['rCart'][2, 0]/pk.AU)
            if plotSOI:
                plotting = True
                if body=='1':
                    jplName = 'mercury'
                elif body=='2':
                    jplName = 'venus'
                if body=='3':
                    jplName = 'earth'
                elif body=='4':
                    jplName = 'mars'
                elif body=='5':
                    jplName = 'jupiter'
                elif body=='6':
                    jplName = 'saturn'
                elif body=='7':
                    jplName = 'uranus'
                elif body=='8':
                    jplName = 'neptune'
                else:
                    print(f'\tNo SOI computed for {body}.')
                    plotting = False

                if plotting:
                    __, __, planet = ephemeris(jplName, 0, mode='jpl')
                    muBody = planet.mu_self
                    muCentral = pk.MU_SUN
                    rSOI = np.sqrt(
                          ephem[body]['rCart'][0, 0]**2
                        + ephem[body]['rCart'][1, 0]**2
                        + ephem[body]['rCart'][2, 0]**2)\
                        * (muBody/muCentral)**(2/5)
                    self.plotSphere(ephem[body]['rCart'][:, 0]/pk.AU,
                                    rSOI/pk.AU)
                    # print(f'\tGravitational parameter {body}: {muBody:.2E} m^3/s^2')
                    print(f'\tRadius of SOI {body}: {rSOI/1e3:.2E} km = {rSOI/pk.AU:.2E} AU')

        # formatting
        if scaling:
            axisEqual3D(ax)
            # ax.set_xlim(-scaling, scaling)
            # ax.set_ylim(-scaling, scaling)
            # ax.set_zlim(-scaling, scaling)

        ax.set_xlabel('x [AU]', labelpad=15)
        ax.set_ylabel('y [AU]', labelpad=15)
        ax.set_zlabel('z [AU]', labelpad=15)
        plt.grid()
        plt.legend(loc='lower right')

        if save==None:
            save = self.save
        if folder==None:
            folder = self.folder
        if save:
            checkFolder(folder)
            plt.savefig(os.path.join(os.getcwd(), folder, 'solarSystem.pdf'),
                        dpi=300)
            plt.savefig(os.path.join(os.getcwd(), folder, 'solarSystem.png'),
                        dpi=300)
        if self.show:
            plt.show()

    def trajectory3D(self, scaling=False, save=None, folder=None):
        '''
        Plots the trajectory and the orbits of the visited bodies
        '''

        print('Plot the trajectory in 3D.')


        # start figure
        fig = plt.figure(figsize=(7, 5))
        ax = fig.gca(projection='3d')

        # Sun
        ax.scatter([0], [0], [0], s=100, color='yellow', label='Sun',
                    marker='o', edgecolor='orange')

        # celestial bodies
        ephem = self.bodyEphemeris
        for i, body in enumerate(self.bodyNames):
            ax.plot(ephem[body]['rCart'][0, :]/pk.AU,
                    ephem[body]['rCart'][1, :]/pk.AU,
                    ephem[body]['rCart'][2, :]/pk.AU,
                    label=body)

        # spacecraft
        ax.plot(self.scEphemeris['rCart'][0, :]/pk.AU,
                self.scEphemeris['rCart'][1, :]/pk.AU,
                self.scEphemeris['rCart'][2, :]/pk.AU,
                label='spacecraft')

        # notable positions
        for i, epoch in enumerate(self.epochs):
            ax.scatter(self.scEphemeris['rCart'][0, self.idxEpo[i]]/pk.AU,
                    self.scEphemeris['rCart'][1, self.idxEpo[i]]/pk.AU,
                    self.scEphemeris['rCart'][2, self.idxEpo[i]]/pk.AU,
                    zorder=4)

        # formatting
        # ax.set_aspect('equal')
        if scaling:
            ax.set_zlim(-2.3, 2.3)
        ax.set_xlabel('x [AU]', labelpad=15)
        ax.set_ylabel('y [AU]', labelpad=15)
        ax.set_zlabel('z [AU]', labelpad=15)
        plt.grid()
        plt.legend()

        if save==None:
            save = self.save
        if folder==None:
            folder = self.folder
        if save:
            checkFolder(folder)
            plt.savefig(os.path.join(os.getcwd(), folder, 'trajectory3D.pdf'),
                        dpi=300)
            plt.savefig(os.path.join(os.getcwd(), folder, 'trajectory3D.png'),
                        dpi=300)
        if self.show:
            plt.show()

    def trajectory3Danimation(self, staticOrbits=True, scaling=False,
                                save=None, folder=None):
        '''
        Plots the trajectory and the orbits of the visited bodies
        '''

        print('Plot animation of the trajectory in 3D.')

        # start figure
        fig = plt.figure(figsize=(8, 8))
        ax = fig.gca(projection='3d')

        # start with an empty plot, format lines and markers
        line0, = plt.plot([], [], [], self.colors[0]+'-', zorder=4,
                    label='Trajectory')
        dot0, = plt.plot([], [], [], self.colors[0]+'o', zorder=4)

        # plot the complete orbits in the beginning
        ephem = self.bodyEphemeris
        dots = []
        for i, body in enumerate(self.bodyNames):
            c = self.colors[i+1]
            if staticOrbits:
                ax.plot(ephem[body]['rCart'][0, :]/pk.AU,
                        ephem[body]['rCart'][1, :]/pk.AU,
                        ephem[body]['rCart'][2, :]/pk.AU,
                        c+'-', label=body)
            dot = plt.plot([], [], [], c+'o', zorder=3)
            dots.append(dot)

        # Sun
        ax.scatter([0], [0], [0], s=100, color='yellow', label='Sun',
                    marker='o', edgecolor='orange')

        # formatting
        ax.set_xlabel('x [AU]', labelpad=15)
        ax.set_ylabel('y [AU]', labelpad=15)
        ax.set_zlabel('z [AU]', labelpad=15)
        if scaling:
            ax.set_xlim(-scaling, scaling)
            ax.set_ylim(-scaling, scaling)
            ax.set_zlim(-scaling, scaling)
        plt.grid(True)

        # this function will be called at every iteration
        scEphem = self.scEphemeris['rCart']/pk.AU
        def update_line(num, line0, dot0, dots):
            line0.set_data(scEphem[0:2, :num])
            line0.set_3d_properties(scEphem[2, :num])
            dot0.set_data(scEphem[0:2, num])
            dot0.set_3d_properties(scEphem[2, num])
            for i, dot in enumerate(dots):
                body = self.bodyNames[i]
                dot[0].set_data(ephem[body]['rCart'][0:2, num]/pk.AU)
                dot[0].set_3d_properties(ephem[body]['rCart'][2, num]/pk.AU)
            return line0,

        nFrame = int(len(scEphem[0, :]))
        line_ani = animation.FuncAnimation(fig, update_line, frames=nFrame,
                            fargs=(line0, dot0, dots),
                            interval=40, repeat_delay=1e3)

        plt.legend()

        if save==None:
            save = self.save
        if folder==None:
            folder = self.folder
        if save==True:
            checkFolder(folder)
            print('Rendering...', end='', flush=True)
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=60, metadata=dict(artist='Leon S'),
                            bitrate=1800)
            line_ani.save(os.path.join(os.getcwd(), folder, 'trajectory3D.mp4'),
                            writer=writer)
            print('Done!')

        plt.show()


    def trajectory2D(self, half=False, scaling=False, save=None, folder=None):
        '''
        Plots the trajectory and the orbits of the visited bodies
        '''

        print('Plot the trajectory in 2D.')


        # start figure
        # fig = plt.figure(figsize=(14, 10))
        if half == True:
            fig = newFigure(height=6.5/2, target='paper', half=half)
        else:
            fig = newFigure(height=6.5, target='paper')
        ax = fig.gca()

        # Sun
        ax.scatter([0], [0], s=80, color='yellow',
                    marker='o', edgecolor='orange', zorder=3)

        # celestial bodies orbits
        ephem = self.bodyEphemeris
        for i, body in enumerate(self.bodyNames):
            ax.plot(ephem[body]['rCart'][0, :]/pk.AU,
                    ephem[body]['rCart'][1, :]/pk.AU,
                    label=body, zorder=2, linewidth=0.8)

        # spacecraft
        ax.plot(self.scEphemeris['rCart'][0, :]/pk.AU,
                self.scEphemeris['rCart'][1, :]/pk.AU,
                label='spacecraft', zorder=3, linewidth=1)

        # notable positions
        for i, epoch in enumerate(self.epochs):
            ax.scatter(self.scEphemeris['rCart'][0, self.idxEpo[i]]/pk.AU,
                    self.scEphemeris['rCart'][1, self.idxEpo[i]]/pk.AU,
                    c='k', alpha=0.5, zorder=4, s=8)

        # formatting
        ax.set_aspect('equal')
        ax.set_xlabel('x [AU]')
        ax.set_ylabel('y [AU]')
        ax.set_xlim((-3.5, 3.5))
        ax.set_ylim((-3.5, 3.5))
        # plt.grid(zorder=0)
        plt.legend(loc='lower left')
        plt.tight_layout()

        if save==None:
            save = self.save
        if folder==None:
            folder = self.folder
        if save:
            checkFolder(folder)
            plt.savefig(os.path.join(os.getcwd(), folder, 'trajectory2D.pdf'),
                        dpi=300)
            plt.savefig(os.path.join(os.getcwd(), folder, 'trajectory2D.png'),
                        dpi=300)
        if self.show:
            plt.show()

    def trajectory2Danimation(self, staticOrbits=True, save=None, folder=None):
        '''
        Plots the trajectory and the orbits of the visited bodies
        '''

        print('Plot animation of the trajectory in 2D.')

        # start figure
        fig = newFigure(height=6.5, target='paper', dpi=300)
        ax = fig.gca()

        # start with an empty plot, format lines and markers
        line0, = plt.plot([], [], self.colors[0]+'-', zorder=4,
                        label='Trajectory')
        dot0, = plt.plot([], [], self.colors[0]+'o', zorder=4)

        # plot the complete orbits in the beginning
        ephem = self.bodyEphemeris
        dots = []
        for i, body in enumerate(self.bodyNames):
            c = self.colors[i+1]
            if staticOrbits:
                ax.plot(ephem[body]['rCart'][0, :]/pk.AU,
                        ephem[body]['rCart'][1, :]/pk.AU,
                        c+'-', label=body)
            dot = plt.plot([], [], c+'o', zorder=3)
            dots.append(dot)

        # Sun
        ax.scatter([0], [0], s=100, color='yellow', label='Sun',
                    marker='o', edgecolor='orange', zorder=3)

        # formatting
        plt.xlabel('x [AU]')
        plt.ylabel('y [AU]')
        plt.grid(True)

        # this function will be called at every iteration
        scEphem = self.scEphemeris['rCart']/pk.AU
        def update_line(num, line0, dot0, dots):
            line0.set_data(scEphem[0, :num], scEphem[1, :num])
            dot0.set_data(scEphem[0, num], scEphem[1, num])
            for i, dot in enumerate(dots):
                body = self.bodyNames[i]
                dot[0].set_data(ephem[body]['rCart'][0, num]/pk.AU,
                                ephem[body]['rCart'][1, num]/pk.AU)
            if num in self.idxEpo:
                ax.scatter(scEphem[0, num], scEphem[1, num], c='k', alpha=0.5,
                            zorder=4)
            return line0,

        nFrame = int(len(scEphem[0, :]))
        line_ani = animation.FuncAnimation(fig, update_line, frames=nFrame,
                            fargs=(line0, dot0, dots),
                            interval=40, repeat_delay=5e3)

        plt.legend(loc=1)
        plt.tight_layout()

        if save==None:
            save = self.save
        if folder==None:
            folder = self.folder
        if save==True:
            checkFolder(folder)
            print('Rendering...', end='', flush=True)
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=30, metadata=dict(artist='Leon S'),
                            bitrate=1800)
            line_ani.save(os.path.join(os.getcwd(), folder, 'trajectory2D.mp4'),
                            writer=writer, dpi=300)
            print('Done!')

        plt.show()

    def hodograph(self, scaling=False, save=None, folder=None):
        '''
        Plots the trajectory and the orbits of the visited bodies
        '''

        print('Plot the hodograph.')

        id = self.idxEpo

        # start figure
        fig = plt.figure(figsize=(10, 10))
        ax = plt.gca()

        # celestial bodies velocities
        ephem = self.bodyEphemeris
        for i, body in enumerate(self.bodyNames):
            ax.plot(ephem[body]['vCyl'][0, :]/1e3,
                    ephem[body]['vCyl'][1, :]/1e3,
                    label=body, zorder=2)

        # points that don't coincide with the s/c velocities
        ax.scatter(ephem[self.bodyNames[0]]['vCyl'][0, 0]/1e3,
                   ephem[self.bodyNames[0]]['vCyl'][1, 0]/1e3,
                   zorder=2)
        ax.scatter(ephem[self.bodyNames[1]]['vCyl'][0, id[1]]/1e3,
                   ephem[self.bodyNames[1]]['vCyl'][1, id[1]]/1e3,
                   zorder=2)

        # spacecraft
        ax.plot(self.scEphemeris['vCyl'][0, 0:id[1]]/1e3,
                self.scEphemeris['vCyl'][1, 0:id[1]]/1e3,
                label='transfer 1', zorder=3)
        ax.scatter(self.scEphemeris['vCyl'][0, 0]/1e3,
                   self.scEphemeris['vCyl'][1, 0]/1e3,
                   label='launch', zorder=3)
        ax.scatter(self.scEphemeris['vCyl'][0, id[1]-1]/1e3,
                   self.scEphemeris['vCyl'][1, id[1]-1]/1e3,
                   label='flyby approach', zorder=3)
        ax.scatter(self.scEphemeris['vCyl'][0, id[1]]/1e3,
                   self.scEphemeris['vCyl'][1, id[1]]/1e3,
                   label='flyby departure', zorder=3)
        ax.plot(self.scEphemeris['vCyl'][0, id[1]:id[2]]/1e3,
                   self.scEphemeris['vCyl'][1, id[1]:id[2]]/1e3,
                   label='transfer 2', zorder=3)
        ax.scatter(self.scEphemeris['vCyl'][0, id[2]]/1e3,
                   self.scEphemeris['vCyl'][1, id[2]]/1e3,
                   label='rendezvous', zorder=3)
        ax.plot(self.scEphemeris['vCyl'][0, id[2]:id[3]]/1e3,
                self.scEphemeris['vCyl'][1, id[2]:id[3]]/1e3,
                label='stay 1', zorder=3)
        ax.scatter(self.scEphemeris['vCyl'][0, id[3]]/1e3,
                   self.scEphemeris['vCyl'][1, id[3]]/1e3,
                   label='departure', zorder=3)
        ax.plot(self.scEphemeris['vCyl'][0, id[3]:id[4]]/1e3,
                self.scEphemeris['vCyl'][1, id[3]:id[4]]/1e3,
                label='transfer 3', zorder=3)
        ax.scatter(self.scEphemeris['vCyl'][0, id[4]]/1e3,
                   self.scEphemeris['vCyl'][1, id[4]]/1e3,
                   label='arrival', zorder=3)

        # formatting
        # ax.set_title('Hodograph')
        ax.set_aspect('equal')
        ax.set_xlabel(r'$V_r$ [km/s]', labelpad=15)
        ax.set_ylabel(r'$V_t$ [km/s]', labelpad=15)
        plt.grid(zorder=0)
        plt.legend()

        if save==None:
            save = self.save
        if folder==None:
            folder = self.folder
        if save:
            checkFolder(folder)
            plt.savefig(os.path.join(os.getcwd(), folder, 'hodograph.pdf'),
                        dpi=300)
            plt.savefig(os.path.join(os.getcwd(), folder, 'hodograph.png'),
                        dpi=300)
        if self.show:
            plt.show()

    def hodographAsteroids(self, scaling=False, save=None, folder=None):
        '''
        Plots the trajectory and the orbits of the visited bodies
        '''

        print('Plot the hodograph.')

        id = self.idxEpo

        # start figure
        fig = newFigure(height=6, target='paper')
        ax = plt.gca()

        # celestial bodies velocities
        ephem = self.bodyEphemeris
        for i, body in enumerate(self.bodyNames):
            ax.plot(ephem[body]['vCyl'][0, :]/1e3,
                    ephem[body]['vCyl'][1, :]/1e3,
                    label=body, zorder=2)

        # points that don't coincide with the s/c velocities
        # ax.scatter(ephem[self.bodyNames[0]]['vCyl'][0, 0]/1e3,
        #            ephem[self.bodyNames[0]]['vCyl'][1, 0]/1e3,
        #            zorder=2)
        # ax.scatter(ephem[self.bodyNames[1]]['vCyl'][0, id[1]]/1e3,
        #            ephem[self.bodyNames[1]]['vCyl'][1, id[1]]/1e3,
        #            zorder=2)

        # spacecraft plots
        # epoch markers
        for i, ii in enumerate(id):
            ax.scatter(self.scEphemeris['vCyl'][0, ii]/1e3,
                       self.scEphemeris['vCyl'][1, ii]/1e3,
                       label='', zorder=4, color='grey', s=15,
                       alpha=.8, linewidths=1)
        # transfer 1
        ax.plot(self.scEphemeris['vCyl'][0, 0:id[1]]/1e3,
                self.scEphemeris['vCyl'][1, 0:id[1]]/1e3,
                label='transfer', zorder=3, color='C5')
        # stay 1
        ax.plot(self.scEphemeris['vCyl'][0, id[1]:id[2]]/1e3,
                   self.scEphemeris['vCyl'][1, id[1]:id[2]]/1e3,
                   label='', zorder=3, color='C5')
        # transfer 2
        ax.plot(self.scEphemeris['vCyl'][0, id[2]:id[3]]/1e3,
                self.scEphemeris['vCyl'][1, id[2]:id[3]]/1e3,
                label='', zorder=3, color='C5')
        # stay 2
        ax.plot(self.scEphemeris['vCyl'][0, id[3]:id[4]]/1e3,
                self.scEphemeris['vCyl'][1, id[3]:id[4]]/1e3,
                label='', zorder=3, color='C5')
        # transfer 3
        ax.plot(self.scEphemeris['vCyl'][0, id[4]:id[5]]/1e3,
                self.scEphemeris['vCyl'][1, id[4]:id[5]]/1e3,
                label='', zorder=3, color='C5')
        # stay 3
        ax.plot(self.scEphemeris['vCyl'][0, id[5]:id[6]]/1e3,
                self.scEphemeris['vCyl'][1, id[5]:id[6]]/1e3,
                label='', zorder=3, color='C5')
        # transfer 4
        ax.plot(self.scEphemeris['vCyl'][0, id[6]:id[7]]/1e3,
                self.scEphemeris['vCyl'][1, id[6]:id[7]]/1e3,
                label='', zorder=3, color='C5')
        # ax.scatter(self.scEphemeris['vCyl'][0, id[4]]/1e3,
        #            self.scEphemeris['vCyl'][1, id[4]]/1e3,
        #            label='', zorder=3, color='grey', s=13)

        # formatting
        # ax.set_title('Hodograph')
        ax.set_aspect('equal')
        ax.set_xlabel(r'$V_r$ [km/s]', labelpad=15)
        ax.set_ylabel(r'$V_t$ [km/s]', labelpad=15)
        plt.grid(zorder=0)
        plt.legend()

        if save==None:
            save = self.save
        if folder==None:
            folder = self.folder
        if save:
            checkFolder(folder)
            plt.savefig(os.path.join(os.getcwd(), folder, 'hodograph.pdf'),
                        dpi=300)
            plt.savefig(os.path.join(os.getcwd(), folder, 'hodograph.png'),
                        dpi=300)
        if self.show:
            plt.show()

    def thrust(self, save=None, folder=None):
        """
        Plot the thrust profile in Cylindrical coordinates
        """

        print('Plot thrust profile.')

        fig = newFigure(height=2.5, target='paper')
        ax = plt.gca()

        # Cylindrical accelerations
        plt.plot(self.tSamples, self.scThrust['thrust'][0, :], ':',
                    label=r'$f_r$')
        plt.plot(self.tSamples, self.scThrust['thrust'][1, :], '--',
                    label=r'$f_\theta$')
        plt.plot(self.tSamples, self.scThrust['thrust'][2, :], '-.',
                    label=r'$f_z$')
        plt.plot(self.tSamples, self.scThrust['thrust'][3, :], '-',
                    label=r'$f_{\mathrm{total}}$', alpha=0.5)
        plt.grid()
        plt.xlabel('time [mjd2000]')
        plt.ylabel(r'$f$ $[m/s^2]$')
        plt.xlim([self.tSamples[0], self.tSamples[-1]])
        plt.ticklabel_format(style='sci', axis='y', scilimits=(-4,-4))
        # plt.ylim([-0.0004, 0.0005])
        plt.legend()
        plt.tight_layout()

        if save==None:
            save = self.save
        if folder==None:
            folder = self.folder
        if save==True:
            checkFolder(folder)
            plt.savefig(os.path.join(os.getcwd(), folder, 'thrust.pdf'), dpi=300)
            plt.savefig(os.path.join(os.getcwd(), folder, 'thrust.png'), dpi=300)
        plt.show()


    def retrieveEpochs(self):
        '''
        Generate a sorted list of epochs of switching (patching) points
        Generate vector of sample points in time [days]
        Eg. Launch, swingby, rendezvous
        '''

        epochs = []
        for trans in self.transfers:
            start = trans.jdDep
            end = trans.jdArr
            epochs.extend([start, end])

        # remove duplicates and sort list
        epochs = list(set(epochs))
        epochs.sort()
        # print(f'Notable epochs: {epochs} mjd2000')
        print(f'Notable epochs: [{" ".join((f"{x:.2f}") for x in epochs)}] mjd2000')

        # gernerate sample points
        tSamples = np.linspace(epochs[0], epochs[-1], self.samples)

        return epochs, tSamples

    def samplePlanets(self, add=[]):
        '''
        Returns the state of the bodies visited during the patched transfer
        Ephemeris source is passed as a list in self.ephems
        The body names defined in the transfer objects must correspond to the
        names or IDs expected by the ephemeris source
        add manually adds another body to be included in the plot
        Output structure: Dictionary with a sub-dictionary for each body,
            containing a (3 x samples) numpy array
        '''

        print('Sample celestial body ephemerides')

        # retrieve names of all visited bodies
        bodyNames = []
        bodyPykep = []
        for trans in self.transfers:
            body1 = trans.departureBody
            body2 = trans.arrivalBody
            bodyNames.extend([body1, body2])

        for entry in add:
            bodyNames.extend(entry)

        # remove duplicates (kepping the order)
        # bodyNames = list(set(bodyNames))
        bodyNames = list(OrderedDict.fromkeys(bodyNames))
        print(f'Visited bodies: {bodyNames}')

        # retrieve the corresponding ephemeris source if mode is 'auto'
        # else default to the spice ephemerides
        # Example syntax for spice: '1234', for jpl: 'mars', for gtoc2: 123
        mode = []
        if self.ephems == 'auto':
            for body in bodyNames:
                if type(body) == str:
                    try:
                        int(body)
                        mode.append('spice')
                    except ValueError:
                        mode.append('jpl')
                else:
                    mode.append('gtoc2')
        else:
            mode = ['spice']*len(bodyNames)

        # create nested dictionary
        states = ['rCart', 'vCart', 'rCyl', 'vCyl']
        bodyEphemeris = {name: {state:np.empty([3, self.samples])
                            for state in states} for name in bodyNames}

        # retrieve values for each body and each state vector
        # for body in bodyNames:
        #     pkBody = pk.planet.spice(body, 'sun', 'eclipj2000')
        #     for i, t in enumerate(self.tSamples):
        #         rCart, vCart = pkBody.eph(t)
        #         bodyEphemeris[body]['rCart'][:, i] = rCart
        #         bodyEphemeris[body]['vCart'][:, i] = vCart
        #         bodyEphemeris[body]['rCyl'][:, i] = Pcart2cyl(rCart)
        #         bodyEphemeris[body]['vCyl'][:, i] = Vcart2cyl(vCart, rCart)

        # retrieve values for each body and each state vector
        for j, body in enumerate(bodyNames):
            for i, t in enumerate(self.tSamples):
                stateCyl, stateCart, __ = ephemeris(body, t, mode=mode[j])
                bodyEphemeris[body]['rCart'][:, i] = stateCart[0:3]
                bodyEphemeris[body]['vCart'][:, i] = stateCart[3:6]
                bodyEphemeris[body]['rCyl'][:, i] = stateCyl[0:3]
                bodyEphemeris[body]['vCyl'][:, i] = stateCyl[3:6]

        return bodyEphemeris, bodyNames

    def sampleTrajectory(self):
        '''
        Returns a dictionary with the sampled trajectory
        Switches between the low thrust legs at the switching points in 'epochs'
        '''

        print('Sample the spacecraft trajectory.')

        states = ['rCart', 'vCart', 'rCyl', 'vCyl']
        scState = {state:np.empty([3, self.samples]) for state in states}

        # indeces that are closest to the switching points
        idxEpo = []
        for epoch in self.epochs:
            idxEpo.append((np.abs(self.tSamples - epoch)).argmin())

        # sample the segments consecutively
        i = 0
        idxEpo[-1] += 1
        for j, transfer in enumerate(self.transfers):
            for t in np.linspace(0, transfer.tofSec, idxEpo[j+1]-idxEpo[j]):
                rCyl = [transfer.r(t), transfer.t(t), transfer.z(t)]
                vCyl = [transfer.rDot(t), transfer.tDot(t), transfer.zDot(t)]
                scState['rCyl'][:, i] = rCyl
                scState['vCyl'][:, i] = vCyl
                scState['rCart'][:, i] = Pcyl2cart(rCyl)
                scState['vCart'][:, i] = Vcyl2cart(vCyl, rCyl)
                i += 1

        idxEpo[-1] -= 1

        return scState, idxEpo

    def sampleTrajectoryThrust(self):
        '''
        Returns a dictionary with the thrust in Cylindrical coordinates
            [fr, ft, fz, ftotal]
        '''

        print('Sample thrust profile.')

        scThrust = {'thrust':np.empty([4, self.samples])}

        # sample the segments consecutively
        i = 0
        idxEpo = np.copy(self.idxEpo)
        idxEpo[-1] += 1
        for j, transfer in enumerate(self.transfers):
            for t in np.linspace(0, transfer.tofSec, idxEpo[j+1]-idxEpo[j]):
                thrust = [transfer.fr(t), transfer.ft(t), transfer.fz(t),
                            transfer.fTotal(t)]
                scThrust['thrust'][:, i] = thrust
                i += 1

        return scThrust
