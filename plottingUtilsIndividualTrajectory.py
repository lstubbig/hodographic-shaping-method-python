import matplotlib as mlt
# mlt.use('TkAgg')
# mlt.use('Qt5Agg')
import matplotlib.pyplot as plt

import numpy as np
import pykep as pk

from utils import *
from conversions import *

class plotting(object):
    """
    Provide visualization for hodographicShaping trajectories
    Samples trajectory at initialization
    Methods to plot various characteristics (3D trajectory, hodograph, etc.)
    """

    def __init__(self, trajectory, samples=100, folder='graveyeard', save=False,
                    ephemSource = 'jpl'):
        '''
        Create plotting object
        trajectory needs to be of type 'hodographicShaping'
        ephemSource needs to correspond to the one used for trajectory (due to
        planet names)
        '''

        print('\nBegin plotting.')
        print('Sampling at', samples, 'points.')

        self.samples = samples
        self.trajectory = trajectory
        self.folder = folder
        self.save = save
        self.ephemSource = ephemSource

        if self.save==True:
            checkFolder(self.folder)

        # sample planets and trajectory
        self.plPosCart, self.plPosCyl, self.plVelCart, self.plVelCyl = \
                self.samplePlanets(trajectory, samples=samples)
        self.traPosCart, self.traPosCyl = \
                self.sampleTrajectoryPosition(trajectory, samples=samples)
        self.traVelCart, self.traVelCyl = \
                self.sampleTrajectoryVelocity(trajectory, samples=samples)
        self.traAccCyl, self.traAccCart = \
                self.sampleTrajectoryAcceleration(trajectory, samples=samples)

    def trajectory3D(self, save=None, folder=None, scaling=True):
        """
        Plot the given trajectory in 3D
        """

        print('Plot 3D trajectory')

        # start figure
        fig = newFigure(height=6.4)
        ax = fig.gca(projection='3d')

        # Sun
        ax.scatter([0], [0], [0], s=100, color='yellow', label='Sun', marker='o', edgecolor='orange',)

        # Departure planet
        ax.plot(self.plPosCart['xDep']/pk.AU, self.plPosCart['yDep']/pk.AU, self.plPosCart['zDep']/pk.AU, label='Departure planet', c='C0')
        ax.scatter(self.plPosCart['xDep'][0]/pk.AU,  self.plPosCart['yDep'][0]/pk.AU,  self.plPosCart['zDep'][0]/pk.AU, c='k')
        ax.scatter(self.plPosCart['xDep'][-1]/pk.AU, self.plPosCart['yDep'][-1]/pk.AU, self.plPosCart['zDep'][-1]/pk.AU, c='k')

        # Arrival planet
        ax.plot(self.plPosCart['xArr']/pk.AU, self.plPosCart['yArr']/pk.AU, self.plPosCart['zArr']/pk.AU, label='Arrival planet', c='C3')
        ax.scatter(self.plPosCart['xArr'][0]/pk.AU,  self.plPosCart['yArr'][0]/pk.AU,  self.plPosCart['zArr'][0]/pk.AU, c='k')
        ax.scatter(self.plPosCart['xArr'][-1]/pk.AU, self.plPosCart['yArr'][-1]/pk.AU, self.plPosCart['zArr'][-1]/pk.AU, c='k')

        # Trajectory
        ax.plot(self.traPosCart['x']/pk.AU, self.traPosCart['y']/pk.AU, self.traPosCart['z']/pk.AU, label='Trajectory', c='C1')
        ax.scatter(self.traPosCart['x'][0]/pk.AU, self.traPosCart['y'][0]/pk.AU, self.traPosCart['z'][0]/pk.AU, label='launch', c='C2')
        ax.scatter(self.traPosCart['x'][-1]/pk.AU, self.traPosCart['y'][-1]/pk.AU, self.traPosCart['z'][-1]/pk.AU, label='arrival', c='C3')

        # formatting
        ax.set_aspect('equal')
        if scaling:
            ax.set_zlim(-1.5, 1.5)

        # plt.title('Orbits and trajectory')
        ax.set_xlabel('x [AU]', labelpad=15)
        ax.set_ylabel('y [AU]', labelpad=15)
        ax.set_zlabel('z [AU]', labelpad=15)
        plt.grid()
        plt.legend()

        if save==None:
            save = self.save
        if folder==None:
            folder = self.folder
        if save==True:
            checkFolder(folder)
            plt.savefig(os.path.join(os.getcwd(), folder, 'trajectory3D.pdf'), dpi=300)
            plt.savefig(os.path.join(os.getcwd(), folder, 'trajectory3D.png'), dpi=300)

        plt.show()

    def trajectory2D(self, save=None, folder=None, quiver=False):
        '''
        Two dimensional plot in the ecliptic plane
        '''

        fig = newFigure(height=6.4)

        # Sun
        sun = plt.scatter([0], [0], s=100, color='yellow', label='Sun', marker='o', edgecolor='orange')
        # arrival planet
        plot1 = plt.plot(self.plPosCart['xArr']/pk.AU, self.plPosCart['yArr']/pk.AU, label='Arrival Planet', color='C3', zorder=1)
        plot0 = plt.scatter(self.plPosCart['xArr'][0]/pk.AU, self.plPosCart['yArr'][0]/pk.AU, color='k', zorder=2)
        plot0 = plt.scatter(self.plPosCart['xArr'][-1]/pk.AU, self.plPosCart['yArr'][-1]/pk.AU, color='k', zorder=2)
        # departure planet
        plot1 = plt.plot(self.plPosCart['xDep']/pk.AU, self.plPosCart['yDep']/pk.AU, label='Departure Planet', color='C0', zorder=1)
        plot1 = plt.scatter(self.plPosCart['xDep'][0]/pk.AU, self.plPosCart['yDep'][0]/pk.AU, color='C2', label='launch', zorder=2)
        plot1 = plt.scatter(self.plPosCart['xArr'][-1]/pk.AU, self.plPosCart['yArr'][-1]/pk.AU, color='C3', label='arrival', zorder=2)
        # trajectory
        plot1 = plt.plot(self.traPosCart['x']/pk.AU, self.traPosCart['y']/pk.AU, label='Trajectory', color='C1', zorder=1)
        plot0 = plt.scatter(self.traPosCart['x'][0]/pk.AU, self.traPosCart['y'][0]/pk.AU, color='k', zorder=2)
        plot0 = plt.scatter(self.traPosCart['x'][-1]/pk.AU, self.traPosCart['y'][-1]/pk.AU, color='k', zorder=2)

        plt.xlabel('$x$ [AU]')
        plt.ylabel('$y$ [AU]')

        plt.grid()
        ax = plt.gca()
        ax.set_axisbelow(True)
        plt.legend()
        plt.axis('equal')
        if save==None:
            save = self.save
        if folder==None:
            folder = self.folder
        if save==True:
            checkFolder(folder)
            plt.savefig(os.path.join(os.getcwd(), folder, 'trajectory2D.pdf'), dpi=300)
            plt.savefig(os.path.join(os.getcwd(), folder, 'trajectory2D.png'), dpi=300)

        plt.show()


    def trajectory3Danimation(self, save=None, folder=None):
        """
        Animation of the flown trajectory
        """

        print('Show animated trajectory.')
        import matplotlib.animation as animation

        # data = np.array([x, y])
        data = np.vstack((self.traPosCart['x'],
                        self.traPosCart['y'],
                        self.traPosCart['z']))
        dataDep = np.vstack((self.plPosCart['xDep'],
                        self.plPosCart['yDep'],
                        self.plPosCart['zDep']))
        dataArr = np.vstack((self.plPosCart['xArr'],
                        self.plPosCart['yArr'],
                        self.plPosCart['zArr']))
        data /= pk.AU
        dataDep /= pk.AU
        dataArr /= pk.AU

        # create figure
        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca(projection='3d')

        # start with an empty plot
        line0, = plt.plot([], [], [], "C1-", zorder=3)
        dot0, = plt.plot([], [], [], "C1o", zorder=3)
        dot1, = plt.plot([], [], [], "C0o", zorder=3)
        dot2, = plt.plot([], [], [], "C3o", zorder=3)

        # Sun
        sun = ax.scatter([0], [0], [0], s=100, color='yellow', label='Sun',
                            marker='o', edgecolor='orange')
        # Departure planet
        planet1 = ax.plot(self.plPosCart['xDep']/pk.AU,
                        self.plPosCart['yDep']/pk.AU,
                        self.plPosCart['zDep']/pk.AU,
                        label='Departure planet', c='C0')
        # Arrival planet
        planet2 = ax.plot(self.plPosCart['xArr']/pk.AU,
                        self.plPosCart['yArr']/pk.AU,
                        self.plPosCart['zArr']/pk.AU,
                        label='Arrival planet', c='C3')

        # formatting
        ax.set_xlabel('x [AU]', labelpad=15)
        ax.set_ylabel('y [AU]', labelpad=15)
        ax.set_zlabel('z [AU]', labelpad=15)
        ax.set_aspect('equal')
        # ax.set_zlim(-0.05, 0.05)
        ax.set_zlim(-1.5, 1.5)
        plt.grid(True)
        # plt.title("Low-thrust trajectory")

        # this function will be called at every iteration
        def update_line(num, data, line, dot0, dot1, dot2):
            line.set_data(data[0:2, :num])
            line.set_3d_properties(data[2, :num])
            dot0.set_data(data[0:2, num])
            dot0.set_3d_properties(data[2, num])
            dot1.set_data(dataDep[0:2, num])
            dot1.set_3d_properties(dataDep[2, num])
            dot2.set_data(dataArr[0:2, num])
            dot2.set_3d_properties(dataArr[2, num])
            return line,

        nFrame = int(len(self.traPosCart['x']))
        line_ani = animation.FuncAnimation(fig, update_line, frames=nFrame,
                                    fargs=(data, line0, dot0, dot1, dot2),
                                    interval=20, repeat_delay=1e3)

        if save==None:
            save = self.save
        if folder==None:
            folder = self.folder
        if save==True:
            checkFolder(folder)
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=60, metadata=dict(artist='Leon S'),
                            bitrate=1800)
            line_ani.save(os.path.join(os.getcwd(), folder, 'trajectory3D.mp4'),
                            writer=writer)
            # line_ani.save(os.path.join(os.getcwd(), folder,
                # 'trajectory3D.mp4'), fps=30, extra_args=['-vcodec', 'libx264'])

        plt.show()

    def hodograph(self, twoDplot=False, save=None, folder=None):
        """
        Plot the trajectory's hodograph
        Plot the given trajectory in 2D as subplot if twoDplot is set to True
        """

        print('Plot hodograph')
        # Hoodgraph and orbits
        if twoDplot:
            figHodoOrbit = newFigure(height=7)
        else:
            figHodoOrbit = newFigure(height=3)

        # Hodograph
        if twoDplot:
            plt.subplot(2, 1, 1)
        #departure planet
        plot1 = plt.plot(self.plVelCyl['VrDep']/1E3, self.plVelCyl['VtDep']/1E3, label='Departure Planet', color='C0', zorder=1)
        plot0 = plt.scatter(self.plVelCyl['VrDep'][-1]/1E3, self.plVelCyl['VtDep'][-1]/1E3, color='k', zorder=2)
        #arrival planet
        plot1 = plt.plot(self.plVelCyl['VrArr']/1E3, self.plVelCyl['VtArr']/1E3, label='Arrival Planet', color='C3', zorder=1)
        plot0 = plt.scatter(self.plVelCyl['VrArr'][0]/1E3, self.plVelCyl['VtArr'][0]/1E3, color='k', zorder=2)
        # trajectory
        plot1 = plt.plot(self.traVelCyl['vr']/1E3, self.traVelCyl['vt']/1E3, label='Trajectory', color='C1', zorder=1)
        plot1 = plt.scatter(self.traVelCyl['vr'][0]/1E3, self.traVelCyl['vt'][0]/1E3, color='C2', label='launch', zorder=2)
        plot1 = plt.scatter(self.traVelCyl['vr'][-1]/1E3, self.traVelCyl['vt'][-1]/1E3, color='C3', label='arrival', zorder=2)
        plt.xlabel('$V_r$ [km/s]')
        plt.ylabel('$V_t$ [km/s]')
        plt.grid()
        ax = plt.gca()
        ax.set_axisbelow(True)
        plt.legend()
        plt.axis('equal')

        # Positions
        if twoDplot:
            plt.title('Hodograph')
            plt.subplot(2, 1, 2)
            # trajectory
            plot1 = plt.plot(self.traPosCart['x']/pk.AU, self.traPosCart['y']/pk.AU, label='Trajectory', color='C1', zorder=1)
            plot0 = plt.scatter(self.traPosCart['x'][0]/pk.AU, self.traPosCart['y'][0]/pk.AU, color='k', zorder=2)
            plot0 = plt.scatter(self.traPosCart['x'][-1]/pk.AU, self.traPosCart['y'][-1]/pk.AU, color='k', zorder=2)
            # arrival planet
            plot1 = plt.plot(self.plPosCart['xArr']/pk.AU, self.plPosCart['yArr']/pk.AU, label='Arrival Planet', color='C3', zorder=1)
            plot0 = plt.scatter(self.plPosCart['xArr'][0]/pk.AU, self.plPosCart['yArr'][0]/pk.AU, color='k', zorder=2)
            plot0 = plt.scatter(self.plPosCart['xArr'][-1]/pk.AU, self.plPosCart['yArr'][-1]/pk.AU, color='k', zorder=2)
            # departure planet
            plot1 = plt.plot(self.plPosCart['xDep']/pk.AU, self.plPosCart['yDep']/pk.AU, label='Departure Planet', color='C0', zorder=1)
            plot1 = plt.scatter(self.plPosCart['xDep'][0]/pk.AU, self.plPosCart['yDep'][0]/pk.AU, color='C2', label='launch', zorder=2)
            plot1 = plt.scatter(self.plPosCart['xArr'][-1]/pk.AU, self.plPosCart['yArr'][-1]/pk.AU, color='C3', label='arrival', zorder=2)
            plt.xlabel('$x$ [AU]')
            plt.ylabel('$y$ [AU]')

            plt.grid()
            ax = plt.gca()
            ax.set_axisbelow(True)
            plt.legend()
            plt.title('Orbit')
            plt.axis('equal')

        plt.tight_layout()

        if save==None:
            save = self.save
        if folder==None:
            folder = self.folder
        if save==True:
            checkFolder(folder)
            plt.savefig(os.path.join(os.getcwd(), folder, 'hodograph.pdf'), dpi=300)
            plt.savefig(os.path.join(os.getcwd(), folder, 'hodograph.png'), dpi=300)

        plt.show()

    def thrust(self, save=None, folder=None):
        """
        Plot the thrust profile in Cylindrical coordinates
        """

        print('Plot thrust')

        fig = newFigure(height=3)

        samplePoints = self.tSampleSec

        # Cylindrical accelerations
        plot1 = plt.plot(self.tSample, self.trajectory.fr(samplePoints), ':', label=r'$f_r$')
        plot1 = plt.plot(self.tSample, self.trajectory.ft(samplePoints), '--', label=r'$f_\theta$')
        plot1 = plt.plot(self.tSample, self.trajectory.fz(samplePoints), '-.', label=r'$f_z$')
        plot1 = plt.plot(self.tSample, self.trajectory.fTotal(samplePoints), '-', label=r'$f_{\mathrm{total}}$', alpha=0.5)
        plt.grid()
        plt.xlabel('time [mjd2000]')
        plt.ylabel(r'$f$ $[m/s^2]$')
        plt.xlim([self.tSample[0], self.tSample[-1]])
        # plt.ylim([-0.0004, 0.0005])
        plt.title('Thrust acceleration')
        plt.legend()

        if save==None:
            save = self.save
        if folder==None:
            folder = self.folder
        if save==True:
            checkFolder(folder)
            plt.savefig(os.path.join(os.getcwd(), folder, 'thrust.pdf'), dpi=300)
            plt.savefig(os.path.join(os.getcwd(), folder, 'thrust.png'), dpi=300)
        plt.show()

    def thrustCart(self, save=None, folder=None):
        """
        Plot the thrust profile in Cartesian coordinates
        """

        print('Plot thrust (Cartesian)')

        fig = plt.figure(figsize=(10, 5))

        samplePoints = self.tSampleSec

        # Cartesian accelerations
        plot1 = plt.plot(self.tSample, self.traAccCart['ax'], ':', label=r'$f_x$')
        plot1 = plt.plot(self.tSample, self.traAccCart['ay'], '--', label=r'$f_y$')
        plot1 = plt.plot(self.tSample, self.traAccCart['az'], '-.', label=r'$f_z$')
        plot1 = plt.plot(self.tSample, self.traAccCart['total'], '-', label=r'$f_{\mathrm{total}}$', alpha=0.5)
        plt.grid()
        plt.xlabel('time [mjd2000]')
        plt.ylabel(r'$f$ $[m/s^2]$')
        plt.xlim([self.tSample[0], self.tSample[-1]])
        # plt.ylim([-0.0004, 0.0005])
        plt.title('Thrust acceleration (Cartesian)')
        plt.legend()

        if save==None:
            save = self.save
        if folder==None:
            folder = self.folder
        if save==True:
            checkFolder(folder)
            plt.savefig(os.path.join(os.getcwd(), folder, 'thrustCart.pdf'), dpi=300)
            plt.savefig(os.path.join(os.getcwd(), folder, 'thrustCart.png'), dpi=300)
        plt.show()

    def figure119(self, save=None, folder=None):
        """
        Plot the thrust profile next to the 3D trajectory
        Recreates Figure 11.9 in [Gondelach, 2012]
        """

        print('Plot trajectory and thrust, recreating Figure 11.9')

        samplePoints = self.tSampleSec

        # initialize figure
        fig = plt.figure(figsize=(10, 4))
        gs = mlt.gridspec.GridSpec(1, 2, width_ratios=[3, 2])
        ax0 = plt.subplot(gs[0], projection='3d')

        # plot 3D trajectory
        ax0.plot(self.plPosCart['xDep']/pk.AU, self.plPosCart['yDep']/pk.AU, self.plPosCart['zDep']/pk.AU, label='Earth', c='b')
        ax0.plot(self.plPosCart['xArr']/pk.AU, self.plPosCart['yArr']/pk.AU, self.plPosCart['zArr']/pk.AU, label='Mars', c='k')
        ax0.plot(self.traPosCart['x']/pk.AU, self.traPosCart['y']/pk.AU, self.traPosCart['z']/pk.AU, label='Transfer', c='r')

        # axis formatting
        ax0.set_xlim(-2, 2)
        ax0.set_xticks([-2, -1, 0, 1, 2])
        ax0.set_ylim(-2, 2)
        ax0.set_yticks([-2, 0, 2])
        ax0.set_zlim(-0.06, 0.05)
        ax0.view_init(30, -95)
        ax0.xaxis.pane.fill = False
        ax0.yaxis.pane.fill = False
        ax0.zaxis.pane.fill = False
        ax0.grid(False)
        ax0.set_xlabel('x [AU]')
        ax0.set_ylabel('y [AU]')
        ax0.set_zlabel('z [AU]', labelpad=10)
        ax0.tick_params(axis='z', pad=8)
        # plt.legend()

        # plot thrust profile
        ax1 = plt.subplot(gs[1])

        tDays = np.linspace(0, self.trajectory.tof, self.samples)
        ax1.plot(tDays, self.trajectory.fr(samplePoints), '-b', label='Radial')
        ax1.plot(tDays, self.trajectory.ft(samplePoints), '-r', label='Normal')
        ax1.plot(tDays, self.trajectory.fz(samplePoints), '-g', label='Axial')
        ax1.plot(tDays, self.trajectory.fTotal(samplePoints), '--k', label='Total')
        ax1.set_xlabel('Time [days]')
        ax1.set_xticks([0, 200, 400, 600, 800, 1000, 1200])
        ax1.set_ylabel('Thrust acceleration [m/s^2]')
        ax1.set_ylim([-5e-5, 20e-5])
        ax1.set_xlim(left=tDays[0])
        ax1.ticklabel_format(style='sci', axis='y', scilimits=(-5,-5))
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.legend()

        fig.tight_layout()

        if save==None:
            save = self.save
        if folder==None:
            folder = self.folder
        if save==True:
            checkFolder(folder)
            plt.savefig(os.path.join(os.getcwd(), folder, '119.pdf'), dpi=300)
            plt.savefig(os.path.join(os.getcwd(), folder, '119.png'), dpi=300)

        plt.show()

    def stateVectorsAll(self, save=None, folder=None):
        """
        Plot the spacecraft's state vectors ver time
        Velocity, position and acceleration in cylindrical and cartesian coordinates
        """

        print('Plot position and velocity (cylindrical and cartesian)')

        fig = plt.figure(figsize=(12, 15))

        # Cartesian velocities
        nPlots = 6
        plt.subplot(nPlots, 2, 1)
        plot1 = plt.plot(self.tSample, self.traVelCart['vx'], color='C0')
        plt.grid()
        plt.xlabel('time [days]')
        plt.ylabel('$V_x$ [m/s]')
        plt.title('Cartesian Velocities')

        plt.subplot(nPlots, 2, 3)
        plot1 = plt.plot(self.tSample, self.traVelCart['vy'], color='C0')
        plt.grid()
        plt.xlabel('time [days]')
        plt.ylabel('$V_y$ [m/s]')

        plt.subplot(nPlots, 2, 5)
        plot1 = plt.plot(self.tSample, self.traVelCart['vz'], color='C0')
        plt.grid()
        plt.xlabel('time [days]')
        plt.ylabel('$V_z$ [m/s]')

        # Cylindrical velocities
        plt.subplot(nPlots, 2, 2)
        plot1 = plt.plot(self.tSample, self.traVelCyl['vr'], color='C1')
        plt.grid()
        plt.xlabel('time [days]')
        plt.ylabel(r'$V_r$ [m/s]')
        plt.title('Cylindrical Velocities')

        plt.subplot(nPlots, 2, 4)
        plot1 = plt.plot(self.tSample, self.traVelCyl['vt'], color='C1')
        plt.grid()
        plt.xlabel('time [days]')
        plt.ylabel(r'$V_\theta$ [m/s]')

        plt.subplot(nPlots, 2, 6)
        plot1 = plt.plot(self.tSample, self.traVelCyl['vz'], color='C1')
        plt.grid()
        plt.xlabel('time [days]')
        plt.ylabel(r'$V_z$ [m/s]')

        # Cartesian positions
        plt.subplot(nPlots, 2, 7)
        plot1 = plt.plot(self.tSample, self.traPosCart['x']/pk.AU, color='C2')
        plt.grid()
        plt.xlabel('time [days]')
        plt.ylabel('$x$ [AU]')
        plt.title('Cartesian Positions')

        plt.subplot(nPlots, 2, 9)
        plot1 = plt.plot(self.tSample, self.traPosCart['y']/pk.AU, color='C2')
        plt.grid()
        plt.xlabel('time [days]')
        plt.ylabel('$y$ [AU]')

        plt.subplot(nPlots, 2, 11)
        plot1 = plt.plot(self.tSample, self.traPosCart['z']/pk.AU, color='C2')
        plt.grid()
        plt.xlabel('time [days]')
        plt.ylabel('$z$ [AU]')

        # Cylindrical positions
        plt.subplot(nPlots, 2, 8)
        plot1 = plt.plot(self.tSample, self.traPosCyl['r']/pk.AU, color='C3')
        plt.grid()
        plt.xlabel('time [days]')
        plt.ylabel(r'$r$ [AU]')
        plt.title('Cylindrical Positions')

        plt.subplot(nPlots, 2, 10)
        plot1 = plt.plot(self.tSample, self.traPosCyl['t']*180/np.pi, color='C3')
        plt.grid()
        plt.xlabel('time [days]')
        plt.ylabel(r'$\theta$ [deg]')

        plt.subplot(nPlots, 2, 12)
        plot1 = plt.plot(self.tSample, self.traPosCyl['z']/pk.AU, color='C3')
        plt.grid()
        plt.xlabel('time [days]')
        plt.ylabel(r'$z$ [AU]')

        plt.tight_layout()

        if save==None:
            save = self.save
        if folder==None:
            folder = self.folder
        if save==True:
            checkFolder(folder)
            plt.savefig(os.path.join(os.getcwd(), folder, 'state.pdf'), dpi=300)
            plt.savefig(os.path.join(os.getcwd(), folder, 'state.png'), dpi=300)

        plt.show()

    def stateVectorsCylindrical(self, save=None, folder=None):
        """
        Plot the spacecraft's state vectors ver time
        Velocity, position and acceleration in cylindrical and cartesian coordinates
        """

        print('Plot cylindrical state vectors')

        fig = plt.figure(figsize=(12, 12))

        nPlots = 3

        # Cylindrical positions
        plt.subplot(nPlots, 3, 1)
        plot1 = plt.plot(self.tSample, self.traPosCyl['r']/pk.AU, color='C3')
        plt.grid()
        plt.xlabel('time [days]')
        plt.ylabel(r'$r$ [AU]')
        plt.title('Cylindrical Positions')

        plt.subplot(nPlots, 3, 4)
        plot1 = plt.plot(self.tSample, self.traPosCyl['t']*180/np.pi, color='C3')
        plt.grid()
        plt.xlabel('time [days]')
        plt.ylabel(r'$\theta$ [deg]')

        plt.subplot(nPlots, 3, 7)
        plot1 = plt.plot(self.tSample, self.traPosCyl['z']/pk.AU, color='C3')
        plt.grid()
        plt.xlabel('time [days]')
        plt.ylabel(r'$z$ [AU]')

        # Cylindrical velocities
        plt.subplot(nPlots, 3, 2)
        plot1 = plt.plot(self.tSample, self.traVelCyl['vr'], color='C1')
        plt.grid()
        plt.xlabel('time [days]')
        plt.ylabel(r'$V_r$ [m/s]')
        plt.title('Cylindrical Velocities')

        plt.subplot(nPlots, 3, 5)
        plot1 = plt.plot(self.tSample, self.traVelCyl['vt'], color='C1')
        plt.grid()
        plt.xlabel('time [days]')
        plt.ylabel(r'$V_\theta$ [m/s]')

        plt.subplot(nPlots, 3, 8)
        plot1 = plt.plot(self.tSample, self.traVelCyl['vz'], color='C1')
        plt.grid()
        plt.xlabel('time [days]')
        plt.ylabel(r'$V_z$ [m/s]')

    def stateVectorsCylindricalInclPlanets(self, save=None, folder=None):
        """
        Plot the spacecraft's and planets' state vectors over time
        Velocity, position and acceleration in cylindrical and cartesian coordinates
        """

        print('Plot cylindrical state vectors')

        fig = plt.figure(figsize=(12, 12))

        nPlots = 3

        # Cylindrical positions
        plt.subplot(nPlots, 3, 1)
        plot1 = plt.plot(self.tSample, self.traPosCyl['r']/pk.AU, label='Trajectory', c='C1')
        plot1 = plt.plot(self.tSample, self.plPosCyl['rDep']/pk.AU, label='Departure planet', c='C0')
        plot1 = plt.plot(self.tSample, self.plPosCyl['rArr']/pk.AU, label='Arrival planet', c='C3')
        plt.grid()
        plt.legend()
        plt.xlabel('time [days]')
        plt.ylabel(r'$r$ [AU]')
        plt.title('Cylindrical Positions')

        plt.subplot(nPlots, 3, 4)
        tsaw = self.traPosCyl['t']*180/np.pi
        for i in range(0, 6):
            tsaw[tsaw > 180] = tsaw[tsaw > 180] - 360       # make saw pattern
        plot1 = plt.plot(self.tSample, tsaw, label='Trajectory', c='C1')
        plot1 = plt.plot(self.tSample, self.plPosCyl['tDep']*180/np.pi, label='Departure planet', c='C0')
        plot1 = plt.plot(self.tSample, self.plPosCyl['tArr']*180/np.pi, label='Arrival planet', c='C3')
        plt.grid()
        plt.legend()
        plt.xlabel('time [days]')
        plt.ylabel(r'$\theta$ [deg]')

        plt.subplot(nPlots, 3, 7)
        plot1 = plt.plot(self.tSample, self.traPosCyl['z']/pk.AU, label='Trajectory', c='C1')
        plot1 = plt.plot(self.tSample, self.plPosCyl['zDep']/pk.AU, label='Departure planet', c='C0')
        plot1 = plt.plot(self.tSample, self.plPosCyl['zArr']/pk.AU, label='Arrival planet', c='C3')
        plt.grid()
        plt.legend()
        plt.xlabel('time [days]')
        plt.ylabel(r'$z$ [AU]')

        # Cylindrical velocities
        plt.subplot(nPlots, 3, 2)
        plot1 = plt.plot(self.tSample, self.traVelCyl['vr'], label='Trajectory', c='C1')
        plot1 = plt.plot(self.tSample, self.plVelCyl['VrDep'], label='Departure planet', c='C0')
        plot1 = plt.plot(self.tSample, self.plVelCyl['VrArr'], label='Arrival planet', c='C3')
        plt.grid()
        plt.legend()
        plt.xlabel('time [days]')
        plt.ylabel(r'$V_r$ [m/s]')
        plt.title('Cylindrical Velocities')

        plt.subplot(nPlots, 3, 5)
        plot1 = plt.plot(self.tSample, self.traVelCyl['vt'], label='Trajectory', c='C1')
        plot1 = plt.plot(self.tSample, self.plVelCyl['VtDep'], label='Departure planet', c='C0')
        plot1 = plt.plot(self.tSample, self.plVelCyl['VtArr'], label='Arrival planet', c='C3')
        plt.grid()
        plt.legend()
        plt.xlabel('time [days]')
        plt.ylabel(r'$V_\theta$ [m/s]')

        plt.subplot(nPlots, 3, 8)
        plot1 = plt.plot(self.tSample, self.traVelCyl['vz'], label='Trajectory', c='C1')
        plot1 = plt.plot(self.tSample, self.plVelCyl['VzDep'], label='Departure planet', c='C0')
        plot1 = plt.plot(self.tSample, self.plVelCyl['VzArr'], label='Arrival planet', c='C3')
        plt.grid()
        plt.legend()
        plt.xlabel('time [days]')
        plt.ylabel(r'$V_z$ [m/s]')

        # Cylindrical accelerations
        plt.subplot(nPlots, 3, 3)
        plot1 = plt.plot(self.tSample, self.traAccCyl['ar'], color='C1')
        plt.grid()
        plt.xlabel('time [days]')
        plt.ylabel(r'$a_r$ [m/s^2]')
        plt.title('Cylindrical Accelerations')

        plt.subplot(nPlots, 3, 6)
        plot1 = plt.plot(self.tSample, self.traAccCyl['at'], color='C1')
        plt.grid()
        plt.xlabel('time [days]')
        plt.ylabel(r'$a_\theta$ [m/s^2]')

        plt.subplot(nPlots, 3, 9)
        plot1 = plt.plot(self.tSample, self.traAccCyl['az'], color='C1')
        plt.grid()
        plt.xlabel('time [days]')
        plt.ylabel(r'$a_z$ [m/s^2]')

        plt.tight_layout()

        if save==None:
            save = self.save
        if folder==None:
            folder = self.folder
        if save==True:
            checkFolder(folder)
            plt.savefig(os.path.join(os.getcwd(), folder, 'stateCylindricalInclPlanets.pdf'), dpi=300)
            plt.savefig(os.path.join(os.getcwd(), folder, 'stateCylindricalInclPlanets.png'), dpi=300)

        plt.show()

    def stateVectorsCartesian(self, save=None, folder=None):
        """
        Plot the spacecraft's state vectors ver time
        Velocity, position and acceleration in cylindrical and cartesian coordinates
        """

        print('Plot cartesian state vectors')

        fig = plt.figure(figsize=(12, 12))

        nPlots = 3

        # Cartesian positions
        plt.subplot(nPlots, 2, 1)
        plot1 = plt.plot(self.tSample, self.traPosCart['x']/pk.AU, color='C3')
        plt.grid()
        plt.xlabel('time [days]')
        plt.ylabel(r'$x$ [AU]')
        plt.title('Cartesian Positions')

        plt.subplot(nPlots, 2, 3)
        plot1 = plt.plot(self.tSample, self.traPosCart['y']/pk.AU, color='C3')
        plt.grid()
        plt.xlabel('time [days]')
        plt.ylabel(r'$y$ [deg]')

        plt.subplot(nPlots, 2, 5)
        plot1 = plt.plot(self.tSample, self.traPosCart['z']/pk.AU, color='C3')
        plt.grid()
        plt.xlabel('time [days]')
        plt.ylabel(r'$z$ [AU]')

        # Cartesian velocities
        plt.subplot(nPlots, 2, 2)
        plot1 = plt.plot(self.tSample, self.traVelCart['vx'], color='C1')
        plt.grid()
        plt.xlabel('time [days]')
        plt.ylabel(r'$V_z$ [m/s]')
        plt.title('Cartesian Velocities')

        plt.subplot(nPlots, 2, 4)
        plot1 = plt.plot(self.tSample, self.traVelCart['vy'], color='C1')
        plt.grid()
        plt.xlabel('time [days]')
        plt.ylabel(r'$V_y$ [m/s]')

        plt.subplot(nPlots, 2, 6)
        plot1 = plt.plot(self.tSample, self.traVelCart['vz'], color='C1')
        plt.grid()
        plt.xlabel('time [days]')
        plt.ylabel(r'$V_z$ [m/s]')

        plt.tight_layout()

        if save==None:
            save = self.save
        if folder==None:
            folder = self.folder
        if save==True:
            checkFolder(folder)
            plt.savefig(os.path.join(os.getcwd(), folder, 'stateCartesian.pdf'), dpi=300)
            plt.savefig(os.path.join(os.getcwd(), folder, 'stateCartesian.png'), dpi=300)

        plt.show()


    def samplePlanets(self, trajectory, samples=100):
        """
        Return a dictionary with sampled position vectors of the departure and
        arrival planets of the given trajectory
        """

        # define planets
        if self.ephemSource == 'jpl':
            planetDep = pk.planet.jpl_lp(trajectory.departureBody)
            planetArr = pk.planet.jpl_lp(trajectory.arrivalBody)
        elif self.ephemSource == 'spice':
            planetDep = pk.planet.spice(trajectory.departureBody, 'sun', 'eclipj2000')
            planetArr = pk.planet.spice(trajectory.arrivalBody, 'sun', 'eclipj2000')
        else:
            print('ERROR: This is not a valid source of ephemerides.')

        # time variable [days]
        self.tSample = np.linspace(self.trajectory.jdDep, self.trajectory.jdArr, samples)
        tSample = self.tSample

        # init planet velocity vectors
        tof = self.trajectory.tof
        VrDep = np.linspace(0, tof, samples)
        VtDep = np.linspace(0, tof, samples)
        VzDep = np.linspace(0, tof, samples)
        VrArr = np.linspace(0, tof, samples)
        VtArr = np.linspace(0, tof, samples)
        VzArr = np.linspace(0, tof, samples)

        VxDep = np.linspace(0, tof, samples)
        VyDep = np.linspace(0, tof, samples)
        VzDep = np.linspace(0, tof, samples)
        VxArr = np.linspace(0, tof, samples)
        VyArr = np.linspace(0, tof, samples)
        VzArr = np.linspace(0, tof, samples)

        # init position vectors
        xDep = np.linspace(0, tof, samples)
        yDep = np.linspace(0, tof, samples)
        zDep = np.linspace(0, tof, samples)
        xArr = np.linspace(0, tof, samples)
        yArr = np.linspace(0, tof, samples)
        zArr = np.linspace(0, tof, samples)

        rDep = np.linspace(0, tof, samples)
        tDep = np.linspace(0, tof, samples)
        zDep = np.linspace(0, tof, samples)
        rArr = np.linspace(0, tof, samples)
        tArr = np.linspace(0, tof, samples)
        zArr = np.linspace(0, tof, samples)

        # retrieve and convert planet state vectors
        for i in range(0, len(tSample)):
            epochSample = pk.epoch(tSample[i], 'mjd2000')

            # Departure planet
            rCart, vCart = planetDep.eph(epochSample)
            vCyl = Vcart2cyl(vCart, rCart)
            rCyl = Pcart2cyl(rCart)
            xDep[i] = rCart[0]
            yDep[i] = rCart[1]
            zDep[i] = rCart[2]
            rDep[i] = rCyl[0]
            tDep[i] = rCyl[1]
            zDep[i] = rCyl[2]
            VrDep[i] = vCyl[0]
            VtDep[i] = vCyl[1]
            VxDep[i] = vCart[0]
            VyDep[i] = vCart[1]
            VzDep[i] = vCart[2]

            # Arrival planet
            rCart, vCart = planetArr.eph(epochSample)
            vCyl = Vcart2cyl(vCart, rCart)
            rCyl = Pcart2cyl(rCart)
            xArr[i] = rCart[0]
            yArr[i] = rCart[1]
            zArr[i] = rCart[2]
            rArr[i] = rCyl[0]
            tArr[i] = rCyl[1]
            zArr[i] = rCyl[2]
            VrArr[i] = vCyl[0]
            VtArr[i] = vCyl[1]
            VxArr[i] = vCart[0]
            VyArr[i] = vCart[1]
            VzArr[i] = vCart[2]

        # dictionary with cartesian positions
        planetCartesianPositions = {'xDep' : xDep,
                           'yDep' : yDep,
                           'zDep' : zDep,
                           'xArr' : xArr,
                           'yArr' : yArr,
                           'zArr' : zArr}
        planetCylindricalPositions = {'rDep' : rDep,
                           'tDep' : tDep,
                           'zDep' : zDep,
                           'rArr' : rArr,
                           'tArr' : tArr,
                           'zArr' : zArr}
        planetCartesianVelocities = {'VxDep' : VxDep,
                           'VyDep' : VyDep,
                           'VzDep' : VzDep,
                           'VxArr' : VxArr,
                           'VyArr' : VyArr,
                           'VzArr' : VzArr}
        planetCylindricalVelocity = {'VrDep' : VrDep,
                           'VtDep' : VtDep,
                           'VzDep' : VzDep,
                           'VrArr' : VrArr,
                           'VtArr' : VtArr,
                           'VzArr' : VzArr}
        print('Done sampling planets.')

        return planetCartesianPositions, planetCylindricalPositions, planetCartesianVelocities, planetCylindricalVelocity

    def sampleTrajectoryPosition(self, trajectory, samples=100):
        """
        Returns Cartesian position vectors of the full trajectory
        I.e. from t=0 to t=tof
        """

        # time vector
        self.tSampleSec = np.linspace(0, self.trajectory.tofSec, samples)
        tSampleSec = self.tSampleSec

        # sample and compute position vectors
        xTra = np.linspace(0, self.trajectory.tofSec, samples)
        yTra = np.linspace(0, self.trajectory.tofSec, samples)
        zTra = np.linspace(0, self.trajectory.tofSec, samples)
        tTra = np.linspace(0, self.trajectory.tofSec, samples)
        rTra = np.linspace(0, self.trajectory.tofSec, samples)
        zTra = np.linspace(0, self.trajectory.tofSec, samples)
        for i in range(0, len(tSampleSec)):
            ti = tSampleSec[i]
            rTra[i], tTra[i], zTra[i] = [self.trajectory.r(ti), self.trajectory.t(ti), self.trajectory.z(ti)]
            xTra[i], yTra[i], zTra[i] = Pcyl2cart([rTra[i], tTra[i], zTra[i]])

        # dictionary with cartesian positions
        trajectoryCartPositions = {'x' : xTra,
                               'y' : yTra,
                               'z' : zTra}
        trajectoryCylPositions = {'r' : rTra,
                               't' : tTra,
                               'z' : zTra}
        print('Done sampling trajectory position.')

        return trajectoryCartPositions, trajectoryCylPositions

    def sampleTrajectoryVelocity(self, trajectory, samples=100):
        """
        Returns Cartesian velocity vectors of the full trajectory
        I.e. from t=0 to t=tof
        """

        # time vector
        tSampleSec = self.tSampleSec

        # cartesian velocities
        xTraVel = np.linspace(0, self.trajectory.tofSec, samples)
        yTraVel = np.linspace(0, self.trajectory.tofSec, samples)
        zTraVel = np.linspace(0, self.trajectory.tofSec, samples)
        rTraVel = np.linspace(0, self.trajectory.tofSec, samples)
        tTraVel = np.linspace(0, self.trajectory.tofSec, samples)
        zTraVel = np.linspace(0, self.trajectory.tofSec, samples)
        for i in range(0, len(tSampleSec)):
            vCyl = [self.trajectory.rDot(tSampleSec[i]), self.trajectory.tDot(tSampleSec[i]), self.trajectory.zDot(tSampleSec[i])]
            rCyl = [self.trajectory.r(tSampleSec[i]), self.trajectory.t(tSampleSec[i]), self.trajectory.z(tSampleSec[i])]
            vCart = Vcyl2cart(vCyl, rCyl)
            xTraVel[i] = vCart[0]
            yTraVel[i] = vCart[1]
            zTraVel[i] = vCart[2]
            rTraVel[i] = vCyl[0]
            tTraVel[i] = vCyl[1]
            zTraVel[i] = vCyl[2]

        # dictionaries
        trajectoryVelocitiesCart = {'vx' : xTraVel,
                                    'vy' : yTraVel,
                                    'vz' : zTraVel}
        trajectoryVelocitiesCyl = {'vr' : rTraVel,
                                   'vt' : tTraVel,
                                   'vz' : zTraVel}

        print('Done sampling trajectory velocity.')

        return trajectoryVelocitiesCart, trajectoryVelocitiesCyl

    def sampleTrajectoryAcceleration(self, trajectory, samples=100):
        """
        Returns cylindrical acceleration vectors of the full trajectory
        """

        # initialize vectors
        rTraAcc = np.linspace(0, 1, samples)
        tTraAcc = np.linspace(0, 1, samples)
        zTraAcc = np.linspace(0, 1, samples)
        xTraAcc = np.linspace(0, 1, samples)
        yTraAcc = np.linspace(0, 1, samples)
        totalTraAcc = np.linspace(0, 1, samples)

        x = self.traPosCart['x']
        y = self.traPosCart['y']
        z = self.traPosCart['z']

        # sample acceleration vectors
        for i in range(0, len(self.tSampleSec)):
            ti = self.tSampleSec[i]
            aCyl = [self.trajectory.rDDot(ti), self.trajectory.tDDot(ti), self.trajectory.zDDot(ti)]
            rTraAcc[i] = aCyl[0]
            tTraAcc[i] = aCyl[1]
            zTraAcc[i] = aCyl[2]
            rCart = [x[i], y[i], z[i]]
            aCart = Acyl2Acart(aCyl, rCart)
            xTraAcc[i] = aCart[0]
            yTraAcc[i] = aCart[1]
            totalTraAcc[i] = np.sqrt(xTraAcc[i]**2 + yTraAcc[i]**2 + zTraAcc[i]**2)

        # dictionaries
        trajectoryAccelerationsCyl = {'ar' : rTraAcc,
                                      'at' : tTraAcc,
                                      'az' : zTraAcc}

        trajectoryAccelerationsCart = {'ax' : xTraAcc,
                                       'ay' : yTraAcc,
                                       'az' : zTraAcc,
                                       'total' : totalTraAcc}

        print('Done sampling trajectory acceleration.')

        return trajectoryAccelerationsCyl, trajectoryAccelerationsCart
