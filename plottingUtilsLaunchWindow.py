import matplotlib as mlt
# mlt.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

from utils import *

class plottingGridSearch(object):
    """
    Plotting the results of a grid search
    Takes a numpy dictionary
    """

    def __init__(self, results, save=False, folder='graveyard',
                    externalMonitor=-1, show=True):

        # retrieve results
        self.deltaVs = results['deltaVs']
        self.depDates = results['depDates']
        self.Ns = results['Ns']
        self.tofs = results['tofs']

        # settings
        self.folder = folder
        self.save = save
        self.externalMonitorFlag = externalMonitor
        self.show = show

    def resultContours(self, save=None, folder=None, levels=50):
        '''
        Contour plots of time of flight and departure dates for each N
        '''

        print('Plotting contour')
        nPlots = len(self.Ns)

        XdepDates, Ytofs = np.meshgrid(self.depDates, self.tofs)

        plt.figure(figsize = (10, 4*nPlots))

        index = 0
        for i in self.Ns:
            # plt.subplot(nPlots, 1, i+1, sharex=True)
            plt.subplot(nPlots, 1, index+1)
            plot1 = plt.contourf(XdepDates, Ytofs,
                                self.deltaVs[:, :, index]/1e3, levels)
            cbar = plt.colorbar()
            cbar.ax.set_ylabel('DeltaV [km/s]')
            plt.title('DeltaV for N =' + str(i))
            plt.ylabel('Time of flight [days]')
            index += 1

        plt.xlabel('Departure Date [julian days]')
        plt.ylabel('Time of flight [days]')

        if save==None:
            save = self.save
        if folder==None:
            folder = self.folder
        if save==True:
            checkFolder(folder)
            plt.savefig(os.path.join(os.getcwd(), folder, 'DeltaV.pdf'), dpi=300)
            plt.savefig(os.path.join(os.getcwd(), folder, 'DeltaV.png'), dpi=300)

        plt.tight_layout()
        plt.show()

    def resultContoursImshow(self, save=None, folder=None, maxVal=None):
        '''
        Contour plots of time of flight and departure dates for each N
        Using imshow
        '''

        print('Plotting contour (imshow)')
        nPlots = len(self.Ns)

        XdepDates, Ytofs = np.meshgrid(self.depDates, self.tofs)

        plotDeltaV = self.deltaVs/1e3

        if maxVal:
            plotDeltaV[plotDeltaV > maxVal] = np.nan

        plt.figure(figsize = (10, 4*nPlots))

        index = 0
        for i in self.Ns:
            # plt.subplot(nPlots, 1, i+1, sharex=True)
            plt.subplot(nPlots, 1, index+1)
            plot1 = plt.imshow(plotDeltaV[:, :, index],
                            extent=[self.depDates[0], self.depDates[-1],
                                    self.tofs[0], self.tofs[-1]],
                            origin='lower', aspect='auto')
            cbar = plt.colorbar()
            cbar.ax.set_ylabel('DeltaV [km/s]')
            plt.title('DeltaV for N =' + str(i))
            plt.ylabel('Time of flight [days]')
            index += 1

        plt.xlabel('Departure Date [julian days]')
        plt.ylabel('Time of flight [days]')

        if save==None:
            save = self.save
        if folder==None:
            folder = self.folder
        if save==True:
            checkFolder(folder)
            plt.savefig(os.path.join(os.getcwd(), folder, 'DeltaVimshow.pdf'), dpi=300)
            plt.savefig(os.path.join(os.getcwd(), folder, 'DeltaVimshow.png'), dpi=300)

        plt.tight_layout()
        plt.show()

    def resultContourBest(self, save=None, folder=None, levels=50,
                            zRange = None, colorMap = 'viridis'):
        """
        Contour plot of the best trajectories
        """

        print('Plotting best contour (contourf)')

        XdepDates, Ytofs = np.meshgrid(self.depDates, self.tofs)

        # chose minimum DeltaV from different N
        # convert to km/s
        Vplot = np.amin(self.deltaVs, axis=2)/1e3

        # Cut off large and small values
        if zRange:
            print(zRange)
            Vplot[Vplot < zRange[0]] = zRange[0]
            Vplot[Vplot > zRange[1]] = zRange[1]

        plt.figure(figsize = (10, 5))
        plotLast = plt.contourf(XdepDates, Ytofs , Vplot, levels, cmap = colorMap)

        cbar = plt.colorbar()
        cbar.ax.set_ylabel('DeltaV [km/s]')

        # if self.zRange:
        #     plt.clim(self.zRange)

        plt.xlabel('Departure Date [julian days]')
        plt.ylabel('Time of flight [days]')
        plt.title('Minimum DeltaV')

        if save==None:
            save = self.save
        if folder==None:
            folder = self.folder
        if save==True:
            checkFolder(folder)
            plt.savefig(os.path.join(os.getcwd(), folder, 'bestDeltaV.pdf'), dpi=300)
            plt.savefig(os.path.join(os.getcwd(), folder, 'bestDeltaV.png'), dpi=300)

        plt.show()

    def resultContourBestImshow(self, save=None, folder=None, levels=50,
                                colorMap = 'viridis'):
        """
        Contour plot of the best trajectories
        """

        print('Plotting best contour (imshow)')

        XdepDates, Ytofs = np.meshgrid(self.depDates, self.tofs)

        # chose minimum DeltaV from different N
        # convert to km/s
        Vplot = np.amin(self.deltaVs, axis=2)/1e3

        plt.figure(figsize = (10, 5))

        plot1 = plt.imshow(Vplot,
                        extent=[self.depDates[0], self.depDates[-1],
                                self.tofs[0], self.tofs[-1]],
                        origin='lower', aspect='auto', cmap=colorMap)
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('DeltaV [km/s]')

        plt.xlabel('Departure Date [julian days]')
        plt.ylabel('Time of flight [days]')
        plt.title('Minimum DeltaV')

        if save==None:
            save = self.save
        if folder==None:
            folder = self.folder
        if save==True:
            checkFolder(folder)
            plt.savefig(os.path.join(os.getcwd(), folder,
                        'bestDeltaVimshow.pdf'), dpi=300)
            plt.savefig(os.path.join(os.getcwd(), folder,
                        'bestDeltaVimshow.png'), dpi=300)

        plt.show()

    def resultBestN(self, save=None, folder=None, levels=50,
                                colorMap = 'viridis'):
        """
        Plot the best N for each point in the grid search
        """

        print('Plotting best Ns')
        fig = newFigure(height=3.5, half=False, target='paper')

        XdepDates, Ytofs = np.meshgrid(self.depDates, self.tofs)

        # chose minimum DeltaV from different N
        # convert to km/s
        Nplot = np.argmin(self.deltaVs, axis=2)

        plot1 = plt.imshow(Nplot,
                        extent=[self.depDates[0], self.depDates[-1],
                                self.tofs[0], self.tofs[-1]],
                        origin='lower', aspect='auto', cmap=colorMap)
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('N')

        plt.xlabel('Departure Date [julian days]')
        plt.ylabel('Time of flight [days]')
        plt.title('Number of revolutions for minimum DeltaV')

        if save==None:
            save = self.save
        if folder==None:
            folder = self.folder
        if save==True:
            checkFolder(folder)
            plt.savefig(os.path.join(os.getcwd(), folder,
                        'bestNimshow.pdf'), dpi=300)
            plt.savefig(os.path.join(os.getcwd(), folder,
                        'bestNimshow.png'), dpi=300)

        plt.show()

    def resultContourBest114(self, save=None, folder=None, colorMap='viridis'):
        """
        Contour plot of the best trajectories
        Recreates Figure 11.4 in [Gondelach, 2012]
        """

        print('Plotting best contour, recreating Figure 11.4')

        XdepDates, Ytofs = np.meshgrid(self.depDates, self.tofs)

        # chose minimum DeltaV from different N
        # convert to km/s
        Vplot = np.amin(self.deltaVs, axis=2)/1e3

        plt.figure(figsize = (10, 4))

        cmap = plt.get_cmap(colorMap)
        cmap = mlt.colors.ListedColormap(['mediumblue', 'royalblue',
                        'dodgerblue', 'deepskyblue', 'turquoise', 'chartreuse'])
        cmap.set_over(color='firebrick')
        cmap.set_under(color='r')

        bounds = [6, 7, 8, 10, 15, 20, 40]
        norm = mlt.colors.BoundaryNorm(bounds, cmap.N)
        ax = plt.gca()
        plotBest = plt.contourf(XdepDates, Ytofs , Vplot,
                    levels=bounds, cmap=cmap, norm=norm, extend='max')


        cmap3 = mlt.colors.ListedColormap(['r', 'g', 'b', 'c'])
        cmap3.set_over('0.35')
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('DeltaV [km/s]')

        plt.xlabel('Departure Date [MJD2000]')
        plt.ylabel('Time of flight [days]')
        plt.title('Minimum DeltaV')

        if save==None:
            save = self.save
        if folder==None:
            folder = self.folder
        if save==True:
            checkFolder(folder)
            plt.savefig(os.path.join(os.getcwd(), folder, '114.pdf'), dpi=300)
            plt.savefig(os.path.join(os.getcwd(), folder, '114.png'), dpi=300)

        plt.show()

    def resultContourBest118(self, save=None, folder=None, colorMap='viridis'):
        """
        Contour plot of the best trajectories
        Recreates Figure 11.8 in [Gondelach, 2012]
        """

        print('Plotting best contour, recreating Figure 11.4')

        XdepDates, Ytofs = np.meshgrid(self.depDates, self.tofs)

        # chose minimum DeltaV from different N
        # convert to km/s
        Vplot = np.amin(self.deltaVs, axis=2)/1e3

        plt.figure(figsize = (10, 4))

        cmap = plt.get_cmap(colorMap)
        cmap = mlt.colors.ListedColormap(['midnightblue', 'mediumblue',
                                'royalblue', 'dodgerblue', 'deepskyblue',
                                'turquoise', 'chartreuse'])

        bounds = [5, 6, 7, 8, 10, 15, 20, 40]
        norm = mlt.colors.BoundaryNorm(bounds, cmap.N)
        ax = plt.gca()
        plotBest = plt.contourf(XdepDates, Ytofs , Vplot, levels=bounds,
                                cmap=cmap, norm=norm)

        cbar = plt.colorbar()
        cbar.ax.set_ylabel('DeltaV [km/s]')

        plt.xlabel('Departure Date [MJD2000]')
        plt.ylabel('Time of flight [days]')
        plt.title('Minimum DeltaV')

        if save==None:
            save = self.save
        if folder==None:
            folder = self.folder
        if save==True:
            checkFolder(folder)
            plt.savefig(os.path.join(os.getcwd(), folder, '118.pdf'), dpi=300)
            plt.savefig(os.path.join(os.getcwd(), folder, '118.png'), dpi=300)

        plt.show()

    def resultContourBestPartial(self, save=None, folder=None,
                limits = [7000, 11000, 500, 2000], zRange = None,
                colorMap = 'viridis', levels = 50):
        """
        Contour plot of the best trajectories with the option to zoom
        Pay attention that the limits exist in the axis vectors
        """

        print('Plotting best contour')

        # crop to the desired part of the data
        xMin = limits[0]
        xMax = limits[1]
        yMin = limits[2]
        yMax = limits[3]
        xMinIndex = np.where(self.depDates == xMin)
        xMaxIndex = np.where(self.depDates == xMax)
        yMinIndex = np.where(self.tofs == yMin)
        yMaxIndex = np.where(self.tofs == yMax)
        plotDeltaVs = self.deltaVs[int(yMinIndex[0]):int(yMaxIndex[0]),
                                    int(xMinIndex[0]):int(xMaxIndex[0])]
        XdepDates, Ytofs = np.meshgrid(
                    self.depDates[int(xMinIndex[0]):int(xMaxIndex[0])],
                    self.tofs[int(yMinIndex[0]):int(yMaxIndex[0])])

        # chose minimum DeltaV from different N
        # convert to km/s
        Vplot = np.amin(plotDeltaVs, axis=2)/1e3

        # Cut off large and small values
        if zRange:
            print(zRange)
            Vplot[Vplot < zRange[0]] = zRange[0]
            Vplot[Vplot > zRange[1]] = zRange[1]

        plt.figure(figsize = (10, 5))
        plotLast = plt.contourf(XdepDates, Ytofs , Vplot, levels, cmap=colorMap)

        cbar = plt.colorbar(extend='max')
        # format with arrow
        # cmap = mlt.colors.ListedColormap(['r', 'g', 'b', 'c'])
        # cmap.set_over('r')
        cbar.ax.set_ylabel('DeltaV [km/s]')

        plt.xlabel('Departure Date [julian days]')
        plt.ylabel('Time of flight [days]')
        plt.title('Minimum DeltaV')

        if save==None:
            save = self.save
        if folder==None:
            folder = self.folder
        if save==True:
            checkFolder(folder)
            plt.savefig(os.path.join(os.getcwd(), folder,
                            'bestDeltaVpartialContour.pdf'), dpi=300)
            plt.savefig(os.path.join(os.getcwd(), folder,
                            'bestDeltaVpartialContour.png'), dpi=300)

        # plt.tight_layout()
        plt.show()

    def resultSurfaces(self, save=None, folder=None):
        '''
        Contour plots ofe time of flight and departure dates for each N
        '''

        print('Plotting contour')
        yPlots = np.ceil(len(self.Ns)/2)
        xPlots = 2

        XdepDates, Ytofs = np.meshgrid(self.depDates, self.tofs)

        fig = plt.figure(figsize = (20, 4*yPlots))

        index = 0
        for i in self.Ns:
            ax = plt.subplot(yPlots, xPlots, index+1, projection='3d')
            plot1 = ax.plot_surface(XdepDates, Ytofs,
                                    self.deltaVs[:, :, index]/1e3,
                                    cmap='viridis')
            plt.title('DeltaV for N =' + str(i))
            ax.set_xlabel('Departure Date [julian days]')
            ax.set_ylabel('Time of flight [days]')
            ax.set_zlabel('DeltaV [km/s]');
            index += 1

        if save==None:
            save = self.save
        if folder==None:
            folder = self.folder
        if save==True:
            checkFolder(folder)
            plt.savefig(os.path.join(os.getcwd(), folder,
                        'DeltaVsurfaces.pdf'), dpi=300)
            plt.savefig(os.path.join(os.getcwd(), folder,
                        'DeltaVsurfaces.png'), dpi=300)

        plt.tight_layout()
        plt.show()

    def resultSurfaceBest(self, save=None, folder=None):
        """
        Contour plot of the best trajectories
        """

        print('Plotting best contour')
        levels = 50

        XdepDates, Ytofs = np.meshgrid(self.depDates, self.tofs)

        fig = plt.figure(figsize = (10, 5))
        ax = fig.gca(projection='3d')
        plotLast = ax.plot_surface(XdepDates, Ytofs ,
                        np.amin(self.deltaVs, axis=2)/1e3, cmap='viridis')
        m = mlt.cm.ScalarMappable(cmap=plotLast.cmap, norm=plotLast.norm)
        m.set_array(np.amin(self.deltaVs, axis=2)/1e3)
        plt.colorbar(m)

        # ax.set_zlim(0, 200)

        ax.set_xlabel('Departure Date [julian days]')
        ax.set_ylabel('Time of flight [days]')
        ax.set_zlabel('DeltaV [km/s]');
        plt.title('Minimum DeltaV')

        if save==None:
            save = self.save
        if folder==None:
            folder = self.folder
        if save==True:
            checkFolder(folder)
            plt.savefig(os.path.join(os.getcwd(), folder,
                            'bestDeltaVsurface.pdf'), dpi=300)
            plt.savefig(os.path.join(os.getcwd(), folder,
                            'bestDeltaVsurface.png'), dpi=300)

        plt.show()

    def resultImshowBest(self, save=None, folder=None, maxVal=None):
        """
        Plot the best trajectories as an image
        """

        print('Plotting best imshow')

        plotDeltaV = self.deltaVs/1e3

        if maxVal:
            plotDeltaV[plotDeltaV > maxVal] = np.nan

        # plt.figure(figsize = (10, 5))
        newFigure(height=3.5, half=False, target='paper')
        plotLast = plt.imshow(np.nanmin(plotDeltaV, axis=2),
                    origin='lower', extent=[self.depDates[0],
                    self.depDates[-1], self.tofs[0], self.tofs[-1]],
                    aspect='auto')

        cbar = plt.colorbar()
        cbar.ax.set_ylabel('DeltaV [km/s]')

        plt.xlabel('Departure Date [mjd2000]')
        plt.ylabel('Time of flight [days]')
        # plt.title('Minimum DeltaV')
        plt.tight_layout()

        if save==None:
            save = self.save
        if folder==None:
            folder = self.folder
        if save==True:
            checkFolder(folder)
            plt.savefig(os.path.join(os.getcwd(), folder,
                                        'bestDeltaVimshow.pdf'), dpi=300)
            plt.savefig(os.path.join(os.getcwd(), folder,
                                        'bestDeltaVimshow.png'), dpi=300)

        # plt.tight_layout()
        if self.show == True:
            plt.show()

    def resultImshowBestPartial(self, save=None, folder=None,
                                    limits = [7000, 11000, 500, 2000]):
        """
        Plot the best trajectories as an image with the option to zoom
        """

        print('Plotting best imshow')

        # crop to the desired part of the data
        xMin = limits[0]
        xMax = limits[1]
        yMin = limits[2]
        yMax = limits[3]
        xMinIndex = np.where(self.depDates == xMin)
        xMaxIndex = np.where(self.depDates == xMax)
        yMinIndex = np.where(self.tofs == yMin)
        yMaxIndex = np.where(self.tofs == yMax)
        plotDeltaVs = self.deltaVs[int(yMinIndex[0]):int(yMaxIndex[0]),
                            int(xMinIndex[0]):int(xMaxIndex[0])]

        # create figure
        plt.figure(figsize = (10, 5))
        plotLast = plt.imshow(np.amin(plotDeltaVs, axis=2)/1e3,
                    origin='lower', extent=[xMin, xMax, yMin, yMax],
                    aspect = 'auto')

        cbar = plt.colorbar()
        cbar.ax.set_ylabel('DeltaV [km/s]')

        plt.xlabel('Departure Date [julian days]')
        plt.ylabel('Time of flight [days]')
        plt.title('Minimum DeltaV')

        if save==None:
            save = self.save
        if folder==None:
            folder = self.folder
        if save==True:
            checkFolder(folder)
            plt.savefig(os.path.join(os.getcwd(), folder,
                            'bestDeltaVimshow.pdf'), dpi=300)
            plt.savefig(os.path.join(os.getcwd(), folder,
                            'bestDeltaVimshow.png'), dpi=300)

        # plt.tight_layout()
        plt.show()
