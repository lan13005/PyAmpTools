import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import mplhep as hep
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import gaussian_kde
import time
import matplotlib.colors as colors
from pyamptools.utility.general import prettyLabels
from pyamptools.utility.ift.io import load_sample_pkl


def plot_amplitude(ati_samples, wave, nifty_pkl_files, half_bin_width=None, overlay_saturated_resonances=None, masses=None, draw_bands_else_samples=False, kde_kwargs={}, mle_kwargs={}, figsize=(18, 15)):
    """
    Plot the amplitude samples and the AmpTools results for a given wave

    Args:
        ati_samples (pd.DataFrame): dataframe with the amptools results
        wave (str): wave to plot or pair of waves to plot phases for (e.g. 'S0' or 'S0 P0')
        nifty_pkl_files (list): list of paths to pickeled files of Dict {'mass': [masses], 'amplitude1': [[amplitudes]], ...}
        half_bin_width (float): half of the bin width
        overlay_saturated_resonances (dictionary): dictionary of list of lists (mass, intensities, phases) to overlay saturated resonances
        masses (np.array): array of masses (bin centers) used to draw resonance curves
        figsize (tuple): size of the plotted figure

    Returns:
        fig (matplotlib.figure.Figure): figure object
        axes (list): list of axes objects
    """

    ###############################################################################################
    small_size = 15
    big_size = 20
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.style.use([hep.styles.ATLAS])
    plt.rc("font", size=small_size)  # controls default text sizes
    plt.rc("axes", titlesize=small_size)  # fontsize of the axes title
    plt.rc("axes", labelsize=big_size)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=small_size)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=small_size)  # fontsize of the tick labels
    plt.rc("legend", fontsize=small_size)  # legend fontsize
    plt.rc("figure", titlesize=big_size)  # fontsize of the figure title
    ###############################################################################################

    nrows = int(np.ceil(np.sqrt(len(nifty_pkl_files))))
    ncols = int(np.ceil(len(nifty_pkl_files) / nrows))
    print(f"Creating {nrows}x{ncols} grid of plots for {len(nifty_pkl_files)} NIFTy samples")
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True)
    axes = axes.flatten() if len(nifty_pkl_files) > 1 else [axes]

    is_phase_col = " " in wave

    for pkl_file, ax in zip(nifty_pkl_files, axes):
        # Naming can be confusing. x is the masses for the fit results whereas masses
        #    is the x spacing for the resonacne curves
        nifty_samples = load_sample_pkl(pkl_file)

        #################################
        ###### PLOT NIFTY RESULTS #######
        #################################

        x = nifty_samples["mass"].unique()
        y = nifty_samples.groupby("mass")[wave]
        y_mean, y_std = y.mean(), y.std()
        y_mean = y_mean.values
        y_std = y_std.values

        # draw bands, aggregating samples
        if draw_bands_else_samples:
            ax.plot(x, y_mean, alpha=1.0, color="orange")
            ax.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.8, color="orange")

        # Draw individual samples as lines
        else:
            kde = KDE(nifty_samples[["mass", wave]].values)
            is_intensity = " " not in wave
            for i, group in nifty_samples.groupby("iteration"):
                _x = group["mass"]
                _y = group[wave]
                ax.scatter(_x, _y, color="blueviolet", s=10, zorder=3, marker="x", linewidth=1, alpha=0.8)

                kde.add_curve(kde.scaler.transform(group[["mass", wave]].values), is_intensity)
            _kde_kwargs = {"bw": 0.1, "nlevels": 20, "cmap": "viridis", "cbar": False}
            _kde_kwargs.update(kde_kwargs)
            kde.plot(ax, is_intensity=is_intensity, **_kde_kwargs)

        #################################
        ###### PLOT AMPTOOLS RESULTS ####
        #################################

        if is_phase_col:
            _y = np.rad2deg(ati_samples[wave])
            _y = np.mod(_y + 180, 360) - 180
            _yerr = np.rad2deg(ati_samples[wave + " err"])
        else:
            _y = ati_samples[wave]
            _yerr = ati_samples[wave + " err"]

        _mle_kwargs = {"markersize": 6, "fmt": "o", "label": "AmpTools", "c": "black", "zorder": 4}
        _mle_kwargs.update(mle_kwargs)
        ax.errorbar(ati_samples["mass"], _y, yerr=_yerr, **_mle_kwargs)

        if is_phase_col:
            ax.set_ylim(-180, 180)
        else:
            ax.set_ylim(0, ati_samples[wave].max() * 1.1)
        ax.set_xlim(x.min(), x.max())

        ## Overlay potential resonance curves maximally saturating the partial wave at the resonance's mass
        if overlay_saturated_resonances is None:
            continue

        if is_phase_col:
            amp1, amp2 = wave.split()
            L1, L2 = amp1[0], amp2[0]
            assert L1 != L2, "Logic failure if Ls are the same"
            for resonances, sign in zip([overlay_saturated_resonances[L1], overlay_saturated_resonances[L2]], [1, -1]):
                for resonance in resonances:  # resonances ~ List[mass, normed_peak, intensities, phases]
                    color, mass0, intensity, phase = resonance
                    _phase = phase.copy()
                    # _phase -= _phase.mean()
                    _phase *= sign
                    shift = _phase[np.argmin(np.abs(masses - mass0))] - y_mean[np.argmin(np.abs(x - mass0))]
                    ax.plot(masses - half_bin_width, _phase - shift, c=color, zorder=4)
        else:
            L = wave[0]
            resonances = overlay_saturated_resonances[L]
            for resonance in resonances:
                color, mass0, intensity, phase = resonance
                scale_to_match_peak = y_mean[np.argmin(np.abs(x - mass0))] / intensity.max()
                ax.plot(masses - half_bin_width, intensity * scale_to_match_peak, c=color, zorder=4)

    [axes[i].set_xlabel("Mass (GeV)") for i in range(ncols * (nrows - 1), ncols * nrows)]

    if is_phase_col:
        amp1, amp2 = wave.split()
        [axes[i].set_ylabel("Phase (rad)") for i in range(0, ncols * nrows, ncols)]
        plt.suptitle(f"$\phi$({prettyLabels[amp1]}, {prettyLabels[amp2]})", y=0.96, fontsize=25)
    else:
        [axes[i].set_ylabel("Intensity") for i in range(0, ncols * nrows, ncols)]
        plt.suptitle(f"Wave: {prettyLabels[wave]}", y=0.96, fontsize=25)

    plt.subplots_adjust(wspace=0, hspace=0)

    return fig, axes


class KDE:
    """
    This class manages how the KDE is constructed. N Posterior samples are drawn with M kinematic bins.
    For each N samples:
        Interpolate the data so that the distance between each point is less than a user specified distance
            This creates more data points with regular 2D spacing. Store + Accumuldate all points in an array
    Get KDE of all the interpolated data points
    When plotting, do a column-wise (kinematic-bin) normalization so that the max value of each column is 1
    """

    def __init__(self, data, n_points=100):
        """
        data needs to include all samples drawn from the posterior! Otherwise we will not be able
          to perform corrections for multi-modal solutions
        """

        # The exact range doesn't matter for KDE as long as they are the same for x-y axes
        #  We take this opportunity to normalize to the range for the relative phase plots
        #  so that it is easier to apply corrections / symmetries
        feature_range = (-180, 180)
        self.scaler = MinMaxScaler(feature_range=feature_range)
        # Get rough scale for the average hypotenuse length and partition it finely
        self.desired_dist = np.sqrt(feature_range[0] ** 2 + feature_range[1] ** 2) / 50

        self.data = self.scaler.fit_transform(data)
        self.n_points = n_points
        self.grid_coords = self.get_grid(self.data, n_points=n_points)

        self.rescaled_grid_coords = self.scaler.inverse_transform(np.vstack([self.grid_coords[0].ravel(), self.grid_coords[1].ravel()]).T).T

        self.Z = None

        self._xs = []
        self._ys = []

    def add_curve(self, data, is_intensity):
        # Interpolate the data so that the distance between each point is less than a desired distance
        # since the x-y axes (after MinMaxScalaer) are both on [0,1] average hypotenuse is sqrt(2).
        _x, _y = self.interpolate_linear(data[:, 0], data[:, 1], self.data, self.desired_dist, is_intensity=is_intensity)
        self._xs.extend(_x)
        self._ys.extend(_y)

    def calculate_kde(self, bw=0.1):
        _start = time.time()
        data = np.vstack([self._xs, self._ys])
        self.Z = gaussian_kde(data, bw_method=bw)(self.grid_coords)
        print(f"KDE| density estimation time: {time.time() - _start:.2f} s")

    def plot(self, ax, bw=0.1, pullup=True, nlevels=10, is_intensity=True, cmap="viridis", cbar=False):
        """
        Plot the KDE of the data

        Args:
            ax (matplotlib.axes._subplots.AxesSubplot): Axis to plot the KDE on
            bw (float): Bandwidth for the KDE
            pullup (bool): Whether to normalize the KDE so that the maximum value of each column (mass bin) is 1.
                This makes some sense since we do binned fits. Each normalized column would represent roughly
                the posterior distribution in the bin. The KDE smoothing and interpolation allows X-axis coorelations
                to be introduced.
            nlevels (int): Number of contour levels to plot
            is_intensity (bool): Whether to plot the intensity
            cmap (str): Colormap to use
            cbar (bool): Whether to plot the colorbar
        """

        self.calculate_kde(bw=bw)

        # column-wise normalization so that the maximum value of of each column is 1
        Z = self.Z
        if pullup:
            Z = Z.reshape(self.n_points, self.n_points)
            Z = Z / np.max(Z, axis=0)
            Z = Z.ravel()

        _start = time.time()
        self.plot_kde(self.rescaled_grid_coords, Z, ax, is_intensity=is_intensity, nlevels=nlevels, cmap=cmap, cbar=cbar)
        print(f"KDE| Plot time: {time.time() - _start:.2f} s")

    def get_kde(self):
        return self.Z.reshape(self.n_points, self.n_points)

    @staticmethod
    def interpolate_linear(x, y, full_xy_data, desired_dist, is_intensity):
        """
        Perform linear interpolation so that the distance between each point is less than a desired distance.
        This creates more points when the y-values are changing rapidly and fewer points when the y-values are changing slowly

        Args:
            x (List): x-coordinates of the data
            y (List): y-coordinates of the data
            desired_dist (float): Desired distance between each point

        Returns:
            np.ndarray: New x-coordinates with additional linearly interpolated points
            np.ndarray: New y-coordinates with additional linearly interpolated points
        """

        new_x = [x[0]]
        new_y = [y[0]]

        # x_bin_width = x[1] - x[0]

        for i in range(1, len(x)):
            # Handle intensity and relative phase plots differently

            # Calculate the distance between the current and previous point
            dist = np.hypot(x[i] - x[i - 1], y[i] - y[i - 1])

            # is_phase = not is_intensity

            # For Relative Phase plots
            # Check 1: # Qualitativlely worse
            #     No interpolation if there are other samples with nearby points (in mass) that is within 5 degrees
            #     of each other with the opposite sign. Attempts to handle the trivial ambiguity making jumps
            # y_samples = full_xy_data[:, 1][ np.abs(full_xy_data[:, 0]- x[i]) <= 1.5 * x_bin_width]
            # if is_phase and any( np.abs(y_samples + y[i]) < 1 ):
            #     continue

            if dist > desired_dist:
                # Calculate the number of points to interpolate
                num_points_to_add = int(np.ceil(dist / desired_dist)) - 1
                # Interpolate new x and y values
                for j in range(1, num_points_to_add + 1):
                    fraction_of_distance = j / (num_points_to_add + 1)
                    new_x_point = x[i - 1] + (x[i] - x[i - 1]) * fraction_of_distance
                    new_y_point = y[i - 1] + (y[i] - y[i - 1]) * fraction_of_distance
                    new_x.append(new_x_point)
                    new_y.append(new_y_point)

            new_x.append(x[i])
            new_y.append(y[i])

        return np.array(new_x), np.array(new_y)

    @staticmethod
    def get_grid(data, n_points=100):
        """
        Get a grid of coordinates to evaluate a 2D KDE

        Args:
            data (2D np.ndarray): Data to evaluate the KDE of: [n_samples, 2]
            n_points (int): Number of grid points

        Returns:
            Xgrid (np.ndarray): Grid of x-coordinates to evaluate the KDE
            Ygrid (np.ndarray): Grid of y-coordinates to evaluate the KDE
            grid_coords (np.ndarray): Grid of coordinates to evaluate the KDE
        """

        # Create a grid to evaluate KDE
        xmin = np.min(data[:, 0]) * (1.05 if np.min(data[:, 0]) < 0 else 0.95)
        xmax = np.max(data[:, 0]) * (0.95 if np.max(data[:, 0]) < 0 else 1.05)
        ymin = np.min(data[:, 1]) * (1.05 if np.min(data[:, 1]) < 0 else 0.95)
        ymax = np.max(data[:, 1]) * (0.95 if np.max(data[:, 1]) < 0 else 1.05)
        xgrid = np.linspace(xmin, xmax, n_points)
        ygrid = np.linspace(ymin, ymax, n_points)
        Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
        grid_coords = np.vstack([Xgrid.ravel(), Ygrid.ravel()])

        return grid_coords

    @staticmethod
    def plot_kde(grid_coords, Z, ax, is_intensity=True, nlevels=15, cmap="viridis", cbar=False):
        plt.sca(ax)

        # get viridis colormap but on a smaller range for intensity plot
        # else get the requested cmap for the relative phase plots
        if is_intensity:
            original_cmap = plt.get_cmap(cmap)
            colors_range = original_cmap(np.linspace(0, 1.0, 40))
            cmap = colors.LinearSegmentedColormap.from_list("trunc({n},{a:.2f},{b:.2f})".format(n=cmap, a=0.3, b=0.6), colors_range)

        plt.tricontour(grid_coords[0], grid_coords[1], Z, levels=nlevels, linewidths=1.5, cmap=cmap, alpha=1.0)
        cntr2 = plt.tricontourf(grid_coords[0], grid_coords[1], Z, levels=nlevels, cmap=cmap)

        # Set the alpha for each level after plotting
        for i, collection in enumerate(cntr2.collections):
            if i < len(cntr2.levels) and cntr2.levels[i] <= np.min(Z):
                collection.set_alpha(0.0)
            else:
                # Set a gradient of alphas based on the level index, modify as needed
                collection.set_alpha(0.3 + (i / len(cntr2.levels)) * 0.2)

        if cbar:
            plt.colorbar(cntr2, ax=ax)

        # Plot the Ridge that the maximum of the KDE approximated posterior follows
        n_points = int(np.sqrt(Z.shape[0]))
        y_peak_loc = np.argmax(Z.reshape(n_points, n_points), axis=0)
        _X = grid_coords[0].reshape(n_points, n_points)[0]
        _Y = grid_coords[1].reshape(n_points, n_points)[:, 0]
        ridge_xs = [_X[x] for x in range(len(y_peak_loc))]
        ridge_ys = [_Y[y] for y in y_peak_loc]
        plt.plot(ridge_xs, ridge_ys, color="black", linewidth=1, linestyle="dashed")
