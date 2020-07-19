"""
This module contains functions for plotting various datasets associated with Gaussian Processes and Dirichlet Processes.
"""

from src.utils.constants import NP_DTYPE
from src.utils.types import validate_positive, validate_np_array

from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as color_map
import matplotlib.pyplot as plot
import numpy as np


def set_figure(fig=None, axes_shape=None):
    """
    This function validates fig or creates a new figure.
    :param fig: An existing matplotlib Figure window.
    :param axes_shape: The desired configuration of the axes in the figure.
    :return: Returns a valid matplotlib Figure and Axes.
    """

    # Validated axes_shape.
    if axes_shape is not None:
        assert isinstance(axes_shape, tuple) or isinstance(axes_shape, list), \
            'Shape must be provided as a list or tuple.'
        assert len(axes_shape) == 2, 'Can only specify number of rows and columns of subplots.'

    # Set current figure to one provided if it is valid. Else create new figure.
    if isinstance(fig, Figure):

        if axes_shape is not None:
            # Check figure has enough axes.
            assert len(fig.axes) == np.prod(axes_shape), 'Figure must contain %s axes.' % np.prod(axes_shape)

        plot.figure(fig.number)
        ax = fig.axes
    else:
        if axes_shape is not None:
            # Check figure has enough axes.
            [fig, ax] = plot.subplots(nrows=axes_shape[0], ncols=axes_shape[1], sharex='col', sharey='row')
        else:
            fig = plot.figure()
            ax = fig.axes

    return fig, ax


def plot_data(y, y_predicted=None, fig=None, add_labels=True):
    """
    This function plots the variance of the data. It plots YYT, where Y is [N x D] matrix.
    :param y: The observed data. Must be provided as [N x D] numpy array, where N is the number of samples and D is
    the dimensionality of the observed data.
    :param y_predicted: The predicted data if comparison to observed is desired.
    Must be provided as [N x D] numpy array.
    :param fig: An existing matplotlib Figure window within which to produce the plot.
    :param add_labels: Add title labels to the plot.
    :return: Returns the matplotlib Figure that contains the plot.
    """

    # Confirm y is 2D numpy array.
    assert isinstance(y, np.ndarray)
    assert y.ndim == 2

    # Define n.
    n = y.shape[0]

    if y_predicted is None:
        # Set figure.
        fig = set_figure(fig)

        # For 2D array, np.matmul is same as np.dot.
        plot.imshow(np.matmul(y, y.T), interpolation='nearest', aspect='auto', extent=(0, n, n, 0), origin='upper')

        if add_labels:
            plot.title('YYT - Observed Data')

    else:
        # Validate y_predicted.
        assert isinstance(y_predicted, np.ndarray)
        assert y.shape == y_predicted.shape

        # Set figure.
        [fig, ax] = set_figure(fig, axes_shape=(2, 1))

        plot.sca(ax[0])  # Set current subplot.
        # For 2D array, np.matmul is same as np.dot.
        plot.imshow(np.matmul(y, y.T), interpolation='nearest', aspect='auto', extent=(0, n, n, 0), origin='upper')

        if add_labels:
            plot.title('YYT - Observed Data')

        plot.sca(ax[1])  # Set current subplot.
        # For 2D array, np.matmul is same as np.dot.
        plot.imshow(np.matmul(y_predicted, y_predicted.T), interpolation='nearest', aspect='auto', extent=(0, n, n, 0),
                    origin='upper')

        if add_labels:
            plot.title('YYT - Predicted Data')

    return fig


def plot_ard(ard_hp, true_ard=None, fig=None, add_labels=True):
    """
    This function plots the ARD weights for an unknown covariance function for a GP.
    :param ard_hp: The ARD weights to plot. Must be provided as [D x Q] numpy array, where D is the dimensionality of
    the observed data and Q is the dimensionality of the latent (or input) space.
    :param true_ard: The actual ARD weights if comparision is desired. Must be provided as
    :param fig:
    :param add_labels:
    :return:
    """

    # Confirm ARD weights is 2D numpy array.
    assert isinstance(ard_hp, np.ndarray)
    assert ard_hp.ndim == 2

    # Define d and q.
    [d, q] = np.shape(ard_hp)

    if true_ard is None:
        # Set figure.
        fig = set_figure(fig)

        plot.imshow(ard_hp.T, interpolation='nearest', aspect='auto', extent=(0, d, q, 0), origin='upper',
                    cmap=color_map.Blues)  # , vmin=0.0, vmax=1.0)
        # plot.imshow(ard_hp.T, interpolation='nearest', aspect='auto', extent=(0, d, q, 0), origin='upper',
        #             cmap=color_map.Blues, vmin=0.0, vmax=1.0)
        # plot.imshow(ard_hp.T, interpolation='nearest', aspect='auto', extent=(0, d, q, 0), origin='upper',
        #             cmap=color_map.Blues, vmin=0.0)  # , vmax=1.0)
        # plot.imshow(ard_hp.T, interpolation='nearest', aspect='auto', extent=(0, d, q, 0), origin='upper',
        #             cmap=color_map.Blues, vmax=0.0)  # , vmax=1.0)

        # # Log scale.
        # plot.imshow(ard_hp.T, interpolation='nearest', aspect='auto', extent=(0, d, q, 0), origin='upper',
        #             cmap=color_map.Blues, norm=LogNorm(vmin=0.000001, vmax=1))

        if add_labels:
            plot.xticks(range(d + 1))
            plot.gca().set_xticklabels([])  # Remove Y-dimension indices from x-axis.
            plot.yticks(range(q + 1))
            plot.xlabel('Y-Dimension')
            plot.ylabel('X-Dimension')
            plot.title('ARD Weights')

    else:
        # Validate true_ard.
        assert isinstance(true_ard, np.ndarray)
        assert ard_hp.shape == true_ard.shape

        # Set figure.
        [fig, ax] = set_figure(fig, axes_shape=(2, 1))

        plot.sca(ax[0])  # Set current subplot.
        plot.imshow(true_ard.T, interpolation='nearest', aspect='auto', extent=(0, d, q, 0), origin='upper',
                    cmap=color_map.Blues)  # , vmin=0.0, vmax=1.0)
        if add_labels:
            plot.xticks(range(d))
            plot.yticks(range(q))
            plot.ylabel('X-Dimension')
            plot.title('True ARD Weights')

        plot.sca(ax[1])  # Set current subplot.
        plot.imshow(ard_hp.T, interpolation='nearest', aspect='auto', extent=(0, d, q, 0), origin='upper',
                    cmap=color_map.Blues)  # , vmin=0.0, vmax=1.0)
        if add_labels:
            plot.xticks(range(d))
            plot.yticks(range(q))
            plot.xlabel('Y-Dimension')
            plot.ylabel('X-Dimension')
            plot.title('Optimized ARD Weights')

    return fig


def plot_phi(phi, fig=None, add_labels=True):
    """
    This function plots a stacked bar chart of the probabilities of the multinomial variational distribution for a DP.
    :param phi: The probibilities of the multinomial variational distribution for a Dirichlet Process.
    Must be provided as [N x T] numpy array, where N is the number of samples and T is the truncation level of the DP.
    :param fig: An existing matplotlib Figure window within which to produce the plot.
    :param add_labels: Add x-axis and title labels to the plot.
    :return: Returns the matplotlib Figure that contains the plot.
    """

    # Confirm phi is 2D numpy array.
    assert isinstance(phi, np.ndarray)
    assert phi.ndim == 2

    # Set figure.
    fig = set_figure(fig)

    # Setup variables for stacking bar plot.
    [n, t] = phi.shape
    index = np.arange(n)  # 0 to n.
    width = 0.25  # the width of the bars: can also be len(x) sequence
    current_val = np.zeros(n)

    # Loop through each stick of the DP.
    for i in range(t):
        plot.bar(index, phi[:, i], width, bottom=current_val, color=color_map.jet(1.0 * i / t))
        current_val += phi[:, i]
    plot.xticks(index + width/2.0, ('{:02d}'.format(x) for x in range(1, n + 1)))
    plot.yticks(np.arange(0.0, 1.01, 0.1))
    if add_labels:
        plot.xlabel('Y-Dimension')
        plot.gca().set_xticklabels([])  # Remove Y-dimension indices from x-axis.
        plot.title(r'$\phi$')

    return fig


def plot_phi_matrix(phi, fig=None, add_labels=True):
    """
    This function plots the the probabilities of the multinomial variational distribution for a DP.
    :param phi: The probibilities of the multinomial variational distribution for a Dirichlet Process.
    Must be provided as [N x T] numpy array, where N is the number of samples and T is the truncation level of the DP.
    :param fig: An existing matplotlib Figure window within which to produce the plot.
    :param add_labels: Add x-axis, y-axis, and title labels to the plot.
    :return: Returns the matplotlib Figure that contains the plot.
    """

    # Confirm ARD weights is 2D numpy array.
    assert isinstance(phi, np.ndarray)
    assert phi.ndim == 2

    # Define n and t.
    [n, t] = np.shape(phi)

    # Set figure.
    fig = set_figure(fig)

    plot.imshow(phi.T, interpolation='nearest', aspect='auto', extent=(0, n, t, 0), origin='upper',
                cmap=color_map.Blues, vmin=0.0, vmax=1.0)

    if add_labels:
        plot.xticks(range(n + 1))
        plot.gca().set_xticklabels([])  # Remove Y-dimension indices from x-axis.
        plot.yticks(range(t + 1))
        plot.xlabel('Y-Dimension')
        plot.ylabel('DP Atom')
        plot.title('DP Assignments')

    return fig


def plot_2d_latent_space(session, kernel, ard_weights, x_mean, x_u, var_img_dim=25, x_dims=None, on_threshold=0.001,
                         fig=None, add_labels=True):
    """
    TODO
    This function plots the 2D latent space of the provided model.
    :param model: The Bayesian GPLVM model.
    :param session: The current TensorFlow Session.
    :param var_img_dim: The number of samples to predict for the variance in each dimension. So total samples are the
    square of this number.
    :param x_dims: The latent dimensions to use for the plot.
    :param on_threshold: The threshold for determing if a latent dimension is considered on.
    :param fig: An existing matplotlib Figure window within which to produce the plot.
    :param add_labels: Add axes and title labels to the plot.
    :return: Returns the matplotlib Figure that contains the plot.
    """

    # Set figure.
    fig = set_figure(fig)

    # Get ARD weights, q(X), and inducing inputs from model.
    num_latent_dims = x_mean.shape[1]

    on_dims = np.nonzero(ard_weights > on_threshold)[0] if x_dims is None else x_dims
    assert np.size(on_dims) == 2, 'Number of \'on\' latent dimensions is more than 2.'
    x_0 = np.hstack((x_mean[:, on_dims[0]], x_u[:, on_dims[0]]))
    x_1 = np.hstack((x_mean[:, on_dims[1]], x_u[:, on_dims[1]]))
    scale_factor = 1.1
    x_0_min = np.min(x_0) * scale_factor
    x_0_max = np.max(x_0) * scale_factor
    x_1_min = np.min(x_1) * scale_factor
    x_1_max = np.max(x_1) * scale_factor

    xv, yv = np.meshgrid(np.linspace(x_0_min, x_0_max, var_img_dim), np.linspace(x_1_min, x_1_max, var_img_dim))
    x_test = np.zeros((var_img_dim * var_img_dim, num_latent_dims), dtype=NP_DTYPE)
    x_test[:, on_dims[0]] = xv.flatten()
    x_test[:, on_dims[1]] = yv.flatten()

    var_diag = session.run(kernel.covariance_diag(input_0=x_test, include_noise=False, include_jitter=False))

    plot.imshow(var_diag.reshape(var_img_dim, var_img_dim), interpolation='bicubic', aspect='auto',
                extent=(x_0_min, x_0_max, x_1_min, x_1_max), origin='lower', cmap='RdBu')
    plot.scatter(x_mean[:, on_dims[0]], x_mean[:, on_dims[1]], marker='o', color='C2')
    plot.scatter(x_u[:, on_dims[0]], x_u[:, on_dims[1]], marker='+', color='C1')
    plot.plot(x_mean[:, on_dims[0]], x_mean[:, on_dims[1]], color='C2', linestyle='--')

    if add_labels:
        plot.xlabel('X[%s]' % on_dims[0])
        plot.ylabel('X[%s]' % on_dims[1])
        plot.title('q(X) and Inducing Input')

    return fig


def plot_2d_latent_path(x_mean, inducing_input, var_img_dim=25, add_labels=True):
    """
    This function plots the 2D latent space of the provided model.
    :param model: The GP-DP model.
    :param session: The current TensorFlow Session.
    :param var_img_dim: The number of samples to predict for the variance in each dimension. So total samples are the
    square of this number.
    :param add_labels: Add axes and title labels to the plot.
    :return: Returns the matplotlib Figure that contains the plot.
    """

    num_latent_dims = x_mean.shape[1]
    assert num_latent_dims == inducing_input.shape[1]

    # Create new figure.
    fig = plot.figure()

    x_0 = np.hstack((x_mean[:, 0], inducing_input[:, 0]))
    x_1 = np.hstack((x_mean[:, 1], inducing_input[:, 1]))
    scale_factor = 1.1
    x_0_min = np.min(x_0) * scale_factor
    x_0_max = np.max(x_0) * scale_factor
    x_1_min = np.min(x_1) * scale_factor
    x_1_max = np.max(x_1) * scale_factor

    xv, yv = np.meshgrid(np.linspace(x_0_min, x_0_max, var_img_dim), np.linspace(x_1_min, x_1_max, var_img_dim))
    x_test = np.zeros((var_img_dim * var_img_dim, num_latent_dims), dtype=types.NP_DTYPE)
    x_test[:, 0] = xv.flatten()
    x_test[:, 1] = yv.flatten()

    _, predict_covar = model.predict(x_test)
    var_diag = np.diag(session.run(predict_covar))

    plot.imshow(var_diag.reshape(var_img_dim, var_img_dim), interpolation='bicubic', aspect='auto',
                extent=(x_0_min, x_0_max, x_1_min, x_1_max), origin='lower', cmap='RdBu')
    plot.scatter(x_mean[:, 0], x_mean[:, 1], marker='o', color='C2')
    plot.scatter(x_u[:, 0], x_u[:, 1], marker='+', color='C1')
    plot.plot(x_mean[:, 0], x_mean[:, 1], color='C2', linestyle='--')

    if add_labels:
        plot.xlabel('X[%s]' % 0)
        plot.ylabel('X[%s]' % 1)
        plot.title('q(X) and Inducing Input')

    return fig

    # Add 3D axes.
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_mean[:, 0], x_mean[:, 1], x_mean[:, 2])

    if add_labels:
        # TODO
        pass

    return fig


def plot_3d_latent_path(model, session, add_labels=True):
    """
    This function plots the 3D latent space of the provided model.
    :param model: The GP-DP model.
    :param session: The current TensorFlow Session.
    :param add_labels: Add axes and title labels to the plot.
    :return: Returns the matplotlib Figure that contains the plot.
    """

    # Get mean of X from q(X).
    x_mean, _ = session.run(model.q_x)

    # Create new figure.
    fig = plot.figure()

    # Add 3D axes.
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_mean[:, 0], x_mean[:, 1], x_mean[:, 2])

    if add_labels:
        # TODO
        pass

    return fig


def plot_covariance(data, axis=0, show_colorbar=True):
    """
    This function plots the covariance matrix of the provided data where a single variable is defined along the
    specified axis.
    :param data: The 2D numpy array whose covariance is to be plotted.
    :param axis: The axis which defines a variable. If 0, then each row is a variable and the columns contain the
    observations. If 1, then each column is a variable and the rows contain the observations.
    :param show_colorbar:
    :return: Returns the matplotlib Figure that contains the covariance plot.
    """

    # TODO: Allow for colormap to be specified.

    # Validate input. Confirm data is a 2D numpy array.
    validate_np_array(data, num_dims=2)
    n, d = data.shape

    # Set whether each row or column is a variable for calculating covariance.
    row_variables = True
    num_variables = n
    if axis == 1:
        row_variables = False
        num_variables = d

    # Create new figure.
    fig = plot.figure()

    # Plot covariance.
    plot.imshow(np.cov(data, rowvar=row_variables), interpolation='nearest', aspect='auto',
                extent=(0, num_variables, num_variables, 0), origin='upper')

    # Plot colorbar, if specified.
    if show_colorbar:
        plot.colorbar()

    return fig


def web_plot_covariance(data, fig_axis, colorbar_axis=None, axis=0):
    """
    TODO
    :param data:
    :param fig_axis:
    :param colorbar_axis:
    :param axis:
    """

    # TODO: Allow for colormap to be specified.

    # Validate input.
    assert isinstance(fig_axis, Axes), 'Provided axis for covariance figure is not a valid matplotlib Axes.'
    validate_np_array(data, num_dims=2)  # Confirm data is a 2D numpy array.

    # Set whether each row or column is a variable for calculating covariance.
    # TODO: Should validate that axis is either 0 or 1.
    row_variables = True
    if axis == 1:
        row_variables = False

    # Plot covariance.
    axes_image = fig_axis.matshow(np.cov(data, rowvar=row_variables))

    # Plot colorbar, if Matplotlib axis is provided.
    if colorbar_axis is not None:
        assert isinstance(colorbar_axis, Axes), 'Provided axis for colorbar figure is not a valid matplotlib Axes.'
        plot.colorbar(mappable=axes_image, cax=colorbar_axis)


def plot_eigenvalues(data, axis=0, num_vals=10):
    """
    TODO
    :param data:
    :param axis:
    :param num_vals:
    :return: Returns the matplotlib Figure that contains the plot of eigenvalues.
    """

    # Validate input. Confirm data is a 2D numpy array.
    validate_positive(num_vals, dtype=int)
    validate_np_array(data, num_dims=2)
    n, d = data.shape

    # Set whether each row or column is a variable for calculating covariance.
    row_variables = True
    num_variables = n
    if axis == 1:
        row_variables = False
        num_variables = d

    eig_values, _ = np.linalg.eig(np.cov(data, rowvar=row_variables))
    num_eigs = min(num_vals, num_variables)
    largest_eig_values = np.sort(eig_values)[::-1][:num_eigs]  # Get first num_eigs largest eigenvalues.

    # Confirm all eigenvalues are real.
    largest_eig_values = np.real_if_close(largest_eig_values, tol=1000)
    assert np.all(np.isreal(largest_eig_values)), 'Not all the largest eigenvalues are real.'

    # Create new figure.
    fig = plot.figure()

    # Plot largest eigenvalues as vertical bar plot.
    plot.bar(np.arange(num_eigs), largest_eig_values, width=0.55)
    plot.title('{} Largest Eigenvalues of Covariance Matrix'.format(num_eigs))
    plot.ylabel('Eigenvalue')

    return fig
