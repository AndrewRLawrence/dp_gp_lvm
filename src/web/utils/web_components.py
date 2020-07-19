"""
This module defines various low-level components for building a web GUI in python 3.
"""

from src.utils.constants import UOB_BLUE, UOB_GREY, DataSetKeys, ResultKeys
from src.utils.types import validate_np_array
import src.visualisation.plotters as vis

from io import BytesIO
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plot
from multiprocessing.pool import ThreadPool
import numpy as np
import remi.gui as gui
import threading


class NavigationBar(gui.HBox):
    """
    This class defines a navigation bar GUI widget.
    """

    def __init__(self, headings_list, active_index, click_func):
        """
        Constructor
        :param headings_list: The list of headings, provided as a list of strings.
        """

        def nav_bar_heading(text):
            """
            This function wraps a remi.gui.Label with appropriate style settings to define navigation bar heading.
            :param text: The text of the heading.
            :return: A remi.gui.Label with the provided text and style settings for navigation bar heading.
            """

            return gui.Label(text=text, style={'font-weight': '250', 'line-height': '1.1', 'color': '#FFFFFF',
                                               'font-size': '18px', 'margin-top': '12px', 'margin-bottom': '12px',
                                               'margin-left': '10px', 'margin-right': '10px', 'text-shadow': 'none',
                                               'text-align': 'left'})

        # Validate input.
        assert callable(click_func), 'Click function must be a callable function.'
        assert isinstance(headings_list, list), 'Navigation bar headings must be provided as a list.'
        self.headings_list = headings_list
        self.num_headings = len(headings_list)
        assert 0 <= active_index < self.num_headings, 'Active index must between 0 and {}'.format(self.num_headings)
        self.active_index = active_index

        # Base class constructor with appropriate settings for navigation bar.
        super(NavigationBar, self).__init__(width='100%', margin='0px', style={'text-align': 'left',
                                                                               'background-color': UOB_BLUE,
                                                                               'align-content': 'left'})

        # Define label for each heading and set the click function for each.
        heading_labels = [nav_bar_heading(heading) for heading in headings_list]
        for heading in heading_labels:
            heading.onclick.connect(click_func)

        # Define dictionary with the heading strings as keys and the remi.gui.Labels as the values and append to self.
        self.append(dict(zip(headings_list, heading_labels)))

        # Set active index.
        self.get_child(headings_list[active_index]).style['font-weight'] = '550'

    def set_active_index(self, new_active_index):
        """
        This function allows the active index of the navigation bar to be updated.
        :param new_active_index: An integer representing the index of the new active page.
        """

        # Validate input.
        assert 0 <= new_active_index < self.num_headings, \
            'New active index must between 0 and {}'.format(self.num_headings)

        # Update font weightings.
        self.get_child(self.headings_list[self.active_index]).style['font-weight'] = '250'
        self.get_child(self.headings_list[new_active_index]).style['font-weight'] = '550'

        # Update active index.
        self.active_index = new_active_index

    def get_active_index(self):
        """
        This function returns the active index of the navigation bar.
        :return: The current active index of the navigation bar.
        """
        return self.active_index


class MatplotlibFigure(gui.Image):
    """
    This class defines a GUI widget for displaying a matplotlib figure.
    """

    def __init__(self, **kwargs):
        """
        TODO
        :param kwargs:
        """

        self._update_counter = 0
        super(MatplotlibFigure, self).__init__('/{}/get_image_data?update_index={}'.format(
            id(self), self._update_counter), **kwargs)

        self._buffer = None
        self._buffer_lock = threading.Lock()
        self._fig = Figure(figsize=(4, 4))  # TODO: Allow size to be set.
        self.ax = self._fig.add_subplot(111)

        self.redraw()

    def redraw(self):
        """
        This function overrides the base class redraw function and defines how to redraw matplotlib figure.
        """

        canvas = FigureCanvasAgg(self._fig)
        buffer = BytesIO()
        # canvas.print_figure(buffer, format='png')  # PNG is only format supported by Agg backend.
        canvas.print_png(buffer)  # PNG is the only format supported by the Agg backend of matplotlib.
        with self._buffer_lock:
            if self._buffer is not None:
                self._buffer.close()
            self._buffer = buffer

        self._update_counter += 1
        self.set_image('{}/get_image_data?update_index={}'.format(id(self), self._update_counter))

        super(MatplotlibFigure, self).redraw()

    def get_image_data(self, update_index):
        """
        TODO
        :param update_index:
        :return:
        """

        ret_val = None

        with self._buffer_lock:
            if self._buffer is not None:
                self._buffer.seek(0)
                ret_val = [self._buffer.read(), {'Content-type': 'image/png'}]
            # No else needed as return value is already None.

        return ret_val


class DataSetAnalysis(gui.VBox):
    """
    This class defines a GUI widget for analysing a data set.
    """

    def __init__(self, data, label_text):
        """
        Constructor
        :param data: The raw data of the data set, either as a numpy array or a NpzFile object.
        :param label_text: The name of the data set.
        """

        # Validate input.
        assert isinstance(data, np.ndarray) or isinstance(data, np.lib.npyio.NpzFile), \
            'Data is not provided as a numpy array or an NpzFile.'

        # Base class constructor with appropriate settings for data set analysis page.
        super(DataSetAnalysis, self).__init__(width='50%', margin='auto',
                                              style={'display': 'block', 'overflow': 'auto', 'align-content': 'center',
                                                     'margin-top': '20px'})

        # Define thread pool for analysing data for the chosen plot type.
        self._thread_pool = ThreadPool(processes=1)

        dataset_title = h3(label_text)
        dataset_title.style['text-align'] = 'center'

        self._full_data = data
        data_keys = None
        plot_keys = ['Covariance', 'Eigenvalues', 'Histogram', 'Scatter']
        selection_container = None
        self._subset_dropdown = None
        self._subset_select_button = None

        # Build subset pulldown if data is a NpzFile oject.
        if isinstance(data, np.lib.npyio.NpzFile):
            data_keys = list(data.keys())
            self._current_data = data.get(data_keys[0])

            # Define row container for subset selection.
            selection_container = gui.HBox(width='100%', margin='auto',
                                           style={'display': 'block', 'overflow': 'auto', 'align-content': 'center',
                                                  'margin-top': '10px'})

            subset_label = h5('Subset: ')
            subset_label.style['margin-top'] = '8px'

            # Define drop down with different subsets.
            self._subset_dropdown = gui.DropDown(width='250px', style={'margin-right': '15px'})
            self._subset_dropdown.add_class('form-control dropdown')
            self._subset_dropdown.append(dict(zip(data_keys,
                                                  [gui.DropDownItem(subset_str) for subset_str in data_keys])))

            self._subset_select_button = gui.Button('Select', width='100px', style={'box-shadow': 'none'})
            self._subset_select_button.add_class('btn btn-primary')
            self._subset_select_button.onclick.connect(self._subset_select_button_clicked)  # Listener function

            selection_container.append([subset_label, self._subset_dropdown, self._subset_select_button])
        else:
            self._current_data = data

        # Confirm current data is a 2D numpy array.
        validate_np_array(self._current_data, num_dims=2)

        # Add attributes for current data subset.
        attributes_container = gui.HBox(width='100%', margin='auto',
                                        style={'display': 'block', 'overflow': 'auto', 'align-content': 'center',
                                               'text-align': 'center', 'margin-top': '10px'})
        self._num_observations_label = h5('Number of Observations (N): {}'.format(self._current_data.shape[0]))
        self._num_dimensions_label = h5('Number of Dimensions (D): {}'.format(self._current_data.shape[1]))
        self._dtype_label = h6('DType: {}'.format(self._current_data.dtype))
        attributes_container.append([h4('Attributes: '),
                                     self._num_observations_label,
                                     self._num_dimensions_label,
                                     self._dtype_label])

        # Add plot section.
        plot_container = gui.VBox(width='100%', margin='auto',
                                  style={'display': 'block', 'overflow': 'auto', 'align-content': 'center',
                                         'text-align': 'center', 'margin-top': '10px'})
        plot_control_container = gui.HBox(width='100%', margin='auto',
                                          style={'display': 'block', 'overflow': 'auto', 'align-content': 'center',
                                                 'margin-top': '10px'})
        plot_figures_container = gui.HBox(width='100%', margin='auto',
                                          style={'display': 'block', 'overflow': 'auto', 'align-content': 'center',
                                                 'margin-top': '10px'})
        plot_label = h5('Plot Type: ')
        plot_label.style['margin-top'] = '8px'

        # Define drop down with different plot types.
        self._plot_dropdown = gui.DropDown(width='200px', style={'margin-right': '15px'})
        self._plot_dropdown.add_class('form-control dropdown')
        self._plot_dropdown.append(dict(zip(plot_keys, [gui.DropDownItem(plot_str) for plot_str in plot_keys])))

        self._plot_select_button = gui.Button('Select', width='100px', style={'box-shadow': 'none'})
        self._plot_select_button.add_class('btn btn-primary')
        self._plot_select_button.onclick.connect(self._plot_select_button_clicked)  # Listener function

        # Add MatplotlibFigure widgets.
        self._mpl_data_fig = MatplotlibFigure(width=250, height=250)
        self._mpl_colorbar = MatplotlibFigure(width=50, height=250)
        ax_image = self._mpl_data_fig.ax.imshow(np.cov(self._current_data, rowvar=True), interpolation='nearest',
                                                aspect='auto',
                                                extent=(0, self._current_data.shape[0], self._current_data.shape[0], 0),
                                                origin='upper')
        self._mpl_data_fig.ax.set_title(self._plot_dropdown.get_key())
        plot.colorbar(mappable=ax_image, cax=self._mpl_colorbar.ax)
        self._mpl_data_fig.redraw()
        self._mpl_colorbar.redraw()

        plot_control_container.append([plot_label, self._plot_dropdown, self._plot_select_button])
        plot_figures_container.append([self._mpl_data_fig, self._mpl_colorbar])
        plot_container.append([plot_control_container, plot_figures_container])

        if selection_container is not None:
            self.append([dataset_title, selection_container, attributes_container, plot_container])
        else:
            self.append([dataset_title, attributes_container, plot_container])

    def __del__(self):
        """
        Destructor
        """
        # Terminate the worker processes immediately without completing outstanding work.
        self._thread_pool.terminate()
        self._thread_pool.join()

    def _update_current_data(self, dropdown_key):
        """
        TODO
        :param dropdown_key:
        """
        self._current_data = self._full_data.get(dropdown_key)
        self._num_observations_label.set_text('Number of Observations (N): {}'.format(self._current_data.shape[0]))
        self._num_dimensions_label.set_text('Number of Dimensions (D): {}'.format(self._current_data.shape[1]))
        self._dtype_label = h6('DType: {}'.format(self._current_data.dtype))

        # Update plot as well.
        self._update_plot(self._plot_dropdown.get_key())

    def _update_plot(self, dropdown_key):
        """
        TODO
        :param dropdown_key:
        """
        ax_image = self._mpl_data_fig.ax.imshow(np.cov(self._current_data, rowvar=True), interpolation='nearest',
                                                aspect='auto',
                                                extent=(0, self._current_data.shape[0], self._current_data.shape[0], 0),
                                                origin='upper')
        self._mpl_data_fig.ax.set_title(dropdown_key)
        plot.colorbar(mappable=ax_image, cax=self._mpl_colorbar.ax)
        self._mpl_data_fig.redraw()
        self._mpl_colorbar.redraw()

    def _plot_covariance(self):
        """
        TODO
        """
        vis.web_plot_covariance(self._current_data, self._mpl_data_fig.ax, colorbar_axis=self._mpl_colorbar.ax, axis=0)
        # self._mpl_data_fig.ax.set_title('Covariance')

    def _subset_select_button_clicked(self, event_emitter):
        """
        This listener function defines the action when the data subset select button is clicked.
        :param event_emitter: The source of the event. In this case, it should be the _subset_select_button.
        """

        assert event_emitter is self._subset_select_button, 'Event emitter was not the expected select button.'
        self._update_current_data(self._subset_dropdown.get_key())  # This updates plot as well.

    def _plot_select_button_clicked(self, event_emitter):
        """
        This listener function defines the action when the plot select button is clicked.
        :param event_emitter: The source of the event. In this case, it should be the _plot_select_button.
        :return:
        """

        assert event_emitter is self._plot_select_button, 'Event emitter was not the expected select button.'
        self._update_plot(self._plot_dropdown.get_key())


def nav_bar(headings_list, active_index=0):
    """
    This function creates a row container with the headings provided in headings_list.
    :param headings_list: The list of headings, provided as a list of strings.
    :param active_index:
    :return: A remi.gui.HBox representing the navigation bar.
    """

    def nav_bar_heading(text):
        """
        This function wraps a remi.gui.Label with appropriate style settings to define navigation bar heading.
        :param text: The text of the heading.
        :return: A remi.gui.Label with the provided text and style settings for navigation bar heading.
        """

        return gui.Label(text=text, style={'font-weight': '250', 'line-height': '1.1', 'color': '#FFFFFF',
                                           'font-size': '18px', 'margin-top': '12px', 'margin-bottom': '12px',
                                           'margin-left': '10px', 'margin-right': '10px', 'text-shadow': 'none',
                                           'text-align': 'left'})

    def nav_bar_active_heading(text):
        """
        This function wraps a remi.gui.Label with appropriate style settings to define navigation bar active heading.
        :param text: The text of the heading.
        :return: A remi.gui.Label with the provided text and style settings for navigation bar active heading.
        """

        return gui.Label(text=text, style={'font-weight': '550', 'line-height': '1.1', 'color': '#FFFFFF',
                                           'font-size': '18px', 'margin-top': '12px', 'margin-bottom': '12px',
                                           'margin-left': '10px', 'margin-right': '10px', 'text-shadow': 'none',
                                           'text-align': 'left'})

    # Validate input.
    assert isinstance(headings_list, list), 'Navigation bar headings must be provided as a list.'
    assert 0 <= active_index < len(headings_list), 'Active index must between 0 and {}'.format(len(headings_list))

    bar = gui.HBox(width='100%', margin='0px', style={'text-align': 'left', 'background-color': UOB_BLUE, 'align-content': 'left'})
    # bar = gui.Widget(width='100%', layout_orientation=gui.Widget.LAYOUT_HORIZONTAL, margin='0px',
    #                 style={'text-align': 'left', 'background-color': UOB_BLUE, 'align-content': 'left'})
    for index in range(len(headings_list)):
        if index == active_index:
            bar.append(nav_bar_active_heading(headings_list[index]))
        else:
            bar.append(nav_bar_heading(headings_list[index]))

    # row_container = gui.Widget(width='50%', layout_orientation=gui.Widget.LAYOUT_HORIZONTAL, margin='auto',
    #                            style={'display': 'block', 'overflow': 'auto', 'align-content': 'center',
    #                                   'margin-top': '50px'})

    return bar


def footer(text):
    """
    This function creates a row container with the headings provided in headings_list.
    :param text: The text of the footer.
    :return: A remi.gui.HBox representing the footer.
    """

    footer_box = gui.HBox(width='100%', margin='0px', style={'text-align': 'center', 'background-color': UOB_GREY,
                                                             'vertical-align': 'bottom', 'bottom': '0%',
                                                             'position': 'fixed'})

    footer_text = gui.Label(text=text, style={'font-weight': '1500', 'line-height': '1.1', 'color': '#FFFFFF',
                                              'font-size': '14px', 'margin-top': '15px', 'margin-bottom': '15px',
                                              'margin-left': '10px', 'margin-right': '10px', 'text-shadow': 'none',
                                              'text-align': 'center'})

    footer_box.append(footer_text)

    return footer_box


def h1(text):
    """
    This function wraps a remi.gui.Label with appropriate style settings to define Heading 1.
    :param text: The text of the heading.
    :return: A remi.gui.Label with the provided text and style settings for a Heading 1.
    """

    # # Could also do.
    # heading = gui.Label(text=text)
    # heading.add_class('h1')

    return gui.Label(text=text, style={'font-weight': '500', 'line-height': '1.1', 'color': UOB_GREY,
                                       'font-size': '47px', 'margin-top': '25px', 'margin-bottom': '11px',
                                       'margin-left': '10px', 'margin-right': '10px', 'text-shadow': 'none'})


def h2(text):
    """
    This function wraps a remi.gui.Label with appropriate style settings to define Heading 2.
    :param text: The text of the heading.
    :return: A remi.gui.Label with the provided text and style settings for a Heading 2.
    """

    return gui.Label(text=text, style={'font-weight': '500', 'line-height': '1.1', 'color': UOB_GREY,
                                       'font-size': '35px', 'margin-top': '25px', 'margin-bottom': '11px',
                                       'margin-left': '10px', 'margin-right': '10px', 'text-shadow': 'none'})


def h3(text):
    """
    This function wraps a remi.gui.Label with appropriate style settings to define Heading 3.
    :param text: The text of the heading.
    :return: A remi.gui.Label with the provided text and style settings for a Heading 3.
    """

    return gui.Label(text=text, style={'font-weight': '500', 'line-height': '1.1', 'color': UOB_GREY,
                                       'font-size': '27px', 'margin-top': '25px', 'margin-bottom': '11px',
                                       'margin-left': '10px', 'margin-right': '10px', 'text-shadow': 'none'})


def h4(text):
    """
    This function wraps a remi.gui.Label with appropriate style settings to define Heading 4.
    :param text: The text of the heading.
    :return: A remi.gui.Label with the provided text and style settings for a Heading 4.
    """

    return gui.Label(text=text, style={'font-weight': '500', 'line-height': '1.1', 'color': UOB_GREY,
                                       'font-size': '20px', 'margin-top': '14px', 'margin-bottom': '11px',
                                       'margin-left': '10px', 'margin-right': '10px', 'text-shadow': 'none'})


def h5(text):
    """
    This function wraps a remi.gui.Label with appropriate style settings to define Heading 5.
    :param text: The text of the heading.
    :return: A remi.gui.Label with the provided text and style settings for a Heading 5.
    """

    return gui.Label(text=text, style={'font-weight': '500', 'line-height': '1.1', 'color': UOB_GREY,
                                       'font-size': '15px', 'margin-top': '14px', 'margin-bottom': '11px',
                                       'margin-left': '10px', 'margin-right': '10px', 'text-shadow': 'none'})


def h6(text):
    """
    This function wraps a remi.gui.Label with appropriate style settings to define Heading 6.
    :param text: The text of the heading.
    :return: A remi.gui.Label with the provided text and style settings for a Heading 6.
    """

    return gui.Label(text=text, style={'font-weight': '500', 'line-height': '1.1', 'color': UOB_GREY,
                                       'font-size': '13px', 'margin-top': '14px', 'margin-bottom': '11px',
                                       'margin-left': '10px', 'margin-right': '10px', 'text-shadow': 'none'})
