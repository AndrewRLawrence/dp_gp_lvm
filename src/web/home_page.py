"""
This module defines the web GUI for the home page of the repo to be able to navigate the different models, data sets,
experiments, and results.
"""

from src.data_io.data_set_reader import read_data_set_file
from src.utils.constants import UOB_GREY, UOB_BLUE
import src.web.utils.web_components as wc

from multiprocessing.pool import ThreadPool
import numpy as np
from os.path import isfile
import remi.gui as gui
from remi import App, start


class HomePage(App):
    """
    This class defines the layout and behaviour of the home page for the web GUI of the repo.
    """

    def __init__(self, *args):
        # The version of remi installed in the virtual environment uses a custom CSS file based off University of Bath
        #  colour-scheme.

        # TODO: Make these attributes private, i.e., refactor to _name.

        # Define thread pool for data set reading and for trained model reading and graph construction.
        self.thread_pool = ThreadPool(processes=2)  # TODO: Determine what is a good number of processes.
        self.async_data_set_reader = None
        self.async_model_reader_builder = None

        # Define lists of different models and data sets in repo.
        self.models = ['DP-GP-LVM', 'Bayesian GP-LVM', 'MRD', 'Fully Independent MRD', 'GP-LVM']
        self.data_sets = ['Synthetic Data', 'PoseTrack', 'CMU MOCAP', 'Horse MOCAP', 'MNIST Digits',
                          'Skin Cancer MNIST', 'Colorectal Histology MNIST', 'Frey Faces']

        # Define placeholder for selected and loaded data set model.
        self._selected_dataset = None
        self._data_set = None
        self._data_set_analysis = None
        self._selected_model = None
        self._model = None

        # Define list of different headings for navigation bar for web interface.
        self.nav_bar_headings = ['Home', 'Models', 'Data Sets', 'Experiments', 'Analysis', 'About']
        self.num_pages = len(self.nav_bar_headings)
        self.nav_bar = wc.NavigationBar(headings_list=self.nav_bar_headings,
                                        active_index=0,
                                        click_func=self.nav_bar_clicked)
        self.footer = wc.footer('© Andrew R. Lawrence, 2019. All rights reserved.')
        super(HomePage, self).__init__(*args)

    def _repo_page(self):

        # Define root widget.
        main_container = gui.Widget(width='100%', style={'vertical-align': 'top', 'align-content': 'center'})

        # Define heading text.
        title = wc.h1('DP-GP-LVM')
        title.style['text-align'] = 'center'
        description = wc.h6('This repository tracks a Python 3 Bayesian non-parametric library developed using '
                            'TensorFlow. This repo includes various models: GPs, DPs, GP-LVMs, MRD, and DP-GP-LVM, as '
                            'well as various experiments with these models using synthetic and real-world data sets. '
                            'TensorFlow was chosen as the preferred machine learning library as it is open source and '
                            'heavily supported. Additionally, TensorFlow supports numerous platforms and features '
                            'automatic differentiation to calculate gradients throughout the graph.')
        description.style['text-align'] = 'justify-all'
        instructions = wc.h4('There are a few ways to explore this repository. The user can start by choosing a data '
                             'set or a model on their respective pages, which can be accessed using the navigation '
                             'bar at the top of every page. Further instructions are provided on the Data Sets and '
                             'Models pages.')
         # 'Choose a data set from the pulldown menu before loading/defining the appropriate model. '
         # 'Afterwards, the user can set experiment and model parameters, load results or load a '
         # 'trained model, or choose to train a new model.')
        instructions.style['text-align'] = 'justify-all'

        # Define close button.
        close_container = gui.HBox(width='50%', margin='auto',
                                   style={'display': 'block', 'overflow': 'auto', 'align-content': 'center',
                                          'margin-top': '100px', 'margin-bottom': '50px'})
        self.close_button = gui.Button('Close Application', width='200px',
                                       style={'box-shadow': 'none', 'font-weight': '550', 'font-size': '20px'})
        self.close_button.add_class('btn btn-outline-danger')
        # self.close_button.add_class('btn btn-danger')
        self.close_button.onclick.connect(self.close_button_clicked)  # Listener function for mouse click.
        close_container.append(self.close_button)

        # Build up the webpage.
        main_container.append([self.nav_bar, title, description, instructions, close_container, self.footer])

        # Return the root widget.
        return main_container

    def _model_page(self):

        # Define root widget.
        main_container = gui.Widget(width='100%', style={'vertical-align': 'top', 'align-content': 'center'})

        # Define title and instructions text.
        models_title = wc.h2('Bayesian Non-Parametric Models')
        models_title.style['text-align'] = 'center'
        models_instructions = wc.h4('Choose a model from the pulldown menu. Afterwards, the user can choose to load a '
                                    'existing model trained for a specific data set to visualise results and generate '
                                    'data from the model. Or the user can train a new model by (1) setting model '
                                    'hyperparameters, (2) choosing a data set, and (3) setting experiment parameters.')
        models_instructions.style['text-align'] = 'justify-all'

        # Define row container for model selection.
        selection_container = gui.HBox(width='50%', margin='auto',
                                       style={'display': 'block', 'overflow': 'auto',
                                              'align-content': 'center', 'margin-top': '50px'})

        model_label = wc.h5('Model: ')
        model_label.style['margin-top'] = '8px'

        # Define drop down with different models.
        self.model_dropdown = gui.DropDown(width='250px', style={'margin-right': '15px'})
        self.model_dropdown.add_class("form-control dropdown")
        self.model_dropdown.append(dict(zip(self.models,
                                            [gui.DropDownItem(model_str) for model_str in self.models])))

        self.model_select_button = gui.Button('Select', width='100px', style={'box-shadow': 'none'})
        self.model_select_button.add_class('btn btn-primary')
        self.model_select_button.onclick.connect(self.select_button_clicked)  # Listener function for mouse click.

        # Build up the webpage.
        selection_container.append([model_label, self.model_dropdown, self.model_select_button])
        main_container.append([self.nav_bar, models_title, models_instructions, selection_container, self.footer])

        # Return the root widget.
        return main_container

    def _data_set_page(self):

        # Define root widget.
        main_container = gui.Widget(width='100%', style={'vertical-align': 'top', 'align-content': 'center'})

        # Define title and instructions text.
        datasets_title = wc.h2('Data Sets')
        datasets_title.style['text-align'] = 'center'
        datasets_instructions = wc.h4('Choose a data set from the pulldown menu before loading/defining the '
                                      'appropriate model. Afterwards, the user is presented with information about the '
                                      'data set, e.g., number of dimensions and number of observations. The user can '
                                      'also visualise the data. Then, with a loaded data set, the user can (1) set '
                                      'experiment parameters and set model hyperparameters, (2a) load results or load '
                                      'a trained model, or (2b) choose to train a new model.')
        datasets_instructions.style['text-align'] = 'justify-all'

        # Define row container for data set selection.
        selection_container = gui.HBox(width='50%', margin='auto',
                                       style={'display': 'block', 'overflow': 'auto',
                                              'align-content': 'center', 'margin-top': '50px'})

        dataset_label = wc.h5('Data Set: ')
        dataset_label.style['margin-top'] = '8px'

        # Define drop down with different data sets.
        self.dataset_dropdown = gui.DropDown(width='250px', style={'margin-right': '15px'})
        self.dataset_dropdown.add_class("form-control dropdown")
        self.dataset_dropdown.append(dict(zip(self.data_sets,
                                              [gui.DropDownItem(dataset_str) for dataset_str in self.data_sets])))

        self.dataset_select_button = gui.Button('Select', width='100px', style={'box-shadow': 'none'})
        self.dataset_select_button.add_class('btn btn-primary')
        self.dataset_select_button.onclick.connect(self.select_button_clicked)  # Listener function for mouse click.

        # Build up the webpage.
        selection_container.append([dataset_label, self.dataset_dropdown, self.dataset_select_button])
        main_container.append([self.nav_bar, datasets_title, datasets_instructions, selection_container, self.footer])

        # Return the root widget.
        return main_container

    def _experiment_page(self):

        # Define root widget.
        main_container = gui.Widget(width='100%', style={'vertical-align': 'top', 'align-content': 'center'})

        # Build up the webpage.
        main_container.append([self.nav_bar, self.footer])

        # Return the root widget.
        return main_container

    def _analysis_page(self):
        """
        This function defines the structure of the analysis page of the web application.
        :return: The root remi.gui.Widget object for the analysis page.
        """

        # Define root widget.
        main_container = gui.Widget(width='100%', style={'vertical-align': 'top', 'align-content': 'center'})

        # Define title and instructions text.
        analysis_title = wc.h2('Analysis')
        analysis_title.style['text-align'] = 'center'
        analysis_instructions = wc.h4('Once a model and/or data set has been loaded. This page can be used to see model'
                                      'hyperparameters and data set properties (such as number of dimensions), and '
                                      'to visualise the data and model latent spaces and hyperparameters.')
        analysis_instructions.style['text-align'] = 'justify-all'

        # # Add MatplotlibFigure widget.
        # self.mpl = wc.MatplotlibFigure(width=250, height=250)
        # self.mpl.ax.set_title('Test')
        # self.mpl.ax.scatter(np.linspace(0, 1, 25), np.random.standard_normal(25))
        # self.mpl.redraw()

        # Build up the webpage.
        main_container.append([self.nav_bar, analysis_title, analysis_instructions, self.footer])

        # Return the root widget.
        return main_container

    def _about_page(self):
        """
        This function defines the structure of the about page of the web application.
        :return: The root remi.gui.Widget object for the about page.
        """

        # Define root widget.
        main_container = gui.Widget(width='100%', style={'vertical-align': 'top', 'align-content': 'center'})

        # Define authors section.
        authors_title = wc.h2('Authors')
        authors_title.style['text-align'] = 'center'

        # Center University of Bath logo.
        uob_container = gui.HBox(width='50%', margin='auto',
                                 style={'display': 'block', 'overflow': 'auto', 'align-content': 'center'})
        uob_container.append(gui.Image('http://127.0.0.1:8000/web/assets/uob-logo-grey-transparent.png', height='75px'))

        # Define abstract section.
        abstract_title = wc.h2('Abstract')
        abstract_title.style['text-align'] = 'center'
        abstract = wc.h6('This repository contains the inplementation of our non-parametric Bayesian latent variable '
                         'model (DP-GP-LVM) capable of learning dependency structures across dimensions in a '
                         'multivariate setting. '
                         'Our approach is based on flexible Gaussian process priors for the generative mappings and '
                         'interchangeable Dirichlet process priors to learn the structure. The introduction of the '
                         'Dirichlet process as a specific structural prior allows our model to circumvent issues '
                         'associated with previous Gaussian process latent variable models. Inference is performed by '
                         'deriving an efficient variational bound on the marginal log-likelihood of the model. '
                         'We demonstrate the efficacy of our approach via analysis of discovered structure and '
                         'superior quantitative performance on missing data imputation. These experiments can also be '
                         'found in this repository.')
        abstract.style['text-align'] = 'justify-all'

        # TODO: Add graphical model.

        # TODO: Add references for our paper(s) and maybe other papers for other models and data sets.

        # Define acknowledgements section.
        acknowledgements_title = wc.h2('Acknowledgements')
        acknowledgements_title.style['text-align'] = 'center'
        acknowledgements = wc.h6('TODO')
        acknowledgements.style['text-align'] = 'justify-all'

        # Center EU, CAMERA, and CDE logos.
        ack_logos_container = gui.HBox(width='50%', margin='auto',
                                       style={'display': 'block', 'overflow': 'auto', 'align-content': 'center'})
        ack_logos_container.append([gui.Image('http://127.0.0.1:8000/web/assets/logo_ce-en-rvb-hr.jpg',
                                              height='50px', margin='20px'),
                                    gui.Image('http://127.0.0.1:8000/web/assets/CAMERA_logo.png',
                                              height='50px', margin='20px'),
                                    gui.Image('http://127.0.0.1:8000/web/assets/cde_logo.png',
                                              height='50px', margin='20px')])

        # Build up the webpage.
        main_container.append([self.nav_bar, authors_title, uob_container, abstract_title, abstract,
                               acknowledgements_title, acknowledgements, ack_logos_container, self.footer])

        # Return the root widget.
        return main_container

    def _closing_page(self):
        """
        This function defines the structure of the closing page of the web application.
        :return: The root remi.gui.Widget object for the closing page.
        """

        # Define root widget.
        main_container = gui.Widget(width='100%', height='100%', margin='0px',
                                    style={'vertical-align': 'middle', 'background-color': UOB_BLUE,
                                           'position': 'fixed'})

        closing_label = gui.Label('Closing Web Application',
                                  style={'font-weight': '500', 'color': '#FFFFFF', 'font-size': '50px',
                                         'text-shadow': 'none', 'position': 'absolute', 'top': '0px', 'bottom': '0px',
                                         'left': '0px', 'right': '0px', 'text-align': 'center', 'height': '25%',
                                         'margin': 'auto'})
        closing_instructions = gui.Label('See the terminal for final messages and confirmation that the application '
                                         'closed successfully.',
                                         style={'font-weight': '500', 'color': '#FFFFFF', 'font-size': '15px',
                                                'text-shadow': 'none', 'position': 'absolute', 'top': '0px',
                                                'bottom': '0px', 'left': '0px', 'right': '0px', 'text-align': 'center',
                                                'height': '5%', 'margin': 'auto'})

        # Build up the webpage.
        main_container.append([closing_label, closing_instructions, self.footer])

        # Return the root widget.
        return main_container

    def idle(self):
        """
        This function defines the idle loop for the web application, which runs every update call. It checks to see
        if the functions run in the thread pool have completed.
        """

        # Update analysis page once data set has been loaded.
        if self._data_set is None and self.async_data_set_reader is not None:
            if self.async_data_set_reader.ready():
                print('Data set has been read.')
                self._data_set = self.async_data_set_reader.get()

                # Instantiate a DataSetAnalysis GUI widget object and add to analysis page.
                self._data_set_analysis = wc.DataSetAnalysis(self._data_set, self._selected_dataset)
                self.pages.get(self.nav_bar_headings[4]).append(self._data_set_analysis)
            else:
                print('Data not ready, still reading data set.')
        # No else needed as data set has already been loaded.

        # Update analysis page once model has been loaded and TensorFlow graph has been constructed.
        if self._model is None and self.async_model_reader_builder is not None:
            if self.async_model_reader_builder.ready():
                pass
            else:
                pass
        # No else needed as model has already been loaded and TensorFlow graph has been constructed.

        # # If necessary, update MatplotlibFigure in data set analysis section.
        # if self._data_set_analysis is not None:
        #     # if self._data_set_analysis.update_ready:
        #     if self._data_set_analysis.async_plotter.ready():
        #         # Update figure.
        #         self._data_set_analysis.update_plot()
        # # No else needed as plot does not exist yet or is up-to-date.

        # # Update MatplotlibFigure widget.
        # self.mpl.ax.clear()
        # self.mpl.ax.set_title('Test')
        # self.mpl.ax.scatter(np.linspace(0, 1, 25), np.random.standard_normal(25))
        # self.mpl.redraw()

    def main(self):
        """
        This function defines the main loop for the web application. It defines the layout of the web page.
        """

        # Define each page.
        self.pages = dict(zip(self.nav_bar_headings,
                              [self._repo_page(), self._model_page(), self._data_set_page(),
                               self._experiment_page(), self._analysis_page(), self._about_page()]))

        # Return the root widget, which is first page.
        return self.pages.get(self.nav_bar_headings[0])

    def on_close(self):
        """
        This function overrides the base class on_close function and defines actions to perform before terminating.
        """
        print('Closing web application.')
        # Terminate the worker processes immediately without completing outstanding work.
        self.thread_pool.terminate()
        self.thread_pool.join()
        super(HomePage, self).on_close()

    def set_current_page(self, page_index):
        """
        This function updates the root widget to that of the specified page_index.
        :param page_index: Index of page to be set as new root.
        """

        # Validate input then set new root widget.
        assert 0 <= page_index < self.num_pages, 'Page index must between 0 and {}'.format(self.num_pages)
        self.nav_bar.set_active_index(new_active_index=page_index)
        self.set_root_widget(self.pages.get(self.nav_bar_headings[page_index]))

    def nav_bar_clicked(self, event_emitter):
        """
        This listener function defines the action when one of the labels in the navigation bar is clicked.
        :param event_emitter: The source of the event. In this case, it is one of the labels in the navigation bar.
        """

        # Determine text of event emitter, i.e., the label that was clicked.
        assert isinstance(event_emitter, gui.Label), 'Event emitter should be a remi.gui.Label.'
        label_text = event_emitter.get_text()
        assert label_text in self.nav_bar_headings, 'Text of label is not in navigation bar.'

        self.set_current_page(page_index=self.nav_bar_headings.index(label_text))

    def close_button_clicked(self, event_emitter):
        """
        This listener function defines the action when the close button is clicked.
        :param event_emitter: The source of the event. In this case, it should be the close_button.
        """

        assert event_emitter is self.close_button, 'Event emitter was not the expected close button.'
        self.set_root_widget(self._closing_page())
        print('Close button was clicked.')
        self.close()  # Closes the application.

    def select_button_clicked(self, event_emitter):
        """
        This listener function defines the action when the select button is clicked.
        :param event_emitter: The source of the event. In this case, it should be the select_button.
        """

        # Determine which select button was clicked.
        if event_emitter is self.model_select_button:

            # Disable model select button and pulldown menu.
            self.model_dropdown.attributes['disabled'] = 'true'
            self.model_select_button.attributes['disabled'] = 'true'
            self._selected_model = self.model_dropdown.get_key()

            print('Model selection button was clicked with {} selected in model dropdown.'.format(self._selected_model))

            # Define vertical container for loading model.
            trained_model_container = gui.VBox(width='50%', margin='auto',
                                               style={'display': 'block', 'overflow': 'auto',
                                                      'align-content': 'center', 'margin-top': '40px'})

            trained_model_title = wc.h3('Load Trained {}'.format(self._selected_model))
            trained_model_title.style['text-align'] = 'center'
            trained_model_instructions = wc.h5('Click button to load model file.')  # TODO: Finish these instructions.
            trained_model_instructions.style['text-align'] = 'justify-all'

            # Update model page to allow for loading trained model.
            self.model_file_button = gui.Button('Load Trained Model', width='150px', style={'box-shadow': 'none'})
            self.model_file_button.add_class('btn btn-primary')
            self.model_file_button.onclick.connect(self.open_file_selection_dialog)  # Listener function.

            trained_model_container.append([trained_model_title, trained_model_instructions, self.model_file_button])

            # TODO: Add ability to define specify model hyperparameters to train new one.
            new_model_container = gui.VBox(width='50%', margin='auto',
                                           style={'display': 'block', 'overflow': 'auto',
                                                  'align-content': 'center', 'margin-top': '40px'})

            new_model_title = wc.h3('Specify Hyperparameters for new instance of {}'.format(self._selected_model))
            new_model_title.style['text-align'] = 'center'
            new_model_instructions = wc.h5('Under Construction. Not implemented yet.')  # TODO: Implement and update.
            new_model_instructions.style['text-align'] = 'justify-all'

            new_model_container.append([new_model_title, new_model_instructions])

            # Add loading trained model section and specifying new model to models page.
            self.pages.get(self.nav_bar_headings[1]).append([trained_model_container, new_model_container])

        elif event_emitter is self.dataset_select_button:

            # Disable data set select button and pulldown menu.
            self.dataset_dropdown.attributes['disabled'] = 'true'
            self.dataset_select_button.attributes['disabled'] = 'true'
            self._selected_dataset = self.dataset_dropdown.get_key()

            print('Data set selection button was clicked with {} selected in data set dropdown.'.format(
                self._selected_dataset))

            # Define vertical container for loading data set.
            dataset_container = gui.VBox(width='50%', margin='auto',
                                               style={'display': 'block', 'overflow': 'auto',
                                                      'align-content': 'center', 'margin-top': '40px'})

            dataset_title = wc.h3('Load {} Data Set'.format(self._selected_dataset))
            dataset_title.style['text-align'] = 'center'
            dataset_instructions = wc.h5('Click button to load data set file.')  # TODO: Finish these instructions.
            dataset_instructions.style['text-align'] = 'justify-all'

            # Update data sets page to allow for loading data set.
            self.dataset_file_button = gui.Button('Load Data Set', width='150px', style={'box-shadow': 'none'})
            self.dataset_file_button.add_class('btn btn-primary')
            self.dataset_file_button.onclick.connect(self.open_file_selection_dialog)  # Listener function.

            dataset_container.append([dataset_title, dataset_instructions, self.dataset_file_button])

            self.pages.get(self.nav_bar_headings[2]).append(dataset_container)

        else:
            # Do nothing.
            pass

    def open_file_selection_dialog(self, event_emitter):
        """
        TODO
        :param event_emitter:
        :return:
        """

        selection_folder = None

        try:
            if event_emitter is self.model_file_button:
                print('Model file selection button was clicked to load trained {}.'.format(self._selected_model))
                subfolders = ['dp_gp_lvm', 'bgplvm', 'mrd', 'fi_mrd', 'gplvm']
                selection_folder = '../results/{}/'.format(subfolders[self.models.index(self._selected_model)])
        except AttributeError:
            # Do nothing as that button has not been defined yet.
            pass

        try:
            if event_emitter is self.dataset_file_button:
                print('Data set file selection button was clicked to load {} data set.'.format(self._selected_dataset))
                subfolders = [None, None, 'cmu_mocap', None, None, 'skin_cancer_mnist', None, 'frey_faces']
                selection_folder = './data_io/data_sets/{}/'.format(
                    subfolders[self.data_sets.index(self._selected_dataset)])
        except AttributeError:
            # Do nothing as that button has not been defined yet.
            pass

        if selection_folder is not None:

            try:
                # Open file selection dialog at appropriate folder.
                file_selection_dialog = gui.FileSelectionDialog(title='File Selection',
                                                                message='Select desired file.',
                                                                multiple_selection=False,
                                                                selection_folder=selection_folder,
                                                                allow_file_selection=True,
                                                                allow_folder_selection=False)

                # Reformat file selection dialog to make it look nicer.
                # file_selection_dialog.style['width'] = '700px'
                file_selection_keys = list(file_selection_dialog.children.keys())
                file_selection_dialog.get_child(file_selection_keys[0]).set_style(
                    {'font-weight': '500', 'line-height': '1.1', 'color': UOB_BLUE, 'font-size': '28px',
                     'margin-top': '20px', 'text-shadow': 'none'})
                file_selection_dialog.get_child(file_selection_keys[1]).set_style(
                    {'font-weight': '500', 'line-height': '1.1', 'color': UOB_GREY, 'font-size': '18px',
                     'margin-top': '10px', 'margin-bottom': '12px', 'text-shadow': 'none'})
                file_selection_dialog.conf.set_text('OK')
                file_selection_dialog.conf.set_size(100, 35)
                file_selection_dialog.conf.add_class('btn btn-primary')
                file_selection_dialog.cancel.set_text('Cancel')
                file_selection_dialog.cancel.set_size(100, 35)
                file_selection_dialog.cancel.add_class('btn btn-primary')

                file_selection_dialog.fileFolderNavigator.controlsContainer.set_size('100%', '50px')
                file_selection_dialog.fileFolderNavigator.controlBack.add_class('btn btn-primary')
                file_selection_dialog.fileFolderNavigator.controlGo.add_class('btn btn-primary')
                file_selection_dialog.fileFolderNavigator.controlGo.set_text('Go')

                # Disable text field and buttons to navigate folders; thereofre user is only allowed to choose files
                # in folder to which the dialog box opens.
                file_selection_dialog.fileFolderNavigator.pathEditor.attributes['disabled'] = 'true'
                file_selection_dialog.fileFolderNavigator.controlBack.attributes['disabled'] = 'true'
                file_selection_dialog.fileFolderNavigator.controlGo.attributes['disabled'] = 'true'

                file_selection_dialog.confirm_value.connect(self.on_file_selection_dialog_confirm)  # Listener function.
                file_selection_dialog.show(self)  # Show file selection dialog window.
            except FileNotFoundError as err:
                # Print error, but then do nothing as dialog window will not be shown.
                print("FileNotFoundError: {0}".format(err))
        else:
            # Do nothing. So ignore that call.
            pass

    def on_file_selection_dialog_confirm(self, event_emitter, filelist):
        """
        TODO
        :param event_emitter:
        :param filelist:
        :return:
        """

        # Confirm event emitter is of type remi.gui.FileSelectionDialog
        assert isinstance(event_emitter, gui.FileSelectionDialog), \
            'Event emitter was not the expected file selection dialog.'

        # Confirm filelist has only one file. The FileSelectionDialog initialised should not support multiple selection.
        assert len(filelist) == 1, 'Only one file should have been selected.'

        # Confirm file exists.
        selected_file = filelist[0]
        assert isfile(selected_file), 'Selected file does not exist.'

        if 'data_sets' in selected_file:
            self.dataset_file_button.attributes['disabled'] = 'true'
            self.async_data_set_reader = self.thread_pool.apply_async(func=np.load, args=(selected_file, ))
        else:
            # TODO: Add function to load file and build TF graph to thread pool.
            # self.async_model_reader_builder = None
            pass

        # Want to open file and go to appropriate tab.
        self.set_current_page(page_index=4)  # Go to analysis page. So user can define and view plots.


if __name__ == '__main__':

    # Create HTTP web server and start application.
    start(HomePage, debug=True, address='127.0.0.1', port=8085, start_browser=False,
          username=None, password=None, title='DP-GP-LVM', multiple_instance=False,
          update_interval=0.1)
