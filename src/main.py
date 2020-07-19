"""
This module defines the GUI and web server to navigate the different models, data sets, experiments, and results.
"""

import src.web.assets.host_images as hi
from src.web.home_page import HomePage

import argparse
# import matplotlib.pyplot as plot
# import mpld3
from remi import start


# class MyApp(App):
#     def __init__(self, *args):
#         super(MyApp, self).__init__(*args)
#
#         # import os
#         # local_str = os.path.dirname(os.path.abspath(__file__)) + '/web/'
#
#         # res_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'res')
#         # # static_file_path can be an array of strings allowing to define
#         # #  multiple resource path in where the resources will be placed
#         # super(MyApp, self).__init__(*args, static_file_path={'res': STYLE_PATH})
#         # super(MyApp, self).__init__(*args, static_file_path={'my_resources': local_str,
#         #                                                      'res': local_str})
#
#     def idle(self):
#         # idle loop, you can place here custom code
#         # avoid to use infinite iterations, it would stop gui update
#         pass
#
#     def main(self):
#
#         # Define custom HTML HEAD tags.
#         # Load Boostrap Resources from Online Source
#         # One could download the files and put them into res folder for access without internet connection
#
#         # Not all the Bootstrap functionality will work!! Just basic styling is possible.
#
#         # For valid Bootstrap Classes check:  https://www.w3schools.com/bootstrap/bootstrap_ref_all_classes.asp
#
#         web_link_str = 'https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css'
#         # web_link_str = 'http://127.0.0.1:8000/src/web/style.min.css'
#         import os
#         local_str = os.path.dirname(os.path.abspath(__file__)) + '/web/bootstrap.min.css'
#         # url_str = "url('/my_resources:subfolder/other_subfolder/image.png')"
#         # url_str = "url('{/my_sources:uob.css}')"
#         # url_str = 'file://{}'.format(local_str)
#         # local_str = './web/uob.css'
#         url_str = 'file://' + pathname2url(local_str)
#         # url_str = pathname2url(local_str)
#         print(url_str)
#
#         # Update page header to different style file.
#         print('{} is file: {}'.format(local_str, os.path.isfile(local_str)))
#         # self.page.children['head'].children['internal_css'] = '<link rel="stylesheet" href="{}">\n'.format(web_link_str)
#
#         print('%r' % self.page)
#
#         # self.page.children['head'].add_child('internal_css', "<link href='{}' rel='stylesheet' />\n".format(local_str))
#         # self.page.children['head'].add_child('internal_css', "<link href='{}' rel='stylesheet' />\n".format(web_link_str))
#
#         # self.page.children['head'].add_child('mycss', "<link href='{}' rel='stylesheet' />\n".format(web_link_str))
#         # self.page.children['head'].add_child('mycss', my_css_head)
#
#         # creating a container VBox type, vertical (you can use also HBox or Widget)
#         main_container = gui.VBox(width='200px', height='250px', style={'margin': '0px auto', 'padding': '10px'})
#
#         # Drop Down
#         self.dd = gui.DropDown(width='250px')
#         self.dd.style.update({'font-size': 'large'})
#         self.dd.add_class("form-control dropdown")
#         self.item1 = gui.DropDownItem("First Choice")
#         self.item2 = gui.DropDownItem("Second Item")
#         self.dd.append(self.item1, 'item1')
#         self.dd.append(self.item2, 'item2')
#
#         # Table
#         myList = [('ID', 'Lastname', 'Firstname', 'ZIP', 'City'),
#                   ('1', 'Pan', 'Peter', '99999', 'Neverland'),
#                   ('2', 'Sepp', 'Schmuck', '12345', 'Examplecity')]
#
#         self.tbl = gui.Table.new_from_list(content=myList, width='400px', height='100px', margin='10px')
#         self.tbl.add_class("table table-striped")
#
#         # Buttons
#
#         # btn adds basic design to a button like rounded corners and stuff
#         # btn-success, btn-danger and similar adds theming based on the function
#         # if you use btn-success without btn, the button will be standard, but with green background
#
#         self.bt1 = gui.Button("OK", width="50px")
#         self.bt1.add_class("btn btn-primary")  # Bootstrap Class:  btn-success
#         self.bt1.set_style({'box-shadow': 'none'})
#
#         self.bt2 = gui.Button("KO", width="50px")
#         self.bt2.add_class("btn btn-danger")  # Bootstrap Class:  btn btn-danger
#
#         # Build up the gui
#         main_container.append(self.bt1, 'btn1')
#         main_container.append(self.bt2, 'btn2')
#         main_container.append(self.tbl, 'tbl')
#         main_container.append(self.dd, 'dd')
#
#         # returning the root widget
#         return main_container


if __name__ == '__main__':

    # Get command line arguments.
    parser = argparse.ArgumentParser(description='Web interface for DP-GP-LVM repository.')
    parser.add_argument('--address', type=str, default='127.0.0.1',
                        help='The IPv4 address to which the web server should bind.')
    parser.add_argument('--port', type=int, default=8085,
                        help='The port number under which the application serves the content.')
    parser.add_argument('--username', type=str, default=None)
    parser.add_argument('--password', type=str, default=None)
    parser.add_argument('--debug', action='store_true', help='Debug flag to set level of log messages in terminal.')
    parser.add_argument('--start_browser', action='store_true')
    args = parser.parse_args()

    # TODO: Determine reasonable update interval and if we want multiple instances.

    # Run server on a new thread to host images.
    hi.run_on_new_thread()

    # Create HTTP web server and start HomePage app.
    start(HomePage, debug=args.debug, address=args.address, port=args.port, start_browser=args.start_browser,
          username=args.username, password=args.password, title='DP-GP-LVM', multiple_instance=False,
          update_interval=0.1)

    # # mpl3d example.
    # plot.plot([3, 1, 4, 1, 5], 'ks-', mec='w', mew=5, ms=20)
    # mpld3.show()
