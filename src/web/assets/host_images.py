"""
This module serves the images in this directory.
"""

import http.server
from socketserver import TCPServer
import threading


def run_on_new_thread():
    """
    This function creates and runs an HTTP server on a new thread to host images in this directory.
    """
    server_address = '127.0.0.1'
    port_number = 8000

    TCPServer.allow_reuse_address = True  # So two servers can be running simultaneously.
    httpd = TCPServer((server_address, port_number), http.server.SimpleHTTPRequestHandler)

    thread = threading.Thread(target=httpd.serve_forever)
    thread.daemon = True
    try:
        print('Serving images at {}:{}.'.format(server_address, port_number))
        thread.start()
    except KeyboardInterrupt:
        httpd.shutdown()
        httpd.socket.close()
