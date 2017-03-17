#!/usr/bin/python

import argparse
import cv2
import Image
import threading
import time
import StringIO
import sys

from BaseHTTPServer import BaseHTTPRequestHandler,HTTPServer
from SocketServer import ThreadingMixIn

capture = None # Global for capture device
last_received = None # Global for latest recived line

def receiving(capture):
    global last_received
    t = threading.currentThread() # Get current thread running function
    while getattr(t, "do_receive", True): # Watch for a stop signal
        retval = capture.grab()
        rc,last_received = capture.retrieve()

class StreamHandler(BaseHTTPRequestHandler):
    """Handle stream connection."""
    def mjpg_cb(self):
        self.send_response(200)
        self.send_header('Pragma:', 'no-cache')
        self.send_header('Cache-Control:', 'no-cache')
        self.send_header('Content-Encoding:', 'identify')
        self.send_header('Content-type','multipart/x-mixed-replace; boundary=--jpgboundary')
        self.end_headers()
        while True:
            try:
                # rc,img = capture.read()
                # retval = capture.grab()
                # rc,img = capture.retrieve()
                # if not rc:
                #     continue
                img = last_received
                imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                jpg = Image.fromarray(imgRGB)
                tmpFile = StringIO.StringIO()
                jpg.save(tmpFile,'JPEG')
                self.wfile.write("--jpgboundary")
                self.send_header('Content-type','image/jpeg')
                self.send_header('Content-length',str(tmpFile.len))
                self.end_headers()
                jpg.save(self.wfile,'JPEG')
                time.sleep(0.05)
            except KeyboardInterrupt:
                break

    def do_GET(self):
        if self.path.endswith('.mjpg'):
            self.mjpg_cb()

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""

class ArgParser(argparse.ArgumentParser):
    """Argument parser class"""

    def set(self):
        """Setup parser for sroscore"""

        self.add_argument(
            '-i', '--input',
            action='store',
            default="http://172.17.0.1:8080/stream?topic=/usb_cam/image_raw",
            help='path to input video (URL_STEAM|FILE_PATH)')
        self.add_argument(
            '-p', '--port',
            action='store',
            type=int,
            default=8081,
            help='path to input video (URL_STEAM|FILE_PATH)')
        self.add_argument(
            '-a', '--address',
            action='store',
            default="localhost",
            help='path to input video (URL_STEAM|FILE_PATH)')
        self.add_argument(
            '--version',
            action='version',
            version='%(prog)s 0.0')

def main(argv = sys.argv):
    arg_parser = ArgParser(
        prog='openface_streamer',
        description='OpenFace Streamer')
    arg_parser.set()
    args, argv = arg_parser.parse_known_args(argv)

    global capture
    capture = cv2.VideoCapture(args.input)
    try:
        server = ThreadedHTTPServer((args.address, args.port), StreamHandler)
        print "server started"
        t = threading.Thread(target=receiving, args=(capture,))
        t.start()
        print "capture started"
        server.serve_forever()
    except KeyboardInterrupt:
        capture.release()
        server.socket.close()
        t.do_receive = False
        t.join()

if __name__ == '__main__':
    main()
