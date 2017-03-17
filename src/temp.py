#!/usr/bin/python

import argparse
import cv2
import Image
import threading
import time
import StringIO
import sys

from flask import Flask, render_template, Response

capture = None # Global for capture device
last_received = None # Global for latest recived line

def receiving(capture):
    global last_received
    t = threading.currentThread() # Get current thread running function
    while getattr(t, "do_receive", True): # Watch for a stop signal
        retval = capture.grab()
        rc,last_received = capture.retrieve()

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

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def gen():
    while True:
        try:
            ret, jpeg = cv2.imencode('.jpg', last_received)
            frame = jpeg.tobytes()
            time.sleep(0.05)
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        except KeyboardInterrupt:
            break

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def main(argv = sys.argv):
    arg_parser = ArgParser(
        prog='openface_streamer',
        description='OpenFace Streamer')
    arg_parser.set()
    args, argv = arg_parser.parse_known_args(argv)

    global capture
    capture = cv2.VideoCapture(args.input)
    try:
        print "server started"
        t = threading.Thread(target=receiving, args=(capture,))
        t.start()
        print "capture started"
        app.run(host='0.0.0.0', debug=True)
    except KeyboardInterrupt:
        capture.release()
        t.do_receive = False
        t.join()

if __name__ == '__main__':
    main()
