#!/usr/bin/python

import cv2
import Image
import threading
from BaseHTTPServer import BaseHTTPRequestHandler,HTTPServer
from SocketServer import ThreadingMixIn
import StringIO
import time
capture=None

last_received = None # Global for latest recived line

def receiving(capture):
    global last_received

    t = threading.currentThread() # Get current thread running function
    while getattr(t, "do_receive", True): # Watch for a stop signal
        retval = capture.grab()
        rc,last_received = capture.retrieve()

class CamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.endswith('.mjpg'):
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
                        # continue
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
            return
        if self.path.endswith('.html'):
            self.send_response(200)
            self.send_header('Content-type','text/html')
            self.end_headers()
            self.wfile.write('<html><head></head><body>')
            self.wfile.write('<img src="http://127.0.0.1:8080/cam.mjpg"/>')
            self.wfile.write('</body></html>')
            return


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""

def main():
    global capture
    capture = cv2.VideoCapture('http://172.17.0.1:8080/stream?topic=/usb_cam/image_raw')
    # capture.set(cv2.CAP_PROP_BUFFERSIZE, 3)
    # capture = cv2.VideoCapture(0)
    # capture.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 320);
    # capture.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 240);
    # capture.set(cv2.cv.CV_CAP_PROP_SATURATION,0.2);
    global img
    try:
        server = ThreadedHTTPServer(('localhost', 8081), CamHandler)
        print "server started"
        t = threading.Thread(target=receiving, args=(capture,))
        t.start()
        server.serve_forever()
    except KeyboardInterrupt:
        capture.release()
        server.socket.close()
        t.do_receive = False
        t.join()

if __name__ == '__main__':
    main()


# while True:
#     ret, frame = cap.read()
#     cv2.imshow('Video', frame)
#
#     if cv2.waitKey(1) == 27:
#         exit(0)
