#Script to output video with annotated frames.

#Takes as input a video file
#Outputs a video file with the annotated faces

import numpy as np
import cv2
import sys

from openface_healper import OpenFaceAnotater


def main(argv = sys.argv):
	cap = cv2.VideoCapture(argv[1])
	openface_anotater = OpenFaceAnotater(argv)
	bbs = []
	fourcc = cv2.cv.CV_FOURCC(*'DIVX')
	out = cv2.VideoWriter('/root/video/annotated.avi',fourcc, 25.0, (1280,720))

	while(cap.isOpened()):
		# Capture frame-by-frame
		ret, frame = cap.read()
		if ret==True:
			# Our operations on the frame come here
			bbs = []
			img = openface_anotater.predict(frame, bbs, multiple=True, scale=0.375)
			out.write(img)
			if cv2.waitKey(20) & 0xFF == ord('q'):
				break
		else:
			break

	# When everything done, release the capture
	cap.release()
	out.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()
