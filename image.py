import numpy as np
import cv2
import cv2.cv as cv
from matplotlib import pyplot as plt
import sys, getopt


def img_process(inputfile, basefile, outputfile):
	#find the background and set the alpha to be transparent
	image_base = cv2.imread(basefile, 1)
	image = cv2.imread(inputfile, 1)

	base_tmp = cv2.cvtColor(image_base, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(base_tmp, (5,5), 0)
	circles = cv2.HoughCircles(blur, cv.CV_HOUGH_GRADIENT, 1, 10,
								param1=200, param2=100, minRadius=300, maxRadius=0)
	circles = np.uint16(np.around(circles))
	max_radius = 0
	max_center_x = 0
	max_center_y = 0
	for i in circles[0,:]:
		if i[2] > max_radius:
			max_radius = i[2]
			max_center_x = i[0]
			max_center_y = i[1]
		# draw the outer circle
	cv2.circle(image_base,(max_center_x,max_center_y),max_radius,(0,255,0),2)
	# dst = image - image_base
	# cv2.imshow('image', image_base)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	cv2.imwrite(outputfile, image_base)


def main(argv):
   inputfile = ''
   outputfile = ''
   try:
      opts, args = getopt.getopt(argv,"hi:o:b:",["ifile=","ofile=","bfile="])
   except getopt.GetoptError:
      print 'image.py -i <inputfile> -o <outputfile> -b<basefile>'
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print 'image.py -i <inputfile> -o <outputfile> -b<basefile>'
         sys.exit()
      elif opt in ("-i", "--ifile"):
        inputfile = arg
      elif opt in ("-b", "--bfile"):
        basefile = arg
      elif opt in ("-o", "--ofile"):
         outputfile = arg
   print 'Input file is ', inputfile
   print 'Output file is ', outputfile
   img_process(inputfile, basefile, outputfile)

if __name__ == "__main__":
   main(sys.argv[1:])