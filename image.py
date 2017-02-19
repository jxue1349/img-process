import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys, getopt

# def img_process(inputfile, basefile, outputfile):
# 	#find the background and set the alpha to be transparent
# 	template = cv2.imread(basefile, 0)
# 	img = cv2.imread(inputfile, 0)
# 	img = cv2.medianBlur(img,5)
# 	template = cv2.medianBlur(template,5)
# 	cimg = img - template

# 	cv2.imshow('detected circles',cimg)
# 	cv2.waitKey(0)
# 	cv2.destroyAllWindows()
# 	cv2.imwrite(outputfile, cimg)




def img_process(inputfile, basefile, outputfile):
	#find the background and set the alpha to be transparent
	image_base = cv2.imread(basefile, 1)
	image = cv2.imread(inputfile, 1)

	base_tmp = cv2.cvtColor(image_base, cv2.COLOR_BGR2GRAY)
	# blur = cv2.GaussianBlur(base_tmp, (5,5), 0)
	blur = cv2.blur(base_tmp,(100,100))
	thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)[1]
	cv2.imshow('thresh', thresh)
	cv2.namedWindow('thresh',cv2.WINDOW_NORMAL)
	cv2.resizeWindow('thresh', 60, 60)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 1, 10, param1=100, param2=20, minRadius=300, maxRadius=0)
	circles = np.uint16(np.around(circles))
	max_radius = 0
	max_center_x = 0
	max_center_y = 0
	xc=0
	yc=0
	cc=0
	for i in circles[0,:]:
		# cv2.circle(thresh,(i[0],i[1]),i[2],(0,255,0),2)
		xc+=i[0]
		yc+=i[1]
		cc+=1
		if i[2] > max_radius:
			max_radius = i[2]
			# max_center_x = i[0]
			# max_center_y = i[1]
		# draw the outer circle
	cv2.circle(image_base,(int(xc/cc),int(yc/cc)),max_radius+100,(0,255,0),2)
	cv2.imshow('image', image_base)
	cv2.namedWindow('image',cv2.WINDOW_NORMAL)
	cv2.resizeWindow('image', 60,60)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	# cv2.imwrite(outputfile, image_base)



# def img_process(inputfile, basefile, outputfile):
# 	template = cv2.imread(basefile, 0)
# 	img = cv2.imread(inputfile, 0)

# 	img2 = img.copy()
# 	w, h = template.shape[::-1]
# 	# All the 6 methods for comparison in a list
# 	methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR','cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
# 	# methods = ['cv2.TM_SQDIFF_NORMED']
# 	for meth in methods:
# 	    img = img2.copy()
# 	    method = eval(meth)
# 	    # Apply template Matching
# 	    res = cv2.matchTemplate(img,template,method)
# 	    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
# 	    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
# 	    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
# 	        top_left = min_loc
# 	    else:
# 	        top_left = max_loc
# 	    bottom_right = (top_left[0] + w, top_left[1] + h)
# 	    cv2.rectangle(img,top_left, bottom_right, 255, 2)
# 	    plt.subplot(121),plt.imshow(res,cmap = 'gray')
# 	    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
# 	    plt.subplot(122),plt.imshow(img,cmap = 'gray')
# 	    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
# 	    plt.suptitle(meth)
# 	    plt.show()




# def img_process(inputfile, basefile, outputfile):
# 	img1 = cv2.imread(basefile, 0)
# 	img2 = cv2.imread(inputfile, 0)
# 	img3 = img2.copy()

# 	# Initiate STAR detector
# 	orb = cv2.ORB_create()
	
# 	# find the keypoints and descriptors with SIFT
# 	kp1, des1 = orb.detectAndCompute(img1,None)
# 	kp2, des2 = orb.detectAndCompute(img2,None)

# 	# create BFMatcher object
# 	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# 	# Match descriptors.
# 	matches = bf.match(des1,des2)

# 	# Sort them in the order of their distance.
# 	matches = sorted(matches, key = lambda x:x.distance)

# 	# Draw first 10 matches.
# 	img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],img3,flags=2)



# 	# # Initiate SIFT detector
# 	# sift = cv2.xfeatures2d.SIFT_create()

# 	# # find the keypoints and descriptors with SIFT
# 	# kp1, des1 = sift.detectAndCompute(img1,None)
# 	# kp2, des2 = sift.detectAndCompute(img2,None)

# 	# # BFMatcher with default params
# 	# bf = cv2.BFMatcher()
# 	# matches = bf.knnMatch(des1,des2, k=2)

# 	# # Apply ratio test
# 	# good = []
# 	# for m,n in matches:
# 	#     if m.distance < 0.5*n.distance:
# 	#         good.append([m])

# 	# # cv2.drawMatchesKnn expects list of lists as matches.
# 	# img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,img3,flags=2)


# 	plt.imshow(img3),plt.show()




def main(argv):
   inputfile = ''
   outputfile = ''
   try:
      opts, args = getopt.getopt(argv,"hi:o:b:",["ifile=","ofile=","bfile="])
   except getopt.GetoptError:
      print('image.py -i <inputfile> -o <outputfile> -b<basefile>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print('image.py -i <inputfile> -o <outputfile> -b<basefile>')
         sys.exit()
      elif opt in ("-i", "--ifile"):
        inputfile = arg
      elif opt in ("-b", "--bfile"):
        basefile = arg
      elif opt in ("-o", "--ofile"):
         outputfile = arg
   print('Input file is ', inputfile)
   print('Output file is ', outputfile)
   img_process(inputfile, basefile, outputfile)

if __name__ == "__main__":
   main(sys.argv[1:])