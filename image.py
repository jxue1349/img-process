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
	image_base = cv2.imread(basefile, 1)
	image = cv2.imread(inputfile, 1)

	# grayscale, then blur, then threshold (max constrast) to remove fine features.
	base_tmp = cv2.cvtColor(image_base, cv2.COLOR_BGR2GRAY)
	blur = cv2.blur(base_tmp,(100,100))
	thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY)[1]

	# find circle amongst remaining image blobs.
	circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 1, 10, param1=100, param2=20, minRadius=0, maxRadius=0)
	circles = np.uint16(np.around(circles))

	max_radius = 0
	xc=0
	yc=0
	cc=0
	# get the largest enclosing circle's radius and average the found circles' centers
	for i in circles[0,:]:
		# cv2.circle(image_base,(i[0],i[1]),i[2],(0,255,0),2)
		xc+=i[0]
		yc+=i[1]
		cc+=1
		if i[2] > max_radius:
			max_radius = i[2]
	xc = int(xc/cc)
	yc = int(yc/cc)
	
	# draw the outer circle
	# cv2.circle(image_base,(xc,yc), rad,(0,255,0),2)
	
	#crop orginal image to found target, adjust radius to compensate for earlier blur
	rad= int(max_radius*(1+(100/max_radius)))
	target = image_base[yc-rad:yc+rad,xc-rad:xc+rad]
	cv2.imwrite(outputfile, target)

	#adaptively find the threshold, reapply threshold to highlight the holes.
	base_tmp = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(base_tmp, (1,1), 0)
	
	val, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	hithrs = cv2.threshold(blur, val+15, 255, cv2.THRESH_BINARY)[1]
	lothrs = cv2.threshold(blur, val-50, 255, cv2.THRESH_BINARY)[1]
	
	cv2.imwrite('thresh.png', thresh)
	kernel = np.ones((int(rad/60),int(rad/60)),np.uint8)
	kernel2 = np.ones((int(rad/40),int(rad/40)),np.uint8)

	grad1 = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel)
	grad2 = hithrs - lothrs

	plt.subplot(221),plt.imshow(hithrs)
	plt.title('High Threshold'), plt.xticks([]), plt.yticks([])
	plt.subplot(222),plt.imshow(blur)
	plt.title('Grayscale'), plt.xticks([]), plt.yticks([])	
	plt.subplot(223),plt.imshow(lothrs)
	plt.title('Lo Threshold'), plt.xticks([]), plt.yticks([])	
	plt.subplot(224),plt.imshow(grad2)
	plt.title('hi-lo'), plt.xticks([]), plt.yticks([])	
	plt.suptitle('compare thresholds')
	plt.show()



	opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
	closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
	close2 = cv2.morphologyEx(grad2, cv2.MORPH_CLOSE, kernel)
	open2 = cv2.morphologyEx(grad2, cv2.MORPH_OPEN, kernel)

	print('radius', rad)
	plt.subplot(241),plt.imshow(thresh)
	plt.title('Adaptive Threshold'), plt.xticks([]), plt.yticks([])
	# plt.subplot(242),plt.imshow(grad1)
	# plt.title('Morph Gradient'), plt.xticks([]), plt.yticks([])
	plt.subplot(243),plt.imshow(opening)
	plt.title('Opening'), plt.xticks([]), plt.yticks([])
	plt.subplot(244),plt.imshow(closing)
	plt.title('Closing'), plt.xticks([]), plt.yticks([])
	plt.subplot(245),plt.imshow(grad2)
	plt.title('Hi-Lo 20'), plt.xticks([]), plt.yticks([])
	plt.subplot(246),plt.imshow(close2)
	plt.title('Close Hi-Lo'), plt.xticks([]), plt.yticks([])
	plt.subplot(247),plt.imshow(open2)
	plt.title('Open Hi-Lo'), plt.xticks([]), plt.yticks([])

	openclose = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel2)
	plt.subplot(242),plt.imshow(openclose)
	plt.title('Openclose thresh'), plt.xticks([]), plt.yticks([])

	openclose = cv2.morphologyEx(close2, cv2.MORPH_OPEN, kernel2)
	plt.subplot(248),plt.imshow(openclose)
	plt.title('Openclose hilo'), plt.xticks([]), plt.yticks([])
	# closeopen = cv2.morphologyEx(open2, cv2.MORPH_CLOSE, kernel)
	# plt.subplot(248),plt.imshow(closeopen)
	# plt.title('Closeopen hilo'), plt.xticks([]), plt.yticks([])

	plt.suptitle('compare methods')
	plt.show()



	#generate template matching template
	smallrad = int(rad*.65)
	templ = np.zeros((2*smallrad,2*smallrad,3), np.uint8)
	templ[:] = (255, 255, 255)
	cv2.circle(templ,(smallrad,smallrad),int(rad*.03),(0,0,0),5)
	cv2.imwrite('templ.png', templ)

	#template match to find center of target
	templ = cv2.imread('templ.png', 0)
	img = cv2.imread('thresh.png', 0)
	img = blur = cv2.blur(img,(10,10))
	img2 = thresh.copy()
	w = 100
	h = 100
	# All the 6 methods for comparison in a list
	methods = ['cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF_NORMED']
	for meth in methods:
	    img = img2.copy()
	    method = eval(meth)
	    # Apply template Matching
	    res = cv2.matchTemplate(img,templ,method)
	    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
	    print('min,minloc,  max,maxloc',min_val, min_loc, max_val, max_loc)
	    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
	    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
	        top_left = min_loc
	    else:
	        top_left = max_loc
	    # bottom_right = (top_left[0] + w, top_left[1] + h)
	    # cv2.rectangle(img,top_left, bottom_right, 255, 2)
	    ctr = (top_left[0] + smallrad, top_left[1] + smallrad)
	    cv2.circle(img,ctr,w, (0,0,0),-1)
	    plt.subplot(131),plt.imshow(templ)
	    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
	    plt.subplot(132),plt.imshow(res,cmap = 'gray')
	    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
	    plt.subplot(133),plt.imshow(img)
	    # plt.subplot(122),plt.imshow(img,cmap = 'gray')
	    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
	    plt.suptitle(meth)
	    plt.show()



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