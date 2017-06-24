from sklearn.cluster import KMeans
import matplotlib.pyplot as plot
import argparse 
import cv2
import helper
from tqdm import tqdm

########################
pbar = tqdm(total = 100)
########################

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True)
ap.add_argument("-c", "--clusters", required = True, type = int)
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

########################
pbar.update(10)
########################

plot.figure()
plot.axis("off")
plot.imshow(image)

image = image.reshape(image.shape[0] * image.shape[1], 3);

########################
pbar.update(20)
########################

clt = KMeans(n_clusters = args["clusters"])
clt.fit(image)

########################
pbar.update(40)
########################

hist = helper.centroid_histogram(clt)
bar = helper.plot_colors(hist, clt.cluster_centers_)
maxbar = helper.plot_max()

########################
pbar.update(30)
pbar.close()
########################

p = plot.figure()
p2 = p.add_subplot(211)
p2.axis("off")
p2.imshow(bar)

p3 = p.add_subplot(212)
p3.axis("off")
p3.imshow(maxbar)
plot.show()


####################### mouse callback 
# px_ar = list()
# def click_px(event,x,y,flags,param) :	
# 	if event == cv2.EVENT_LBUTTONDOWN :
# 		print(x,y)

# cv2.namedWindow("image")
# cv2.setMouseCallback("image", click_px, image)

# while True:
# 	cv2.imshow("image", image)
# 	key = cv2.waitKey(1) & 0xFF
# 	if key == ord("c"):
# 		break
# cv2.destroyAllWindows()
#######################



