
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os


mypath="/home/shrehal/train_images"
onlyfiles = [ f for f in os.listdir(mypath) if isfile(join(mypath,f)) ] 
#onlyfiles is list of image names

images = np.empty(len(onlyfiles), dtype=object)
height=[]
width=[]
#images is the empty numpy array 
for n in range(0, len(onlyfiles)):
    images[n] = cv2.imread( join(mypath,onlyfiles[n]) )
    height.append(images[n].shape[0])
    width.append(images[n].shape[1])
    
height=np.array(height)
width=np.array(width)

H=np.max(height)
W=np.max(width)

for i in range (len(images)) :
    img_grey = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
    blur_removed = cv2.GaussianBlur(img_grey,(19,19),0)
    ret,finetune = cv2.threshold(blur_removed,0,255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU )
    #plt.imshow(finetune)
    #plt.show()
    
    
    top= int((H-finetune.shape[0])/2)
    bottom=H-finetune.shape[0]-top
    left= int((W-finetune.shape[1])/2)
    right= W-finetune.shape[1]-left
    
    # padding done
    padded_image = cv2.copyMakeBorder( finetune,top, bottom, left, right, cv2.BORDER_ISOLATED)
    # to invert the image-- have to see if there's a better command available
    final_image = cv2.bitwise_not(padded_image)
    #plt.imshow(final_image)
    #plt.show()

    n=onlyfiles[i]
    folder = '/home/shrehal/pre_processed'
    cv2.imwrite(os.path.join(folder , n), final_image)




