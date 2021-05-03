# Python libraries
import cv2 
import numpy as np

def draw_keypoints(image, kp):
    """
    Function to draw keypoints on the image.

    INPUT
        image(numpy.ndarray):   Numpy image. This is the complete image with object to be detected.
        kp                  :   Keypoints to be plotted on the image

    RETURN
        Returns the image with the keypoints plotted on the image.
    """
    # Draw Key points
    return cv2.drawKeypoints(image,kp ,image, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


def draw_detections(image, obj_label, confidence, bbox, mask=None):
    """
    Function to draw the rectangle, label and mask on the image.

    INPUT
        image(numpy.ndarray):   Image on which the info is drawn.
        obj_label(str)      :   Label to put on the top of the image.
        confidence(float)   :   Detection confidence from the object detection model.
        bbox(numpy.array)   :   4 corners of the bounding box on the image
        mask(numpy.array)   :   Array of the mask

    RETURN
        <numpy.ndarray>
        Image with the bbox and label on the image
    """
    clone = image.copy()
    # Choose a random color
    color = np.random.uniform(0, 255, size=(1, 3)).flatten()

    # Detection parameters
    (startX, startY, endX, endY) = bbox

    # Put rectangle around the objects detected
    cv2.rectangle(clone, (startX, startY), (endX, endY), color, 2)

    # Put label and confidence
    y = startY - 15 if startY - 15 > 15 else startY + 15
    label = "{}: {:.2f}%".format(obj_label, confidence)
    cv2.putText(clone, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # If mask is there then put mask also in the image
    if mask is not None:
        # Put mask around the detection
        roi = clone[startY:endY, startX:endX]
        roi = roi[mask]
        # Color for the mask
        blended = ((0.4 * color) + (0.6 * roi)).astype("uint8")
        clone[startY:endY, startX:endX][mask] = blended
        
    return clone
    
def draw_compare_features(image1, image2, kp1, kp2, goodmatches):
    """
    Function to draw matching keypoints on the two images. 

    INPUT
        image1(numpy.ndarray):   First Image on which the info is drawn.
        image2(numpy.ndarray):   Second Image on which the info is drawn.
        kp1(list)            :   List of Keypoints for image 1
        kp2(list)            :   List of Keypoints for image 2.   
        goodmatches(list)    :   List of good matches of keypoints between the two images. 

    RETURN
        <numpy.ndarray>
        2 images, stiched together with mathces marked.
    """
    # Drawing parameters
    draw_params = dict(matchColor = (0,255,0),
        singlePointColor = (255,0,0),
        flags = cv2.DrawMatchesFlags_DEFAULT)
    
    return cv2.drawMatchesKnn(image1,kp1,image2,kp2,goodmatches,None,**draw_params)