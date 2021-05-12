# Python libraries
import cv2 
import numpy as np


# Global Variable
COLORS = dict()
def __add_color(label):
    if label not in COLORS.keys():
        COLORS[label] = np.random.uniform(0, 255, size=(1, 3)).flatten()


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
    __add_color(obj_label)

    # Detection parameters
    (startX, startY, endX, endY) = bbox

    # Put rectangle around the objects detected
    cv2.rectangle(clone, (startX, startY), (endX, endY), COLORS[obj_label], 2)

    # Put label and confidence
    y = startY - 10 if startY - 10 > 10 else startY + 10
    label = "{}: {:.2f}%".format(obj_label, confidence)
    cv2.putText(clone, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # If mask is there then put mask also in the image
    if mask is not None:
        # Put mask around the detection
        roi = clone[startY:endY, startX:endX]
        roi = roi[mask]
        # Color for the mask
        blended = ((0.4 * COLORS[obj_label]) + (0.6 * roi)).astype("uint8")
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

def get_videotimestamp(cameraCapture):
    """
    Function to get the timestamps of the video. 

    INPUT
        cameraCapture(<class 'cv2.VideoCapture'>):    Video capture object for the video currenty read. 

    RETURN
        <str>
        Current timestamp of the video in format H:M:S.MS
    """
    seconds = 0
    minutes = 0
    hours = 0
    milliseconds = cameraCapture.get(cv2.CAP_PROP_POS_MSEC)
    seconds = milliseconds//1000
    milliseconds = milliseconds%1000
    if seconds >= 60:
        minutes = seconds//60
        seconds = seconds % 60
    if minutes >= 60:
        hours = minutes//60
        minutes = minutes % 60
    return "{}:{}:{}.{}".format(int(hours), int(minutes), int(seconds), int(milliseconds))

def pp_detectionlist(dectList):
    """
    Function to Pretty Print (PP) detection from single image.

    INPUT
        dectList(list): Output the list of dictonary with {label, confidence, box}
    """
    for detection in dectList:
        obj = detection["label"]
        confidence = detection["confidence"]
        print("[DETECTED] {}: {:.2f}".format(obj, confidence))

def draw_metadata(image, metadict):
    """
    Function to put textual data on the bottom of the screen

    INPUT
        image(numpy.ndarray):   Image on which the info is drawn.
        metadict(dict):         Data on 
    """
    (height, width, channel) = image.shape
    # loop over the info tuples and draw them on our frame
    itr = 0
    for k, v in metadict.items():
        text = "{}: {}".format(k, v)
        cv2.putText(image, text, (10, height - ((itr * 30) + 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        itr += 1
