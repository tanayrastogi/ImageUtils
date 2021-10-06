# Python libraries
import cv2 
import numpy as np
import math

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


def draw_detections(image, bbox, **kwargs):
    """
    Function to draw the rectangle, label and mask on the image.

    INPUT
        image(numpy.ndarray):   Image on which the info is drawn.
        bbox(numpy.array)   :   4 corners of the bounding box on the image

        KWARGS Elements
        obj_label(str)      :   Label to put on the top of the image.
        confidence(float)   :   Detection confidence from the object detection model.
        mask(numpy.array)   :   Array of the mask
        directions(float)   :   heading direction of the object
        id(int):            :   object ID used for tracking
        color(tuple)        :   color of the box

    RETURN
        <numpy.ndarray>
        Image with the bbox and label on the image
    """
    clone = image.copy()
    # Choose a random color
    if "obj_label" not in kwargs:
        obj_label= "_"
    else:
        obj_label = kwargs["obj_label"]
    __add_color(obj_label)

    # Detection parameters
    (startX, startY, endX, endY) = bbox

    # Put rectangle around the objects detected
    if "color" not in kwargs:
        cv2.rectangle(clone, (startX, startY), (endX, endY), COLORS[obj_label], 2)
    else:
        cv2.rectangle(clone, (startX, startY), (endX, endY), kwargs["color"], 2)

    # Put label and confidence
    y = startY - 10 if startY - 10 > 10 else startY + 10
    label = "{}".format(obj_label)
    if "confidence" in kwargs:
        label += " {:.2f}%".format(kwargs["confidence"])
    if "direction" in kwargs:
        label += " {}".format(kwargs["directions"]) + u"\N{DEGREE SIGN}"
    cv2.putText(clone, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # If mask is there then put mask also in the image
    if "mask" in kwargs:
        mask = kwargs["mask"]
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

def get_videotimestamp(cameraCapture, ret_type="str"):
    """
    Function to get the timestamps of the video. 

    INPUT
        cameraCapture(<class 'cv2.VideoCapture'>):    Video capture object for the video currenty read. 
        ret_type(str):                                Return Type. Defines the type of return ("str", "datetime")

    RETURN
        <str> or <datetime.datetime>
        Current timestamp of the video either in string or datetime format, based on the type.
    """
    import datetime

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

    if milliseconds < 10:
        ts = "{}:{}:{}.00{}".format(int(hours), int(minutes), int(seconds), int(milliseconds))
    elif milliseconds < 100:
        ts = "{}:{}:{}.0{}".format(int(hours), int(minutes), int(seconds), int(milliseconds))
    else: 
        ts = "{}:{}:{}.{}".format(int(hours), int(minutes), int(seconds), int(milliseconds))

    if ret_type=="str":
        return ts
    elif ret_type=="datetime":
        return datetime.datetime.strptime(ts, "%H:%M:%S.%f")
    else:
        raise ValueError("Choose either str or datetime as ret_type")

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

def draw_metadata(image, **kwargs):
    """
    Function to put textual data on the bottom of the screen

    INPUT
        image(numpy.ndarray):   Image on which the info is drawn.
        kwargs(dict):           Data on the image
    """
    (height, width, channel) = image.shape
    # loop over the info tuples and draw them on our frame
    itr = 0
    for k, v in kwargs.items():
        text = "{}: {}".format(k, v)
        cv2.putText(image, text, (10, height - ((itr * 30) + 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        itr += 1


def draw_marker(image):
    """
    Function to draw XY-axis in image plane on the image.

    INPUT
        image(numpy.ndarray):   Image on which the info is drawn.
    """
    height, width, channel = image.shape

    # Center of image
    cx = int(width/2)
    cy = int(height/2)
    
    # Draw Center of the Image
    cv2.circle(image, (cx, cy),
                5, (0, 0, 255), -1) 
    # Draw line
    cv2.line(image, (cx, 0), (cx, height),
                (255, 0, 0), 1)
    cv2.line(image, (0, cy), (width, cy),
                (255, 0, 0), 1)
    # Put Coordinates 
    cv2.putText(image, "+X", (cx-50, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
    cv2.putText(image, "+Y", (10, cy), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)


def read_srt(filepath: str) -> list:
    """Function to read the meta data from OpenCamera .srt files 
    with location and timestamp data

    INPUT:
        filepath(str):  Path for file to read. 

    RETURN:
        List of dictonary with keys 
                start_time: Start time of the data acq for video
                end_time:   Start time of the data acq for video
                date:       Timestamp of video
                lat:        Latitude in degree
                lng:        Longitude in degree
                alt:        Altitude above sea-level in meters
                heading:    Heading angle from North in degree
    """
    import datetime
    import re 
    
    # Return list
    metaData = list()

    # Check if the file exits
    import os
    if os.path.isfile(filepath):
        pass 
    else:
        raise Exception("File does not exits!")

    with open(filepath, "r") as f:
        data = f.readlines()

    counter = 1
    for itr, line in enumerate(data):
        try:
            if int(line) == counter:
                counter += 1
                loc    = re.split('[°\'\"\,\ ]', data[itr+3])
                s_time = data[itr+1].split("-->")[0].split(" ")[0]
                e_time = data[itr+1].split("-->")[1].split("\n")[0]  
                metaData.append(dict(start_time= datetime.datetime.strptime(s_time, "%H:%M:%S,%f"),
                                     end_time  = datetime.datetime.strptime(e_time, " %H:%M:%S,%f"),
                                     date      = data[itr+2].split("\n")[0],
                                     lat       = float(int(loc[:3][0]) + int(loc[:3][1])/60 + int(loc[:3][2])/3600),
                                     lng       = float(int(loc[5:8][0]) + int(loc[5:8][1])/60 + int(loc[5:8][2])/3600),
                                     alt       = loc[10],
                                     heading   = loc[12]
                                    )
                                )
        except ValueError:
            pass
    return metaData


# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
	# initialize the list of picked indexes	
	pick = []
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")


def draw_polygon(frame, pts, color=None): # Polygon Box
    bbox = np.array(pts, np.int32)
    bbox = bbox.reshape((-1,1,2))
    if color is None:
        cv2.polylines(frame, [bbox], True, (0, 0, 255), 1)
    else:
        cv2.polylines(frame, [bbox], True, color ,1)


def draw_line(frame, pts): # Line
    point1 = pts[0] 
    point2 = pts[1]
    cv2.line(frame, point1, point2, (0, 0, 255), 1)


def get_fps(video):
    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    # With webcam get(CV_CAP_PROP_FPS) does not work.
    if int(major_ver)  < 3 :
        return video.get(cv2.cv.CV_CAP_PROP_FPS)
    else :
        return video.get(cv2.CAP_PROP_FPS)
    

def get_number_of_frames(video):
    video.set(cv2.CAP_PROP_POS_AVI_RATIO,1)
    num_of_frames = video.get(cv2.CAP_PROP_POS_FRAMES)
    video.set(cv2.CAP_PROP_POS_AVI_RATIO,0)
    assert video.get(cv2.CAP_PROP_POS_FRAMES) == float(0); "Some error in getting number of frames!"

    return num_of_frames




def distance_from_camera(bbox, image_shape, real_life_size):
    """
    Calculates the distance of the object from the camera.

    PARMS
    bbox: Bounding box [px]
    image_shape: Size of the image (width, height) [px]
    real_life_size: Height of the object in real world [cms]
    """

    ## REFERENCE FOR GOPRO
    # Focal Length and Image Size
    # https://clicklikethis.com/gopro-sensor-size/
    # https://gethypoxic.com/blogs/technical/gopro-hero9-teardown
    # https://www.sony-semicon.co.jp/products/common/pdf/IMX677-AAPH5_AAPJ_Flyer.pdf
    # http://photoseek.com/2013/compare-digital-camera-sensor-sizes-full-frame-35mm-aps-c-micro-four-thirds-1-inch-type/
    # https://www.gophotonics.com/products/cmos-image-sensors/sony-corporation/21-209-imx677

    # Camera Sensor:                    Sony IMX677
    # Camera Sensor array pixel size:   1.12[um] X 1.12[um]
    # Camera Resolution:                5663(H) X 4223(V)
    # Camera Sensor dimensions:         6.343[mm/H] X 4.730[mm/V] 
    # Camera Focal Length:              2.92 mm  

    #     5633(px)
    #  4  -------------------
    #  2  -                 -
    #  2  -                 -
    #  3  -                 -
    #  (p -                 -
    #  x) -                 -
    #     -------------------

    # REFERNCE FOR CALCULATION
    # https://www.scantips.com/lights/subjectdistance.html

    # GoPro Intrensic Camera Settings #
    ###################################
    focal_length_mm = 5.21
    unit_pixel_length = 1.12
    sen_res = (5663, 4223) 
    sensor_height_mm = (unit_pixel_length*sen_res[1])/1000 
    sensor_width_mm = (unit_pixel_length*sen_res[0])/1000
    ###################################

    # Calculation
    image_height_px = image_shape[0]
    image_width_px  = image_shape[1]

    (startX, startY, endX, endY) = bbox
    height_of_object_px = endY - startY
    width_of_object_px  = endX - startX

    obj_height_on_sensor_mm = (sensor_height_mm * height_of_object_px) / image_height_px
    return (real_life_size * focal_length_mm)/obj_height_on_sensor_mm



def lat_lng_from_camera(origin, heading_angle, distance, 
                        EARTH_RADIUS_KMS = 6378.1):
    """
    Calculates the latitude and longitude (in degree) of the object from the camera.

    PARMS
    origin: Origin point in (lat, lng)
    heading_angle: Heading angle of the object from the camera
    distance: Distance of the object from camera [kms]
    EARTH_RADIUS_KMS: Radius of earth in kms
    """
    lat1 = math.radians(origin[0])
    lng1 = math.radians(origin[1])
    brng = math.radians(heading_angle)   

    lat2 = math.asin(math.sin(lat1)*math.cos(distance/EARTH_RADIUS_KMS) + math.cos(lat1)*math.sin(distance/EARTH_RADIUS_KMS)*math.cos(brng))
    lng2 = lng1 + math.atan2(math.sin(brng)*math.sin(distance/EARTH_RADIUS_KMS)*math.cos(lat1), math.cos(distance/EARTH_RADIUS_KMS)-math.sin(lat1)*math.sin(lat2))
    
    lat2 = math.degrees(lat2)
    lng2 = math.degrees(lng2)
    return (lat2, lng2)

def get_latlon(image, detection, origin, heading_angle,
               real_life_size=150):
    
    # Distance in cms
    distance = distance_from_camera(detection["bbox"], image.shape, real_life_size)

    # Latitude and logitude of object
    return lat_lng_from_camera(origin, heading_angle, distance/100000)



def heading(start_point, end_point):
    # Reference: 
    # https://www.igismap.com/formula-to-find-bearing-or-heading-angle-between-two-points-latitude-longitude/
    # https://www.movable-type.co.uk/scripts/latlong.html
    
    # Convert all the points to
    start_point["lat"] = math.radians(start_point["lat"])
    start_point["lon"] = math.radians(start_point["lon"])
    end_point["lat"] = math.radians(end_point["lat"])
    end_point["lon"] = math.radians(end_point["lon"])

    dL = end_point["lon"]-start_point["lon"]
    Y = (math.cos(start_point["lat"])*math.sin(end_point["lat"])) - (math.sin(start_point["lat"])*math.cos(end_point["lat"])*math.cos(dL))
    X = math.cos(end_point["lat"])*math.sin(dL)
   
    return math.degrees(math.atan2(X, Y))           # in degrees


def heading_from_camera(bbox, image_shape):
    """
    Calculates the heading angle (in degree) of the object from the camera.

    PARMS
    bbox: Bounding box [px]
    image_shape: Size of the image (width, height) [px]
    """
    # GoPro Intrensic Camera Settings #
    ###################################
    focal_length_mm = 5.21
    unit_pixel_length = 1.12
    sen_res = (5663, 4223) 
    sensor_height_mm = (unit_pixel_length*sen_res[1])/1000 
    sensor_width_mm = (unit_pixel_length*sen_res[0])/1000
    ###################################

    # Image Center
    (cX, cY) = image_shape[1]/2, image_shape[0]/2 
    # Object Center
    (startX, startY, endX, endY) = bbox
    (centerX, centerY) = (startX+endX)/2, (startY+endY)/2

    # Distance between the two points
    distance = math.sqrt((centerX - cX)**2 + (centerY - cY)**2)

    # Focal Length in px
    img_width_px = image_shape[1]
    f_px = (focal_length_mm * img_width_px)/ (sensor_width_mm)

    # Heading Angle
    angle = math.degrees(math.asin(distance/f_px))
    if centerX > cX:
        return angle
    else:
        return -angle