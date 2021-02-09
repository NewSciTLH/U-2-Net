from PIL import Image
import numpy as np
import os
import json
import io
from google.cloud import logging, storage, vision
from pathlib import Path
import tempfile

my_file = Path("/home/ericd/storagekey.json")
if my_file.is_file():
    storage_client = storage.Client.from_service_account_json(my_file)
else:
    storage_client = storage.Client()

log_name = 'humanLandmark'
logging_client = logging.Client()
logger = logging_client.logger(log_name)
vision_client = vision.ImageAnnotatorClient()




def minimum_bounding_box(img,alpha=1,mode=0):
    """ Calculates the minimum bounding box for an image """
    yproj = img.mean(axis=1)
    xproj = img.mean(axis=0)
    if mode == 0:
    	_get_idx = lambda x: np.where(x>alpha)[0]
    else:
    	_get_idx = lambda x: np.where(x<alpha)[0]

    def _get_bounds_on_proj(proj):
        idx = _get_idx(proj)
        return(idx[0],
               idx[-1])

    x1,x2 = _get_bounds_on_proj(xproj)
    y1,y2 = _get_bounds_on_proj(yproj)
    return([y1,x1,y2,x2])

def crop_img_from_bbox(img,bbox):
    y1,x1,y2,x2 = bbox
    return(img[y1:y2,x1:x2])


def get_angle(eye1, eye2, nose):
    '''Getting the angle to rotate the images: Rotation Correction algorithm'''
    v = eye2 - eye1
    m = v[1] / v[0]
    xs = (nose[0] + nose[1] * m - eye1[1] * m + eye1[0] * m * m) / (1 + m * m)
    ys = m * (xs - eye1[0]) + eye1[1]
    vec = nose - np.array([xs, ys])
    angle = np.arctan2(- vec[1], vec[0]) * (180 / np.pi) + 90
    angle = 360 + angle if angle < 0 else angle
    return angle


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # bucket_name = "your-bucket-name"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)


def downloadBlob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    """TODO: Consider speeding this up by streaming to buffer"""
    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    logger.log_text(f"{source_blob_name} downloaded to  {destination_file_name}")


def MessageToJsonFacialLandmarks(response_fl):
    face_list = []
    for face in response_fl.face_annotations:

        bounding_poly_vertices = []
        for v in face.bounding_poly.vertices:
            bounding_poly_vertices.append({"x": v.x, "y": v.y})

        fd_bounding_poly_vertices = []
        for v in face.fd_bounding_poly.vertices:
            fd_bounding_poly_vertices.append({"x": v.x, "y": v.y})

        landmarks = {}
        unknown_count = 1
        for landmark in face.landmarks:
            t = landmark.type
            if t == 0:
                t = f"unknown{unknown_count}"
                unknown_count += 1
            landmarks[t] = {"x": landmark.position.x,
                            "y": landmark.position.y,
                            "z": landmark.position.z}
        face_dict = {}
        face_dict["bounding_poly_vertices"] = bounding_poly_vertices
        face_dict["fd_bounding_poly_vertices"] = fd_bounding_poly_vertices
        face_dict["landmarks"] = landmarks
        face_dict["roll_angle"] = face.roll_angle  # John
        face_dict["pan_angle"] = face.pan_angle
        face_dict["tilt_angle"] = face.tilt_angle
        face_dict["detection_confidence"] = face.detection_confidence
        face_dict["landmarking_confidence"] = face.landmarking_confidence
        face_dict["joy_likelihood"] = face.joy_likelihood
        face_dict["sorrow_likelihood"] = face.sorrow_likelihood
        face_dict["anger_likelihood"] = face.anger_likelihood
        face_dict["surprise_likelihood"] = face.surprise_likelihood
        face_dict["under_exposed_likelihood"] = face.under_exposed_likelihood
        face_dict["blurred_likelihood"] = face.blurred_likelihood
        face_dict["headwear_likelihood"] = face.headwear_likelihood
        face_list.append(face_dict)
    return face_list


def APIHumanLandmarks(input_image):
    """ Calls G object recognition for obtaining facial landmarks"""
    # Opening the original image, correcting the orientation, and cropping it
    try:
        image = input_image
        cropped_image = image
    except Exception as e:
        logger.log_text(f"Image corrupted/preprocess failed {str(e)} ", severity='ERROR')
        return None, None, None, None

    b = io.BytesIO()
    cropped_image.save(b, format='PNG')
    b = b.getvalue()

    try:
        response_fl = vision_client.annotate_image({'image': {'content': b}, 'features': [{'type': vision.enums.Feature.Type.FACE_DETECTION}, ], })
    except Exception as e:
        logger.log_text(f"Problem getting API response: {str(e)}", severity='ERROR')
        return None, None, None, None

    try:
        human_landmarks = MessageToJsonFacialLandmarks(response_fl)
    except Exception as e:
        logger.log_text(f"Problem building json file: {str(e)}", severity='ERROR')
        return None, None, None, None

    try:
        left_eye = (response_fl.face_annotations[0].landmarks[0].position.x,
                    response_fl.face_annotations[0].landmarks[0].position.y)
        right_eye = (response_fl.face_annotations[0].landmarks[1].position.x,
                     response_fl.face_annotations[0].landmarks[1].position.y)

        nose = (response_fl.face_annotations[0].landmarks[7].position.x,
                response_fl.face_annotations[0].landmarks[7].position.y)

        eyes = left_eye, right_eye
        
        angle = get_angle(np.array([left_eye[0], left_eye[1]]),
                          np.array([right_eye[0], right_eye[1]]),
                          np.array([nose[0], nose[1]]))
        #print(angle)  # john

        angle = round(angle % 360, 2)
        #print(angle)

    except Exception as e:
        logger.log_text(f"Problem getting landmarks or angle: {str(e)}", severity='ERROR')
        return None, None, None, None
    return angle, human_landmarks, eyes, cropped_image

    
def frame( crop_dict, base_dict):
    #current code: expands base, rotates crop
    #goal: expands and rotates crop, base untouch
    #base_left = (135,200) 
    #base_right =  (165,200)
    # we dont want to modify the mask, we can rescale the input image
    #Get (middle point of average eyes,  eyes' average distance) and (middle point of input image's eyes,  distance between input image's eyes).
    
    dx=0
    dy=0
    base_angle = base_dict['angle']
    base_left, base_right = base_dict['eyes']
    base_center_x, base_center_y = base_dict['center']
    base_center = complex(base_center_x, base_center_y)
    dist_base = base_dict['dist']
    mask = base_dict['image']
    h_base = mask.height #350
    w_base = mask.width  #300

    crop_left, crop_right = crop_dict['eyes']
    angle = crop_dict['angle']
    im_center_x, im_center_y = crop_dict['center']
    im_center = complex(im_center_x, im_center_y)
    dist_crop = crop_dict['dist']
    im = crop_dict['image']
    h_im = im.height
    w_im = im.width

    #first we resize the crop
    ratio = dist_crop/dist_base
    bigCrop = im.resize((int(w_im*ratio), int(h_im*ratio) ))
    #then we rotate it with center the middle of the eyes
    center = ( int(im_center_x*ratio),  int(im_center_y*ratio)  )
    im_r = bigCrop.rotate(base_angle-angle, center=center)# we may have to switch the angle
    #now we translate
    dcenter = base_center-im_center
    x_l = max(0, dcenter.real)-dcenter.real
    y_l = max(0, dcenter.imag)-dcenter.imag
    x_h = min(w_base, w_im +dcenter.real) - dcenter.real
    y_h = min(h_base, h_im +dcenter.imag) - dcenter.imag
    
    crop = im_r.crop(( int(x_l), int(y_l), int(x_h), int(y_h) ))
    #then we glue under the mask
    mask2 = mask.copy()
    mask.paste(crop, (int(x_l + dcenter.real), int(y_l + dcenter.imag) ))
    array_m = np.array(mask2)[:,:,3]
    array_b = np.array(mask)
    array_b[:,:,3] = array_m
    final = Image.fromarray(array_b)
    return final
        
def tempfunction( k, bucket, crop_name, key,  crop_dict, base_dict):
    # saves the actual crop to the bucket
    #base is the final product
    #image is the crop
    #base_left=(115+3*4,160), base_right=(185-3*4,160)
    final = frame(crop_dict, base_dict)
    
    #final, crop, boxCoord, angle, center = frame(crop_dict, base_dict)
    final.save(f'/tmp/{key}crop_{k}_temp', 'png')
    return final
    #upload_blob(bucket, f'/tmp/{key}crop_{k}_temp', crop_name+'_temp')
    #crop.save(f'/tmp/{key}crop_{k}', 'png')
    #upload_blob(bucket, f'/tmp/{key}crop_{k}', crop_name)
    #updateBlobMetadata(bucket, crop_name, boxCoord, angle, center)

    #for i in range(5):
    #    final = frame(image,  orientation, dx, dy, im_center, dist_im, h_im, w_im, angle, base_left=(115+3*i,160), base_right=(185-3*i,160))
    #    final.save(f'/tmp/crop_{k}', 'png')
    #    upload_blob(bucket, f'/tmp/crop_{k}', crop_name+str(i))


def expanded_bb( final_points):
    # The function expects the height and width of an image (I called it cropped because 
    # I was working with simple crop, but any image), and then the "final points" was a 
    # 2-d array with the position of the landmarks (the first dimension were the y and 
    # the second dimension the x), and the scale parameters is because I increase the 
    # asmall bounding box by 15% (I am not sure you want to use it).
    # read the json/consider x0-x1 for all hat objects, take the max and compare/use the better distance
    left, right = final_points
    left_x, left_y = left
    right_x, right_y = right
    base_center_x = (left_x+right_x)/2
    base_center_y = (left_y+right_y)/2 
    dist_base = abs(complex(left_x, left_y)-complex(right_x, right_y ) )
    return (int(base_center_x), int(base_center_y) ), dist_base

    


def human_eyes(crop_image_path):
        # Downloading original image    
    split_path = crop_image_path.split('/')
    file = None
    try:
        key = split_path[-2]
        key_m = split_path[-2]+'m.png'
        bucket = split_path[0]
        source_blob_name = crop_image_path.replace(bucket + '/', '')
        source_blob_name_m = crop_image_path.replace(bucket + '/', '').replace('crop_of_subject', 'final')
    except Exception as e:
        logger.log_text(f"Wrong path structure {str(e)} on {crop_image_path}", severity='ERROR')
        return 'wrong input'
    with tempfile.TemporaryDirectory() as tmpdirname:
        print('created temporary directory', tmpdirname)
        #if not os.path.exists("/tmp"):
        #    os.makedirs("/tmp")
        tmp_local_path = tmpdirname + '/' + key
        tmp_local_path_m = tmpdirname + '/' + key_m
        tmp_labels = tmpdirname + '/' + f'{key}_label'
        try:
            downloadBlob(bucket, source_blob_name, tmp_local_path)
            downloadBlob(bucket, source_blob_name_m, tmp_local_path_m)
        except Exception as e:
            logger.log_text(f"problem downloading image {str(e)} on {key}", severity='ERROR')
            return 'wrong input'
        try:
            input_image = Image.open(tmp_local_path) #.convert('RGB')
            input_image_m = Image.open(tmp_local_path_m)
        except Exception as e:
            logger.log_text(f"Problem opening the image {str(e)} on {key}", severity='ERROR')
            return 'image corrupted'
        angle, human_landmarks, eyes, littleImage = APIHumanLandmarks(input_image)
        angle_m, human_landmarks_m, eyes_m, littleImage_m = APIHumanLandmarks(input_image_m)
        if not eyes:
            logger.log_text(f"Missing output on API Human Landmarks on {key}", severity='ERROR')
        im_center, dist_im = expanded_bb(  final_points=eyes)  #test
        im_center_m, dist_im_m = expanded_bb(  final_points=eyes_m)  #test
        crop_dict={'angle':angle,'hl': human_landmarks, 'eyes':eyes, 'image':littleImage,'center':im_center, 'dist':dist_im}
        mask_dict={'angle':angle_m,'hl': human_landmarks_m, 'eyes':eyes_m, 'image':littleImage_m,'center':im_center_m, 'dist':dist_im_m}
        file =  frame(crop_dict, mask_dict)
        file.save(tmp_labels, 'png')
    return file

if __name__ == "__main__":
    import time
    start = time.time()
    input_image_path = "divvyup_store/socks/600000/crop_of_subject"
    # input_image_path = "model_staging/orientation/image_testing/humans/dat99.jpeg"
    human_eyes(input_image_path)
    end = time.time()
    print(f'Total time: {end - start}')
    """
    Steps:
    1.-> input = (input_image_path,orientation,bbs)
    2.-> test for folder
    3.-> get metadata
    4.-> call landmarkdetector
    5.-> check weights
    6.-> Initialize predictor
    7.-> test for image name
    8.-> download image
    9.-> try to open and rotate, crop the image from metadata
    information
    10.-> make prediction
    11.-> move to cpu, separate predictions, get pixels for eyes,
    apply gaussian filter
    12,13,14.-> get local max and make final predictions for
    eye_1, eye_2, nose
    15.-> get angle
    16.-> save jsoncito
    17.-> update metadata with rotation angles
    """
