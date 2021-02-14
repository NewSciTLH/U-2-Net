from PIL import Image
import numpy as np
import os
import json
import io
from google.cloud import logging, storage, vision, bigquery
from pathlib import Path
import tempfile
from skimage import img_as_ubyte
from skimage.transform import rescale, resize, rotate
import cv2
import importlib
#import utils
#importlib.reload(utils)
#%matplotlib inline
#import matplotlib.pyplot as plt
import pdb
import time
import math
from utils import detector
import importlib
importlib.reload(detector)
import torch
import torchvision.transforms as transforms
from particle_swarm import RandomSearch

my_file = Path("/home/ericd/storagekey.json")
if my_file.is_file():
    storage_client = storage.Client.from_service_account_json(my_file)
    queryclient = bigquery.Client.from_service_account_json("/home/ericd/bqkey.json")
else:
    storage_client = storage.Client()
    queryclient = bigquery.Client()

dataset_id = 'divvyup_metadata'  # replace with your dataset ID
table_id = 'reconciliation_output'  # replace with your table ID
table_ref = queryclient.dataset(dataset_id).table(table_id)
table = queryclient.get_table(table_ref)  # API request   
    
    
    
    
log_name = 'Reconciliation'
logging_client = logging.Client()
logger = logging_client.logger(log_name)
vision_client = vision.ImageAnnotatorClient()


def get_angle2(eyes):
    eye1 = eyes['left_eye']
    eye2 = eyes['right_eye']
    eye1_l, eye1_r = eye1
    eye2_l, eye2_r = eye2
    eye1_l=float(eye1_l)
    eye1_r=float(eye1_r)
    eye2_l=float(eye2_l)
    eye2_r=float(eye2_r)
    print()
    if eye1_l == eye2_l:
        if eye1_r > eye2_r:
            angle = -90
        elif eye1_r < eye2_r:
            angle = 90
        else:
            angle = 0
    else:
        angle = math.atan((eye2_r-eye1_r)/(eye2_l-eye1_l))* (180 / math.pi)
    return -angle
        
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
        cropped_image = input_image.copy()
    except Exception as e:
        logger.log_text(f"Image corrupted/preprocess failed {str(e)} ", severity='ERROR')
        return None, None#, None, None

    b = io.BytesIO()
    cropped_image.save(b, format='PNG')
    b = b.getvalue()

    try:
        response_fl = vision_client.annotate_image({'image': {'content': b}, 'features': [{'type': vision.enums.Feature.Type.FACE_DETECTION}, ], })
    except Exception as e:
        logger.log_text(f"Problem getting API response: {str(e)}", severity='ERROR')
        return None, None#, None, None

    #try:
    #    human_landmarks = MessageToJsonFacialLandmarks(response_fl)
    #except Exception as e:
    #    logger.log_text(f"Problem building json file: {str(e)}", severity='ERROR')
    #    return None, None, None, None

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
        angle2 = get_angle2({'left_eye':left_eye, 'right_eye':right_eye})
        print(angle,' angles',angle2 )

    except Exception as e:
        logger.log_text(f"Problem getting landmarks or angle: {str(e)}", severity='ERROR')
        return None, None#, None, None
    return angle,  eyes

    
def frame(crop_dict, base_dict):
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
    mask = base_dict['image'].copy()
    h_base = mask.height #350
    w_base = mask.width  #300

    crop_left, crop_right = crop_dict['eyes']
    angle = crop_dict['angle']
    im_center_x, im_center_y = crop_dict['center']
    im_center = complex(im_center_x, im_center_y)
    dist_crop = crop_dict['dist']
    im = crop_dict['image'].copy()
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
    
    crop = im_r.crop(( int(x_l), int(y_l), int(x_h), int(y_h) ))#underlying rbg
    #then we glue under the mask
    #rbg crop, rgba mask, coord
    final = exchangerbg(mask, crop, (int(x_l + dcenter.real), int(y_l + dcenter.imag) ))
    
    #mask2 = mask.copy()
    #mask.paste(crop, (int(x_l + dcenter.real), int(y_l + dcenter.imag) ))#into 4 channel
    #array_m = np.array(mask2)[:,:,3]
    #array_b = np.array(mask)
    #array_b[:,:,3] = array_m
    #final = Image.fromarray(array_b)
    return final

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




def lap(image):
    #transforms image for cv2 to compute laplacian
    rgb_org_im = img_as_ubyte(image)     
    origin = np.float32(cv2.cvtColor(rgb_org_im, cv2.COLOR_BGR2GRAY))
    originL = cv2.Laplacian(origin, -1)
    return originL


def APIPetLandmarks(input_image,pet='dogs'):
    det = detector.Detector(torch.cuda.is_available())
    trheechannel = input_image.copy()
    trheechannel = trheechannel.convert('RGB')
    eyes_dict = det.get_landmarks(im_files=trheechannel,subject_class=pet)#im_files, subject_class
    angle = get_angle2(eyes_dict)
    eyes = eyes_dict['left_eye'], eyes_dict['right_eye']
    return angle, eyes  
    
def first_reconciliation(input_image, input_image_m, key, tmp_labels, label='human'):
    #finds landmarks on both images and identify the images with those coordinates
    
    if label=='human':
        angle,  eyes = APIHumanLandmarks(input_image)
        angle_m, eyes_m = APIHumanLandmarks(input_image_m)
    else:
        angle,  eyes = APIPetLandmarks(input_image, label)
        angle_m, eyes_m = APIPetLandmarks(input_image_m, label)
    
    if not eyes: 
        logger.log_text(f"Missing output on API Human Landmarks on {key}", severity='ERROR')
    im_center, dist_im = expanded_bb(  final_points=eyes)  #test
    im_center_m, dist_im_m = expanded_bb(  final_points=eyes_m)  #test
    crop_dict={'angle':angle, 'eyes':eyes, 'image':input_image,'center':im_center, 'dist':dist_im}
    mask_dict={'angle':angle_m, 'eyes':eyes_m, 'image':input_image_m,'center':im_center_m, 'dist':dist_im_m}
    file =  frame(crop_dict, mask_dict)
    file.save(tmp_labels, 'png')
    return file,  mask_dict


def exchangerbg(four_channel_img, trhee_channel_img, coord=None):
    mask2 = four_channel_img.copy()
    four_channel_img.paste(trhee_channel_img, coord)#into 4 channel
    array_m = np.array(mask2)[:,:,3]
    array_b = np.array(four_channel_img)
    array_b[:,:,3] = array_m
    final = Image.fromarray(array_b)
    return final  
    


def third_reconciliation(crop, mask):
    #X=crop, Y=underlying rbg from mask
    Y = mask.copy()
    X = crop.copy()
    X = X.convert('RGB')
    Y = Y.convert('RGB')
    model = RandomSearch(20,{'x':[-.15,.15],'y':[-.15,.15],'scale':[.75,1.25],'theta':[-10,10]},max_iters=100)
    X,Y = transforms.ToTensor()(X), transforms.ToTensor()(Y) #[:-1], transforms.ToTensor()(Y)[:-1]
    best,best_params = model(X,Y)
    T = model.transform(X,params=best_params)
    print(T.size())
    tensor_to_pil = transforms.ToPILImage()(T)#.squeeze_(0))
    final_4 = exchangerbg(mask, tensor_to_pil)
    return final_4
    #X.permute(1,2,0)
    
    
def second_reconciliation(file, mask_dict):
    #from file we use the  rbg, mask comes from mask dict
    mask = np.array(mask_dict['image']).copy()    
    crop = np.array(file).copy()
    l_crop = lap(crop[:,:,:3]) #the underlying rbg we want to improve
    min_bbox = minimum_bounding_box( mask[:,:,3], alpha=0, mode=0) #([y1,x1,y2,x2])#at this moment I also know the center of both images
    mask_crop = crop_img_from_bbox( mask, min_bbox).copy()
    l_mask = lap(mask_crop[:,:,:3])*mask_crop[:,:,3]
    newdict = {'lap_crop':l_crop,'lap_mask':l_mask}
    rec = get_filter_weights( newdict, file, mask_dict, min_bbox)
    return rec
   

def get_filter_weights(laplacians, file, mask_dict,  min_bbox):
    #we take the original image, 
    #we multiply by the mask and we crop, 
    #then we use convolution on the target image.
    center = complex(mask_dict['center'][0], mask_dict['center'][1])
    image = laplacians['lap_crop'].copy()
    target = laplacians['lap_mask'].copy()
    h_im = 350
    w_im = 300
    vector=[]
    y1,x1,y2,x2 = min_bbox
    conv_val = -900
    res = 0
    rot = 0
    results = None
    ncenter = center
    for i in range(-30, 30):
        resized_cropT = resize(image, (h_im+i,h_im+i), anti_aliasing=True)
        dcenter =complex(int(i*center.real/300), int(i*center.imag/350))
        new_center = center + dcenter
        for a in range(-20,20):#default -10,10 with angle /20
            rot_cropT = rotate(resized_cropT/255, angle= a/40, center = (new_center.real, new_center.imag) )*255
            dx,dy = dcenter.real, dcenter.imag
            big_out = cv2.filter2D(rot_cropT, -1, target, anchor=(0,0))
            if big_out.shape[0]<target.shape[0]+1 or big_out.shape[1]<target.shape[1]+1:
                continue
            
            out = big_out[:-target.shape[0],:-target.shape[1]]
            maxmeasure = np.max(out)#np.max(outT[:-rot_cropT.shape[0],:-rot_cropT.shape[1]])
            if conv_val < maxmeasure:
                conv_val = maxmeasure
                res = i
                rot = a
                results = out
                ncenter = new_center
    print((h_im+res,h_im+res), rot/20)
    l_c = laplacians['lap_crop'].copy()
    l_m = laplacians['lap_mask'].copy()
    assert len(results)
    loc = np.where(results == conv_val)
    assert len(loc[0]) == 1
    Y1,X1,Y2,X2 =  (loc[0][0], loc[1][0],
                                    loc[0][0]+l_m.shape[0],
                                    loc[1][0]+l_m.shape[1])
    y1,x1,y2,x2 =  min_bbox #([y1,x1,y2,x2])
    previous_prediction =  file.copy()
    proposal_size = previous_prediction.resize((h_im+res,h_im+res))
    proposal_rot = proposal_size.rotate(rot/20, center=(ncenter.real, ncenter.imag))
    pre_np = np.array(proposal_rot)
    mask = mask_dict['image'].copy()
    
    mask_np = np.array(mask)
    mask_np[y1:y2,x1:x2,:3] = pre_np[Y1:Y2,X1:X2,:3]
    print(np.max(mask_np), np.max(mask_np[:,:,:3]), np.max(mask_np[:,:,3]),np.min(mask_np[:,:,3]),np.mean(mask_np[:,:,3]))
    final = Image.fromarray(mask_np)
    return final

def temp_pair2(im_dic):
    # [ {"key":,"key_m":,"bucket":,"mask":,"crop":,"classes":}]
    key, key_m,  bucket_name, source_blob_name_m, source_blob_name, classes = im_dic["key"], im_dic["key_m"], im_dic["bucket"], im_dic["mask"], im_dic["crop"], im_dic["classes"]
    return key, key_m, source_blob_name, source_blob_name_m, bucket_name, classes

def human_eyes(crop_image_path):
    # based on image it applies reconciliation    
    file = None
    human = True
    key, key_m, source_blob_name, source_blob_name_m, bucket_name,  classes =  temp_pair2(crop_image_path)
    # [ {"key":,"key_m":,"bucket":,"mask":,"crop":,"classes":}]
    destination_blob_name = f'reconciliation_test/{key}/{key}'
    destination_blob_name_2 = f'reconciliation_test/{key}/{key}_2'
    destination_blob_name_3 = f'reconciliation_test/{key}/{key}_3'
    destination_blob_name_4 = f'reconciliation_test/{key}/{key}_4'

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_local_path_o = tmpdirname + '/' + key+'out.png'
        tmp_local_path_o2 = tmpdirname + '/' + key+'out2.png'
        tmp_local_path_o3 = tmpdirname + '/' + key+'out3.png'
        tmp_local_path_o4 = tmpdirname + '/' + key+'out4.png'

        print('created temporary directory', tmpdirname)
        results = {'key' : key,
             'simple_crop':bucket_name+'/'+source_blob_name,
             'final_crop':bucket_name+'/'+source_blob_name_m}
        tmp_local_path = tmpdirname + '/' + key
        tmp_local_path_m = tmpdirname + '/' + key_m
        tmp_labels = tmpdirname + '/' + f'{key}_label'
        try:
            downloadBlob(bucket_name, source_blob_name, tmp_local_path)
            downloadBlob(bucket_name, source_blob_name_m, tmp_local_path_m)
        except Exception as e:
            print(str(e))
            logger.log_text(f"problem downloading image {str(e)} on {key}", severity='ERROR')
            return 'wrong input',None, None
        try:
            input_image = Image.open(tmp_local_path) #.convert('RGB')
            input_image_m = Image.open(tmp_local_path_m)
        except Exception as e:
            print(str(e))
            logger.log_text(f"Problem opening the image {str(e)} on {key}", severity='ERROR')
            return 'image corrupted'
        #class#'cats''dogs''''
        try:
            startl = time.time()
            file,  mask_dict  = first_reconciliation(input_image, input_image_m, key, tmp_labels, classes)
            file.save(tmp_local_path_o)
            upload_blob('model_staging', tmp_local_path_o, destination_blob_name)#to determine
            results['rec_1']='model_staging/'+destination_blob_name,
            middle = time.time()
            print(f'end of first reconciliation {middle-startl}')
            firststep = file.copy()
            file_2 =  second_reconciliation(firststep, mask_dict)
            file_2.save(tmp_local_path_o2)
            upload_blob('model_staging', tmp_local_path_o2, destination_blob_name_2)#to determine
            results['rec_2']='model_staging/'+destination_blob_name_2,
            fo = time.time()
            print(f'second reconciliation lasted {fo-middle}')
        except Exception as e:
            print(str(e))
            logger.log_text(f"Problem with first reconciliation {str(e)} on {key}", severity='ERROR')
        try:
            file_3 = second_reconciliation(input_image,  {'image':input_image_m})#['image']
            file_3.save(tmp_local_path_o3)
            upload_blob('model_staging', tmp_local_path_o3, destination_blob_name_3)#to determine
            results['rec_3']='model_staging/'+destination_blob_name_3,
            ro = time.time()
            print(f'third reconciliaton lated {ro-fo}')
        except Exception as e:
            print(str(e))
            logger.log_text(f"Problem with third reconciliation {str(e)} on {key}", severity='ERROR')
        try:        
            file_4 = third_reconciliation(input_image, input_image_m)
            file_4.save(tmp_local_path_o4)
            upload_blob('model_staging', tmp_local_path_o4, destination_blob_name_4)#to determine
            results['rec_4'] = 'model_staging/'+destination_blob_name_4
        except Exception as e:
            print(str(e))
            logger.log_text(f"Problem with third reconciliation {str(e)} on {key}", severity='ERROR')

        #return file, file_2, file_3, file_4

        rows_to_insert = []
        rows_to_insert.append(results)
        errors = queryclient.insert_rows(table, rows_to_insert)  # API request
        assert errors == []
        print('querry submitted!')

    


def temp_pair(crop_image_path):
    # based on image it applies reconciliation    
    split_path = crop_image_path.split('/')
    try:
        key = split_path[-2]
        key_m = split_path[-2]+'m.png'
        bucket = split_path[0]
        source_blob_name = crop_image_path.replace(bucket + '/', '')
        source_blob_name_m = crop_image_path.replace(bucket + '/', '').replace('crop_of_subject', 'final')
    except Exception as e:
        logger.log_text(f"Wrong path structure {str(e)} on {crop_image_path}", severity='ERROR')
        return 'wrong input'
    return key, key_m, source_blob_name, source_blob_name_m, 'divvyup_store', 'human'
    
        
if __name__ == "__main__":
    start = time.time()
    input_image_path = "divvyup_store/socks/600000/crop_of_subject"
    dataloader = input_image_path#this function queries a table and returns a pair of strings
    print('first process')
    human_eyes(dataloader)
    #firststep, secstep, thirdstep, fourstep = human_eyes(dataloader)
    #print(firststep.size, secstep.size, thirdstep.size, fourstep.size,)
    end = time.time()
    print(f'Total time: {end - start}')
