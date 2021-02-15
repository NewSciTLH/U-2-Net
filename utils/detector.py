from PIL import Image
import torch
import torchvision
import torchvision.transforms.functional as F
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
# some_file.py
import sys
import utils.landmark_utils as utils


class Detector:
    """Class that has get_landmark() function to find eye locations
    
    ...
    Parameters
    ----------
        cuda : bool
            If True, the model will run using GPU acceleration if a CUDA
            GPU is available. If False, the model will run on CPU
    
    Attributes
    ----------
        device : string
            The device that the model will use to execute. If cuda is 
            True, the GPU will be used
        model_paths : dict
            Paths to model checkpoints for each subject class
        n_label_dict : dict
            The number of landmarks for each subject class
    """
    
    def __init__(self, cuda=True):
        
        self.device = 'cuda:0' if cuda else 'cpu'
        
        self.model_paths = {
            'cats':'cat_model_epoch_015.pth', 
            'dogs': 'dog_model_epoch_015.pth', 
            'humans':'human_model_epoch_015.pth'
        }
        
        self.n_label_dict = {'cats':9, 'dogs': 8, 'humans':4, 'all':3}


    def get_landmarks(self, im_files, subject_class):
        """Predicts landmarks for dogs, cats, or humans
        
        ...
        Parameters
        ----------
        im_files : list
            Paths to the images that will be landmarked
        subject_class : str
            The type of subject in the image. Options are 
            {'dogs', 'cats', 'humans'}
        
        Returns
        -------
        eyes : dict
            Dictionary containing the locations of the 
            right and left eyes as np.array's. The format
            is [x, y] for each eye, and the x and y values
            are returned as percentages of the image's total
            resolution. 
        """
        
        if type(im_files) != type([]) :
            im_files = [im_files]
        
        # Load model from checkpoints
        checkpoint_dir = '/home/ericd/U-2-Net/checkpoints'#Path('checkpoints/')
        model = torchvision.models.segmentation.fcn_resnet50(num_classes=self.n_label_dict[subject_class]+1)
        state_dict = torch.load(checkpoint_dir+'/'+self.model_paths[subject_class], map_location=torch.device(self.device))
        model.load_state_dict(state_dict)
        model = model.to(torch.device(self.device))
        
        eyes = {}
        for im_file in im_files:
            # Downscale image for faster computation
            #only for one file
            #im = Image.open(im_file)
            im = im_file.copy()
            
            im = F.resize(im, 300)

            # Predict landmarks
            out, probs = utils.calculate_segmentation_from_image(model, F.to_tensor(im).to(self.device))
            segmentation = utils.predict_segmentation(probs.cpu())
            landmarks = utils.landmarks_DBSCAN(segmentation)

            # Return eye locations as percentage of image size
            im_x, im_y = im.size[0], im.size[1]
            l_eye = np.array([landmarks[0][0][1]/im_x, landmarks[0][0][0]/im_y])
            r_eye = np.array([landmarks[1][0][1]/im_x, landmarks[1][0][0]/im_y])
            #only for one file   
            #eyes[im_file] = {'right_eye': r_eye, 'left_eye': l_eye}
            eyes = {'right_eye': r_eye, 'left_eye': l_eye}
        
        return eyes
    
    