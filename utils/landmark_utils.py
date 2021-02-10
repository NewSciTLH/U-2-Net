import os
import numpy as np
from PIL import Image
from itertools import product
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import pandas as pd
import torch
import cv2
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.cluster import DBSCAN
import matplotlib.patches as patches
from sklearn.neighbors import NearestNeighbors


class head():
    def __init__(self, left_eye=None,right_eye=None, distance=400):
        self.l = left_eye
        self.r = right_eye
        self.distance = distance
        
    def toVect(self, landmark):
        out = np.array([[landmark[0][self.l]], [landmark[1][self.r]] ])
        return out 
        
    
def clusterHead(left_eyes, right_eyes, fullHeads=False):
    #We use NN to cluster head objects: eyes and nose, assuming there is at least one pair of eyes
    if not left_eyes or not right_eyes :
        heads = {}
        if fullHeads:
            for headsita in list(range(len(left_eyes))):
                newHead = head(left_eye = headsita)
                heads[headsita] = newHead
            for headsita in list(range(len(right_eyes))):
                newHead = head(right_eye = headsita)
                heads[headsita] = newHead
    elif len(left_eyes)>1:
        neigh = NearestNeighbors(n_neighbors=2)
        neigh.fit(left_eyes)
        distances, from_right_to_left =neigh.kneighbors(right_eyes) 
        index_taken = {} #[inr, distances[inr][0]]
        queue = list(range(len(right_eyes)))
        heads = {}
        j = -1
        # we examine the terms and correct previous choices
        while queue:
            index_right_eye = queue[0]
            queue = queue[1:]
            # we grab the closest left eye to the inr
            index_left_eye = from_right_to_left[index_right_eye][0]
            if (index_left_eye)==[] and fullHeads:
                # if the point is asolated
                newHead = head( right_eye=index_right_eye)
                heads[j] = newHead
                j = j-1

            elif index_left_eye not in index_taken:
                #new index
                newHead = head(left_eye = index_left_eye, right_eye=index_right_eye, distance = distances[index_right_eye][0])
                heads[index_left_eye] = newHead
                index_taken[index_left_eye] = [index_right_eye, distances[index_right_eye][0]]
            else:
                # we need to compare distances
                newdist = distances[index_right_eye][0]
                olddist = index_taken[index_left_eye][1]
                if olddist<newdist:
                    # wrong left eye
                    index_left_eye = from_right_to_left[index_right_eye][1]
                    newdist = distances[index_right_eye][1]
                    olddist = index_taken.get(index_left_eye, [[],None])[1]
                    if index_left_eye not in index_taken:
                        newHead = head(left_eye = index_left_eye, right_eye=index_right_eye, distance = distances[index_right_eye][1])
                        heads[index_left_eye] = newHead
                        index_taken[index_left_eye] = [index_right_eye, distances[index_right_eye][1]]
                    elif olddist < newdist and fullHeads: # olddist<newdist
                        newHead = head( right_eye=index_right_eye)
                        heads[j] = newHead
                        j = j-1
                    else:
                        queue = queue+[index_taken[index_left_eye][0]]
                        newHead = head(left_eye = index_left_eye, right_eye=index_right_eye, distance = newdist)
                        heads[index_left_eye] = newHead
                        index_taken[index_left_eye] = [index_right_eye, distances[index_right_eye][1]]
                else:
                    # correct left eye already taken
                    queue = queue+[index_taken[index_left_eye][0]]
                    newHead = head(left_eye = index_left_eye, right_eye=index_right_eye, distance = newdist)
                    heads[index_left_eye] = newHead
                    index_taken[index_left_eye] = [index_right_eye, newdist]
        if fullHeads:
            missingheads = set(list(range(len(right_eyes)))).difference(index_taken)
        else:
            missingheads = []
        for headsita in missingheads:
            newHead = head(left_eye = headsita)
            heads[headsita] = newHead
    else:
        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(right_eyes)
        distances, from_right_to_left = neigh.kneighbors(left_eyes)
        newHead = head(left_eye = 0, right_eye = from_right_to_left[0][0])
        heads = {0:newHead}
    return heads

def show_sample(sample, ax=None, color_labels=False, is_tensor=False, **kwargs):
    """Shows a sample with landmarks"""
    if not ax:
        ax = plt.gca()
    color_list = cm.Set1.colors[: len(sample["landmarks"])]
    label_color = color_list if color_labels else "r"
    if is_tensor:
        ax.imshow(sample["image"].permute(1, 2, 0))
    else:
        ax.imshow(sample["image"])
    ax.scatter(
        sample["landmarks"][:, 0],
        sample["landmarks"][:, 1],
        s=20,
        marker=".",
        c=label_color,
    )
    ax.axis("off")
    #     ax.set_title(f'Sample #{sample["index"]}')
    return ax


def show_sample_with_mask(sample, color_labels=False, is_tensor=False, **kwargs):
    """Shows a sample with landmarks and mask"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    color_list = cm.Set1.colors[: len(sample["landmarks"])]
    label_color = color_list if color_labels else "r"
    if is_tensor:
        ax1.imshow(sample["image"].permute(1, 2, 0))
    else:
        ax1.imshow(sample["image"])
    ax1.scatter(
        sample["landmarks"][:, 0],
        sample["landmarks"][:, 1],
        s=20,
        marker=".",
        c=label_color,
    )
    ax1.axis("off")

    ax2.imshow(sample["mask"], cmap="gray")
    ax2.axis("off")
    #     ax.set_title(f'Sample #{sample["index"]}')
    return fig, (ax1, ax2)


def show_multiple_samples(samples, **kwargs):
    """Shows multiple samples with landmarks"""
    n = len(samples)
    n_cols = 4 if n > 4 else n
    n_rows = int(np.ceil(n / 4))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))
    for i, ax in enumerate(axs.flatten()):
        if i < n:
            ax = show_sample(samples[i], ax=ax, **kwargs)
        else:
            ax.axis("off")
    return fig, axs


def show_random_sample(dataset, n_samples=4, seed=None, **kwargs):
    """Shows a random sample of images with landmarks."""
    if seed:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    index_list = rng.randint(0, len(dataset), size=n_samples)
    fig, axs = show_multiple_samples([dataset[i] for i in index_list], **kwargs)
    return fig, axs


def multi_neighborhood_mask(image, landmarks):
    """
    Creates a mask in a 3 by 3 neighborhood of each landmark with a unique label
    """
    w, h = image.size
    mask = np.zeros((w, h))
    for mask_index, (x, y) in enumerate(landmarks):
        for i, j in product([-1, 0, 1], [-1, 0, 1]):
            mask[int(x + i), int(y + j)] = mask_index + 1
    return mask.T


def gen_all_masks(samples, root_dir, mask_dir, path_sub):
    """Generate all the masks
    Args:
        samples: The dataset object. Note it must be the cropped version
        root_dir: Location of root data dir
        mask_dir: A dir to story the masks. Note it must have same name as image dir
                    with path_sub[0] replace
        path_sub: A list of tuples which is used for replacement of image path. 
    """
    if not os.path.exists(root_dir + mask_dir):
        os.mkdir(root_dir + mask_dir)
    for i in tqdm(range(len(samples))):
        h, w = samples.landmark_frame.iloc[i, 1:3]
        mask = multi_neighborhood_mask(w, h, samples[i]["landmarks"])
        mask_path = samples.img_paths[i].replace(*path_sub[0]).replace(*path_sub[1])
        folder, file = os.path.split(mask_path)
        if not os.path.exists(folder):
            os.mkdir(folder)

        np.save(mask_path, mask)


def samples_to_dataframe(samples, landmark_names):
    """Creates a dataframe with the landmarks data and image size. 
    Note: this function loops over and opens every image and 
    thus it takes a while to run. The dataframe only need to be created
    once. In the future much faster operation can be performed 
    on the dataframe rather than looping over each sample. This will 
    improve development
    
    Also this code depends on the ordering of height and width returned
    by skimage defined in the dataset creation step. I only bring this 
    up because PIL and skimage are opposite. 
    
    (width, height) for PIL and (height, width) for skimage. 
    """

    df = pd.DataFrame(
        index=range(len(samples)),
        columns=["image_name", "height", "width", *landmark_names],
    )

    for i in tqdm(range(len(samples))):
        record = {}
        record["image_name"] = os.path.split(samples.img_paths[i])[-1]
        record["height"] = samples[i]["image"].shape[0]
        record["width"] = samples[i]["image"].shape[1]
        for key, value in zip(landmark_names, samples[i]["landmarks"].ravel()):
            record[key] = value
        df.iloc[i] = record

    return df


def crop_landmarks(df):
    """
    Input: landmark dataframe
    Output: cropped landmark dataframe
    """
    cropped_df = df.copy(deep=True)

    for i, row in tqdm(df.iterrows()):
        w, h = row["width"], row["height"]
        landmarks = np.array(row[3:]).reshape(-1, 2)
        cropped_landmarks = deepcopy(landmarks)
        for k, (Lx, Ly) in enumerate(landmarks):
            if ((h - 1) - Ly) <= 0:  # Bottom
                cropped_landmarks[k, 1] = (h - 1) - 1
            if Ly <= 0:  # Top
                cropped_landmarks[k, 1] = 1
            if ((w - 1) - Lx) <= 0:  # Right
                cropped_landmarks[k, 0] = (w - 1) - 1
            if Lx <= 0:  # Left
                cropped_landmarks[k, 0] = 1
        cropped_df.iloc[i, 3:] = cropped_landmarks.flatten()
    return cropped_df


def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return colors.LinearSegmentedColormap.from_list(cmap_name, color_list, N)


def calculate_landmarks_from_probs(tensor):
    landmarks = np.zeros((len(tensor) - 1, 2))

    for i, mask in enumerate(tensor):
        M = cv2.moments(mask.numpy())
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        if i > 0:
            landmarks[i - 1] = cx, cy
    return landmarks


def calculate_landmarks_from_segmentation(seg):
    labels = np.unique(seg)

    landmarks = np.zeros((len(labels) - 1, 2))

    for i, label in enumerate(labels):
        mask = (seg.numpy() == label).astype(np.float)
        M = cv2.moments(mask)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        if i > 0:
            landmarks[i - 1] = cx, cy
    return landmarks


def landmarks_DBSCAN(seg):
    """Use DBSCAN to cluster segmentation into landmarks"""

    seg = seg.numpy()
    seg_labels = np.unique(seg)

    landmarks = []

    for k, seg_label in enumerate(seg_labels):
        # Skip background
        if seg_label == 0:
            continue
        # Grab segmentation labels
        X = np.argwhere(seg == seg_label)
        # Perform DBSCAN
        db = DBSCAN(eps=10, min_samples=25).fit(X)
        labels = db.labels_
        # Create mask for core samples
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        locations = np.zeros((n_clusters_, 2))
        # For each DB cluster calculate the landmark
        for i, cluster in enumerate(set(labels)):
            if cluster == -1:
                continue
            class_member_mask = labels == cluster
            xy = X[class_member_mask & core_samples_mask]
            locations[i] = xy.mean(axis=0)

        # Sort locations by position from left to right.
        locations = locations[locations[:, 1].argsort()]
        landmarks.append(locations)

    return np.array(landmarks)


def calculate_segmentation_from_image(model, image):
    """Note image must be in the correct tensor form. """
    output = model(image.unsqueeze(0))["out"].detach().squeeze()
    probabilities = torch.nn.Softmax(dim=0)(output)
    return output, probabilities


def predict_segmentation(probabilities):
    return np.argmax(probabilities, axis=0)


def show_tensor(tensor):
    n = len(tensor)
    n_cols = 4 if n > 4 else n
    n_rows = int(np.ceil(n / 4))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))
    #     fig, axs = plt.subplots(2, 5, figsize=(10, 5))

    for i, ax in enumerate(axs.flatten()):
        if i < tensor.shape[0]:
            ax.imshow(tensor[i])
            ax.axis("off")
            ax.set_title(f"Label #{i}")
        else:
            ax.axis("off")
    fig.tight_layout()
    return fig, axs


def show_segmentation(image, landmarks, segmentation):
    n_labels = len(landmarks)

    if n_labels == 9:
        label_colors = cm.tab10.colors[1:]
    else:
        label_colors = cm.tab10.colors[1 : n_labels + 1]

    fig, ax = plt.subplots(figsize=(10, 10))

    cmap = colors.ListedColormap(cm.tab10.colors)
    boundaries = np.arange(-0.5, 10.5, 1)
    norm = colors.BoundaryNorm(boundaries, cmap.N)

    cmap = "tab10"
    ax.imshow(image)
    im = ax.imshow(segmentation, cmap=cmap, norm=norm, interpolation="none", alpha=0.5,)

    ax.scatter(landmarks[:, 0], landmarks[:, 1], s=200, c=label_colors, marker="o")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_ticks(np.arange(11))
    return fig, ax


def show_segmentation_DBSCAN(image, landmarks, segmentation):
    n_labels = len(landmarks)

    if n_labels == 9:
        label_colors = cm.tab10.colors[1:]
    else:
        label_colors = cm.tab10.colors[1 : n_labels + 1]

    fig, ax = plt.subplots(figsize=(10, 10))

    cmap = colors.ListedColormap(cm.tab10.colors)
    boundaries = np.arange(-0.5, 10.5, 1)
    norm = colors.BoundaryNorm(boundaries, cmap.N)

    cmap = "tab10"
    ax.imshow(image)
    im = ax.imshow(segmentation, cmap=cmap, norm=norm, interpolation="none", alpha=0.5,)
    for k, land in enumerate(landmarks):
        ax.scatter(land[:, 1], land[:, 0], s=200, c=f"C{k+1}", marker="o")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_ticks(np.arange(11))
    return fig, ax


def show_landmarks_by_subject(image, landmarks, bbox=False):

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)

    for k, subject in enumerate(landmarks.transpose(1, 0, 2)):
        if bbox:
            x_min, y_min = subject.min(axis=0)
            x_max, y_max = subject.max(axis=0)

            width = x_max - x_min
            height = y_max - y_min

            rect = patches.Rectangle(
                (y_min, x_min),
                height,
                width,
                linewidth=3,
                edgecolor="r",
                facecolor="none",
            )
            ax.add_patch(rect)
        ax.scatter(subject[:, 1], subject[:, 0], s=200, c=f"C{k}", marker="o")

    return fig, ax
