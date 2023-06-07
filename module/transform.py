import cv2
import numpy as np

means = np.array([51.85614499, 72.03732053, 81.29577039])
stds = np.array([3.90285906, 3.30087346, 3.03204477])

def normalize_transform(images, masks):
    """Normalize and transpose mask (to align with image)
    
    Args:
        - images: one or a list of
        - masks:  one or a list of
    
    Returns:
        transformed inputs
    """
    images = (images - means) / stds
    if isinstance(masks, list):
        masks = [np.array(i.T[:, :, np.newaxis], dtype=np.float32) for i in masks]
    else:
        masks = np.array(masks.T[:, :, np.newaxis], dtype=np.float32)
    return images, masks

resize = lambda images,width,height: [cv2.resize(img,dsize=(width,height),interpolation=cv2.INTER_NEAREST) for img in images]
blur = lambda images: [cv2.medianBlur(img,3) for img in images]

def normalize_rescale(images, masks, width=768//2,height=768//2):
    """Normalize, transpose mask (to align with image), scale and blur
    
    Args:
        - images: a list of
        - masks:  a list of
    
    Returns:
        transformed inputs
    """
    images = resize(images,width,height)
    masks = resize(masks,width,height)
    images = (np.array(images) - means)/stds
    masks = np.array(np.array([i.T for i in masks]), dtype=np.float32)[..., np.newaxis]
    return images, masks

def normalize_rescale_blur(images, masks, width=768//2,height=768//2):
    """Normalize, transpose mask (to align with image), scale and blur
    
    Args:
        - images: a list of
        - masks:  a list of
    
    Returns:
        transformed inputs
    """
    images = blur(resize(images,width,height))
    masks = blur(resize(masks,width,height))
    images = (np.array(images) - means)/stds
    masks = np.array(np.array([i.T for i in masks]), dtype=np.float32)[..., np.newaxis]
    return images, masks


