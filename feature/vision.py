import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.cm as mpl_color_map
import os
import copy
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt

def save_class_activation_images(image, activation_map, file_name='try', path='visualization'):
    """
        Saves cam activation map and activation map on the original image

    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        file_name (str): File name of the exported image
    """
    tensor = (image + 1) /2
    tensor = tensor.repeat(3, 1, 1)
    tensor = tensor.permute(1, 2, 0)
    org_img = Image.fromarray((tensor*255).cpu().numpy().astype(np.uint8))
    
    if not os.path.exists(path):
        os.makedirs(path)
    # Grayscale activation map
    heatmap, heatmap_on_image = apply_colormap_on_image(org_img, activation_map, 'hsv')
    # Save colored heatmap
    path_to_file = os.path.join(path, file_name+'_Cam_Heatmap.png')
    save_image(heatmap, path_to_file)
    # Save heatmap on iamge
    path_to_file = os.path.join(path, file_name+'_Cam_On_Image.png')
    save_image(heatmap_on_image, path_to_file)
    # SAve grayscale heatmap
    path_to_file = os.path.join(path, file_name+'_Cam_Grayscale.png')
    save_image(activation_map.astype(np.uint8), path_to_file)

def apply_colormap_on_image(org_im, activation, colormap_name):
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    # Get colormap
    color_map = plt.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4
    heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))

    # Apply heatmap on iamge
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image


def save_image(im, path):
    """
        Saves a numpy matrix or PIL image as an image
    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image
    """
    if isinstance(im, (np.ndarray, np.generic)):
        im = format_np_output(im)
        im = Image.fromarray(im)
    im.save(path)

def format_np_output(np_arr):
    """
        This is a (kind of) bandaid fix to streamline saving procedure.
        It converts all the outputs to the same format which is 3xWxH
        with using sucecssive if clauses.
    Args:
        im_as_arr (Numpy array): Matrix of shape 1xWxH or WxH or 3xWxH
    """
    # Phase/Case 1: The np arr only has 2 dimensions
    # Result: Add a dimension at the beginning
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
    # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
    # Result: Repeat first channel and convert 1xWxH to 3xWxH
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
    # Phase/Case 3: Np arr is of shape 3xWxH
    # Result: Convert it to WxHx3 in order to make it saveable by PIL
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
    # Phase/Case 4: NP arr is normalized between 0-1
    # Result: Multiply with 255 and change type to make it saveable by PIL
    if np.max(np_arr) <= 1:
        np_arr = (np_arr*255).astype(np.uint8)
    return np_arr

def show_network(model):
    for name, module in model.named_modules():
        print(name)

class Display():
    def __init__(self, model, module_name, img_size=(224, 224)):
        
        super().__init__()
        self.img_size = img_size
        self.model = model
        self.module_name = module_name


    def forward(self, image):
        # setup hooks for feature map
        feature_map_hook = []

        def hook(module, fea_in, fea_out):
            feature_map_hook.append(fea_out)

        self.model.eval()
        with torch.no_grad():
            out = image.clone()
            # get the output of the final layer
            for name, module in self.model.named_modules():
                # get goal module and set up hook
                if name == self.module_name:
                    module.register_forward_hook(hook=hook)
                    break

            # get pred score and feature map
            final_out = self.model(out)
            feature_map = feature_map_hook[0]
        return feature_map, final_out

    def __call__(self, image):
        return self.forward(image)

    def generate_image(self, image):
        """
        image: PIL
        """
        with torch.no_grad():
            # get final output and feature map
            image = image.unsqueeze(0)
            feature_map, output = self.forward(image)
            # get the target output
            output = output[0]
            feature_map = feature_map.squeeze()
            channel_means = feature_map.mean(dim=[1, 2])
            weight = F.softmax(channel_means, dim=0)

            # Create empty numpy array for cam
            cam = np.ones(feature_map.shape[1:], dtype=np.float32)
            # Multiply each weight with its conv output and then, sum
            
            for i in range(len(feature_map)):

                w = weight[i] ###
                cam += w.cpu().data.numpy() * feature_map[i, :, :].cpu().data.numpy()

            cam = np.maximum(cam, 0)
            cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
            cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
            cam = np.uint8(Image.fromarray(cam).resize((224, 224), Image.LANCZOS))

        return cam


    def save(self, image, file='mamformer'):
        cam = self.generate_image(image)
        save_class_activation_images(image, cam, file)