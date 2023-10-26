# -*- coding: utf-8 -*-
# https://github.com/QuocThangNguyen/deep-metric-learning-tsinghua-dogs/blob/master/src/scripts/visualize_tsne.py
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import cv2
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as PathEffects
from matplotlib.colorbar import Colorbar
import matplotlib.colors as mcolors
import seaborn as sns

import argparse
import os
from multiprocessing import cpu_count
import time
from pprint import pformat
import logging
import sys
from typing import Dict, Any, List, Tuple
# from networks.mobilenetv3_SN2 import SupConMobileNetV3Large
from util_mo import *
import torch.backends.cudnn as cudnn
from due import dkl_Phison_mo, dkl_Phison_mo_s2
from due.wide_resnet_Phison_old import WideResNet
from lib.datasets_mo import get_dataset
from gpytorch.likelihoods import SoftmaxLikelihood
import gpytorch
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(name)s  %(levelname)s: %(message)s',
    datefmt='%y-%b-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)

plt.rcParams['figure.figsize'] = (32, 32)
plt.rcParams['figure.dpi'] = 150

def set_random_seed(seed: int) -> None:
    """
    Set random seed for package random, numpy and pytorch
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_plot_coordinates(image,x,y,image_centers_area_size,offset):

    image_height, image_width, _ = image.shape
    # compute the image center coordinates on the plot
    center_x = int(image_centers_area_size * x) + offset
    # in matplotlib, the y axis is directed upward
    # to have the same here, we need to mirror the y coordinate
    center_y = int(image_centers_area_size * (1 - y)) + offset

    # knowing the image center, compute the coordinates of the top left and bottom right corner
    xmin = center_x - int(image_width / 2)
    ymin = center_y - int(image_height / 2)
    xmax = xmin + image_width
    ymax = ymin + image_height

    return xmin, ymin, xmax, ymax    

def plot_scatter_all(args,args_due, X, y, X_test, y_test):    
    random.seed(1)
    tx = X[:, 0]
    ty = X[:, 1]
    
    tx_test = X_test[:, 0]
    ty_test = X_test[:, 1]
    
    figure = plt.figure()
    ax = plt.subplot(aspect="equal")
    
    for idx, num in enumerate(set(y)):
        if idx != num:
            item = np.where(np.asarray(y) == num)[0]
            for i in item:
                y[i] = idx

    colors_per_class = {}
    for i in set(y):
        colors_per_class[i] = [random.randrange(0, 255) for i in range(3)]
    
    for label in colors_per_class:
        indices = [i for i, l in enumerate(y) if l == label]
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)
        color = np.array([colors_per_class[label][::-1]], dtype=np.float) / 255        
        ax.scatter(current_tx, current_ty, lw=0, s=40, c=color, label=label)
    
    colors_per_class = {}
    colors_per_class[len(set(y))] = [0, 0, 255]
    colors_per_class[len(set(y))+1] = [0, 255, 0]
    
    for idx, label in enumerate(colors_per_class):
        indices = [i for i, l in enumerate(y_test) if l == label]
        current_tx = np.take(tx_test, indices)
        current_ty = np.take(ty_test, indices)
        color = np.array([colors_per_class[label][::-1]], dtype=np.float) / 255   
        if args_due.test_tsne == True:
            if idx == 0:
                ax.scatter(current_tx, current_ty, marker="o", c=color, label=label, s=380, edgecolors="k", linewidths=3) # good new component
            else:
                ax.scatter(current_tx, current_ty, marker="X", c=color, label=label, s=380, edgecolors="k", linewidths=3) # bad new component


    ax.axis("tight")
    ax.axis("off")
    ax.legend(loc='best')
    if args_due.test_tsne == True:
        tnse_points_path= os.path.join(args["output_dir"], "{}/{}_{}_train+test_component_HBE+_component_tsne.pdf".format(args_due.output_inference_dir,args_due.output_inference_dir, args["embedding_layer"]))
    else:
        tnse_points_path= os.path.join(args["output_dir"], "{}/{}_{}_train_component_HBE+_component_tsne.pdf".format(args_due.output_inference_dir,args_due.output_inference_dir, args["embedding_layer"]))
    plt.savefig(tnse_points_path, dpi=150)

def plot_scatter(args,args_due, X, y, X_test, y_test, tsne_initial_inducing_points):    
    tx = X[:, 0]
    ty = X[:, 1]
    tx_test = X_test[:, 0]
    ty_test = X_test[:, 1]
    
    figure = plt.figure()
    ax = plt.subplot(aspect="equal")

#     colors_per_class = {}
#     colors_per_class[0] = [0, 0, 255]
#     colors_per_class[1] = [255, 0, 0]
    for idx, num in enumerate(set(y)):
        if idx != num:
            item = np.where(np.asarray(y) == num)[0]
            for i in item:
                y[i] = idx
#     import pdb;pdb.set_trace()
    colors_per_class = {}
    for i in set(y):
        colors_per_class[i] = [random.randrange(0, 255) for i in range(3)]
    
    for label in colors_per_class:
        indices = [i for i, l in enumerate(y) if l == label]
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)
        color = np.array([colors_per_class[label][::-1]], dtype=np.float) / 255        
        ax.scatter(current_tx, current_ty, lw=0, s=40, c=color, label=label)
        
    colors_per_class = {}
    colors_per_class[len(set(y))] = [0, 0, 255]
    colors_per_class[len(set(y))+1] = [0, 255, 0]
    for idx, label in enumerate(colors_per_class):
        indices = [i for i, l in enumerate(y_test) if l == label]
        current_tx = np.take(tx_test, indices)
        current_ty = np.take(ty_test, indices)
        color = np.array([colors_per_class[label][::-1]], dtype=np.float) / 255    
        if args_due.test_tsne == True:
            if idx == 0:
                ax.scatter(current_tx, current_ty, marker="o", c=color, label=label, s=380, edgecolors="k", linewidths=3) # good new component
            else:
                ax.scatter(current_tx, current_ty, marker="X", c=color, label=label, s=380, edgecolors="k", linewidths=3) # bad new component
    
    ax.axis("tight")
    ax.axis("off")
    ax.legend(loc='best')
    if args_due.test_tsne == True:
        tnse_points_path= os.path.join(args["output_dir"], "{}/{}_{}_train+test_component_HBE+_class_tsne.pdf".format(args_due.output_inference_dir, args_due.output_inference_dir,args["embedding_layer"]))
    else:
        tnse_points_path= os.path.join(args["output_dir"], "{}/{}_{}_train_component_HBE+_class_tsne.pdf".format(args_due.output_inference_dir,args_due.output_inference_dir, args["embedding_layer"]))
    plt.savefig(tnse_points_path, dpi=150)
    
def plot_scatter_new(args,args_due, X, y, X_test, y_test):    
    random.seed(1)
    tx = X[:, 0]
    ty = X[:, 1]
    
    tx_test = X_test[:, 0]
    ty_test = X_test[:, 1]
    
    figure = plt.figure()
    ax = plt.subplot(aspect="equal")
    
    for idx, num in enumerate(set(y)):
        if idx != num:
            item = np.where(np.asarray(y) == num)[0]
            for i in item:
                y[i] = idx

    colors_per_class = {}
    for i in set(y):
        colors_per_class[i] = [random.randrange(0, 255) for i in range(3)]
    
    for label in colors_per_class:
        indices = [i for i, l in enumerate(y) if l == label]
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)
        color = np.array([colors_per_class[label][::-1]], dtype=np.float) / 255        
        ax.scatter(current_tx, current_ty, lw=0, s=40, c=color, label=label)
    
    colors_per_class = {}
    colors_per_class[len(set(y))] = [0, 0, 255]
    colors_per_class[len(set(y))+1] = [0, 255, 0]
    
    for idx, label in enumerate(colors_per_class):
        indices = [i for i, l in enumerate(y_test) if l == label]
        current_tx = np.take(tx_test, indices)
        current_ty = np.take(ty_test, indices)
        color = np.array([colors_per_class[label][::-1]], dtype=np.float) / 255   
        if args_due.test_tsne == True:
            if idx == 0:
                ax.scatter(current_tx, current_ty, marker="o", c=color, label=label, s=380, edgecolors="k", linewidths=3) # good new component
            else:
                ax.scatter(current_tx, current_ty, marker="X", c=color, label=label, s=380, edgecolors="k", linewidths=3) # bad new component


    ax.axis("tight")
    ax.axis("off")
    ax.legend(loc='best')
    if args_due.test_tsne == True:
        tnse_points_path= os.path.join(args["output_dir"], "{}/{}_{}_train+new_test_component_HBE+_component_tsne.pdf".format(args_due.output_inference_dir,args_due.output_inference_dir, args["embedding_layer"]))
    else:
        tnse_points_path= os.path.join(args["output_dir"], "{}/{}_{}_train_component_HBE+_component_tsne.pdf".format(args_due.output_inference_dir,args_due.output_inference_dir, args["embedding_layer"]))
    plt.savefig(tnse_points_path, dpi=150)
def plot_scatter_old(args,args_due, X, y, X_test, y_test):    
    random.seed(1)
    tx = X[:, 0]
    ty = X[:, 1]
    
    tx_test = X_test[:, 0]
    ty_test = X_test[:, 1]
    
    figure = plt.figure()
    ax = plt.subplot(aspect="equal")
    
    for idx, num in enumerate(set(y)):
        if idx != num:
            item = np.where(np.asarray(y) == num)[0]
            for i in item:
                y[i] = idx

    colors_per_class = {}
    for i in set(y):
        colors_per_class[i] = [random.randrange(0, 255) for i in range(3)]
    
    for label in colors_per_class:
        indices = [i for i, l in enumerate(y) if l == label]
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)
        color = np.array([colors_per_class[label][::-1]], dtype=np.float) / 255        
        ax.scatter(current_tx, current_ty, lw=0, s=40, c=color, label=label)
    
    colors_per_class = {}
    colors_per_class[len(set(y))] = [0, 0, 255]
    colors_per_class[len(set(y))+1] = [0, 255, 0]
    
    for idx, label in enumerate(colors_per_class):
        indices = [i for i, l in enumerate(y_test) if l == label]
        current_tx = np.take(tx_test, indices)
        current_ty = np.take(ty_test, indices)
        color = np.array([colors_per_class[label][::-1]], dtype=np.float) / 255   
        if args_due.test_tsne == True:
            if idx == 0:
                ax.scatter(current_tx, current_ty, marker="o", c=color, label=label, s=380, edgecolors="k", linewidths=3) # good new component
            else:
                ax.scatter(current_tx, current_ty, marker="X", c=color, label=label, s=380, edgecolors="k", linewidths=3) # bad new component


    ax.axis("tight")
    ax.axis("off")
    ax.legend(loc='best')
    if args_due.test_tsne == True:
        tnse_points_path= os.path.join(args["output_dir"], "{}/{}_{}_train+old_test_component_HBE+_component_tsne.pdf".format(args_due.output_inference_dir,args_due.output_inference_dir, args["embedding_layer"]))
    else:
        tnse_points_path= os.path.join(args["output_dir"], "{}/{}_{}_train_component_HBE+_component_tsne.pdf".format(args_due.output_inference_dir,args_due.output_inference_dir, args["embedding_layer"]))
    plt.savefig(tnse_points_path, dpi=150)
    
def plot_uncertainty(args,args_due, X, y, X_test, y_test , uncertainty_data, name):    
    random.seed(1)
    tx = X[:, 0]
    ty = X[:, 1]
    
    tx_test = X_test[:, 0]
    ty_test = X_test[:, 1]
    
    # Create a colormap for colors based on uncertainty_data
    colormap = plt.cm.ScalarMappable(cmap='rainbow')
    colormap.set_array(uncertainty_data)

    figure = plt.figure()
    ax = plt.subplot(aspect="equal")
    
    for idx, num in enumerate(set(y)):
        if idx != num:
            item = np.where(np.asarray(y) == num)[0]
            for i in item:
                y[i] = idx
    tx_train_test = np.concatenate((tx, tx_test), axis=0)
    ty_train_test = np.concatenate((ty, ty_test), axis=0)
#     import pdb;pdb.set_trace()
#     colors = ['r' if u > 0.1 else 'b' for u in uncertainty_data]
#     cbar = Colorbar(plt.cm.ScalarMappable(norm=mcolors.Normalize(vmin=min(uncertainty_data), vmax=max(uncertainty_data), cmap='coolwarm')), ax=plt.gca(), label='Uncertainty')

#     cbar.set_ticks([0, 0.1, 0.2]) 
#     import pdb;pdb.set_trace()
#     min_value = np.min(uncertainty_data)
#     max_value = np.max(uncertainty_data)
# #     normalized_values = (uncertainty_data_train - min_value) / (max_value - min_value)
    
#     uncertainty_data_train = uncertainty_data[0:len(tx)]
#     uncertainty_data_test = uncertainty_data[len(tx)::]
#     normalized_values_train = (uncertainty_data_train - min_value) / (max_value - min_value)
#     normalized_values_test = (uncertainty_data_test - min_value) / (max_value - min_value)
    
    ax.scatter(tx, ty, lw=0, s=200, c= colormap.to_rgba(uncertainty_data[0:len(ty)]))
    
    for idx, label in enumerate(y_test):
#         import pdb;pdb.set_trace()
#         indices = [i for i, l in enumerate(y_test) if l == label]
        current_tx = np.take(tx_test, idx)
        current_ty = np.take(ty_test, idx)
        color = uncertainty_data[len(ty)+idx]   
#         if args_due.test_tsne == True:
        if label == 8:
            ax.scatter(current_tx, current_ty, marker="o", c=colormap.to_rgba(color), s=600, edgecolors="k", linewidths=3) # good new component
        else:
            ax.scatter(current_tx, current_ty, marker="X", c=colormap.to_rgba(color), s=600, edgecolors="k", linewidths=3) # bad new component
#     ax.scatter(tx_test, ty_test, lw=0, s=380, c=normalized_values_test)
    plt.colorbar(colormap, label='Uncertainty')
    ax.axis("tight")
    ax.axis("off")
    ax.legend(loc='best')
    
    tnse_points_path= os.path.join(args["output_dir"], "{}/{}_{}_{}_HBE_uncertainty_tsne.pdf".format(args_due.output_inference_dir,args_due.output_inference_dir, args["embedding_layer"], name))
    plt.savefig(tnse_points_path, dpi=150)

def plot_scatter_iip(args,args_due, X, y, X_test, y_test, tsne_initial_inducing_points):    
    random.seed(1)
    tx = X[:, 0]
    ty = X[:, 1]
    
    tx_test = X_test[:, 0]
    ty_test = X_test[:, 1]
    
    tx_ip = tsne_initial_inducing_points[:, 0]
    ty_ip = tsne_initial_inducing_points[:, 1]
    
    figure = plt.figure()
    ax = plt.subplot(aspect="equal")
    
    for idx, num in enumerate(set(y)):
        if idx != num:
            item = np.where(np.asarray(y) == num)[0]
            for i in item:
                y[i] = idx
#     import pdb;pdb.set_trace()
    colors_per_class = {}
    for i in set(y):
        colors_per_class[i] = [random.randrange(0, 255) for i in range(3)]
    
    for label in colors_per_class:
        indices = [i for i, l in enumerate(y) if l == label]
#         import pdb;pdb.set_trace()
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)
        color = np.array([colors_per_class[label][::-1]], dtype=np.float) / 255        
        ax.scatter(current_tx, current_ty, lw=0, s=40, c=color, label=label)
    
    colors_per_class = {}
    colors_per_class[len(set(y))] = [0, 0, 255]
    colors_per_class[len(set(y))+1] = [0, 255, 0]
        
    color = np.array([colors_per_class[len(set(y))][::-1]], dtype=np.float) / 255 
    ax.scatter(tx_ip, ty_ip, marker="o", c=color,label=13,  s=380, edgecolors="k", linewidths=3)
#     import pdb;pdb.set_trace()
    ax.axis("tight")
    ax.axis("off")
    ax.legend(loc='best')

    tnse_points_path= os.path.join(args["output_dir"], "{}/{}_{}_train_component_HBE+_component_tsne_iip.pdf".format(args_due.output_inference_dir,args_due.output_inference_dir, args["embedding_layer"]))
    plt.savefig(tnse_points_path, dpi=150)

def set_model(args, args_due,train_com_loader , num_com):
    # stage 1
    input_size = 224
    num_classes = num_com
    
    if args_due.n_inducing_points is None:
        args_due.n_inducing_points = num_classes
    n_inducing_points = args_due.n_inducing_points
    
    if args_due.coeff == 1:
        from networks.mobilenetv3_SN1 import SupConMobileNetV3Large
    elif args_due.coeff == 3:
        from networks.mobilenetv3_SN3 import SupConMobileNetV3Large
    elif args_due.coeff == 5:
        from networks.mobilenetv3_SN2 import SupConMobileNetV3Large
    elif args_due.coeff == 7:
        from networks.mobilenetv3_SN4 import SupConMobileNetV3Large
    elif args_due.coeff == 0:
        from networks.mobilenetv3 import SupConMobileNetV3Large
    
    feature_extractor =  SupConMobileNetV3Large()

    initial_inducing_points, initial_lengthscale = dkl_Phison_mo.initial_values3(
            train_com_loader, feature_extractor, n_inducing_points
        )

    gp = dkl_Phison_mo.GP(
            num_outputs=num_classes, #可能=conponent 數量 = 23個 
            initial_lengthscale=initial_lengthscale,
            initial_inducing_points=initial_inducing_points,
            kernel=args_due.kernel,
    )

    gpmodel_s1 = dkl_Phison_mo.DKL(feature_extractor, gp)
    likelihood = SoftmaxLikelihood(num_classes=num_classes, mixing_weights=False)
    
    
    ckpt = torch.load(args["checkpoint_path"], map_location='cpu')


    if torch.cuda.is_available():
        gpmodel_s1 = gpmodel_s1.cuda()
        likelihood = likelihood.cuda()
        cudnn.benchmark = True
        gpmodel_s1.load_state_dict(ckpt)
    

    # stage 2 
#     feature_extractor_s1 = gpmodel_s1.feature_extractor
#     feature_extractor_s1.eval()

#     initial_inducing_points, initial_lengthscale = dkl_Phison_mo_s2.initial_values3(
#         train_com_loader, feature_extractor_s1, n_inducing_points*50 # if hparams.n_inducing_points= none ,hparams.n_inducing_points = num_class
#     )

#     print('initial_inducing_points : ', initial_inducing_points.shape)
#     gp = dkl_Phison_mo_s2.GP(
#         num_outputs=num_classes, #可能=conponent 數量 = 23個 
#         initial_lengthscale=initial_lengthscale,
#         initial_inducing_points=initial_inducing_points,
#         kernel=args_due.kernel,
#     )

#     gpmodel = dkl_Phison_mo_s2.DKL(gp)
#     likelihood = SoftmaxLikelihood(num_classes=num_classes, mixing_weights=False)

#     ckpt_gp = torch.load(args["gp_checkpoint_path"], map_location='cpu')

#     if torch.cuda.is_available():
#         gpmodel = gpmodel.cuda()
#         likelihood = likelihood.cuda()
#         cudnn.benchmark = True
#         gpmodel.load_state_dict(ckpt_gp)


    return gpmodel_s1, likelihood
def get_uncertainty(model,likelihood, dataloader, tsne=False):
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    if isinstance(model,torch.nn.DataParallel):
        model = model.module
    
    model.eval()
    model.to(device)
    likelihood.to(device)
    # we'll store the features as NumPy array of size num_images x feature_size
    uncertainty = None
    cls_uncertainty = None
    
    # we'll also store the image labels and paths to visualize them later
    labels = []
    image_paths = []
    name_list = []
    full_name_list = []
    tsne_label_list = []

    print("Start calculate uncertainty")
    for i, data in enumerate(tqdm(dataloader)):

        try:
            img, target, file_path, name, full_name = data
        except:
            img, target, file_path, name = data

#         feat_list = []
#         def hook(module, input, output):
#             feat_list.append(output.clone().detach())

        images = img.to(device)
        target = target.squeeze().tolist()

        if isinstance(target, str):
            labels.append(target)
        else:
            if isinstance(target, int):
                labels.append(target)
            else:
                for p in file_path:
                    image_paths.append(p)
                for lb in target:
                    labels.append(lb)        
        try:
            if name is not None:
                if isinstance(name, list) is False:
                    name = name.squeeze().tolist()
                if isinstance(name, int):
                    name_list.append(name)
                if isinstance(name, list):
                    for n in name:
                        name_list.append(n)
        
            if full_name:
                for k in full_name:
                    full_name_list.append(k)

            if tsne:
                for tsne_lb in tsne_label:
                    tsne_label_list.append(tsne_lb)
        except:
            pass

        with torch.no_grad():                
#             _, output = model(images)

            with gpytorch.settings.num_likelihood_samples(32):
                cls_output , output = model(images)
                output = output.to_data_independent_dist()
                output = likelihood(output).probs.mean(0)
                
#                 cls_output = cls_output.to_data_independent_dist()
#                 cls_output = likelihood_cls(cls_output).probs.mean(0)
            

                
        current_uncertainty = -(output * output.log()).sum(1)

#         current_uncertainty = -(output * output.log()).sum(1)
        current_uncertainty = -(output * output.log()).sum(1) / torch.log(torch.tensor(output.shape[1], dtype=torch.float))
        current_uncertainty = current_uncertainty.cpu().numpy()
        
        cls_current_uncertainty = -(cls_output * cls_output.log()).sum(1) / torch.log(torch.tensor(output.shape[1], dtype=torch.float))

        cls_current_uncertainty = cls_current_uncertainty.cpu().numpy()
        
        if uncertainty is not None:
            uncertainty = np.concatenate((uncertainty, current_uncertainty))
        else:
            uncertainty = current_uncertainty
            
        if cls_uncertainty is not None:
            cls_uncertainty = np.concatenate((cls_uncertainty, cls_current_uncertainty))
        else:
            cls_uncertainty = cls_current_uncertainty
        
    return uncertainty, labels, image_paths, name_list, full_name_list, tsne_label_list ,cls_uncertainty


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualizing embeddings with T-SNE")

    parser.add_argument(
        "-c", "--checkpoint_path",
        type=str,
        default ="",      # checkpoint.pth.tar
        help="Path to model's checkpoint."
    )
    parser.add_argument(
        "-gp_c", "--gp_checkpoint_path",
        type=str,
        default ="",      # checkpoint.pth.tar
        help="Path to model's checkpoint."
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        default="output/",
        help="Directory to save output plots"
    )
    parser.add_argument(
        "--embedding_layer",
        type=str,
        default="shared_embedding",
        help="Which embedding to visualization( encoder or shared_embedding)"
    )
    parser.add_argument(
        "--name",
        type=int,
        default=15,
        help="Test component name"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=1,
        help="Random seed"
    )
    parser.add_argument(
        "--no_spectral_conv",
        action="store_false",
        dest="spectral_conv",
        help="Don't use spectral normalization on the convolutions",
    )

    parser.add_argument(
        "--no_spectral_bn",
        action="store_false",
        dest="spectral_bn",
        help="Don't use spectral normalization on the batch normalization layers",
    )
    parser.add_argument(
        "--no_test_tsne",
        action="store_false",
        dest="test_tsne",
        help="Don't use testing set on T-sne",
    )
    parser.add_argument(
        "--coeff", type=float, default=1, help="Spectral normalization coefficient"
    )
    parser.add_argument("--dropout_rate", type=float, default=0.3, help="Dropout rate")
    parser.add_argument(
        "--n_power_iterations", default=1, type=int, help="Number of power iterations"
    )
    parser.add_argument(
        "--kernel",
        default="RBF",
        choices=["RBF", "RQ", "Matern12", "Matern32", "Matern52"],
        help="Pick a kernel",
    )
    parser.add_argument(
        "--dataset",
        default="fruit_8",
        choices=["CIFAR10", "CIFAR100", "PHISON",'PHISON_regroup','fruit'],
        help="Pick a dataset",
    )
    parser.add_argument(
        "--n_inducing_points", type=int, help="Number of inducing points"
    )
    parser.add_argument(
        "--n_inducing_points_cls", type=int, help="Number of inducing points"
    )
    parser.add_argument(
        "-oid", "--output_inference_dir",
        type=str,
        default="output/",
        help="Directory to save output plots"
    )
    parser.add_argument('--relabel', action='store_true', help='relabel dataset')
    args: Dict[str, Any] = vars(parser.parse_args())
    args_due = parser.parse_args()
    print('test_tsne:',args_due.test_tsne)
    set_random_seed(args["random_seed"])

    # Create output directory if not exists
    if not os.path.isdir(args["output_dir"]+args["output_inference_dir"]):
        os.makedirs(args["output_dir"]+args["output_inference_dir"])
        logging.info(f"Created output directory {args['output_inference_dir']}")

    # Initialize device
    device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Initialized device {device}")

    # Load model's checkpoint
    loc = 'cuda:0'
    
    checkpoint_path: str = args["checkpoint_path"]
    checkpoint: Dict[str, Any] = torch.load(checkpoint_path, map_location="cuda:0")
    logging.info(f"Loaded checkpoint at {args['checkpoint_path']}")
    ds = get_dataset(args_due.dataset ,args_due.random_seed , root="./data")
#     input_size ,num_classes , train_com_loader, train_loader, test_dataset ,train_cls_dataset,train_com_dataset = ds
    input_size ,num_classes , train_com_loader, train_loader, test_dataset ,train_cls_dataset,train_com_dataset, test_com_dataset = ds

#     clust = Clustimage(method='pca')
#     clust.load(f'/root/notebooks/DUE/clust/{args_due.random_seed}_pretrain_all_clustimage_model')
#     num_com = len(set(clust.results['labels']))+2
    # Intialize model
    gpmodel_s1, likelihood = set_model(args, args_due, train_com_loader , num_classes)

#     import pdb;pdb.set_trace()

    feature_extractor = gpmodel_s1.feature_extractor.encoder
    inducing_points, initial_lengthscale = dkl_Phison_mo.initial_values2(
        train_com_loader, feature_extractor, args_due.n_inducing_points
    )
#     inducing_points = gpmodel.com_out.variational_strategy.base_variational_strategy.inducing_points
#     import pdb;pdb.set_trace()
    # Initialize dataset and dataloader

#     training_loader, validation_loader, test_loader = CreateTSNEdataset_regroup(args["random_seed"], tsne=True)
    training_loader, train_com_loader, new_test_loader, old_test_loader, test_loader = CreateTSNEdataset_regroup_fruit_8(args["random_seed"], tsne=True)


    # Calculate embeddings from images in reference set
    start = time.time()
    embeddings_train, labels_train, _, name_list_train, _, _ = get_features_trained_weight(gpmodel_s1, train_com_loader, embedding_layer=args["embedding_layer"], tsne=True)
    
    embeddings_test, labels_test, _, name_list_test, _, _ = get_features_trained_weight(gpmodel_s1, test_loader, embedding_layer=args["embedding_layer"], tsne=True)  
    
    new_embeddings_test, new_labels_test, _, new_name_list_test, _, _ = get_features_trained_weight(gpmodel_s1, new_test_loader, embedding_layer=args["embedding_layer"], tsne=True)   
    old_embeddings_test, old_labels_test, _, old_name_list_test, _, _ = get_features_trained_weight(gpmodel_s1, old_test_loader, embedding_layer=args["embedding_layer"], tsne=True) 
    
    uncertainty_train, labels_train, _, name_list_train, _, _,cls_uncertainty_train = get_uncertainty(gpmodel_s1, likelihood, train_com_loader, tsne=True)
    uncertainty_test, labels_test, _, name_list_test, _, _,cls_uncertainty_test = get_uncertainty(gpmodel_s1, likelihood, test_loader, tsne=True)
    new_uncertainty_test, new_labels_test, _, new_name_list_test, _, _,new_cls_uncertainty_test = get_uncertainty(gpmodel_s1, likelihood, new_test_loader, tsne=True)
    old_uncertainty_test, old_labels_test, _, old_name_list_test, _, _,old_cls_uncertainty_test = get_uncertainty(gpmodel_s1, likelihood, old_test_loader, tsne=True)
    
#     uncertainty_data = np.concatenate((uncertainty_train, uncertainty_test), axis=0)
    uncertainty_data = np.concatenate((uncertainty_train, new_uncertainty_test, old_uncertainty_test), axis=0)
    new_uncertainty_data = np.concatenate((uncertainty_train, new_uncertainty_test), axis=0)
    old_uncertainty_data = np.concatenate((uncertainty_train, old_uncertainty_test), axis=0)
    
    end = time.time()
#     logging.info(f"Calculated {len(embeddings_train)+len(embeddings_test)} embeddings: {end - start} second")

#     import pdb;pdb.set_trace()


    # Train + Test set
    
    inducing_points = inducing_points.cpu().detach().numpy()
    embeddings = np.concatenate((embeddings_train, embeddings_test, new_embeddings_test, old_embeddings_test, inducing_points), axis=0)
    
    name_list_train_ori= name_list_train
    labels_train_ori = labels_train
    
    component_labels_test_mapping = []
    for lb in labels_test:
        if lb == 0:
            component_labels_test_mapping.append(len(set(name_list_train)))
        if lb == 1:
            component_labels_test_mapping.append(len(set(name_list_train))+1)
    
    class_labels_test_mapping = []
    for lb in labels_test:
        if lb == 0:
            class_labels_test_mapping.append(len(set(labels_train)))
        if lb == 1:
            class_labels_test_mapping.append(len(set(labels_train))+1)
    
    new_component_labels_test_mapping = []
    for lb in new_labels_test:
        if lb == 0:
            new_component_labels_test_mapping.append(len(set(name_list_train_ori)))
        if lb == 1:
            new_component_labels_test_mapping.append(len(set(name_list_train_ori))+1)
    
    new_class_labels_test_mapping = []
    for lb in new_labels_test:
        if lb == 0:
            new_class_labels_test_mapping.append(len(set(labels_train_ori)))
        if lb == 1:
            new_class_labels_test_mapping.append(len(set(labels_train_ori))+1)
            
    old_component_labels_test_mapping = []
    for lb in old_labels_test:
        if lb == 0:
            old_component_labels_test_mapping.append(len(set(name_list_train_ori)))
        if lb == 1:
            old_component_labels_test_mapping.append(len(set(name_list_train_ori))+1)
    
    old_class_labels_test_mapping = []
    for lb in old_labels_test:
        if lb == 0:
            old_class_labels_test_mapping.append(len(set(labels_train_ori)))
        if lb == 1:
            old_class_labels_test_mapping.append(len(set(labels_train_ori))+1)
    
    # Init T-SNE
    tsne = TSNE(n_components=2, random_state=12345, perplexity=35, learning_rate=200, n_iter=2000, n_jobs=-1)
    X_transformed = tsne.fit_transform(embeddings)    
    
    
    tsne_train = X_transformed[0:len(embeddings_train)]
#     tsne_test = X_transformed[len(embeddings_train):(len(embeddings_train)+len(embeddings_test))]
    tsne_new_test = X_transformed[(len(embeddings_train)+len(embeddings_test)):((len(embeddings_train)+len(embeddings_test))+len(new_embeddings_test))] 
    tsne_old_test = X_transformed[(len(embeddings_train)+len(embeddings_test)+len(new_embeddings_test)):(len(embeddings_train)+len(embeddings_test)+len(new_embeddings_test)+len(old_embeddings_test))] 
    tsne_initial_inducing_points = X_transformed[(len(embeddings_train)+len(embeddings_test)+len(new_embeddings_test)+len(old_embeddings_test))::]
    
    tsne_all_test = np.concatenate((tsne_new_test, tsne_old_test), axis=0)
    all_component_labels_test_mapping = new_component_labels_test_mapping + old_component_labels_test_mapping

    plot_scatter_all(args, args_due, tsne_train, name_list_train, tsne_all_test, all_component_labels_test_mapping )
#     plot_scatter_iip(args, args_due, tsne_train, name_list_train, tsne_test, component_labels_test_mapping ,tsne_initial_inducing_points)

    plot_scatter_new(args, args_due, tsne_train, name_list_train, tsne_new_test, new_component_labels_test_mapping )
    plot_scatter_old(args, args_due, tsne_train, name_list_train, tsne_old_test, old_component_labels_test_mapping )
    
#     labels = labels_train+labels_test
#     plot_scatter(args, args_due, tsne_train, labels_train, tsne_test, class_labels_test_mapping, tsne_initial_inducing_points)
    name = 'new'
    plot_uncertainty(args, args_due, tsne_train, name_list_train, tsne_new_test, new_component_labels_test_mapping ,new_uncertainty_data ,name)
    name = 'old'
    plot_uncertainty(args, args_due, tsne_train, name_list_train, tsne_old_test, old_component_labels_test_mapping ,old_uncertainty_data,name )
    name = 'all'
    plot_uncertainty(args, args_due, tsne_train, name_list_train, tsne_all_test, all_component_labels_test_mapping ,uncertainty_data ,name)
