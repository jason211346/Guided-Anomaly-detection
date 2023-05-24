# -*- coding: utf-8 -*-

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import cv2
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as PathEffects
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score

import argparse
import os
from multiprocessing import cpu_count
import time
from pprint import pformat
import logging
import sys
from typing import Dict, Any, List, Tuple
from networks.mobilenetv3_HybridExpert import SupConMobileNetV3Large
from util_mo import *
import torch.backends.cudnn as cudnn
from due import dkl_Phison_mo, dkl_Phison_mo_s2
from lib.datasets_mo import get_dataset
from gpytorch.likelihoods import SoftmaxLikelihood
import gpytorch
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(name)s  %(levelname)s: %(message)s',
    datefmt='%y-%b-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)



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

def compute_ece(probs, targets, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    ece = 0
    bin_confidences = []
    bin_accuracies = []
    confidences, _ = torch.max(probs, dim=1)  # Compute the confidence values
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences >= bin_lower) * (confidences < bin_upper)
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            sample_indices = torch.where(in_bin)[0]
            bin_targets = targets[sample_indices]
            bin_probs = probs[sample_indices]
            true_prob_in_bin = (bin_targets == torch.argmax(bin_probs, dim=1)).float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - true_prob_in_bin) * prop_in_bin
            bin_confidences.append(avg_confidence_in_bin.item())
            bin_accuracies.append(true_prob_in_bin.item())
        else:
            bin_confidences.append(None)
            bin_accuracies.append(None)

    return ece, bin_confidences, bin_accuracies

def get_uncertainty(model_s1, model,likelihood, dataloader, tsne=False):
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    if isinstance(model,torch.nn.DataParallel):
        model = model.module
    
    model_s1.eval()
    model.eval()
    model.to(device)
    likelihood.to(device)

    uncertainty = None

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
            with gpytorch.settings.num_likelihood_samples(32):
                _, output = model_s1(images)
                output = model(output)
                output = output.to_data_independent_dist()
                output = likelihood(output).probs.mean(0)

        current_uncertainty = -(output * output.log()).sum(1)

        current_uncertainty = current_uncertainty.cpu().numpy()
        
        
        
        if uncertainty is not None:
            uncertainty = np.concatenate((uncertainty, current_uncertainty))
        else:
            uncertainty = current_uncertainty
        
    return uncertainty, labels, image_paths, name_list, full_name_list, tsne_label_list
def set_model(args, args_due,train_com_loader , num_com):
    # stage 1
    input_size = 224
    num_classes = num_com
    
    if args_due.n_inducing_points is None:
        args_due.n_inducing_points = num_classes
    n_inducing_points = args_due.n_inducing_points
    
    feature_extractor =  SupConMobileNetV3Large()

    initial_inducing_points, initial_lengthscale = dkl_Phison_mo.initial_values(
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
    feature_extractor_s1 = gpmodel_s1.feature_extractor
    feature_extractor_s1.eval()
    
    initial_inducing_points, initial_lengthscale = dkl_Phison_mo_s2.initial_values(
        train_com_loader, feature_extractor_s1, n_inducing_points*50 
    )
    
    print('initial_inducing_points : ', initial_inducing_points.shape)
    gp = dkl_Phison_mo_s2.GP(
        num_outputs=num_classes, 
        initial_lengthscale=initial_lengthscale,
        initial_inducing_points=initial_inducing_points,
        kernel=args_due.kernel,
    )

    gpmodel = dkl_Phison_mo_s2.DKL(gp)
    likelihood = SoftmaxLikelihood(num_classes=num_classes, mixing_weights=False)

    ckpt_gp = torch.load(args["gp_checkpoint_path"], map_location='cpu')
    
    if torch.cuda.is_available():
        gpmodel = gpmodel.cuda()
        likelihood = likelihood.cuda()
        cudnn.benchmark = True
        gpmodel.load_state_dict(ckpt_gp)

    
    return feature_extractor_s1, gpmodel, likelihood




def calculatePerformance(df, file_name):
    df['overkill_rate'] = (df['overkill'] / df['total'] * 100).round(decimals = 5).astype(str) + '%'
    df['leakage_rate'] = (df['leakage'] / df['total'] * 100).round(decimals = 5).astype(str) + '%'
    df = pd.concat([df, pd.DataFrame.from_records([
            {'total':sum(df['total']),
             'good':sum(df['good']),
             'bad':sum(df['bad']),
             'overkill':sum(df['overkill']), 
             'leakage':sum(df['leakage']), 
             'overkill_rate':str(round(100*(sum(df['overkill'])/sum(df['total'])),5))+'%', 
             'leakage_rate': str(round(100*(sum(df['leakage'])/sum(df['total'])),5))+'%', 
            'unknown_rate': str(round(100*(sum(df['unknown'])/sum(df['total'])),5))+'%'}])], sort=False)
    df.to_csv(file_name, index=False)




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
        "--no_calculate_uncertainty",
        action="store_false",
        dest="test_uncertainty",
        help="Don't use testing set on T-sne",
    )
    parser.add_argument(
        "--no_inference",
        action="store_false",
        dest="test_inference",
        help="Don't inference",
    )
    parser.add_argument(
        "--coeff", type=float, default=3, help="Spectral normalization coefficient"
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
        choices=['fruit_8'],
        help="Pick a dataset",
    )
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='num of workers to use')
    parser.add_argument('--dataset2', type=str, default='phison',
                        choices=['cifar10', 'cifar100', 'phison'], help='dataset')
    parser.add_argument('--size', type=int, default=224, help='parameter for RandomResizedCrop')
    parser.add_argument(
        "--n_inducing_points", type=int, help="Number of inducing points"
    )
    parser.add_argument('--relabel', action='store_true', help='relabel dataset')
    parser.add_argument(
        "-oid", "--output_inference_dir",
        type=str,
        default="output/",
        help="Directory to save output plots"
    )
    
    args: Dict[str, Any] = vars(parser.parse_args())
    args_due = parser.parse_args()
    
    print('relabel:',args_due.relabel)
    set_random_seed(args["random_seed"])

    
    # Create output directory if not exists
    if not os.path.isdir(args["output_dir"]):
        os.makedirs(args["output_dir"])
        logging.info(f"Created output directory {args['output_dir']}")
        
    if not os.path.isdir('./output/'+args_due.output_inference_dir):
        os.makedirs('./output/'+args_due.output_inference_dir)

        

    # Initialize device
    device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Initialized device {device}")

    # Load model's checkpoint
    loc = 'cuda:0'
    
    checkpoint_path: str = args["checkpoint_path"]
    checkpoint: Dict[str, Any] = torch.load(checkpoint_path, map_location="cuda:0")
    logging.info(f"Loaded checkpoint at {args['checkpoint_path']}")
    


    # Initialize dataset and dataloader
    df = pd.read_json(f"./bay_other_dataset/{args_due.output_inference_dir}_tau1_tau2_logs.json",lines=True)

    std_threshold_dict = df.iloc[df['target'].idxmax()].tolist()[1]

    exp_2_tau1 = std_threshold_dict['exp_2_tau1']
    exp_2_tau2 = std_threshold_dict['exp_2_tau2']

    print('exp_2_tau1 : ',exp_2_tau1)
    print('exp_2_tau2 : ',exp_2_tau2)

    df_train = pd.DataFrame()
    
    add_test=True    
    if args_due.dataset == 'fruit_8':
        
        # load data
        ds = get_dataset(args_due.dataset ,args_due.random_seed , root="./data" )
    #     input_size, num_classes, train_dataset, test_dataset, train_loader, train_com_loader = ds
        input_size ,num_classes , train_com_loader, train_loader, test_dataset ,train_cls_dataset,train_com_dataset, test_com_dataset = ds

        # Intialize model
        feature_extractor_s1, model ,likelihood  = set_model(args, args_due, train_com_loader , num_classes)
        
        _ ,_ ,_, _, _ ,_,_ ,_,train_com_df, train_df, val_df , _ = get_fruit_8(root="./data" , seed=args["random_seed"])

    train_val_df = pd.concat([train_df, val_df])
    
    for idx, component_name in enumerate(range(num_classes)):
        train_val_loader  = CreateDataset_for_each_component_regroup(args["random_seed"],  train_val_df,component_name)

        if args_due.test_uncertainty == True:
            # Calculate uncertainty from images in reference set

            uncertainty, labels_train, _, name_list_train, _, _ = get_uncertainty(feature_extractor_s1, model, likelihood, train_val_loader, tsne=True)

            uncertainty_mean = np.mean(uncertainty , axis=0)
            uncertainty_std = np.std(uncertainty , axis=0)
            print('uncertainty_mean : ',uncertainty_mean)
            print('uncertainty_std : ',uncertainty_std)

            uncertainty_th = uncertainty_mean + (exp_2_tau1 * uncertainty_std)
            uncertainty_th_2 = uncertainty_mean + (exp_2_tau2 * uncertainty_std)


            df_train = df_train.append({'com':component_name,'TH':uncertainty_th, 'TH_2':uncertainty_th_2 }, ignore_index=True)


        else:
            print('Uncertainty calculations are not performed')
            
    filepath = f'./output/{args_due.output_inference_dir}/uncertainty.csv'
    df_train.to_csv(filepath)

    
