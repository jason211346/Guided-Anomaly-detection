# -*- coding: utf-8 -*-
# https://github.com/QuocThangNguyen/deep-metric-learning-tsinghua-dogs/blob/master/src/scripts/visualize_tsne.py
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import cv2
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as PathEffects
import seaborn as sns
import pandas as pd
from tqdm import tqdm
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
from due import dkl_Phison_mo, dkl_Phison_mo_s2
from due.wide_resnet_Phison_old import WideResNet
from lib.datasets_mo import get_dataset
from gpytorch.likelihoods import SoftmaxLikelihood
import gpytorch
from bayes_opt import BayesianOptimization
from bayes_opt import SequentialDomainReductionTransformer
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(name)s  %(levelname)s: %(message)s',
    datefmt='%y-%b-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)

pbounds = {
           'exp_2_tau1': (0, 15), 'exp_2_tau2': (0, 15),
           }



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

def set_loader(args_due):
    # construct data loader
    if args_due.dataset2 == 'phison':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        raise ValueError('dataset not supported: {}'.format(args_due.dataset2))
    normalize = transforms.Normalize(mean=mean, std=std)

    val_transform = transforms.Compose([
        transforms.Resize([args_due.size, args_due.size]),
        transforms.ToTensor(),
        normalize,
    ])
    if args_due.dataset2 == 'phison':
        if args_due.dataset == 'PHISON':
            train_df, val_df, test_df, train_component_label, val_component_label, test_component_label, train_com_df = CreateDataset_relabel(args_due.random_seed, testing=None)  
        if args_due.dataset == 'PHISON_regroup3':
            train_df, val_df, test_df, train_component_label, val_component_label, test_component_label, train_com_df = CreateDataset_regroup_due_2_seed1212(args_due.random_seed,add_test=True)
        if args_due.dataset == 'fruit_8':
            _ ,_ ,_, _, _ ,_, _,_,_ , train_df, val_df , test_df = get_fruit_8(root="./data" , seed=args_due.random_seed)
            train_component_label = val_component_label = test_component_label = [0,1,2,3,4,5,6,7]

        train_dataset = CustomDataset(train_df, transform=val_transform)
        val_dataset = CustomDataset(val_df, transform=val_transform)
        test_dataset = CustomDataset(test_df, transform=val_transform)
        
        train_val_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])
        
        test_df = pd.concat([train_df, val_df])
        
        test_df_mapping2_label = test_df.copy()    
        name_of_each_component = test_df_mapping2_label['component_name'].value_counts().index.tolist()
        num_of_image_in_each_component = test_df_mapping2_label['component_name'].value_counts().values
        test_component_name_df = pd.DataFrame(list(zip(name_of_each_component, num_of_image_in_each_component)), columns =['component_name', 'total'])

        for name in set(test_df_mapping2_label['component_name'].values):
            temp_data = test_df_mapping2_label.loc[(test_df_mapping2_label["component_name"] == name)]
            for k, v in zip(temp_data['class'].value_counts().keys(), temp_data['class'].value_counts()):
                if k == 0:
                    test_component_name_df.loc[test_component_name_df['component_name'] == name, 'good'] = temp_data['class'].value_counts().sort_index().values[0]
                elif k ==1:
                    try:
                        test_component_name_df.loc[test_component_name_df['component_name'] == name, 'bad'] = temp_data['class'].value_counts().sort_index().values[1]
                    except:
                        print(f"{name} only contains bad label.")
                        test_component_name_df.loc[test_component_name_df['component_name'] == name, 'bad'] = temp_data['class'].value_counts().sort_index().values[0]
        test_component_name_df['good'] = test_component_name_df['good'].fillna(0).astype(int)
        try:
            test_component_name_df['bad'] = test_component_name_df['bad'].fillna(0).astype(int)
            test_component_name_df = test_component_name_df[['component_name', 'total', 'good', 'bad']]    
        except:
            test_component_name_df = test_component_name_df[['component_name', 'total', 'good']]        
        
        col = {'overkill': 0, 'leakage': 0, 'unknown':  0,}
        test_component_name_df = test_component_name_df.assign(**col)
        
    else:
        raise ValueError(args_due.dataset2)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args_due.batch_size, shuffle=False,
        num_workers=args_due.num_workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args_due.batch_size, shuffle=False,
        num_workers=args_due.num_workers, pin_memory=True)
    
    train_val_loader = torch.utils.data.DataLoader(
        train_val_dataset, batch_size=args_due.batch_size, shuffle=False,
        num_workers=args_due.num_workers, pin_memory=True)
    
    return val_loader, test_loader, test_component_name_df, train_component_label, val_component_label, test_component_label ,train_val_loader

def get_uncertainty(model_s1, model,likelihood, dataloader, tsne=False):
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    if isinstance(model,torch.nn.DataParallel):
        model = model.module
    
    model_s1.eval()
    model.eval()
    model_s1.to(device)
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
        gpmodel_s1.load_state_dict(ckpt)
    

    # stage 2 
    feature_extractor_s1 = gpmodel_s1.feature_extractor
    feature_extractor_s1.eval()
    
    initial_inducing_points, initial_lengthscale = dkl_Phison_mo_s2.initial_values(
        train_com_loader, feature_extractor_s1, n_inducing_points*50 # if hparams.n_inducing_points= none ,hparams.n_inducing_points = num_class
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
#         cudnn.benchmark = True
        gpmodel.load_state_dict(ckpt_gp)

    
    return feature_extractor_s1, gpmodel, likelihood


def calculatePerformance(df, file_name):
    df['overkill_rate'] = (df['overkill'] / df['total']).round(decimals = 5).astype(str)
    df['leakage_rate'] = (df['leakage'] / df['total']).round(decimals = 5).astype(str)
    df['unknown_rate'] = (df['unknown'] / df['total']).round(decimals = 5).astype(str)
    df = pd.concat([df, pd.DataFrame.from_records([
            {'total':sum(df['total']),
             'good':sum(df['good']),
             'bad':sum(df['bad']),
             'overkill':sum(df['overkill']), 
             'leakage':sum(df['leakage']), 
             'unknown':sum(df['unknown']), 
             'overkill_rate':round(1*(sum(df['overkill'])/sum(df['total'])),5), 
             'leakage_rate': round(1*(sum(df['leakage'])/sum(df['total'])),5), 
            'unknown_rate': round(1*(sum(df['unknown'])/sum(df['total'])),5)
            }
    ]
    )], sort=False)
    score =  1 / (3 / ( 1/np.exp(float(df['overkill_rate'].tail(1))) + 1/np.exp(float(df['leakage_rate'].tail(1))) + 1/np.exp(float(df['unknown_rate'].tail(1)))))
    df.to_csv(file_name, index=False)
    return score


def EXP1(args_due ,model ,test_loader ,df ,train_component_label, val_component_label, test_component_label ):
    
    model.eval()
    
    df_all = df.copy()
    df_ind = df.copy()
    df_ood = df.copy()
    
    top1 = AverageMeter()
    
    y_pred = []
    y_true = []
    y_true_IND = []
    y_pred_IND = []
    y_true_OOD = []
    y_pred_OOD = []
    
    with torch.no_grad():
        for idx, (images, labels, _, component_name) in enumerate(tqdm(test_loader)):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]
            
            cls_output, component_out = model(images)
            _, prediction=torch.max(cls_output.data, 1)
            acc1 = accuracy(cls_output, labels, topk=(1))
            top1.update(acc1[0].item(), bsz)
            y_pred.extend(prediction.view(-1).detach().cpu().numpy())
            y_true.extend(labels.view(-1).detach().cpu().numpy())
            # ALL test samples
            for idx, (gt, pred, name) in enumerate(list(zip(labels.data, prediction.data, component_name))):
                if gt.item() == 0 and gt.item() != pred.item():
                    if isinstance(name, str):
                        df_all.loc[(df_all["component_name"] == name), 'overkill'] +=1
                    else:
                        df_all.loc[(df_all["component_name"] == name.item()), 'overkill'] +=1
                if gt.item() == 1 and gt.item() != pred.item():
                    if isinstance(name, str):
                        df_all.loc[(df_all["component_name"] == name), 'leakage'] +=1
                    else:
                        df_all.loc[(df_all["component_name"] == name.item()), 'leakage'] +=1
                        
    print(' * EXP1_Acc@1 {top1.avg:.3f}\n'.format(top1=top1))
    # ALL test set
#     calculatePerformance(df_all, file_name=f'./output/{args_due.output_inference_dir}/DUE_dataset_{args_due.output_inference_dir}_EXP1_ALL_test_set_overkill_and_leakage.csv')
    
    return y_true, y_pred

def EXP2(args_due ,model_s1 ,model ,test_loader ,df ,train_component_label, val_component_label, test_component_label ,likelihood ,TH , TH_2,good_num_com):
    
    model.eval()
    model_s1.eval()

    df_all = df.copy()
    df_ind = df.copy()
    df_ood = df.copy()
    
    top1 = AverageMeter()
    
    y_pred = []
    y_true = []
    y_com = []
    y_true_IND = []
    y_pred_IND = []
    y_true_OOD = []
    y_pred_OOD = []
    
    with torch.no_grad():
        for idx, (images, labels, _, component_name) in enumerate(tqdm(test_loader)):
            images = images.float().cuda()
            labels = labels.cuda()
            component_name = component_name.cuda()
            bsz = labels.shape[0]
            
            
            with gpytorch.settings.num_likelihood_samples(1000):
                _, component_out = model_s1(images)
                component_out = model(component_out)
                component_out = component_out.to_data_independent_dist()
                component_out = likelihood(component_out).probs.mean(0)
            
            current_uncertainty = -(component_out * component_out.log()).sum(1)
            
            _, prediction=torch.max(component_out.data, 1)            
            
            unk_list=[]
            bad_list=[]
            for i in range(len(current_uncertainty)):
                uncertainty_th = TH[prediction[i].item()]
                uncertainty_th_2 = TH_2[prediction[i].item()]
        
                if current_uncertainty[i]>=uncertainty_th and current_uncertainty[i]<=uncertainty_th_2:
                    bad_list.append(i)
                if current_uncertainty[i]>uncertainty_th_2:
                    unk_list.append(i)


            prediction = [0 if x < good_num_com else 1 for x in prediction] 
            for i in bad_list:
                prediction[i] = 1    
            for i in unk_list:
                prediction[i] = 2   

            y_true.extend(labels.view(-1).detach().cpu().numpy())
            y_com.extend(component_name)
            
            prediction=torch.Tensor(prediction)

            y_pred.extend(prediction.view(-1).detach().cpu().numpy())
             
            # ALL test samples 
            for idx, (gt, pred, name) in enumerate(list(zip(labels.data, prediction.data, component_name))):
                
                if pred.item() == 2:
                    if isinstance(name, str):
                        df_all.loc[(df_all["component_name"] == name), 'unknown'] +=1
                    else:
                        df_all.loc[(df_all["component_name"] == name.item()), 'unknown'] +=1
                if pred.item() != 2:
                    if gt.item() == 0 and gt.item() != pred.item():
                        if isinstance(name, str):
                            df_all.loc[(df_all["component_name"] == name), 'overkill'] +=1
                        else:
                            df_all.loc[(df_all["component_name"] == name.item()), 'overkill'] +=1
                    if gt.item() == 1 and gt.item() != pred.item():
                        if isinstance(name, str):
                            df_all.loc[(df_all["component_name"] == name), 'leakage'] +=1
                        else:
                            df_all.loc[(df_all["component_name"] == name.item()), 'leakage'] +=1


#     print(' * EXP2_Acc@1 {top1.avg:.3f}\n'.format(top1=top1))
    # ALL test set
#     calculatePerformance(df_all, file_name=f'./output/{args_due.output_inference_dir}/DUE_dataset_{args_due.output_inference_dir}_EXP2_ALL_test_set_overkill_and_leakage_unk.csv')
    
    return y_com, y_pred 

def HybridExpert(y_pred_expert1, y_pred_expert2, y_true, name_label_list,  df, args_due, train_component_label, val_component_label, test_component_label):

    
    gt_label_after_unknown = []
    pred_after_unknown = []
    
    y_pred_all = []
    y_true_all = []
    
    prediction_IND = []
    gt_label_IND = []
    prediction_OOD = []
    gt_label_OOD = []
    
    df_ind = df.copy()
    df_ood = df.copy()
    
    for name in df_ind["component_name"].value_counts().index:
        if name not in train_component_label:
            df_ind = df_ind[df_ind["component_name"] != name]
    for name in df_ood["component_name"].value_counts().index:
        if name not in test_component_label:
            df_ood = df_ood[df_ood["component_name"] != name]
    
            
    for idx, (pred_expert1, pred_expert2, gt_label, name) in enumerate(list(zip(y_pred_expert1, y_pred_expert2, y_true, name_label_list))):
        name = np.int64(name.item())
        if (pred_expert1 != pred_expert2) or (pred_expert2 ==2):

            y_pred_all.append(2)
            y_true_all.append(gt_label)
            
            if isinstance(name, np.int64):
                df.loc[(df["component_name"] == name), 'unknown'] +=1
            else:
                df.loc[(df["component_name"] == ''.join(name)), 'unknown'] +=1

        if (pred_expert1 == pred_expert2) and (pred_expert1 !=2):

            y_pred_all.append(pred_expert2)
            y_true_all.append(gt_label)
            
            gt_label_after_unknown.append(gt_label)
            pred_after_unknown.append(pred_expert2)

            if gt_label == 0 and gt_label != pred_expert2:
                if isinstance(name, np.int64):
                    df.loc[(df["component_name"] == name), 'overkill'] +=1
                else:
                    df.loc[(df["component_name"] == ''.join(name)), 'overkill'] +=1
            elif gt_label == 1 and gt_label != pred_expert2:
                if isinstance(name, np.int64):
                    df.loc[(df["component_name"] == name), 'leakage'] +=1
                else:
                    df.loc[(df["component_name"] == ''.join(name)), 'leakage'] +=1 

    print("HybridExpert Accuracy: {}\n".format(100*round(accuracy_score(gt_label_after_unknown, pred_after_unknown),4)))  
    
    score = calculatePerformance(df, file_name=f'./output/train_val/dataset_{args_due.random_seed}_HBE+_ALL_train_val.csv')
    print(score)
    
    return score


def search(exp_2_tau1, exp_2_tau2):
    params = {}    
    params['exp_2_tau1'] = exp_2_tau1
    params['exp_2_tau2'] = exp_2_tau2
    print('tau1',exp_2_tau1)
    print('tau2',exp_2_tau2)
    df_train = pd.DataFrame()

    for idx, component_name in enumerate(range(num_classes)):
        train_val_com_loader  = CreateDataset_for_each_component_regroup(args["random_seed"],  train_val_df,component_name)

        if args_due.test_uncertainty == True:
            # Calculate uncertainty from images in reference set

            uncertainty, labels_train, _, name_list_train, _, _ = get_uncertainty(model, gp_model, likelihood, train_val_com_loader, tsne=True)

            uncertainty_mean = np.mean(uncertainty , axis=0)
            uncertainty_std = np.std(uncertainty , axis=0)
            print('uncertainty_mean : ',uncertainty_mean)
            print('uncertainty_std : ',uncertainty_std)

            uncertainty_th = uncertainty_mean + (exp_2_tau1 * uncertainty_std)
            uncertainty_th_2 = uncertainty_mean + (exp_2_tau2 * uncertainty_std)


            df_train = df_train.append({'com':component_name,'TH':uncertainty_th , 'TH_2':uncertainty_th_2 }, ignore_index=True)


        else:
            print('Uncertainty calculations are not performed')

    filepath = './output/train_val/uncertainty.csv'

    df_train.to_csv(filepath)
    

    # set TH 
    df = pd.read_csv('./output/train_val/uncertainty.csv')
    TH = df['TH']
    TH_2 = df['TH_2']
    
    if args_due.test_inference == True:
        ####---- train+val set inference ----####       


        test_df_orig = test_component_name_df.copy()    
        test_component_name_df_Expert2 = test_df_orig.copy()
        test_component_name_df_HybridExpert = test_df_orig.copy()


        print("2. Testing with DUE (Expert 2)")  # Discriminative model
        name_label_list, y_pred_expert2  = EXP2(args_due ,model ,gp_model ,train_val_loader, test_component_name_df_Expert2, train_component_label, val_component_label, test_component_label ,likelihood ,TH,TH_2,good_num_com)

        print("3. Testing with Hybrid Expert") 
        score = HybridExpert(y_pred_expert1, y_pred_expert2, y_true, name_label_list, test_component_name_df_HybridExpert, args_due, train_component_label, val_component_label, test_component_label)
        print('score',score)
    else:
        print('inference calculations are not performed')
    
    return score   


parser = argparse.ArgumentParser(description="Visualizing embeddings with T-SNE")

parser.add_argument(
    "-c", "--checkpoint_path",
    type=str,
    default ="",      # checkpoint.pth.tar
    help="Path to model's checkpoint."
)
parser.add_argument(
    "-gpc", "--gp_checkpoint_path",
    type=str,
    default ="",      # checkpoint.pth.tar
    help="Path to model's checkpoint."
)
parser.add_argument(
    "-cls", "--checkpoint_cls_lik_path",
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
parser.add_argument(
    "--n_inducing_points_cls", type=int, help="Number of inducing points"
)
parser.add_argument('--relabel', action='store_true', help='relabel dataset')
parser.add_argument(
    "-oid", "--output_inference_dir",
    type=str,
    default="output/",
    help="Directory to save output plots"
)
parser.add_argument("--leakage_weight", type=float, default=1, help="Dropout rate")

args: Dict[str, Any] = vars(parser.parse_args())
args_due = parser.parse_args()

print('relabel:',args_due.relabel)
set_random_seed(args["random_seed"])


# Create output directory if not exists
if not os.path.isdir('./output/train_val/'):
    os.makedirs('./output/train_val/')
if not os.path.isdir('./bay_fruit_dataset/'):
    os.makedirs('./bay_fruit_dataset/')

# Initialize device
device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.info(f"Initialized device {device}")

# Load model's checkpoint
loc = 'cuda:0'

checkpoint_path: str = args["checkpoint_path"]
checkpoint: Dict[str, Any] = torch.load(checkpoint_path, map_location="cuda:0")
logging.info(f"Loaded checkpoint at {args['checkpoint_path']}")

# load data
ds = get_dataset(args_due.dataset ,args_due.random_seed , root="./data" )

input_size ,num_classes , train_com_loader, train_loader, test_dataset ,train_cls_dataset,train_com_dataset, test_com_dataset = ds

good_num_com = num_classes

model, gp_model ,likelihood  = set_model(args, args_due, train_com_loader , num_classes)
_ ,_ ,_, _, _ ,_, _,_,train_com_df , train_df, val_df , test_df = get_fruit_8(root="./data" , seed=args_due.random_seed)

train_val_df = pd.concat([train_df, val_df])

val_loader, test_loader, test_component_name_df, train_component_label, val_component_label, test_component_label ,train_val_loader = set_loader(args_due)

print("1. Testing with Discriminative Model (Expert 1)")  # Discriminative model
y_true, y_pred_expert1 = EXP1(args_due ,model ,train_val_loader, test_component_name_df, train_component_label, val_component_label, test_component_label)

# Run Bayesian Optimization
start = time.time()
logger = JSONLogger(path=f"./bay_fruit_dataset/{args_due.output_inference_dir}_tau1_tau2_logs.json")

bounds_transformer = SequentialDomainReductionTransformer(minimum_window=0.5)
bayes_optimizer = BayesianOptimization(search, random_state=1212, pbounds=pbounds, bounds_transformer=bounds_transformer)
bayes_optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
bayes_optimizer.maximize(init_points=100, n_iter=100, acq="ei", xi=0)
print('It takes %s minutes' % ((time.time() - start)/60))
print(bayes_optimizer.max)

    
