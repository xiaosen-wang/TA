import json
import torch
import os
import argparse
import time
import attack_mask as attack
from attack_utils import get_model, read_imagenet_data_specify, save_results
from foolbox.distances import l2
import numpy as np
from PIL import Image
import torch_dct

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_folder", "-o", default="results", help="Output folder")
    parser.add_argument(
        "--model_name",
        type=str,
        default="resnet-18",
        help="The name of model you want to attack(resnet-18, inception-v3, vgg-16, resnet-101, densenet-121)"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="images",
        help="The path of dataset"
    )
    parser.add_argument(
         "--csv",
        type=str,
        default="label.csv",
        help="The path of csv information about dataset"
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=20,
        help='The random seed you choose'
    )
    parser.add_argument(
        '--max_queries',
        type=int,
        default=1000,
        help='The max number of queries in model'
    )
    parser.add_argument(
        '--ratio_mask',
        type=float,
        default=0.1,
        help='ratio of mask'
    )
    parser.add_argument(
        '--dim_num',
        type=int,
        default=1,
        help='the number of picked dimensions'
    )
    parser.add_argument(
        '--max_iter_num_in_2d',
        type=int,
        default=2,
        help='the maximum iteration number of attack algorithm in 2d subspace'
    )
    parser.add_argument(
        '--init_theta',
        type=int,
        default=2,
        help='the initial angle of a subspace=init_theta*np.pi/32'
    )
    parser.add_argument(
        '--init_alpha',
        type=float,
        default=np.pi/2,
        help='the initial angle of alpha'
    )
    parser.add_argument(
        '--plus_learning_rate',
        type=float,
        default=0.1,
        help='plus learning_rate when success'
    )
    parser.add_argument(
        '--minus_learning_rate',
        type=float,
        default=0.005,
        help='minus learning_rate when fail'
    )
    parser.add_argument(
        '--half_range',
        type=float,
        default=0.1,
        help='half range of alpha from pi/2'
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    if args.model_name=='inception-v3':
        args.side_length=299
    else:
        args.side_length=224


    ###############################
    print("Load Model: %s" % args.model_name)
    fmodel = get_model(args,device)

    ###############################
    

    ###############################
    print("Load Data")
    # images, labels = samples(fmodel, dataset="imagenet", batchsize=args.n_images)
    images, labels, selected_paths = read_imagenet_data_specify(args, device)
    print("{} images loaded with the following labels: {}".format(len(images), labels))

    ###############################
    print("Attack !")
    time_start = time.time()

    ta_model = attack.TA(fmodel, input_device=device)
    my_advs, q_list, my_intermediates, max_length = ta_model.attack(args,images, labels)
    print('TA Attack Done')
    print("{:.2f} s to run".format(time.time() - time_start))
    print("Results")

    my_labels_advs = fmodel(my_advs).argmax(1)
    my_advs_l2 = l2(images, my_advs)

    for image_i in range(len(images)):
        print("My Adversarial Image {}:".format(image_i))
        label_o = int(labels[image_i])
        label_adv = int(my_labels_advs[image_i])
        print("\t- l2 = {}".format(my_advs_l2[image_i]))
        print("\t- {} queries\n".format(q_list[image_i]))
    save_results(args,my_intermediates, len(images))

    
