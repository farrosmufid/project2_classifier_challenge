'''
this script is for the evaluation of Project 2.

-------------------------------------------
INTRO:
You should write your codes or modify codes between the
two '#####' lines. The codes between two '=======' lines
are used for logging or printing, please keep them unchanged
in your submission.

You need to debug and run your codes in somewhere else (e.g. Jupyter
Notebook). This file is only used for the evaluation stage and for
your submission and marks. Please make sure the codes are running
smoothly using your trained model.

-------------------------------------------
USAGE:
In your final update, please keep the file name as 'python2_test.py'.

>> python project2_test.py
This will run the program on CPU to test on your trained nets.

>> python project2_test.py --cuda
This will run the program on GPU. You can ignore this if you do not have GPU or
CUDA installed.

-------------------------------------------
NOTE:
this file might be incomplete, feel free to contact us
if you found any bugs or any stuff should be improved.
Thanks :)

Email: geng.zhan@sydney.edu.au, dwan3342@uni.sydney.edu.au
'''

# import the packages
import argparse
import logging
import sys
import time
import os

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from network import Network # the network you used

# ==================================
# control input options. DO NOT CHANGE THIS PART.
def parse_args():
    parser = argparse.ArgumentParser(description= \
        'scipt for evaluation of project 2')
    parser.add_argument('--cuda', action='store_true', default=False,
        help='Used when there are cuda installed.')
    parser.add_argument('--output_path', default='./', type=str,
        help='The path that stores the log files.')

    pargs = parser.parse_args()
    return pargs

# Creat logs. DO NOT CHANGE THIS PART.
def create_logger(final_output_path):
    log_file = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=os.path.join(final_output_path, log_file),
                        format=head)
    clogger = logging.getLogger()
    clogger.setLevel(logging.INFO)
    # add handler
    # print to stdout and log file
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    clogger.addHandler(ch)
    return clogger

# evaluation process. DO NOT CHANGE THIS PART.
def eval_net(net, loader, logging):
    net = net.eval()
    if args.cuda:
        net = net.cuda()

    assert os.path.exists(args.output_path + 'project2.pth'), 'InputError: There is no pretrained file project2.pth. Please try again.'

    # use your trained network by default
    if args.cuda:
        net.load_state_dict(torch.load(args.output_path + 'project2.pth', map_location='cuda'))
    else:
        net.load_state_dict(torch.load(args.output_path + 'project2.pth', map_location='cpu'))

    correct = 0
    total = 0
    for data in loader:
        images, labels = data
        if args.cuda:
            images, labels = images.cuda(), labels.cuda()
        outputs = net(images)
        if args.cuda:
            outputs = outputs.cpu()
            labels = labels.cpu()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # print and write to log
    logging.info('=' * 55)
    logging.info('SUMMARY of Project2')
    logging.info('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total))
    logging.info('=' * 55)

# Prepare for writing logs and setting GPU.
# DO NOT CHANGE THIS PART.
args = parse_args()
if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)
# print('using args:\n', args)

logger = create_logger(args.output_path)
logger.info('using args:')
logger.info(args)

# DO NOT change codes above this line
# ==================================


####################################
# Transformation definition
# NOTE:
# Write the test_transform here. We recommend you use
# Normalization, CenterCrop and Resize. Please do not use
# Random operations, which will make your performance worse.

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

####################################

####################################
# Define the test dataset and dataloader.
# You can make some modifications, e.g. batch_size, adding other hyperparameters, etc.

# !! PLEASE KEEP test_image_path AS '../test' WHEN YOU SUBMIT.

test_image_path = '../test'

testset = ImageFolder(test_image_path, test_transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

####################################

# ==================================
# test the network and write to logs.
# use cuda if called with '--cuda'.
# DO NOT CHANGE THIS PART.

network = Network()
if args.cuda:
    network = network.cuda()

# test your trained network
eval_net(network, testloader, logging)
# ==================================
