#! /usr/bin/env python

# Script to run human parsing on an image

import argparse
import cv2
import os
import scipy.io

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script to run human parser.')
    args = parser.parse_args()

    filename = 'example_person.jpg'

    os.system('cp {} ./LIP_JPPNet/datasets/examples/images/'.format(args.image))

    os.system('echo /images/{} > ./LIP_JPPNet/datasets/examples/list/val.txt'.format(filename))

    os.system('sed -i -e \'s/NUM_STEPS = 6/NUM_STEPS = 1/g\' ./LIP_JPPNet/evaluate_parsing_JPPNet-s2.py')

    os.system('cd LIP_JPPNet; python evaluate_parsing_JPPNet-s2.py; cd ..;')

    img = cv2.imread('./LIP_JPPNet/output/parsing/val/{}'.format(os.path.splitext(filename)[0]+'.png'))
    seg_dict = {'segment':img[:,:,0]} 
    scipy.io.savemat('./output/{}'.format(os.path.splitext(filename)[0]+'.mat'), seg_dict)
