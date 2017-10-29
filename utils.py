import os
import csv
import cv2
import pickle
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split



def generate_labels_track1(root="./data/data/"):
    labels = []
    with open(root+"driving_log.csv") as file:
        for row in csv.reader(file):
            labels.append(row)
    labels = labels[1:]
    return labels

def generate_labels_track2(root="./data2/"):
    labels = []
    with open(root+"driving_log.csv") as file:
        for row in csv.reader(file):
            temp = []
            for r in row[:3]:
                t = r.split("/IMG/")
                r = t[0] + "/data2/IMG/" + t[1]
                temp.append(r)
            temp.extend(row[3:])
            labels.append(temp)
    labels = labels[300:]
    return labels

def data_generator(labels, root="./data/data/IMG/", batch_size=32, N=1/2, k=0.2):
    
    '''generate batch of data'''
    
    train_size = len(labels)
    while True: # Loop forever so the generator never terminates
        
        labels = shuffle(labels)
        
        for x in range(0, train_size, batch_size):
            batch = labels[x:(x + batch_size)]
            
        images = []
        angles = []
        
        for b in batch:
            
            if np.random.random() < 0.5:
                # central camera
                name = root + b[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                center_angle = float(b[3])
                images.append(center_image)
                angles.append(center_angle)
            else:
                # flip central camera
                name = root + b[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                center_image = np.fliplr(center_image)
                center_angle = -float(b[3])
                images.append(center_image)
                angles.append(center_angle)
            
        x_train = np.array(images)
        y_train = np.array(angles)
        yield (x_train, y_train)