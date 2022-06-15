import sys
from random import shuffle
import matplotlib.pyplot as plt
from scipy import spatial
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tqdm import tqdm
from pathlib import Path

import scipy
from scipy.stats import wasserstein_distance

import os
import dataset
import features
import paths
import pandas as pd
from csv import writer


def visualize_similar_images(img_paths, max_query_imgs=20, max_matches=3):
    img_paths = img_paths[:min(max_query_imgs, len(img_paths))]

    for i in tqdm(range(len(img_paths))):
        img_path = img_paths[i]
        similar = similar_images_paths(img_path, max_imgs=max_matches)

        qn = Path(img_path).name
        qname = os.path.splitext(qn)[0]

        cnt = 1
        for path, similarity in similar:
            p = Path(path).name 
            pq = os.path.splitext(p)[0]
            List = [qname,pq,similarity]
            
            with open('submission.csv','a') as fo:
                writer_object = writer(fo)
                writer_object.writerow(List)
                fo.close()
        
            cnt += 1


 
    



def similar_images_paths(img_path, max_imgs=100):
    query_features = features.extract_features(img_path)
    stored_features = dataset.get_stored_features()

    max_imgs = min(max_imgs, len(stored_features[0]))
    similarities = []

    for filename, encoding in list(zip(*stored_features)):
        h_distance = spatial.distance.hamming(query_features, encoding)
        c_distance = spatial.distance.cosine(query_features, encoding)
        em_dist = wasserstein_distance(query_features, encoding, u_weights=None, v_weights=None) #earth_mover distance
        similarity = 1 - (h_distance + c_distance + em_dist) / 3
        
        similarities.append((filename, similarity))

    similarities.sort(key=lambda tup: -tup[1])
    return similarities[:max_imgs]


if __name__ == "__main__":
    args = sys.argv
    if len(args) > 1:
        visualize_similar_images(args[1:])
    else:
        paths = dataset.get_file_list(paths.query_images_folder_path)
        visualize_similar_images(paths)
