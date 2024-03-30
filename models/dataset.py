import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset

from tools.utils import Utils


class TripletDataset(Dataset):
    def __init__(self, split, data = 'triplets.csv', transform=None, train=True, seed=21):
        """
        TripletDataset class to load the triplets dataset
        """
        if not os.path.exists(data):
            Utils.generate_triplets(csv_path=data)
        self.data = pd.read_csv(data)
        self.full_len = len(self.data)

        n = int(len(self.data) * split)
        if train:
            self.data = self.data.sample(frac=1, random_state=seed).reset_index(drop=True)
            self.data = self.data.iloc[:n]

        elif not train:
            self.data = self.data.sample(frac=1, random_state=seed).reset_index(drop=True)
            self.data = self.data.iloc[-n:]

        self.__image_downloader()
        self.transform = transform
        self.train = train


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        triplet = []

        for p in range(1, 4):
            images_place_p = [Image.open(row[f'image_{img}_p{p}']) for img in range(1, 6)]
            if self.transform:
                images_place_p = [self.transform(image) for image in images_place_p]
                images_place_p = torch.stack(images_place_p)
            triplet.append(images_place_p)

        return triplet[0], triplet[1], triplet[2], row['response']


    def __image_downloader(self):
        image_folder_path = self.data['image_1_p1'].iloc[0].split('/')[:-3]
        image_folder_path = '/'.join(image_folder_path)

        if not os.path.exists(image_folder_path):
            Utils.download_images(image_folder_path)
    

    def visualize_triplet(self, idx):

        p1, p2, p3, choice = self[idx]


        fig, ax = plt.subplots(3, 5, figsize=(15, 6))
        fig.suptitle(f'Triplet {idx} - Chosen: {choice}')

        for i in range(3):
            px = [p1, p2, p3][i]
            for j in range(5):
                if self.transform is not None:
                    ax[i, j].imshow(px[j].numpy().transpose(1, 2, 0))
                else:
                    ax[i, j].imshow(px[j])
                ax[i, j].axis('off')

        rect = plt.Rectangle((0.11, 0.1+0.27*(2-choice)), 0.81, 1/4, linewidth=1, edgecolor='r', facecolor='none', zorder=10)
        fig.add_artist(rect)
        plt.show()

    
class PlaceDataset(Dataset):
    def __init__(self, data = 'places/Delft_NL_images.csv', pkl = 'places/Delft_NL.pkl', root = '/tudelft.net/staff-umbrella/phdfrancisco/collection/application/summarized', transform=None):
        """
        PlaceDataset class to load the places dataset
        """
        if not os.path.exists(data):
            Utils.generate_places_csv(pkl_file = pkl, csv_file = data)
        self.data = pd.read_csv(data)
        self.root = root
        self.transform = transform
    

    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        images_place = [Image.open(f"{self.root}/{row[f'img_{img}']}") for img in range(1, 6)]
        if self.transform:
            images_place = [self.transform(image) for image in images_place]
            images_place = torch.stack(images_place)

        return row['h3'], images_place

class PairDataset(Dataset):
    # TODO: Implement the PairDataset class for openning pairs of places (5 images each)
    pass


