from collections import namedtuple
import h5py
import json
from PIL import Image

from typing import List, Union, Tuple
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
import os
import numpy as np
import torch

from torchvision.transforms import Compose
import torchvision.transforms as ttf

from abc import ABC, abstractmethod

import pandas as pd


class TensorDataset(Dataset):
    @abstractmethod
    def __getitem__(self, idx) -> Tuple[Tensor]:
        pass


class TensorKeyDataset(Dataset):
    @abstractmethod
    def __getitem__(self, idx) -> Tuple[Tensor, str]:
        pass


class TensorPairSimDataset(Dataset):
    @abstractmethod
    def __getitem__(self, idx) -> Tuple[Tensor, Tensor, float]:
        pass


class MslsSingleCitySingleIndex:
    def __init__(self, root_dir: str, city: str, index_type: str) -> None:
        self.root_dir = root_dir
        self.city = city

        assert index_type in ["database", "query"]
        self.index_type = index_type

        raw_csv_path = os.path.join(root_dir, "train_val", city, index_type, "raw.csv")

        with open(raw_csv_path, "r") as f:
            raw_csv_pd = pd.read_csv(f)

        self.image_keys = list(raw_csv_pd["key"])

        # is_pano is a boolean array
        self.is_pano = np.array(raw_csv_pd["pano"])

        # array of indices of non-pano images
        self.non_pano_indices = np.where(self.is_pano == False)[0]


class MslsSingleCity:
    def __init__(
        self,
        root_dir: str,
        city: str,
        positive_threshold: float = 0.5,
        pair_probs: List[float] = None,
    ) -> None:
        self.root_dir = root_dir
        self.city = city

        if pair_probs is None:
            pair_probs = [0.5, 0.25, 0.25]

        query_db = MslsSingleCitySingleIndex(root_dir, city, "query")
        database_db = MslsSingleCitySingleIndex(root_dir, city, "database")

        # gt label file
        labels_db_file = f"{city}_gt.h5"
        labels_db_file_path = os.path.join(root_dir, "train_val", labels_db_file)

        with h5py.File(labels_db_file_path, "r") as f:
            self.fov_matrix = np.array(f["fov"])

        positive_pairs_idx = np.where(self.fov_matrix > positive_threshold)
        positive_pairs_idx = list(zip(positive_pairs_idx[0], positive_pairs_idx[1]))

        print(
            "Number of positive pairs before removing panos:", len(positive_pairs_idx)
        )

        # if either index is a pano, remove it
        positive_pairs_idx = [
            (i, j)
            for i, j in positive_pairs_idx
            if not (query_db.is_pano[i] or database_db.is_pano[j])
        ]

        print("Number of positive pairs after removing panos:", len(positive_pairs_idx))

        soft_negative_pairs_idx = np.where(
            (self.fov_matrix > 0.0) & (self.fov_matrix < positive_threshold)
        )
        soft_negative_pairs_idx = list(
            zip(soft_negative_pairs_idx[0], soft_negative_pairs_idx[1])
        )

        print(
            "Number of soft negative pairs before removing panos:",
            len(soft_negative_pairs_idx),
        )

        # if either index is a pano, remove the pair
        soft_negative_pairs_idx = [
            (i, j)
            for i, j in soft_negative_pairs_idx
            if not (query_db.is_pano[i] or database_db.is_pano[j])
        ]
        print(
            "Number of soft negative pairs after removing panos:",
            len(soft_negative_pairs_idx),
        )

        # negative pairs are too common to store in memory
        # instead we'll randomly sample and filter them

        np.random.shuffle(positive_pairs_idx)
        np.random.shuffle(soft_negative_pairs_idx)

        # how many more soft negative pairs than positive pairs
        nb_positive_pairs = len(positive_pairs_idx)
        nb_soft_negative_pairs = nb_positive_pairs * pair_probs[1] / pair_probs[0]
        nb_hard_negative_pairs = nb_positive_pairs * pair_probs[2] / pair_probs[0]

        # truncate the soft negative pairs and hard negative pairs
        soft_negative_pairs_idx = soft_negative_pairs_idx[: int(nb_soft_negative_pairs)]

        # randomly sample matrix until we have enough hard negative pairs
        hard_negative_pairs_idx = []
        while len(hard_negative_pairs_idx) < nb_hard_negative_pairs:
            # pick valud from non-pano indices
            i = np.random.choice(query_db.non_pano_indices)
            j = np.random.choice(database_db.non_pano_indices)

            if self.fov_matrix[i, j] == 0.0:
                hard_negative_pairs_idx.append((i, j))

        print("Number of hard negative pairs:", len(hard_negative_pairs_idx))

        self.pairs_idx = (
            positive_pairs_idx + soft_negative_pairs_idx + hard_negative_pairs_idx
        )

        np.random.shuffle(self.pairs_idx)

        print("Number of pairs:", len(self.pairs_idx))


# just the images for a given city and index file ("database.json" or "query.json")
class MslsCityImage(MslsSingleCitySingleIndex, TensorDataset):
    IMAGE_T = Compose(
        [
            ttf.ToTensor(),
            ttf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ttf.CenterCrop(size=(360, 640)),
            ttf.Resize((360, 640), antialias=True),
        ]
    )

    def __init__(
        self,
        root_dir: str,
        city: str,
        index_type: str,
        image_t: Union[Compose, None] = None,
    ) -> None:
        super(MslsCityImage, self).__init__(root_dir, city, index_type)

        if image_t is None:
            self.image_t = MslsCityImage.IMAGE_T
        else:
            self.image_t = image_t

        self.image_paths = self._load_image_paths()

    def _load_image_paths(self) -> List[str]:
        return [
            os.path.join(
                self.root_dir,
                "train_val",
                self.city,
                self.index_type,
                "images",
                f"{key}.jpg",
            )
            for key in self.image_keys
        ]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Image.Image:
        # load image
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")

        # apply transforms
        image = self.image_t(image)

        return image


class MslsCityImageKey(MslsSingleCitySingleIndex, TensorKeyDataset):
    def __init__(
        self,
        root_dir: str,
        city: str,
        index_type: str,
        image_t: Union[Compose, None] = None,
    ) -> None:
        super().__init__(root_dir, city, index_type)

        self._msls_city_dataset = MslsCityImage(root_dir, city, index_type, image_t)

    def __len__(self):
        # only work with non-pano images
        return len(self.non_pano_indices)

    def __getitem__(self, idx) -> Tuple[Tensor, str]:
        i = self.non_pano_indices[idx]
        image = self._msls_city_dataset[i]
        image_key = self.image_keys[i]

        return image, image_key


class MslsCitiesImageKey(TensorKeyDataset):
    def __init__(
        self,
        root_dir: str,
        cities: List[str],
        index_type: str,
        image_t: Union[Compose, None] = None,
    ) -> None:
        self.datasets = [
            MslsCityImageKey(root_dir, city, index_type, image_t) for city in cities
        ]

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)

    def __getitem__(self, idx) -> Tuple[Tensor, str]:
        # find the city that contains the index
        for dataset in self.datasets:
            if idx < len(dataset):
                return dataset[idx]
            else:
                idx -= len(dataset)


class MslsCityImagePair(MslsSingleCity, TensorPairSimDataset):
    def __init__(
        self,
        root_dir: str,
        city: str,
        image_t: Union[Compose, None] = None,
        positive_threshold: float = 0.5,
        pair_probs: Union[List[float], None] = None,
    ) -> None:
        super(MslsCityImagePair, self).__init__(
            root_dir, city, positive_threshold, pair_probs
        )

        # create datasets for both
        self.query_dataset = MslsCityImage(root_dir, city, "query", image_t)
        self.database_dataset = MslsCityImage(
            root_dir,
            city,
            "database",
            image_t,
        )

    def __len__(self):
        return len(self.pairs_idx)

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor, float]:
        query_idx, database_idx = self.pairs_idx[idx]

        query_image = self.query_dataset[query_idx]
        database_image = self.database_dataset[database_idx]

        label = self.fov_matrix[query_idx, database_idx]

        return query_image, database_image, label


# image_mask pairs and similary labels for a given list of cities
class MslsCitiesImagePair(TensorPairSimDataset):
    def __init__(
        self,
        root_dir: str,
        cities: List[str],
        image_t: Union[Compose, None] = None,
        positive_threshold: float = 0.5,
        pair_probs: Union[List[float], None] = None,
    ) -> None:
        self.cities_pairs = {
            city: MslsCityImagePair(
                root_dir,
                city,
                image_t,
                positive_threshold,
                pair_probs,
            )
            for city in cities
        }

    def __len__(self):
        return sum(len(dataset) for dataset in self.cities_pairs.values())

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor, float]:
        for city in self.cities_pairs.keys():
            if idx >= len(self.cities_pairs[city]):
                idx -= len(self.cities_pairs[city])
            else:
                return self.cities_pairs[city][idx]


# images and masks for a given city and index file ("database.json" or "query.json")
class MslsCityImageMask(MslsSingleCitySingleIndex, TensorDataset):
    MASK_T = Compose(
        [
            ttf.ToTensor(),
            ttf.CenterCrop(size=(360, 640)),
            ttf.Resize((360, 640), antialias=True),
        ]
    )

    def __init__(
        self,
        root_dir: str,
        masks_dir: str,
        city: str,
        index_type: str,
        image_t: Union[Compose, None] = None,
        mask_t: Union[Compose, None] = None,
    ) -> None:
        super(MslsCityImageMask, self).__init__(root_dir, city, index_type)

        self._msls_city_dataset = MslsCityImage(root_dir, city, index_type, image_t)

        if mask_t is None:
            self.mask_t = MslsCityImageMask.MASK_T
        else:
            self.mask_t = mask_t

        self.mask_paths = self._load_mask_paths(masks_dir)

    def _load_mask_paths(self, masks_dir: str) -> List[str]:
        return [
            os.path.join(
                masks_dir,
                "train_val",
                self.city,
                self.index_type,
                "images",
                f"{key}.jpg",
            )
            for key in self.image_keys
        ]

    def __len__(self):
        return len(self.mask_paths)

    def __getitem__(self, idx) -> Tensor:
        image = self._msls_city_dataset[idx]
        mask = Image.open(self.mask_paths[idx])

        # apply transforms
        mask = self.mask_t(mask)

        # concatenate the image and mask
        image_mask = torch.cat([image, mask], dim=0)

        return image_mask


class MslsCityImageMaskKey(MslsSingleCitySingleIndex, TensorKeyDataset):
    def __init__(
        self,
        root_dir: str,
        masks_dir: str,
        city: str,
        index_type: str,
        image_t: Union[Compose, None] = None,
        mask_t: Union[Compose, None] = None,
    ) -> None:
        super(MslsCityImageMaskKey, self).__init__(root_dir, city, index_type)

        self._image_mask_dataset = MslsCityImageMask(
            root_dir, masks_dir, city, index_type, image_t, mask_t
        )

    def __len__(self):
        # only work with non-pano images
        return len(self.non_pano_indices)

    def __getitem__(self, idx) -> Tuple[Tensor, str]:
        i = self.non_pano_indices[idx]
        image_mask = self._image_mask_dataset[i]
        image_key = self.image_keys[i]

        return image_mask, image_key


class MslsCitiesImageMaskKey(TensorKeyDataset):
    def __init__(
        self,
        root_dir: str,
        masks_dir: str,
        cities: List[str],
        index_type: str,
        image_t: Union[Compose, None] = None,
        mask_t: Union[Compose, None] = None,
    ) -> None:
        self.datasets = [
            MslsCityImageMaskKey(root_dir, masks_dir, city, index_type, image_t, mask_t)
            for city in cities
        ]

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)

    def __getitem__(self, idx) -> Tuple[Tensor, str]:
        # find the city that contains the index
        for dataset in self.datasets:
            if idx < len(dataset):
                return dataset[idx]
            else:
                idx -= len(dataset)


class MslsCityImageMaskPair(MslsSingleCity, TensorPairSimDataset):
    def __init__(
        self,
        root_dir: str,
        masks_dir: str,
        city: str,
        image_t: Union[Compose, None] = None,
        mask_t: Union[Compose, None] = None,
        positive_threshold: float = 0.5,
        pair_probs: Union[List[float], None] = None,
    ) -> None:
        super(MslsCityImageMaskPair, self).__init__(
            root_dir, city, positive_threshold, pair_probs
        )

        # create datasets for both
        self.query_dataset = MslsCityImageMask(
            root_dir, masks_dir, city, "query", image_t, mask_t
        )
        self.database_dataset = MslsCityImageMask(
            root_dir,
            masks_dir,
            city,
            "database",
            image_t,
            mask_t,
        )

    def __len__(self):
        return len(self.pairs_idx)

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor, float]:
        query_idx, database_idx = self.pairs_idx[idx]

        query_image_mask = self.query_dataset[query_idx]
        database_image_mask = self.database_dataset[database_idx]

        label = self.fov_matrix[query_idx, database_idx]

        return query_image_mask, database_image_mask, label


# image_mask pairs and similary labels for a given list of cities
class MslsCitiesImageMaskPair(TensorPairSimDataset):
    def __init__(
        self,
        root_dir: str,
        masks_dir: str,
        cities: List[str],
        image_t: Union[Compose, None] = None,
        mask_t: Union[Compose, None] = None,
        positive_threshold: float = 0.5,
        pair_probs: Union[List[float], None] = None,
    ) -> None:
        self.cities_pairs = {
            city: MslsCityImageMaskPair(
                root_dir,
                masks_dir,
                city,
                image_t,
                mask_t,
                positive_threshold,
                pair_probs,
            )
            for city in cities
        }

    def __len__(self):
        return sum(len(dataset) for dataset in self.cities_pairs.values())

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor, float]:
        # we assume that the index goes across all cities
        # we iterate over the cities, subtracting the length of each dataset
        # until we have an index that is less than the length of the current city's dataset

        for city in self.cities_pairs.keys():
            if idx >= len(self.cities_pairs[city]):
                # index is still too big
                # subtract the length of the current city's dataset and move on
                idx -= len(self.cities_pairs[city])
            else:
                return self.cities_pairs[city][idx]


class MslsCityImageAct(MslsSingleCitySingleIndex, TensorDataset):
    # activations tensor will have 19 channels (19x360x640)
    # width and height will be the same as the image
    ACTIVATIONS_T = ttf.Compose(
        [
            ttf.ToTensor(),
            ttf.CenterCrop((360, 640)),
        ]
    )

    def __init__(
        self,
        root_dir: str,
        activations_dir: str,
        city: str,
        index_type: str,
        image_t: Union[Compose, None] = None,
        activations_t: Union[Compose, None] = None,
    ) -> None:
        super(MslsCityImageAct, self).__init__(root_dir, city, index_type)

        if activations_t is None:
            self.activations_t = MslsCityImageAct.ACTIVATIONS_T
        else:
            self.activations_t = activations_t

        self._msls_city_dataset = MslsCityImage(root_dir, city, index_type, image_t)

        # get the paths to the activations
        self.activations_paths = self._get_activations_paths(activations_dir)

    def _get_activations_paths(self, activations_dir: str) -> List[str]:
        # get the paths to the activations folder
        activations_paths = [
            os.path.join(
                activations_dir,
                "train_val",
                self.city,
                self.index_type,
                "images",
                key,
            )
            for key in self.image_keys
        ]

        return activations_paths

    def __len__(self):
        return len(self.activations_paths)

    def __getitem__(self, idx) -> Tuple[Tensor]:
        # get the image and activations
        image = self._msls_city_dataset[idx]

        # load the activations from the activations dir
        d = self.activations_paths[idx]

        imgs = []
        for i in range(0, 19):
            img_path = os.path.join(d, f"{i}.jpg")
            img = Image.open(img_path)
            imgs.append(img)

        # convert to numpy array and stack the images
        activations = np.stack([np.array(img) for img in imgs])

        # permute the activations to be channels last because that's what the transform expects
        activations = np.transpose(activations, (1, 2, 0))

        # apply transform
        activations = self.activations_t(activations)

        # concatenate the image and activations
        return torch.cat([image, activations], dim=0)


class MslsCityImageActKey(MslsSingleCitySingleIndex, TensorKeyDataset):
    def __init__(
        self,
        root_dir: str,
        activations_dir: str,
        city: str,
        index_type: str,
        image_t: Union[Compose, None] = None,
        activations_t: Union[Compose, None] = None,
    ) -> None:
        super(MslsCityImageActKey, self).__init__(root_dir, city, index_type)

        self._image_act_dataset = MslsCityImageAct(
            root_dir, activations_dir, city, index_type, image_t, activations_t
        )

    def __len__(self):
        return len(self.non_pano_indices)

    def __getitem__(self, idx) -> Tuple[Tensor, str]:
        i = self.non_pano_indices[idx]

        image_act = self._image_act_dataset[i]
        key = self.image_keys[i]
        return image_act, key


class MslsCitiesImageActKey(TensorKeyDataset):
    def __init__(
        self,
        root_dir: str,
        activations_dir: str,
        cities: List[str],
        index_type: str,
        image_t: Union[Compose, None] = None,
        activations_t: Union[Compose, None] = None,
    ) -> None:
        self.cities_datasets = {
            city: MslsCityImageActKey(
                root_dir,
                activations_dir,
                city,
                index_type,
                image_t,
                activations_t,
            )
            for city in cities
        }

    def __len__(self):
        return sum(len(dataset) for dataset in self.cities_datasets.values())

    def __getitem__(self, idx) -> Tuple[Tensor, str]:
        for city in self.cities_datasets.keys():
            if idx >= len(self.cities_datasets[city]):
                idx -= len(self.cities_datasets[city])
            else:
                return self.cities_datasets[city][idx]


class MslsCityImageActPair(MslsSingleCity, TensorPairSimDataset):
    def __init__(
        self,
        root_dir: str,
        activations_dir: str,
        city: str,
        image_t: Union[Compose, None] = None,
        activations_t: Union[Compose, None] = None,
        positive_threshold: float = 0.5,
        pair_probs: Union[List[float], None] = None,
    ) -> None:
        super(MslsCityImageActPair, self).__init__(
            root_dir, city, positive_threshold, pair_probs
        )

        self.query_dataset = MslsCityImageAct(
            root_dir,
            activations_dir,
            city,
            "query",
            image_t,
            activations_t,
        )

        self.database_dataset = MslsCityImageAct(
            root_dir,
            activations_dir,
            city,
            "database",
            image_t,
            activations_t,
        )

    def __len__(self):
        return len(self.pairs_idx)

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor, float]:
        # get the pair index
        pair_idx = self.pairs_idx[idx]

        # get the query and database image
        query_image_act = self.query_dataset[pair_idx[0]]
        database_image_act = self.database_dataset[pair_idx[1]]

        # get the similarity
        similarity = self.fov_matrix[pair_idx[0], pair_idx[1]]

        # return the pair
        return query_image_act, database_image_act, similarity


class MslsCitiesImageActPair(TensorPairSimDataset):
    def __init__(
        self,
        root_dir: str,
        activations_dir: str,
        cities: List[str],
        image_t: Union[Compose, None] = None,
        activations_t: Union[Compose, None] = None,
        positive_threshold: float = 0.5,
        pair_probs: Union[List[float], None] = None,
    ) -> None:
        self.cities_datasets = {
            city: MslsCityImageActPair(
                root_dir,
                activations_dir,
                city,
                image_t,
                activations_t,
                positive_threshold,
                pair_probs,
            )
            for city in cities
        }

    def __len__(self):
        return sum(len(dataset) for dataset in self.cities_datasets.values())

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor, float]:
        for city in self.cities_datasets.keys():
            if idx >= len(self.cities_datasets[city]):
                idx -= len(self.cities_datasets[city])
            else:
                return self.cities_datasets[city][idx]
