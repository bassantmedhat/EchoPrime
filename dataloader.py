import csv
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import scipy.io
import EchoPrime.video_utils as video_utils

prefix = 'dir_path'
csv_file = 'data/AS_data.csv'

def get_all_files_and_ground_truths(filename):
    """
    Reads a CSV file and retrieves the 'files_paths' and 'views' columns from all rows.

    Args:ss
    - filename (str): The path to the CSV file.

    Returns:
    - list: A list of tuples containing (study_id, file_value, view_GT, AS_label_GT) for each row.
    """
    results = []
    try:
        with open(filename, mode='r') as file:
            reader = csv.DictReader(file)  # Reads rows as dictionaries
            for row in reader:
                study_id = row['study_id']
                file_value = row['files']
                view_GT = row['views']
                AS_label_GT = row['AS_label']
                results.append([study_id, file_value, view_GT, AS_label_GT])
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
    except KeyError as e:
        print(f"Error: Missing expected column in the CSV file: {e}")
    return results

class EchoDataset(Dataset):
    def __init__(self, files, mode="whole_study", frames_to_take=32, frame_stride=2, video_size=224, mean=None, std=None):
        """
        Args:
            files (list): List of study metadata (from CSV).
            mode (str): "whole_study" or "paired_views" (default: "whole_study").
            frames_to_take (int): Number of frames to extract per video.
            frame_stride (int): Step size for frame sampling.
            video_size (int): Size to resize frames to.
            mean (torch.Tensor): Mean for normalization.
            std (torch.Tensor): Standard deviation for normalization.
        """
        self.files = files
        self.mode = mode
        self.frames_to_take = frames_to_take
        self.frame_stride = frame_stride
        self.video_size = video_size
        self.mean = mean if mean is not None else torch.tensor([29.110628, 28.076836, 29.096405]).reshape(3, 1, 1, 1)
        self.std = std if std is not None else torch.tensor([47.989223, 46.456997, 47.20083]).reshape(3, 1, 1, 1)

        if mode == "paired_views":
            self.paired_files = self.create_paired_files()

    def create_paired_files(self):
        """
        Creates a list of paired video files (PSAX and PLAX) with their corresponding view_GT for the paired_views mode.
        """
        paired_files = []
        for study in self.files:
            study_id, mat_paths, view_GT, AS_label_GT = study
            mat_paths = mat_paths.split(',')
            view_GT = view_GT.split(',')
            AS_label_GT = AS_label_GT.split(',')
            AS_label_GT = list(map(lambda x: x.strip(), AS_label_GT))

            # Separate PSAX and PLAX videos with corresponding views
            psax_files = [(mat_paths[i], view_GT[i]) for i in range(len(mat_paths)) if "psax" in view_GT[i].lower()]
            plax_files = [(mat_paths[i], view_GT[i]) for i in range(len(mat_paths)) if "plax" in view_GT[i].lower()]

            # Pair PSAX and PLAX videos, ignore unmatched videos
            num_pairs = min(len(psax_files), len(plax_files))
            for i in range(num_pairs):
                psax_path, psax_view = psax_files[i]
                plax_path, plax_view = plax_files[i]
                paired_files.append((study_id, psax_path, plax_path, psax_view, plax_view, AS_label_GT[i]))

        return paired_files

    def preprocess_video(self, mat_file_path):
        """
        Preprocesses a single video.
        """
        try:
            mat_file_path = prefix.strip() + mat_file_path.strip()
            mat_data = scipy.io.loadmat(mat_file_path)
            pixels = mat_data['cine']

            if pixels.ndim == 3:
                pixels = np.repeat(pixels[..., None], 3, axis=3)

            x = np.zeros((len(pixels), self.video_size, self.video_size, 3))
            for i in range(len(x)):
                x[i] = video_utils.crop_and_scale(pixels[i])

            x = torch.as_tensor(x, dtype=torch.float).permute([3, 0, 1, 2])
            x.sub_(self.mean).div_(self.std)

            if x.shape[1] < self.frames_to_take:
                padding = torch.zeros(
                    (3, self.frames_to_take - x.shape[1], self.video_size, self.video_size),
                    dtype=torch.float,
                )
                x = torch.cat((x, padding), dim=1)

            start = 0
            return x[:, start : (start + self.frames_to_take) : self.frame_stride, :, :]
        except Exception as e:
            print("corrupt file:", mat_file_path)
            print(str(e))
            return None

    def __getitem__(self, idx):
        if self.mode == "whole_study":
            # Process all videos in the study
            study_id, mat_paths, view_GT, AS_label_GT = self.files[idx]
            mat_paths = mat_paths.split(',')
            video_tensor = torch.stack([self.preprocess_video(path) for path in mat_paths if path])
            return study_id, video_tensor, view_GT, AS_label_GT

        elif self.mode == "paired_views":
            # Return paired PSAX and PLAX videos with their views
            study_id, psax_path, plax_path, psax_view, plax_view, AS_label_GT = self.paired_files[idx]
            psax_tensor = self.preprocess_video(psax_path)
            plax_tensor = self.preprocess_video(plax_path)

            if psax_tensor is None or plax_tensor is None:
                raise ValueError(f"Corrupt video in study {study_id}")

            # Stack PSAX and PLAX tensors along the batch dimension
            stacked_tensor = torch.stack([psax_tensor, plax_tensor], dim=0)
            return study_id, stacked_tensor, [psax_view, plax_view], AS_label_GT

    def __len__(self):
        return len(self.paired_files) if self.mode == "paired_views" else len(self.files)


def create_dataloader(csv_file, mode, batch_size=4, shuffle=True, num_workers=2):
    """
    Creates a DataLoader for the EchoDataset.

    Args:
    - csv_file (str): Path to the CSV file containing data information.
    - batch_size (int): Number of samples per batch.
    - shuffle (bool): Whether to shuffle the data at the beginning of each epoch.
    - num_workers (int): Number of subprocesses to use for data loading.

    Returns:
    - DataLoader: PyTorch DataLoader for the EchoDataset.
    """
    files = get_all_files_and_ground_truths(csv_file)
    dataset = EchoDataset(files, mode=mode)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader



