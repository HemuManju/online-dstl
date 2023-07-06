import torch
from torch.utils.data import Dataset


class TorchDataset(Dataset):
    """All subject dataset class.

    Parameters
    ----------
    split_ids : list
        ids list of training or validation or traning data.

    Attributes
    ----------
    split_ids

    """

    def __init__(self, points, seq_length=4):
        super(TorchDataset, self).__init__()
        self.points = points
        self.seq_len = seq_length

    def __getitem__(self, index):
        # Read only specific data and convert to torch tensors
        appended_data = torch.from_numpy(self.points[0 : index + self.seq_len, :]).type(
            torch.float32
        )
        out_appended_data = torch.from_numpy(
            self.points[1 : index + self.seq_len + 1, :]
        ).type(torch.float32)
        input_data = torch.from_numpy(
            self.points[index : index + self.seq_len, :]
        ).type(torch.float32)
        output_data = torch.from_numpy(
            self.points[index + 1 : index + self.seq_len + 1, :]
        ).type(torch.float32)
        return input_data, output_data, appended_data, out_appended_data

    def __len__(self):
        return self.points.shape[0] - self.seq_len
