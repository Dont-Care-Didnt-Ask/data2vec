__all__ = ['ImagenetIterableYTDataset', 'IterableYTDataset']

import math
from torch.utils.data import IterableDataset

import torch.distributed as dist
import torch.utils.data as torch_data
from PIL import Image
from io import BytesIO

import yt.wrapper as yt

def get_local_num_workers():
    num_local = 1
    local_worker_info = torch_data.get_worker_info()
    if local_worker_info is not None:
        num_local = local_worker_info.num_workers
    return num_local


def get_local_worker_id():
    id_local = 0
    local_worker_info = torch_data.get_worker_info()
    if local_worker_info is not None:
        id_local = local_worker_info.id
    return id_local


def get_dist_num_workers():
    num_dist = 1
    if dist.is_initialized():
        num_dist = dist.get_world_size()
    return num_dist


def get_dist_worker_id():
    id_dist = 0
    if dist.is_initialized():
        id_dist = dist.get_rank()
    return id_dist


class IterableYTDataset(IterableDataset):
    def __init__(self, path, num_readers=4, chunk_distributed=True):
        self._table = path
        self._num_readers = num_readers
        self._reader = None
        self._chunk_distributed = chunk_distributed

        self.table = yt.TablePath(path)
        self.client = yt.YtClient(proxy='hahn')
        self.total_rows = self.client.get_attribute(self.table, 'row_count')
        self.global_num_workers = get_dist_num_workers()
        self.global_worker_idx = get_dist_worker_id()

    def set_ranges(self):
        local_num_workers, local_worker_idx = get_local_num_workers(), get_local_worker_id()
        global_worker_idx = self.global_worker_idx * local_num_workers + local_worker_idx

        start = global_worker_idx * len(self)
        end = min(self.total_rows, start + len(self))

        self.table.ranges = [dict(
            lower_limit=dict(row_index=start),
            upper_limit=dict(row_index=end),
        )]

    def __len__(self):
        global_num_workers = self.global_num_workers
        local_num_workers = get_local_num_workers()

        if self._chunk_distributed:
            num_workers = global_num_workers * local_num_workers
        else:
            num_workers = local_num_workers

        return int(math.floor(self.total_rows / num_workers))

    def row_processor(self, row):
        return row

    def generate_data(self):
        "Iterate over yt table"
        self.client = yt.YtClient(proxy='hahn')

        self.set_ranges()
        rows = self.client.read_table(
            self.table,
            format=yt.YsonFormat(attributes=dict(encode_utf8=True), encoding=None),
        )

        for row in rows:
            yield self.row_processor(row)

    def __iter__(self):
        return self.generate_data()


class ImagenetIterableYTDataset(IterableYTDataset):
    def __init__(self, path, transform, num_readers=4, chunk_distributed=True):
        super().__init__(path, num_readers=num_readers, chunk_distributed=chunk_distributed)
        self.transform = transform

    def row_processor(self, row):
        row = {k.decode('utf-8'): v for k, v in row.items()}
        return self.transform(Image.open(BytesIO(row['image'])).convert("RGB")), row['label']
