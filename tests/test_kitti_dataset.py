import importlib.util

import pytest

if importlib.util.find_spec("numpy") is None:
    pytest.skip("numpy is required for evaluateResults tests", allow_module_level=True)

if importlib.util.find_spec("faiss") is None:
    pytest.skip("faiss is required for evaluateResults tests", allow_module_level=True)

import numpy as np

from kitti_dataset import evaluateResults


class DummyDataset:
    def __init__(self, poses, db_split_index=1, sample_inteval=200, image_shape=(8, 8)):
        self.poses = np.array(poses, dtype=float)
        self.db_split_index = db_split_index
        self.sample_inteval = sample_inteval
        self.image_shape = image_shape

    def __getitem__(self, idx):
        dummy_image = np.zeros((3, *self.image_shape), dtype=np.float32)
        return dummy_image, idx

    def __len__(self):
        return len(self.poses)


def test_evaluate_results_top1_recall_without_local_features():
    poses = [
        [0.0] * 12,
        [10.0] * 12,
        [0.1] * 12,  # query close to first pose
    ]
    dataset = DummyDataset(poses)

    # Descriptors place the first database entry closest to the query
    global_descs = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.1, 0.0],
        ],
        dtype=np.float32,
    )

    recall = evaluateResults("00", global_descs, None, dataset, top_k=1)

    assert recall == 1.0


def test_evaluate_results_handles_missing_local_features_gracefully():
    poses = [
        [0.0] * 12,
        [10.0] * 12,
        [0.1] * 12,
    ]
    dataset = DummyDataset(poses)

    global_descs = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.1, 0.0],
        ],
        dtype=np.float32,
    )

    # Local features filled with zeros trigger the geometric verifier early exits
    local_feats = np.zeros((len(global_descs), 1, 4, 4), dtype=np.float32)

    recall = evaluateResults("00", global_descs, local_feats, dataset, top_k=1)

    assert recall == 1.0
