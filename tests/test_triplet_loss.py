#!/usr/bin/env python3
"""Unit tests"""

import numpy as np
import torch
from model.triplet_loss import _pairwise_distances
from model.triplet_loss import _get_triplet_mask
from model.triplet_loss import _get_anchor_positive_triplet_mask
from model.triplet_loss import _get_anchor_negative_triplet_mask
from model.triplet_loss import batch_all_triplet_loss
from model.triplet_loss import batch_hard_triplet_loss


def pairwise_distance_np(embeddings, squared=False):
    """Calculate pairwise distances of the given embeddings in numpy
    Args:
        embeddings (torch.tensor): Embeddings of shape (batch_size, embed_dim)
        squared (bool): Squared euclidean distance matrix
    Returns:
        pdists (torch.tensor): Pairwise distances of shape (batch_size, batch_size)
    """
    num_data = embeddings.shape[0]
    triu = np.triu_indices(num_data, 1)
    upper_tri_pdists = np.linalg.norm(embeddings[triu[1]] - embeddings[triu[0]], axis=1)

    if squared:
        upper_tri_pdists **= 2.0

    pdists = np.zeros((num_data, num_data))
    pdists[triu] = upper_tri_pdists
    pdists = pdists + pdists.T - np.diag(pdists.diagonal())

    return pdists

def test_pairwise_distances():
    """Test the pairwise distances function."""
    num_data = 64
    feat_dim = 6

    embeddings = np.random.randn(num_data, feat_dim)
    embeddings[1] = embeddings[0]  # to get distance 0

    for squared in [True, False]:
        res_np = pairwise_distance_np(embeddings, squared=squared)
        res_pt = _pairwise_distances(torch.as_tensor(embeddings), squared=squared)
        assert np.allclose(res_np, res_pt)

def test_pairwise_distances_are_positive():
    """Test that the pairwise distances are always positive. Use a tricky case
    where numerical errors are common.
    """
    num_data = 64
    feat_dim = 6

    # Create embeddings very close to each other in [1.0 - 2e-7, 1.0 + 2e-7]
    # This will encourage errors in the computation
    embeddings = 1.0 + 2e-7 * np.random.randn(num_data, feat_dim)
    embeddings[1] = embeddings[0]  # to get distance 0
    embeddings = torch.as_tensor(embeddings)

    for squared in [True, False]:
        res_pt = _pairwise_distances(embeddings, squared=squared)
        assert (res_pt >= 0.0).all()

def test_triplet_mask():
    """Test function _get_triplet_mask."""
    num_data = 64
    num_classes = 10

    labels = np.random.randint(0, num_classes, size=(num_data))

    mask_np = np.zeros((num_data, num_data, num_data))
    for i in range(num_data):
        for j in range(num_data):
            for k in range(num_data):
                distinct = (i != j and i != k and j != k)
                valid = (labels[i] == labels[j]) and (labels[i] != labels[k])
                mask_np[i, j, k] = (distinct and valid)

    mask_pt_val = _get_triplet_mask(torch.as_tensor(labels))
    assert np.allclose(mask_np, mask_pt_val)

def test_anchor_positive_triplet_mask():
    """Test function _get_anchor_positive_triplet_mask."""
    num_data = 64
    num_classes = 10

    labels = np.random.randint(0, num_classes, size=(num_data))

    mask_np = np.zeros((num_data, num_data))
    for i in range(num_data):
        for j in range(num_data):
            distinct = (i != j)
            valid = labels[i] == labels[j]
            mask_np[i, j] = (distinct and valid)

    mask_pt_val = _get_anchor_positive_triplet_mask(torch.as_tensor(labels))
    assert np.allclose(mask_np, mask_pt_val)

def test_anchor_negative_triplet_mask():
    """Test function _get_anchor_negative_triplet_mask."""
    num_data = 64
    num_classes = 10

    labels = np.random.randint(0, num_classes, size=(num_data))

    mask_np = np.zeros((num_data, num_data))
    for i in range(num_data):
        for k in range(num_data):
            distinct = (i != k)
            valid = (labels[i] != labels[k])
            mask_np[i, k] = (distinct and valid)

    mask_pt_val = _get_anchor_negative_triplet_mask(torch.as_tensor(labels))
    assert np.allclose(mask_np, mask_pt_val)

def test_simple_batch_all_triplet_loss():
    """Test the triplet loss with batch all triplet mining in a simple case.
    There is just one class in this super simple edge case, and we want to make sure that
    the loss is 0.
    """
    num_data = 10
    feat_dim = 6
    margin = 0.2
    num_classes = 1

    embeddings = np.random.rand(num_data, feat_dim)
    labels = np.random.randint(0, num_classes, size=(num_data))
    labels, embeddings = torch.as_tensor(labels), torch.as_tensor(embeddings)

    for squared in [True, False]:
        loss_np = 0.0
        loss_pt_val = batch_all_triplet_loss(labels,
                                             embeddings,
                                             margin,
                                             squared=squared)
        assert np.allclose(loss_np, loss_pt_val)

def test_batch_all_triplet_loss():
    """Test the triplet loss with batch all triplet mining"""
    num_data = 10
    feat_dim = 6
    margin = 0.2
    num_classes = 5

    embeddings = np.random.rand(num_data, feat_dim)
    labels = np.random.randint(0, num_classes, size=(num_data))

    for squared in [True, False]:
        pdist = pairwise_distance_np(embeddings, squared=squared)

        loss_np = 0.0
        num_positives = 0
        for i in range(num_data):
            for j in range(num_data):
                for k in range(num_data):
                    distinct = (i != j and i != k and j != k)
                    valid = (labels[i] == labels[j]) and (labels[i] != labels[k])
                    if distinct and valid:
                        pos_distance = pdist[i][j]
                        neg_distance = pdist[i][k]

                        loss = np.maximum(0.0, pos_distance - neg_distance + margin)
                        loss_np += loss

                        num_positives += (loss > 0)

        loss_np /= num_positives

        loss_pt_val = batch_all_triplet_loss(torch.as_tensor(labels),
                                             torch.as_tensor(embeddings),
                                             margin,
                                             squared=squared)
        assert np.allclose(loss_np, loss_pt_val)

def test_batch_hard_triplet_loss():
    """Test the triplet loss with batch hard triplet mining"""
    num_data = 50
    feat_dim = 6
    margin = 0.2
    num_classes = 5

    embeddings = np.random.rand(num_data, feat_dim)
    labels = np.random.randint(0, num_classes, size=(num_data))

    for squared in [True, False]:
        pdist = pairwise_distance_np(embeddings, squared=squared)

        loss_np = 0.0
        for i in range(num_data):
            max_pos_dist = np.max(pdist[i][labels == labels[i]])
            min_neg_dist = np.min(pdist[i][labels != labels[i]])

            loss = np.maximum(0.0, max_pos_dist - min_neg_dist + margin)
            loss_np += loss

        loss_np /= num_data

        loss_pt_val = batch_hard_triplet_loss(torch.as_tensor(labels),
                                              torch.as_tensor(embeddings),
                                              margin,
                                              squared=squared)
        assert np.allclose(loss_np, loss_pt_val)
