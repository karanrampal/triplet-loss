#!/usr/bin/env python3
"""Define the triplet loss function
"""

import torch


def _pairwise_distances(embeddings, squared=False):
    """Calculate pairwise distances of the given embeddings
    Args:
        embeddings (torch.tensor): Embeddings of shape (batch_size, embed_dim)
        squared (bool): Squared euclidean distance matrix
    Returns:
        dists (torch.tensor): Pairwise distances of shape (batch_size, batch_size)
    """
    dot_prod = torch.matmul(embeddings, embeddings.T)
    sq_norm = dot_prod.diagonal(0)
    dists = sq_norm.unsqueeze(0) - 2.0 * dot_prod + sq_norm.unsqueeze(1)

    # Due to computation errors some dists may be negative so we make them 0.0
    dists = torch.clamp(dists, min=0.0)

    if not squared:
        # Gradient of sqrt is infinite when dists are 0.0
        mask = dists.eq(0.0).float()
        dists = dists + mask * 1e-16
        dists = (1.0 - mask) * torch.sqrt(dists)

    return dists

def _get_anchor_positive_triplet_mask(labels):
    """Get a 2D mask where mask[a, p] is True iff a and p have same label but not
    the same index.
    Args:
        labels (torch.tensor): Labels with shape (batch_size)
    Returns:
        mask (torch.bool): Mask of shape (batch_size, batch_size)
    """
    # Check that i and j are distinct
    indices_equal = torch.eye(labels.shape[0], dtype=torch.bool)
    indices_not_equal = ~indices_equal

    # Use broadcasting to make the mask
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    mask = labels_equal & indices_not_equal

    return mask

def _get_anchor_negative_triplet_mask(labels):
    """Get a 2D mask where mask[a, n] is True iff a and n have different labels.
    Args:
        labels (torch.tensor): Labels with shape (batch_size)
    Returns:
        mask (torch.bool): Mask of shape (batch_size, batch_size)
    """
    # Use broadcasting to make the mask
    mask = ~(labels.unsqueeze(0) == labels.unsqueeze(1))

    return mask

def _get_triplet_mask(labels):
    """Get a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) are valid.
    A triplet (i, j, k) is valid if i, j, k are not the same indexes and
    labels[i] == labels[j] and labels[i] != labels[k]
    Args:
        labels (torch.tensor): Labels with shape (batch_size,)
    Returns:
        mask (torch.bool): Mask of shape (batch_size, batch_size, batch_size)
    """
    indices_equal = torch.eye(labels.shape[0], dtype=torch.bool)
    indices_not_equal = ~indices_equal
    i_not_equal_j = indices_not_equal.unsqueeze(2)
    i_not_equal_k = indices_not_equal.unsqueeze(1)
    j_not_equal_k = indices_not_equal.unsqueeze(0)

    distinct_indices = (i_not_equal_j & i_not_equal_k) & j_not_equal_k

    label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    label_i_equal_j = label_equal.unsqueeze(2)
    label_i_not_equal_k = ~label_equal.unsqueeze(1)

    valid_labels = label_i_not_equal_k & label_i_equal_j
    mask = valid_labels & distinct_indices

    return mask

def batch_all_triplet_loss(labels, embeddings, margin, squared=False):
    """Calculate triplet loss for a batch of embeddings. We generate all the valid
    triplets.
    Args:
        labels (torch.tensor): Labels of the batch of size (batch_size,)
        embeddings (torch.tensor): Embeddings of shape (batch_size, embed_dim)
        margin (float): Margin for triplet loss
        squared (bool): Squared euclidean distance matrix
    Returns:
        triplet_loss (torch.tensor): Scalar tensor containing the triplet loss
    """
    pdist = _pairwise_distances(embeddings, squared=squared)

    anchor_positive_dist = pdist.unsqueeze(2)
    anchor_negative_dist = pdist.unsqueeze(1)

    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    mask = _get_triplet_mask(labels)
    triplet_loss = mask.float() * triplet_loss
    triplet_loss = torch.clamp(triplet_loss, min=0.0)

    # Count number of valid hard triplets (where triplet_loss > 0)
    valid_hard_triplets = triplet_loss.gt(1e-16)
    num_valid_hard_triplets = valid_hard_triplets.sum()
    triplet_loss = triplet_loss.sum() / (num_valid_hard_triplets + 1e-16)

    return triplet_loss

def batch_hard_triplet_loss(labels, embeddings, margin, squared=False):
    """Calculate triplet loss for a batch of embeddings. For each anchor, we take
    the hardest positive and hardest negative to form a triplet.
    Args:
        labels (torch.tensor): Labels of the batch of size (batch_size,)
        embeddings (torch.tensor): Embeddings of shape (batch_size, embed_dim)
        margin (float): Margin for triplet loss
        squared (bool): Squared euclidean distance matrix
    Returns:
        triplet_loss (torch.tensor): Scalar tensor containing the triplet loss
    """
    pdist = _pairwise_distances(embeddings, squared=squared)

    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels).float()
    anchor_positive_dist = mask_anchor_positive * pdist
    hardest_positive_dist, _ = anchor_positive_dist.max(1, keepdim=True)

    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels).float()
    max_anchor_negative_dist, _ = pdist.max(1, keepdim=True)
    anchor_negative_dist = pdist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)
    hardest_negative_dist, _ = anchor_negative_dist.min(1, keepdim=True)

    loss = torch.clamp(hardest_positive_dist - hardest_negative_dist + margin, min=0.0)
    triplet_loss = loss.mean()

    return triplet_loss
