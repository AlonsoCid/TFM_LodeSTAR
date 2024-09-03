import numpy as np
from scipy.optimize import linear_sum_assignment
import scipy.spatial as sc

def compute_matching(lc, pc, distance_threshold=1):
    """
    Computes the matching between ground truth and predicted objects using the Hungarian algorithm,
    considering only the centroid positions (no size/extent involved).

    Parameters:
    lc: np.array - Ground truth centroids
    pc: np.array - Predicted centroids
    distance_threshold: float (optional) - Maximum allowed distance for matching. Pairs with a higher distance are discarded.

    Returns:
    true_positives: int - Count of matched ground truth and prediction centroids
    pairs_dist: np.array - Distances between matched centroids
    false_negatives: int - Number of unmatched ground truth objects (false negatives)
    false_positives: int - Number of unmatched predicted objects (false positives)
    """
    # Compute the distance matrix between centroids (Euclidean distance)
    costd = sc.distance.cdist(lc, pc, metric='euclidean')

    # Solve the linear assignment problem using the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(costd)

    # Get distances for the matched pairs
    pairs_dist = costd[row_ind, col_ind]

    # Filter pairs by the distance threshold
    valid_mask = pairs_dist < distance_threshold
    true_positives = np.sum(valid_mask)  # Count of valid matches (true positives)
    filtered_pairs_dist = pairs_dist[valid_mask]  # Distances for valid matches

    # Calculate false negatives (unmatched ground truth) and false positives (unmatched predictions)
    false_negatives = len(lc) - true_positives
    false_positives = len(pc) - true_positives

    # Calculate jaccard index
    intersection = true_positives
    union = true_positives + false_negatives + false_positives

    if union == 0:
        return 1.0  # No objects in ground truth or prediction, Jaccard index is 1
    
    jaccard_index = intersection / union

    return true_positives, filtered_pairs_dist, false_negatives, false_positives, jaccard_index