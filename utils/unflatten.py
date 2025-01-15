import numpy as np
from typing import List


def unflatten(source: np.ndarray, lengths: np.ndarray) -> List[np.ndarray]:
    """
    Takes a source array and unflattens it into a list of arrays based on specified lengths.

    Parameters
    ----------
    source : np.ndarray
        The source array to be unflattened.
    lengths : np.ndarray
        Array of integers specifying the length of each subarray.

    Returns
    -------
    List[np.ndarray]
        List of numpy arrays, each with length specified by the lengths parameter.
    """
    # Initialize result list
    result = []
    current_idx = 0

    # Convert lengths to 1D array if it's not already
    lengths = np.ravel(lengths)

    # Handle each trajectory length
    for length in lengths:
        # Get the slice of the source array
        end_idx = current_idx + int(np.asscalar(length)) if hasattr(np, 'asscalar') else current_idx + int(
            length.item())
        result.append(source[current_idx:end_idx])
        current_idx = end_idx

    return result

