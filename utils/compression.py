import numpy as np

def compress_sparse_matrix(matrix, n_value, tol=0.1):
    """
    Compress matrix by storing only values that differ from n_value by more than tol.
    Returns (elements, indices) where indices are [start, end] of each run.
    """
    flat = matrix.flatten()
    non_n = np.where(np.abs(flat - n_value) > tol)[0]
    if len(non_n) == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.int32)

    elements = []
    indices = []
    start = non_n[0]
    current = [flat[start]]
    for i in range(1, len(non_n)):
        if non_n[i] == non_n[i-1] + 1:
            current.append(flat[non_n[i]])
        else:
            elements.extend(current)
            indices.append([start, non_n[i-1]])
            start = non_n[i]
            current = [flat[start]]
    elements.extend(current)
    indices.append([start, non_n[-1]])
    return np.array(elements, dtype=np.float32), np.array(indices, dtype=np.int32)

def restore_sparse_matrix(compressed_data, n_value, original_shape):
    elements, indices = compressed_data
    if len(elements) == 0:
        return np.full(original_shape, n_value, dtype=np.float32)
    total = np.prod(original_shape)
    restored = np.full(total, n_value, dtype=np.float32)
    elem_idx = 0
    for start, end in indices:
        length = end - start + 1
        restored[start:end+1] = elements[elem_idx:elem_idx+length]
        elem_idx += length
    return restored.reshape(original_shape)