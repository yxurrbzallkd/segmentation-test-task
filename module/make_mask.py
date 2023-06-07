import numpy as np

def ships_to_mask(ships,W=768,H=768):
    """
    The run-length format:
    all pixels are numberated as in a flattened image
    (left to right, top to bottom)
    Our data is
        ship_id_1:
            [loc_seq_1, len_seq_1, loc_seq_2, len_seq_2 ...]
        ship_id_2:
            ...
    """
    mask = np.zeros((W*H),dtype=np.uint8) # locations are with respect to a flat mask
    for i in ships:
        for j in range(len(ships[i]) // 2):
            # take all locations and set mask from loc to loc + loc_len to 1
            mask[ships[i][j * 2]:ships[i][j * 2]+ships[i][j * 2+1]] = 1
    mask = mask.reshape((W,H)) # reshape the mask
    return mask
