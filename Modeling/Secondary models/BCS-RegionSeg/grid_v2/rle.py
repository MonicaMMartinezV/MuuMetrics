import numpy as np
def decode_counts(s):
    # COCO compressed RLE string -> list of counts
    cnts, p = [], 0
    while p < len(s):
        x, k, more = 0, 0, True
        while more:
            c = ord(s[p]) - 48
            x |= (c & 0x1f) << (5 * k)
            more = c & 0x20
            p += 1; k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if len(cnts) > 2:
            x += cnts[-2]
        cnts.append(x)
    return cnts
def rle_to_mask(seg):
    h, w = seg['size']
    cnts = decode_counts(seg['counts']) if isinstance(seg['counts'], str) else seg['counts']
    flat = np.zeros(h * w, np.uint8); pos, v = 0, 0
    for c in cnts:
        flat[pos:pos + c] = v; pos += c; v ^= 1
    return flat.reshape((w, h)).T  # column-major
