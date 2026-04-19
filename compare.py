# Compare dumps, report discrepancies
#
# When the dump feature is set, verbose logging is output to stdout.

import sys
kws = ('huffman', 'antialias', 'imdct', 'ms_stereo_l', 'ms_stereo_r', 'reorder', 'change_sign',
       'imdct36', 'imdct_short', 'imdct36_b', 'l3_dct_9_in', 'l3_dct_9_out',
       'dst1', 'dst2', 'dst3', 'dst4', 'leaf', 'leaf2', 'linbits',
       'dct_ii', 'nbands', 't', 't2')

def filter_file(filename):
    with open(filename) as file:
        lines = [line.rstrip().split() for line in file]
    filtered = [s for s in lines if s[0] in kws]
    return filtered

def compare(s1, s2):
    if s1[0] != s2[0]:
        return f'type mismatch {s1[0]} {s2[0]}'
    if len(s1) != len(s2):
        return 'length mismatch'
    for j in range(1, len(s1)):
        v1 = float(s1[j])
        v2 = float(s2[j])
        if abs(v1 - v2) > 1e-6:
            return f'value mismatch {s1[0]} at {j}: {v1} != {v2}'

def main():
    l1 = filter_file(sys.argv[1])
    l2 = filter_file(sys.argv[2])
    print(len(l1), len(l2))
    for i in range(min(len(l1), len(l2))):
        result = compare(l1[i], l2[i])
        if result is not None:
            print(f'line {i}: {result}')
            break



main()