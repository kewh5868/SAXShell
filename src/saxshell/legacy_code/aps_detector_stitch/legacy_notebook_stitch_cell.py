"""Extract of the old APS 5-ID-D detector stitching notebook cell.

This is preserved for reference only. Use
``saxshell.saxs.aps_detector_stitch`` for maintained code.
"""

import numpy as np


def legacy_stitch_after_subtraction(
    sol_102,
    sol_103,
    sol_104,
    emp_102,
    emp_103,
    emp_104,
):
    sub_102 = np.zeros(np.shape(sol_102))
    sub_103 = np.zeros(np.shape(sol_103))
    sub_104 = np.zeros(np.shape(sol_104))

    cap_size = 1
    sub_102[:, 1] = (sol_102[:, 1] - emp_102[:, 1]) / cap_size
    sub_102[:, 2] = (sol_102[:, 2] + emp_102[:, 2]) / cap_size
    sub_102[:, 0] = sol_102[:, 0]

    sub_103[:, 1] = (sol_103[:, 1] - emp_103[:, 1]) / cap_size
    sub_103[:, 2] = (sol_103[:, 2] + emp_103[:, 2]) / cap_size
    sub_103[:, 0] = sol_103[:, 0]

    sub_104[:, 1] = (sol_104[:, 1] - emp_104[:, 1]) / cap_size
    sub_104[:, 2] = (sol_104[:, 2] + emp_104[:, 2]) / cap_size
    sub_104[:, 0] = sol_104[:, 0]

    sub_102 = sub_102[9:, :]
    sub_104 = sub_104[:485, :]

    sub_103_cf = sub_104[-1, 1] / sub_103[0, 1]
    sub_103[:, 1] = sub_103[:, 1] * sub_103_cf
    sub_103[:, 2] = sub_103[:, 2] * sub_103_cf

    sub_102_cf = sub_103[-1, 1] / sub_102[0, 1]
    sub_102[:, 1] = sub_102[:, 1] * sub_102_cf
    sub_102[:, 2] = sub_102[:, 2] * sub_102_cf

    return np.concatenate((sub_104, sub_103, sub_102), axis=0)
