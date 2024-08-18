import numpy as np

def cal_gap(res:dict, opt):
    for alg in res.values():
        alg["gap"] = abs(alg["value"] - opt)
        alg["epsilon"] = 1 /alg["gap"]

