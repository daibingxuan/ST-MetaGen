import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes
from itertools import product
import os
from tqdm import tqdm

# ==============================================================================
# 1. 结构因子水平集函数生成（对应 structureFactors.m）
# ==============================================================================
def structureFactors(group, hkl=None, AB=0, origin=1, verbose=0):
    """
    生成基于晶体空间群的水平集函数（level-set field）
    f <= 0 对应固体区域
    
    参数:
        group: int，空间群编号（立方晶系 195~230）
        hkl: (N,3) array 或 list，米勒指数；为 None 时自动生成 0~3 的所有组合（去掉(0,0,0)）
        AB: int，0或1，部分空间群支持 AB 两种变体
        origin: int，1或2，部分空间群支持双原点选择
        verbose: int，是否打印详细信息
    
    返回:
        results: list of dict，每个元素包含
            'group': 空间群号
            'f': 水平集函数句柄，输入 x,y,z 返回场值
            'HKL': 对应米勒指数元组
            'AB': AB 类型
            'origin': 原点选择
            'exitflag': 退出状态
                0: 成功
                1: 无效空间群号
                2: 当前 HKL 不允许
                3: AB 参数不允许
                4: origin 参数不允许
    """
    valid_AB_groups = [
        195,196,197,198,199,207,209,211,208,210,
        212,213,214,215,216,217,218,219,220
    ]
    if AB == 1 and group not in valid_AB_groups:
        if verbose:
            print(f'    *** No. {group} origin={origin} *AB={AB}* not allowed ***')
        return [{
            'group': group, 'f': None, 'HKL': None,
            'AB': AB, 'origin': origin, 'exitflag': 3
        }]

    valid_origin2_groups = [201, 203, 222, 224, 227, 228]
    if origin == 2 and group not in valid_origin2_groups:
        if verbose:
            print(f'    *** No. {group} *origin={origin}* AB={AB} not allowed ***')
        return [{
            'group': group, 'f': None, 'HKL': None,
            'AB': AB, 'origin': origin, 'exitflag': 4
        }]

    if hkl is None or len(np.atleast_2d(hkl)) == 0:
        miller = [0, 1, 2, 3]
        hkl_list = list(product(miller, repeat=3))
        hkl_list = hkl_list[1:]
    else:
        hkl_arr = np.atleast_2d(hkl)
        hkl_list = [tuple(row) for row in hkl_arr]

    results = []
    for _ in range(len(hkl_list)):
        results.append({
            'group': group, 'f': None, 'HKL': None,
            'AB': AB, 'origin': origin, 'exitflag': 0
        })

    def c(u, v):
        return np.cos(u * v)
    def s(u, v):
        return np.sin(u * v)

    def make_E_O(h, k, l):
        def E_func(p, q, r):
            def field(x, y, z):
                t1 = p(h, x) * q(k, y) * r(l, z)
                t2 = p(h, y) * q(k, z) * r(l, x)
                t3 = p(h, z) * q(k, x) * r(l, y)
                return t1 + t2 + t3
            return field

        def O_func(p, q, r):
            def field(x, y, z):
                t1 = p(h, x) * q(k, z) * r(l, y)
                t2 = p(h, z) * q(k, y) * r(l, x)
                t3 = p(h, y) * q(k, x) * r(l, z)
                return t1 + t2 + t3
            return field
        return E_func, O_func

    for i, (h, k, l) in enumerate(hkl_list):
        res = results[i]
        res['HKL'] = (h, k, l)
        E, O = make_E_O(h, k, l)
        f = None

        if group in {195, 196, 197}:
            if AB == 0:
                f = E(c, c, c)
            else:
                f = E(s, s, s)

        elif group in {198, 199}:
            if AB == 0:
                if (h + k) % 2 == 0 and h == l and k == l:
                    f = E(c, c, c)
                elif (h + k) % 2 == 0 and h + 1 == l and k == h:
                    f = E(c, s, s)
                elif (h + k) % 2 == 1 and h == l + 1 and k == l:
                    f = E(s, c, s)
                elif (h + k) % 2 == 1 and h == l and k == l + 1:
                    f = E(s, s, c)
            else:
                if (h + k) % 2 == 0 and h == l and k == l:
                    f = E(s, s, s)
                elif (h + k) % 2 == 0 and h + 1 == l and k == h:
                    f = E(s, c, c)
                elif (h + k) % 2 == 1 and h == l + 1 and k == l:
                    f = E(c, s, c)
                elif (h + k) % 2 == 1 and h == l and k == l + 1:
                    f = E(c, c, s)

        elif group in {200, 202, 204}:
            f = E(c, c, c)

        elif group == 201:
            if origin == 1:
                if (h + k + l) % 2 == 0:
                    f = E(c, c, c)
                else:
                    f = E(s, s, s)
            else:
                if (h + k) % 2 == 0 and h == l and k == l:
                    f = E(c, c, c)
                elif (h + k) % 2 == 0 and h + 1 == l and k == h:
                    f = E(s, s, c)
                elif (h + k) % 2 == 1 and h == l + 1 and k == l:
                    f = E(c, s, s)
                elif (h + k) % 2 == 1 and h == l and k == l + 1:
                    f = E(s, c, s)

        elif group == 203:
            if origin == 1:
                mod_val = (h + k + l) % 4
                if mod_val == 0:
                    f = E(c, c, c)
                elif mod_val == 1:
                    f1, f2 = E(c, c, c), O(s, s, s)
                    def f(x, y, z): return f1(x,y,z) - f2(x,y,z)
                elif mod_val == 2:
                    f = E(s, s, s)
                elif mod_val == 3:
                    f1, f2 = E(c, c, c), O(s, s, s)
                    def f(x, y, z): return f1(x,y,z) + f2(x,y,z)
            else:
                mod_hk = (h + k) % 4
                if mod_hk == 0 and h == l and k == l:
                    f = E(c, c, c)
                elif mod_hk == 0 and h + 2 == l and k == h:
                    f = E(s, s, c)
                elif mod_hk == 2 and h == l + 2 and k == l:
                    f = E(c, s, s)
                elif mod_hk == 2 and h == l and k == l + 2:
                    f = E(s, c, s)
                elif mod_hk == 2 and h == l and k == l:
                    f1 = E(c,c,c); f2 = E(c,s,s); f3 = E(s,c,s); f4 = E(s,s,c)
                    def f(x,y,z): return f1(x,y,z)+f2(x,y,z)+f3(x,y,z)+f4(x,y,z)
                elif mod_hk == 2 and h == l + 2 and k == h:
                    f1 = E(c,c,c); f2 = E(c,s,s); f3 = E(s,c,s); f4 = E(s,s,c)
                    def f(x,y,z): return f1(x,y,z)-f2(x,y,z)-f3(x,y,z)+f4(x,y,z)
                elif mod_hk == 0 and h + 2 == l and k == l:
                    f1 = E(c,c,c); f2 = E(c,s,s); f3 = E(s,c,s); f4 = E(s,s,c)
                    def f(x,y,z): return f1(x,y,z)+f2(x,y,z)-f3(x,y,z)-f4(x,y,z)
                elif mod_hk == 0 and h == l and k + 2 == l:
                    f1 = E(c,c,c); f2 = E(c,s,s); f3 = E(s,c,s); f4 = E(s,s,c)
                    def f(x,y,z): return f1(x,y,z)-f2(x,y,z)+f3(x,y,z)-f4(x,y,z)

        elif group in {205, 206}:
            if (h + k) % 2 == 0 and h == l and k == l:
                f = E(c, c, c)
            elif (h + k) % 2 == 0 and h + 1 == l and k == h:
                f = E(c, s, s)
            elif (h + k) % 2 == 1 and h == l + 1 and k == l:
                f = E(s, c, s)
            elif (h + k) % 2 == 1 and h == l and k == l + 1:
                f = E(s, s, c)

        elif group in {207, 209, 211}:
            if AB == 0:
                f1, f2 = E(c,c,c), O(c,c,c)
                def f(x,y,z): return f1(x,y,z) + f2(x,y,z)
            else:
                f1, f2 = E(s,s,s), O(s,s,s)
                def f(x,y,z): return f1(x,y,z) - f2(x,y,z)

        elif group == 208:
            if AB == 0:
                if (h + k + l) % 2 == 0:
                    f1, f2 = E(c,c,c), O(c,c,c)
                    def f(x,y,z): return f1(x,y,z) + f2(x,y,z)
                else:
                    f1, f2 = E(c,c,c), O(c,c,c)
                    def f(x,y,z): return f1(x,y,z) - f2(x,y,z)
            else:
                if (h + k + l) % 2 == 0:
                    f1, f2 = E(s,s,s), O(s,s,s)
                    def f(x,y,z): return f1(x,y,z) - f2(x,y,z)
                else:
                    f1, f2 = E(s,s,s), O(s,s,s)
                    def f(x,y,z): return f1(x,y,z) + f2(x,y,z)

        elif group == 210:
            if AB == 0:
                mod_val = (h + k + l) % 4
                if mod_val == 0:
                    f1, f2 = E(c,c,c), O(c,c,c)
                    def f(x,y,z): return f1(x,y,z) + f2(x,y,z)
                elif mod_val == 1:
                    f1, f2 = E(c,c,c), O(s,s,s)
                    def f(x,y,z): return f1(x,y,z) - f2(x,y,z)
                elif mod_val == 2:
                    f1, f2 = E(c,c,c), O(c,c,c)
                    def f(x,y,z): return f1(x,y,z) - f2(x,y,z)
                elif mod_val == 3:
                    f1, f2 = E(c,c,c), O(s,s,s)
                    def f(x,y,z): return f1(x,y,z) + f2(x,y,z)
            else:
                mod_val = (h + k + l) % 4
                if mod_val == 0:
                    f1, f2 = E(s,s,s), O(s,s,s)
                    def f(x,y,z): return f1(x,y,z) - f2(x,y,z)
                elif mod_val == 1:
                    f1, f2 = E(s,s,s), O(c,c,c)
                    def f(x,y,z): return f1(x,y,z) - f2(x,y,z)
                elif mod_val == 2:
                    f1, f2 = E(s,s,s), O(s,s,s)
                    def f(x,y,z): return f1(x,y,z) + f2(x,y,z)
                elif mod_val == 3:
                    f1, f2 = E(s,s,s), O(c,c,c)
                    def f(x,y,z): return f1(x,y,z) + f2(x,y,z)

        elif group == 212:
            hklmod4 = (h + k + l) % 4
            if AB == 0:
                if hklmod4 == 0:
                    if (h+k)%2==0 and h==l and k==l:
                        f1,f2 = E(c,c,c), O(c,c,c)
                        def f(x,y,z): return f1(x,y,z)+f2(x,y,z)
                    elif (h+k)%2==0 and h+1==l and k==h:
                        f1,f2 = E(c,s,s), O(s,c,s)
                        def f(x,y,z): return f1(x,y,z)+f2(x,y,z)
                    elif (h+k)%2==1 and h==l+1 and k==l:
                        f1,f2 = E(s,c,s), O(s,s,c)
                        def f(x,y,z): return f1(x,y,z)+f2(x,y,z)
                    elif (h+k)%2==1 and h==l and k==l+1:
                        f1,f2 = E(s,s,c), O(c,s,s)
                        def f(x,y,z): return f1(x,y,z)+f2(x,y,z)
                elif hklmod4 == 1:
                    if (h+k)%2==0 and h==l and k==l:
                        f1,f2 = E(c,c,c), O(s,s,s)
                        def f(x,y,z): return f1(x,y,z)-f2(x,y,z)
                    elif (h+k)%2==0 and h+1==l and k==h:
                        f1,f2 = E(c,s,s), O(c,s,c)
                        def f(x,y,z): return f1(x,y,z)-f2(x,y,z)
                    elif (h+k)%2==1 and h==l+1 and k==l:
                        f1,f2 = E(s,c,s), O(c,c,s)
                        def f(x,y,z): return f1(x,y,z)-f2(x,y,z)
                    elif (h+k)%2==1 and h==l and k==l+1:
                        f1,f2 = E(s,s,c), O(s,c,c)
                        def f(x,y,z): return f1(x,y,z)-f2(x,y,z)
                elif hklmod4 == 2:
                    if (h+k)%2==0 and h==l and k==l:
                        f1,f2 = E(c,c,c), O(c,c,c)
                        def f(x,y,z): return f1(x,y,z)-f2(x,y,z)
                    elif (h+k)%2==0 and h+1==l and k==h:
                        f1,f2 = E(c,s,s), O(s,c,s)
                        def f(x,y,z): return f1(x,y,z)-f2(x,y,z)
                    elif (h+k)%2==1 and h==l+1 and k==l:
                        f1,f2 = E(s,c,s), O(s,s,c)
                        def f(x,y,z): return f1(x,y,z)-f2(x,y,z)
                    elif (h+k)%2==1 and h==l and k==l+1:
                        f1,f2 = E(s,s,c), O(c,s,s)
                        def f(x,y,z): return f1(x,y,z)-f2(x,y,z)
                elif hklmod4 == 3:
                    if (h+k)%2==0 and h==l and k==l:
                        f1,f2 = E(c,c,c), O(s,s,s)
                        def f(x,y,z): return f1(x,y,z)+f2(x,y,z)
                    elif (h+k)%2==0 and h+1==l and k==h:
                        f1,f2 = E(c,s,s), O(c,s,c)
                        def f(x,y,z): return f1(x,y,z)+f2(x,y,z)
                    elif (h+k)%2==1 and h==l+1 and k==l:
                        f1,f2 = E(s,c,s), O(c,c,s)
                        def f(x,y,z): return f1(x,y,z)+f2(x,y,z)
                    elif (h+k)%2==1 and h==l and k==l+1:
                        f1,f2 = E(s,s,c), O(s,s,c)
                        def f(x,y,z): return f1(x,y,z)+f2(x,y,z)
            else:
                if hklmod4 == 0:
                    if (h+k)%2==0 and h==l and k==l:
                        f1,f2 = E(s,s,s), O(s,s,s)
                        def f(x,y,z): return f1(x,y,z)-f2(x,y,z)
                    elif (h+k)%2==0 and h+1==l and k==h:
                        f1,f2 = E(s,c,c), O(c,s,c)
                        def f(x,y,z): return f1(x,y,z)-f2(x,y,z)
                    elif (h+k)%2==1 and h==l+1 and k==l:
                        f1,f2 = E(c,s,c), O(c,c,s)
                        def f(x,y,z): return f1(x,y,z)-f2(x,y,z)
                    elif (h+k)%2==1 and h==l and k==l+1:
                        f1,f2 = E(c,c,s), O(s,c,c)
                        def f(x,y,z): return f1(x,y,z)-f2(x,y,z)
                elif hklmod4 == 1:
                    if (h+k)%2==0 and h==l and k==l:
                        f1,f2 = E(s,s,s), O(c,c,c)
                        def f(x,y,z): return f1(x,y,z)-f2(x,y,z)
                    elif (h+k)%2==0 and h+1==l and k==h:
                        f1,f2 = E(s,c,c), O(s,c,s)
                        def f(x,y,z): return f1(x,y,z)-f2(x,y,z)
                    elif (h+k)%2==1 and h==l+1 and k==l:
                        f1,f2 = E(c,s,c), O(s,s,c)
                        def f(x,y,z): return f1(x,y,z)-f2(x,y,z)
                    elif (h+k)%2==1 and h==l and k==l+1:
                        f1,f2 = E(c,c,s), O(c,s,s)
                        def f(x,y,z): return f1(x,y,z)-f2(x,y,z)
                elif hklmod4 == 2:
                    if (h+k)%2==0 and h==l and k==l:
                        f1,f2 = E(s,s,s), O(s,s,s)
                        def f(x,y,z): return f1(x,y,z)+f2(x,y,z)
                    elif (h+k)%2==0 and h+1==l and k==l:
                        f1,f2 = E(s,c,c), O(c,s,c)
                        def f(x,y,z): return f1(x,y,z)+f2(x,y,z)
                    elif (h+k)%2==1 and h==k+1 and h==l:
                        f1,f2 = E(c,c,s), O(s,c,c)
                        def f(x,y,z): return f1(x,y,z)+f2(x,y,z)
                    elif (h+k)%2==1 and h==k and h==l+1:
                        f1,f2 = E(s,c,c), O(c,s,c)
                        def f(x,y,z): return f1(x,y,z)+f2(x,y,z)
                elif hklmod4 == 3:
                    if (h+k)%2==0 and h==l and k==l:
                        f1,f2 = E(s,s,s), O(c,c,c)
                        def f(x,y,z): return f1(x,y,z)+f2(x,y,z)
                    elif (h+k)%2==0 and h+1==l and k==h:
                        f1,f2 = E(s,c,c), O(s,c,s)
                        def f(x,y,z): return f1(x,y,z)+f2(x,y,z)
                    elif (h+k)%2==1 and h==l+1 and k==l:
                        f1,f2 = E(c,s,c), O(s,s,c)
                        def f(x,y,z): return f1(x,y,z)+f2(x,y,z)
                    elif (h+k)%2==1 and h==l and k==l+1:
                        f1,f2 = E(c,c,s), O(c,s,s)
                        def f(x,y,z): return f1(x,y,z)+f2(x,y,z)

        elif group == 213:
            hklmod4 = (h + k + l) % 4
            if AB == 0:
                if hklmod4 == 0:
                    if h%2==0 and h==l and k==l:
                        f1,f2 = E(c,c,c), O(c,c,c)
                        def f(x,y,z): return f1(x,y,z)+f2(x,y,z)
                    elif h%2==0 and h+1==k and k==l:
                        f1,f2 = E(s,c,s), O(s,s,c)
                        def f(x,y,z): return f1(x,y,z)+f2(x,y,z)
                    elif h%2==1 and h==k+1 and h==l:
                        f1,f2 = E(s,s,c), O(c,s,s)
                        def f(x,y,z): return f1(x,y,z)+f2(x,y,z)
                    elif h%2==1 and h==k and h==l+1:
                        f1,f2 = E(c,s,s), O(s,c,s)
                        def f(x,y,z): return f1(x,y,z)+f2(x,y,z)
                elif hklmod4 == 1:
                    if h%2==1 and h==l and k==l:
                        f1,f2 = E(c,c,c), O(s,s,s)
                        def f(x,y,z): return f1(x,y,z)+f2(x,y,z)
                    elif h%2==0 and h==k and l==h+1:
                        f1,f2 = E(c,s,s), O(c,s,c)
                        def f(x,y,z): return f1(x,y,z)+f2(x,y,z)
                    elif h%2==1 and h==k+1 and k==l:
                        f1,f2 = E(s,c,s), O(c,c,s)
                        def f(x,y,z): return f1(x,y,z)+f2(x,y,z)
                    elif h%2==0 and h+1==k and h==l:
                        f1,f2 = E(s,s,c), O(s,c,c)
                        def f(x,y,z): return f1(x,y,z)+f2(x,y,z)
                elif hklmod4 == 2:
                    if h%2==0 and h==l and k==l:
                        f1,f2 = E(c,c,c), O(c,c,c)
                        def f(x,y,z): return f1(x,y,z)-f2(x,y,z)
                    elif h%2==0 and h+1==k and k==l:
                        f1,f2 = E(s,c,s), O(s,s,c)
                        def f(x,y,z): return f1(x,y,z)-f2(x,y,z)
                    elif h%2==1 and h==k+1 and h==l:
                        f1,f2 = E(s,s,c), O(c,s,s)
                        def f(x,y,z): return f1(x,y,z)-f2(x,y,z)
                    elif h%2==1 and h==k and h==l+1:
                        f1,f2 = E(c,s,s), O(s,c,s)
                        def f(x,y,z): return f1(x,y,z)-f2(x,y,z)
                elif hklmod4 == 3:
                    if h%2==1 and h==l and k==l:
                        f1,f2 = E(c,c,c), O(s,s,s)
                        def f(x,y,z): return f1(x,y,z)-f2(x,y,z)
                    elif h%2==0 and h==k and h+1==l:
                        f1,f2 = E(c,s,s), O(c,s,c)
                        def f(x,y,z): return f1(x,y,z)-f2(x,y,z)
                    elif h%2==1 and h==k+1 and k==l:
                        f1,f2 = E(s,c,s), O(c,c,s)
                        def f(x,y,z): return f1(x,y,z)-f2(x,y,z)
                    elif h%2==0 and h+1==k and h==l:
                        f1,f2 = E(s,s,c), O(s,c,c)
                        def f(x,y,z): return f1(x,y,z)-f2(x,y,z)
            else:
                if hklmod4 == 0:
                    if h%2==0 and h==l and k==l:
                        f1,f2 = E(s,s,s), O(s,s,s)
                        def f(x,y,z): return f1(x,y,z)-f2(x,y,z)
                    elif h%2==0 and h+1==k and k==l:
                        f1,f2 = E(c,s,c), O(c,c,s)
                        def f(x,y,z): return f1(x,y,z)-f2(x,y,z)
                    elif h%2==1 and h==k+1 and h==l:
                        f1,f2 = E(c,c,s), O(s,c,c)
                        def f(x,y,z): return f1(x,y,z)-f2(x,y,z)
                    elif h%2==1 and h==k and h==l+1:
                        f1,f2 = E(s,c,c), O(c,s,c)
                        def f(x,y,z): return f1(x,y,z)-f2(x,y,z)
                elif hklmod4 == 1:
                    if h%2==1 and h==l and k==l:
                        f1,f2 = E(s,s,s), O(c,c,c)
                        def f(x,y,z): return f1(x,y,z)+f2(x,y,z)
                    elif h%2==0 and h==k and h+1==l:
                        f1,f2 = E(s,c,c), O(s,c,s)
                        def f(x,y,z): return f1(x,y,z)+f2(x,y,z)
                    elif h%2==1 and h==k+1 and k==l:
                        f1,f2 = E(c,s,c), O(s,s,c)
                        def f(x,y,z): return f1(x,y,z)+f2(x,y,z)
                    elif h%2==0 and h+1==k and h==l:
                        f1,f2 = E(c,c,s), O(c,s,s)
                        def f(x,y,z): return f1(x,y,z)+f2(x,y,z)
                elif hklmod4 == 2:
                    if h%2==0 and h==l and k==l:
                        f1,f2 = E(s,s,s), O(s,s,s)
                        def f(x,y,z): return f1(x,y,z)+f2(x,y,z)
                    elif h%2==0 and h+1==k and k==l:
                        f1,f2 = E(c,s,c), O(c,c,s)
                        def f(x,y,z): return f1(x,y,z)+f2(x,y,z)
                    elif h%2==1 and h==k+1 and h==l:
                        f1,f2 = E(c,c,s), O(s,c,c)
                        def f(x,y,z): return f1(x,y,z)+f2(x,y,z)
                    elif h%2==1 and h==k and h==l+1:
                        f1,f2 = E(s,c,c), O(c,s,c)
                        def f(x,y,z): return f1(x,y,z)+f2(x,y,z)
                elif hklmod4 == 3:
                    if h%2==1 and h==l and k==l:
                        f1,f2 = E(s,s,s), O(c,c,c)
                        def f(x,y,z): return f1(x,y,z)-f2(x,y,z)
                    elif h%2==0 and h==k and h+1==l:
                        f1,f2 = E(s,c,c), O(s,c,s)
                        def f(x,y,z): return f1(x,y,z)-f2(x,y,z)
                    elif h%2==1 and h==k+1 and k==l:
                        f1,f2 = E(c,s,c), O(s,s,c)
                        def f(x,y,z): return f1(x,y,z)-f2(x,y,z)
                    elif h%2==0 and h+1==k and h==l:
                        f1,f2 = E(c,c,s), O(c,s,s)
                        def f(x,y,z): return f1(x,y,z)-f2(x,y,z)

        elif group == 214:
            hklMod4 = (h + k + l) % 4
            if AB == 0:
                if hklMod4 == 0:
                    if h%2==0 and h==l and k==l:
                        f1,f2 = E(c,c,c), O(c,c,c)
                        def f(x,y,z): return f1(x,y,z)+f2(x,y,z)
                    elif h%2==0 and h+1==k and k==l:
                        f1,f2 = E(s,c,s), O(s,s,c)
                        def f(x,y,z): return f1(x,y,z)+f2(x,y,z)
                    elif h%2==1 and h==k+1 and h==l:
                        f1,f2 = E(s,s,c), O(c,s,s)
                        def f(x,y,z): return f1(x,y,z)+f2(x,y,z)
                    elif h%2==1 and h==k and h==l+1:
                        f1,f2 = E(c,s,s), O(s,c,s)
                        def f(x,y,z): return f1(x,y,z)+f2(x,y,z)
                elif hklMod4 == 2:
                    if h%2==0 and h==l and k==l:
                        f1,f2 = E(c,c,c), O(c,c,c)
                        def f(x,y,z): return f1(x,y,z)-f2(x,y,z)
                    elif h%2==0 and h+1==k and k==l:
                        f1,f2 = E(s,c,s), O(s,s,c)
                        def f(x,y,z): return f1(x,y,z)-f2(x,y,z)
                    elif h%2==1 and h==k+1 and h==l:
                        f1,f2 = E(s,s,c), O(c,s,s)
                        def f(x,y,z): return f1(x,y,z)-f2(x,y,z)
                    elif h%2==1 and h==k and h==l+1:
                        f1,f2 = E(c,s,s), O(s,c,s)
                        def f(x,y,z): return f1(x,y,z)-f2(x,y,z)
            else:
                if hklMod4 == 0:
                    if h%2==0 and h==l and k==l:
                        f1,f2 = E(s,s,s), O(s,s,s)
                        def f(x,y,z): return f1(x,y,z)-f2(x,y,z)
                    elif h%2==0 and h+1==k and k==l:
                        f1,f2 = E(c,s,c), O(c,c,s)
                        def f(x,y,z): return f1(x,y,z)-f2(x,y,z)
                    elif h%2==1 and h==k+1 and h==l:
                        f1,f2 = E(c,c,s), O(s,c,c)
                        def f(x,y,z): return f1(x,y,z)-f2(x,y,z)
                    elif h%2==1 and h==k and h==l+1:
                        f1,f2 = E(s,c,c), O(c,s,c)
                        def f(x,y,z): return f1(x,y,z)-f2(x,y,z)
                elif hklMod4 == 2:
                    if h%2==0 and h==l and k==l:
                        f1,f2 = E(s,s,s), O(s,s,s)
                        def f(x,y,z): return f1(x,y,z)+f2(x,y,z)
                    elif h%2==0 and h+1==k and k==l:
                        f1,f2 = E(c,s,c), O(c,c,s)
                        def f(x,y,z): return f1(x,y,z)+f2(x,y,z)
                    elif h%2==1 and h==k+1 and h==l:
                        f1,f2 = E(c,c,s), O(s,c,c)
                        def f(x,y,z): return f1(x,y,z)+f2(x,y,z)
                    elif h%2==1 and h==k and h==l+1:
                        f1,f2 = E(s,c,c), O(c,s,c)
                        def f(x,y,z): return f1(x,y,z)+f2(x,y,z)

        elif group in {215, 216, 217}:
            if AB == 0:
                f1,f2 = E(c,c,c), O(c,c,c)
                def f(x,y,z): return f1(x,y,z) + f2(x,y,z)
            else:
                f1,f2 = E(s,s,s), O(s,s,s)
                def f(x,y,z): return f1(x,y,z) + f2(x,y,z)

        elif group in {218, 219}:
            hklMod2 = (h + k + l) % 2
            if AB == 0:
                if hklMod2 == 0:
                    f1,f2 = E(c,c,c), O(c,c,c)
                    def f(x,y,z): return f1(x,y,z) + f2(x,y,z)
                else:
                    f1,f2 = E(c,c,c), O(c,c,c)
                    def f(x,y,z): return f1(x,y,z) - f2(x,y,z)
            else:
                if hklMod2 == 0:
                    f1,f2 = E(s,s,s), O(s,s,s)
                    def f(x,y,z): return f1(x,y,z) + f2(x,y,z)
                else:
                    f1,f2 = E(s,s,s), O(s,s,s)
                    def f(x,y,z): return f1(x,y,z) - f2(x,y,z)

        elif group == 220:
            hklMod4 = (h + k + l) % 4
            if AB == 0:
                if hklMod4 == 0:
                    if h%2==0 and h==l and k==l:
                        f1,f2 = E(c,c,c), O(c,c,c)
                        def f(x,y,z): return f1(x,y,z)+f2(x,y,z)
                    elif h%2==0 and h+1==k and k==l:
                        f1,f2 = E(s,c,s), O(s,s,c)
                        def f(x,y,z): return f1(x,y,z)+f2(x,y,z)
                    elif h%2==1 and h==k+1 and h==l:
                        f1,f2 = E(s,s,c), O(c,s,s)
                        def f(x,y,z): return f1(x,y,z)+f2(x,y,z)
                    elif h%2==1 and h==k and h==l+1:
                        f1,f2 = E(c,s,s), O(s,c,s)
                        def f(x,y,z): return f1(x,y,z)+f2(x,y,z)
                elif hklMod4 == 2:
                    if h%2==0 and h==l and k==l:
                        f1,f2 = E(c,c,c), O(c,c,c)
                        def f(x,y,z): return f1(x,y,z)+f2(x,y,z)
                    elif h%2==0 and h+1==k and k==l:
                        f1,f2 = E(s,c,s), O(s,s,c)
                        def f(x,y,z): return f1(x,y,z)+f2(x,y,z)
                    elif h%2==1 and h==k+1 and h==l:
                        f1,f2 = E(s,s,c), O(c,s,s)
                        def f(x,y,z): return f1(x,y,z)+f2(x,y,z)
                    elif h%2==1 and h==k and h==l+1:
                        f1,f2 = E(c,s,s), O(s,c,s)
                        def f(x,y,z): return f1(x,y,z)+f2(x,y,z)
            else:
                if hklMod4 == 0:
                    if h%2==0 and h==l and k==l:
                        f1,f2 = E(s,s,s), O(s,s,s)
                        def f(x,y,z): return f1(x,y,z)-f2(x,y,z)
                    elif h%2==0 and h+1==k and k==l:
                        f1,f2 = E(c,s,c), O(c,c,s)
                        def f(x,y,z): return f1(x,y,z)-f2(x,y,z)
                    elif h%2==1 and h==k+1 and h==l:
                        f1,f2 = E(c,c,s), O(s,c,c)
                        def f(x,y,z): return f1(x,y,z)-f2(x,y,z)
                    elif h%2==1 and h==k and h==l+1:
                        f1,f2 = E(s,c,c), O(c,s,c)
                        def f(x,y,z): return f1(x,y,z)-f2(x,y,z)
                elif hklMod4 == 2:
                    if h%2==0 and h==l and k==l:
                        f1,f2 = E(s,s,s), O(s,s,s)
                        def f(x,y,z): return f1(x,y,z)-f2(x,y,z)
                    elif h%2==0 and h+1==k and k==l:
                        f1,f2 = E(c,s,c), O(c,c,s)
                        def f(x,y,z): return f1(x,y,z)-f2(x,y,z)
                    elif h%2==1 and h==k+1 and h==l:
                        f1,f2 = E(c,c,s), O(s,c,c)
                        def f(x,y,z): return f1(x,y,z)-f2(x,y,z)
                    elif h%2==1 and h==k and h==l+1:
                        f1,f2 = E(s,c,c), O(c,s,c)
                        def f(x,y,z): return f1(x,y,z)-f2(x,y,z)

        elif group in {221, 225, 229}:
            f1,f2 = E(c,c,c), O(c,c,c)
            def f(x,y,z): return f1(x,y,z) + f2(x,y,z)

        elif group == 222:
            if origin == 1:
                if (h + k + l) % 2 == 0:
                    f1,f2 = E(c,c,c), O(c,c,c)
                    def f(x,y,z): return f1(x,y,z) + f2(x,y,z)
                else:
                    f1,f2 = E(s,s,s), O(s,s,s)
                    def f(x,y,z): return f1(x,y,z) - f2(x,y,z)
            else:
                if h%2==0 and h==l and k==l:
                    f1,f2 = E(c,c,c), O(c,c,c)
                    def f(x,y,z): return f1(x,y,z)+f2(x,y,z)
                elif h%2==0 and h+1==k and k==l:
                    f1,f2 = E(c,s,s), O(c,s,s)
                    def f(x,y,z): return f1(x,y,z)+f2(x,y,z)
                elif h%2==1 and h==k+1 and h==l:
                    f1,f2 = E(s,c,s), O(s,c,s)
                    def f(x,y,z): return f1(x,y,z)+f2(x,y,z)
                elif h%2==1 and h==k and h==l+1:
                    f1,f2 = E(s,s,c), O(s,s,c)
                    def f(x,y,z): return f1(x,y,z)+f2(x,y,z)
                elif h%2==1 and h==k and k==l:
                    f1,f2 = E(c,c,c), O(c,c,c)
                    def f(x,y,z): return f1(x,y,z)-f2(x,y,z)
                elif h%2==1 and h==k+1 and k==l:
                    f1,f2 = E(c,s,s), O(c,s,s)
                    def f(x,y,z): return f1(x,y,z)-f2(x,y,z)
                elif h%2==0 and h+1==k and h==l:
                    f1,f2 = E(s,c,s), O(s,c,s)
                    def f(x,y,z): return f1(x,y,z)-f2(x,y,z)
                elif h%2==0 and h==k and h+1==l:
                    f1,f2 = E(s,s,c), O(s,s,c)
                    def f(x,y,z): return f1(x,y,z)-f2(x,y,z)

        elif group == 223:
            if (h + k + l) % 2 == 0:
                f1,f2 = E(c,c,c), O(c,c,c)
                def f(x,y,z): return f1(x,y,z) + f2(x,y,z)
            else:
                f1,f2 = E(c,c,c), O(c,c,c)
                def f(x,y,z): return f1(x,y,z) - f2(x,y,z)

        elif group == 224:
            if origin == 1:
                if (h + k + l) % 2 == 0:
                    f1,f2 = E(c,c,c), O(c,c,c)
                    def f(x,y,z): return f1(x,y,z) + f2(x,y,z)
                else:
                    f1,f2 = E(s,s,s), O(s,s,s)
                    def f(x,y,z): return f1(x,y,z) + f2(x,y,z)
            else:
                if (h+k)%2==0 and h==l and k==l:
                    f1,f2 = E(c,c,c), O(c,c,c)
                    def f(x,y,z): return f1(x,y,z)+f2(x,y,z)
                elif (h+k)%2==0 and h+1==l and k==h:
                    f1,f2 = E(s,s,c), O(s,s,c)
                    def f(x,y,z): return f1(x,y,z)+f2(x,y,z)
                elif (h+k)%2==1 and h==l+1 and k==l:
                    f1,f2 = E(c,s,s), O(c,s,s)
                    def f(x,y,z): return f1(x,y,z)+f2(x,y,z)
                elif (h+k)%2==1 and h==l and k==l+1:
                    f1,f2 = E(s,c,s), O(s,c,s)
                    def f(x,y,z): return f1(x,y,z)+f2(x,y,z)

        elif group == 226:
            if (h + k + l) % 2 == 0:
                f1,f2 = E(c,c,c), O(c,c,c)
                def f(x,y,z): return f1(x,y,z) + f2(x,y,z)
            else:
                f1,f2 = E(c,c,c), O(c,c,c)
                def f(x,y,z): return f1(x,y,z) - f2(x,y,z)

        elif group == 227:
            if origin == 1:
                mod_val = (h + k + l) % 4
                if mod_val == 0:
                    f1,f2 = E(c,c,c), O(c,c,c)
                    def f(x,y,z): return f1(x,y,z) + f2(x,y,z)
                elif mod_val == 1:
                    f1=E(c,c,c); f2=E(s,s,s); f3=O(c,c,c); f4=O(s,s,s)
                    def f(x,y,z): return f1(x,y,z)-f2(x,y,z)+f3(x,y,z)-f4(x,y,z)
                elif mod_val == 2:
                    f1,f2 = E(s,s,s), O(s,s,s)
                    def f(x,y,z): return f1(x,y,z) + f2(x,y,z)
                elif mod_val == 3:
                    f1=E(c,c,c); f2=E(s,s,s); f3=O(c,c,c); f4=O(s,s,s)
                    def f(x,y,z): return f1(x,y,z)+f2(x,y,z)+f3(x,y,z)+f4(x,y,z)
            else:
                mod_hk = (h + k) % 4
                if mod_hk==0 and h==l and k==l:
                    f1,f2 = E(c,c,c), O(c,c,c)
                    def f(x,y,z): return f1(x,y,z)+f2(x,y,z)
                elif mod_hk==0 and h+2==l and k==h:
                    f1,f2 = E(s,s,c), O(s,s,c)
                    def f(x,y,z): return f1(x,y,z)+f2(x,y,z)
                elif mod_hk==2 and h==l+2 and k==l:
                    f1,f2 = E(c,s,s), O(c,s,s)
                    def f(x,y,z): return f1(x,y,z)+f2(x,y,z)
                elif mod_hk==2 and h==l and k==l+2:
                    f1,f2 = E(s,c,s), O(s,c,s)
                    def f(x,y,z): return f1(x,y,z)+f2(x,y,z)
                elif mod_hk==2 and h==l and k==l:
                    f1=E(c,c,c); f2=E(c,s,s); f3=O(s,c,s); f4=O(s,s,c)
                    f5=O(c,c,c); f6=O(c,s,s); f7=O(s,c,s); f8=O(s,s,c)
                    def f(x,y,z):
                        return (f1(x,y,z)+f2(x,y,z)+f3(x,y,z)+f4(x,y,z)
                                +f5(x,y,z)+f6(x,y,z)+f7(x,y,z)+f8(x,y,z))
                elif mod_hk==2 and h==l+2 and k==h:
                    f1=E(c,c,c); f2=E(c,s,s); f3=O(s,c,s); f4=O(s,s,c)
                    f5=O(c,c,c); f6=O(c,s,s); f7=O(s,c,s); f8=O(s,s,c)
                    def f(x,y,z):
                        return (f1(x,y,z)-f2(x,y,z)-f3(x,y,z)+f4(x,y,z)
                                +f5(x,y,z)-f6(x,y,z)-f7(x,y,z)+f8(x,y,z))
                elif mod_hk==0 and h+2==l and k==l:
                    f1=E(c,c,c); f2=E(c,s,s); f3=O(s,c,s); f4=O(s,s,c)
                    f5=O(c,c,c); f6=O(c,s,s); f7=O(s,c,s); f8=O(s,s,c)
                    def f(x,y,z):
                        return (f1(x,y,z)+f2(x,y,z)-f3(x,y,z)-f4(x,y,z)
                                +f5(x,y,z)+f6(x,y,z)-f7(x,y,z)-f8(x,y,z))
                elif mod_hk==0 and k+2==l and h==l:
                    f1=E(c,c,c); f2=E(c,s,s); f3=O(s,c,s); f4=O(s,s,c)
                    f5=O(c,c,c); f6=O(c,s,s); f7=O(s,c,s); f8=O(s,s,c)
                    def f(x,y,z):
                        return (f1(x,y,z)-f2(x,y,z)+f3(x,y,z)-f4(x,y,z)
                                +f5(x,y,z)-f6(x,y,z)+f7(x,y,z)-f8(x,y,z))

        elif group == 228:
            if origin == 1:
                mod_val = (h + k + l) % 4
                if mod_val == 0:
                    f1,f2 = E(c,c,c), O(c,c,c)
                    def f(x,y,z): return f1(x,y,z) + f2(x,y,z)
                elif mod_val == 1:
                    f1=E(c,c,c); f2=E(s,s,s); f3=O(c,c,c); f4=O(s,s,s)
                    def f(x,y,z): return f1(x,y,z)+f2(x,y,z)-f3(x,y,z)-f4(x,y,z)
                elif mod_val == 2:
                    f1,f2 = E(s,s,s), O(s,s,s)
                    def f(x,y,z): return f1(x,y,z) + f2(x,y,z)
                elif mod_val == 3:
                    f1=E(c,c,c); f2=E(s,s,s); f3=O(c,c,c); f4=O(s,s,s)
                    def f(x,y,z): return f1(x,y,z)-f2(x,y,z)-f3(x,y,z)+f4(x,y,z)
            else:
                mod_hk = (h + k) % 4
                if mod_hk==0 and h==l and k==l:
                    f1,f2 = E(c,c,c), O(c,c,c)
                    def f(x,y,z): return f1(x,y,z)+f2(x,y,z)
                elif mod_hk==0 and h+2==l and k==h:
                    f1,f2 = E(s,s,c), O(s,s,c)
                    def f(x,y,z): return f1(x,y,z)+f2(x,y,z)
                elif mod_hk==2 and h==l+2 and k==l:
                    f1,f2 = E(c,s,s), O(c,s,s)
                    def f(x,y,z): return f1(x,y,z)+f2(x,y,z)
                elif mod_hk==2 and h==l and k==l+2:
                    f1,f2 = E(s,c,s), O(s,c,s)
                    def f(x,y,z): return f1(x,y,z)+f2(x,y,z)
                elif mod_hk==2 and h==l and k==l:
                    f1=E(c,c,c); f2=E(c,s,s); f3=O(s,c,s); f4=O(s,s,c)
                    f5=O(c,c,c); f6=O(c,s,s); f7=O(s,c,s); f8=O(s,s,c)
                    def f(x,y,z):
                        return (f1(x,y,z)+f2(x,y,z)+f3(x,y,z)+f4(x,y,z)
                                -f5(x,y,z)-f6(x,y,z)-f7(x,y,z)-f8(x,y,z))
                elif mod_hk==2 and h==l+2 and k==h:
                    f1=E(c,c,c); f2=E(c,s,s); f3=O(s,c,s); f4=O(s,s,c)
                    f5=O(c,c,c); f6=O(c,s,s); f7=O(s,c,s); f8=O(s,s,c)
                    def f(x,y,z):
                        return (f1(x,y,z)-f2(x,y,z)-f3(x,y,z)+f4(x,y,z)
                                -f5(x,y,z)+f6(x,y,z)+f7(x,y,z)-f8(x,y,z))
                elif mod_hk==0 and h+2==l and k==l:
                    f1=E(c,c,c); f2=E(c,s,s); f3=O(s,c,s); f4=O(s,s,c)
                    f5=O(c,c,c); f6=O(c,s,s); f7=O(s,c,s); f8=O(s,s,c)
                    def f(x,y,z):
                        return (f1(x,y,z)+f2(x,y,z)-f3(x,y,z)-f4(x,y,z)
                                -f5(x,y,z)-f6(x,y,z)+f7(x,y,z)+f8(x,y,z))
                elif mod_hk==0 and k+2==l and h==l:
                    f1=E(c,c,c); f2=E(c,s,s); f3=O(s,c,s); f4=O(s,s,c)
                    f5=O(c,c,c); f6=O(c,s,s); f7=O(s,c,s); f8=O(s,s,c)
                    def f(x,y,z):
                        return (f1(x,y,z)-f2(x,y,z)+f3(x,y,z)-f4(x,y,z)
                                -f5(x,y,z)+f6(x,y,z)-f7(x,y,z)+f8(x,y,z))

        elif group == 230:
            mod_val = (h + k + l) % 4
            if mod_val == 0:
                if h%2==0 and h==l and k==l:
                    f1,f2 = E(c,c,c), O(c,c,c)
                    def f(x,y,z): return f1(x,y,z)+f2(x,y,z)
                elif h%2==0 and h+1==k and k==l:
                    f1,f2 = E(s,c,s), O(s,s,c)
                    def f(x,y,z): return f1(x,y,z)+f2(x,y,z)
                elif h%2==1 and h==k+1 and h==l:
                    f1,f2 = E(s,s,c), O(c,s,s)
                    def f(x,y,z): return f1(x,y,z)+f2(x,y,z)
                elif h%2==1 and h==k and h==l+1:
                    f1,f2 = E(c,s,s), O(s,c,s)
                    def f(x,y,z): return f1(x,y,z)+f2(x,y,z)
            elif mod_val == 2:
                if h%2==0 and h==l and k==l:
                    f1,f2 = E(c,c,c), O(c,c,c)
                    def f(x,y,z): return f1(x,y,z)-f2(x,y,z)
                elif h%2==0 and h+1==k and k==l:
                    f1,f2 = E(s,c,s), O(s,s,c)
                    def f(x,y,z): return f1(x,y,z)-f2(x,y,z)
                elif h%2==1 and h==k+1 and h==l:
                    f1,f2 = E(s,s,c), O(c,s,s)
                    def f(x,y,z): return f1(x,y,z)-f2(x,y,z)
                elif h%2==1 and h==k and h==l+1:
                    f1,f2 = E(c,s,s), O(s,c,s)
                    def f(x,y,z): return f1(x,y,z)-f2(x,y,z)

        else:
            if verbose:
                print(f'    *** invalid space group No. {group} ***')
            res['exitflag'] = 1
            return results

        if f is None:
            if verbose:
                print(f'    *** No. {group} ({h}{k}{l}) not allowed ***')
            res['exitflag'] = 2
        else:
            res['f'] = f
            res['exitflag'] = 0

    return results


# ==============================================================================
# 2. 生成水平集场网格（对应 getLSField2.m）
# ==============================================================================
def getLSField2(f, t, lsType, resPhi, normFlag=0, tile=None, L=1.0):
    """
    生成离散化的水平集场体素网格
    f <= 0 对应固体区域

    参数:
        f: callable，水平集函数句柄
        t: float，等值面偏移量
        lsType: str，水平集类型：'tm' / 'tp' / 'tt'
        resPhi: int，单胞体素分辨率
        normFlag: int，是否归一化场值（1=归一化）
        tile: list[int]，三个方向单胞重复次数
        L: float，单胞边长
    返回:
        field: (Nx, Ny, Nz) numpy数组
    """
    if tile is None:
        tile = [1, 1, 1]
    tile = np.atleast_1d(tile).flatten()
    if len(tile) == 1:
        tile = np.repeat(tile, 3)

    nx = int(tile[0] * resPhi)
    ny = int(tile[1] * resPhi)
    nz = int(tile[2] * resPhi)

    x = np.linspace(0, 2 * np.pi * tile[0] / L, nx)
    y = np.linspace(0, 2 * np.pi * tile[1] / L, ny)
    z = np.linspace(0, 2 * np.pi * tile[2] / L, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='xy')

    field = f(X, Y, Z)

    if normFlag:
        field = field / np.max(np.abs(field))

    if lsType == 'tm':
        field = field - t
    elif lsType == 'tp':
        field = t - field
    elif lsType == 'tt':
        field = field**2 - t**2
    else:
        raise ValueError(f"未知的水平集类型: {lsType}")

    return field


# ==============================================================================
# 3. 等值面提取辅助（对应 phi2fv.m / isosurface）
# ==============================================================================
def phi2fv(phi, level=0.0):
    """
    从水平集场提取三角网格等值面
    对应 MATLAB 的 isosurface / phi2fv
    """
    verts, faces, _, _ = marching_cubes(phi, level=level, allow_degenerate=False)
    return {'vertices': verts, 'faces': faces}


# ==============================================================================
# 4. 等值面可视化（对应 plotIsosurface.m）
# ==============================================================================
def plotIsosurface(geometry, ax=None, show=True, facecolor='red'):
    """
    绘制3D等值面，支持三种输入：
    1. 3D bool数组（体素）
    2. 3D float数组（水平集场）
    3. dict: {'vertices': (N,3), 'faces': (M,3)}

    返回:
        fv: 顶点+面字典
    """
    if isinstance(geometry, dict) and 'vertices' in geometry and 'faces' in geometry:
        fv = geometry
    elif isinstance(geometry, np.ndarray):
        if geometry.dtype == bool:
            padded = np.pad(geometry, pad_width=1, mode='constant', constant_values=False)
            fv = phi2fv(padded.astype(float), level=0.1)
        else:
            fv = phi2fv(geometry, level=0.0)
    else:
        raise TypeError("不支持的几何输入类型")

    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

    mesh = Poly3DCollection(fv['vertices'][fv['faces']])
    mesh.set_facecolor(facecolor)
    mesh.set_edgecolor('none')
    mesh.set_shade(True)
    ax.add_collection3d(mesh)

    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('y', fontsize=14)
    ax.set_zlabel('z', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.view_init(elev=30, azim=45)
    ax.set_aspect('equal')

    verts = fv['vertices']
    ax.set_xlim(verts[:, 0].min(), verts[:, 0].max())
    ax.set_ylim(verts[:, 1].min(), verts[:, 1].max())
    ax.set_zlim(verts[:, 2].min(), verts[:, 2].max())

    if show:
        plt.tight_layout()
        plt.show()

    return fv


# ==============================================================================
# 5. 批量生成所有结构（对应主生成脚本）
# ==============================================================================
def generate_all_structures(
    key_path='familyNameKey.csv',
    res=64,
    skipid_1based=None,
    verbose=1
):
    """
    批量生成所有空间群结构的水平集场（对应原MATLAB主脚本）
    """
    key_df = pd.read_csv(key_path, header=None)
    key_names = key_df.iloc[1:, 1].tolist()
    n = len(key_names)

    if skipid_1based is None:
        skipid_1based = [112,130,157,189,200,211,236,270,283,302]
    skipid = [x-1 for x in skipid_1based]

    structures = []
    for ii in range(n):
        if ii in skipid:
            continue
        kk = key_names[ii]
        if verbose:
            print(kk)
        ss = kk.split('-')
        spaceGroup = int(ss[0])
        origin = int(ss[1])
        AB = int(ss[2])
        HKL = [int(c) for c in ss[3]]
        lsType = ss[-1]

        res_list = structureFactors(spaceGroup, hkl=[HKL], AB=AB, origin=origin, verbose=verbose)
        f = res_list[0]['f']
        if f is None:
            continue

        levelSetField = getLSField2(f, t=0, lsType=lsType, resPhi=res, normFlag=0)
        structures.append({
            'spaceGroup': spaceGroup,
            'origin': origin,
            'AB': AB,
            'HKL': HKL,
            'lsType': lsType,
            'levelSetField': levelSetField
        })
    return structures


# ==============================================================================
# 6. DPP多样性子集生成（对应DPP脚本）
# ==============================================================================
def generate_dpp_subset(
    dpp_path='final_dpp_sets_latent_idx_python.txt',
    key_path='familyNameKey.csv',
    res=42,
    row_idx=0,
    verbose=1
):
    """
    生成DPP筛选后的多样性子集
    row_idx=0: 纯形状多样性
    """
    dpp_ids = np.loadtxt(dpp_path, dtype=int, delimiter=',')
    if dpp_ids.ndim == 1:
        dpp_ids = dpp_ids.reshape(1, -1)
    ids = dpp_ids[row_idx, :]

    key_df = pd.read_csv(key_path, header=None)
    key_names = key_df.iloc[1:, 1].tolist()

    dpp_structures = []
    for idx in ids:
        kk = key_names[idx]
        if verbose:
            print(kk)
        ss = kk.split('-')
        spaceGroup = int(ss[0])
        origin = int(ss[1])
        AB = int(ss[2])
        HKL = [int(c) for c in ss[3]]
        lsType = ss[-1]

        res_list = structureFactors(spaceGroup, hkl=[HKL], AB=AB, origin=origin, verbose=verbose)
        f = res_list[0]['f']
        if f is None:
            continue

        levelSetField = getLSField2(f, t=0, lsType=lsType, resPhi=res, normFlag=0)
        dpp_structures.append({
            'spaceGroup': spaceGroup,
            'origin': origin,
            'AB': AB,
            'HKL': HKL,
            'lsType': lsType,
            'levelSetField': levelSetField
        })
    return dpp_structures


# if __name__ == '__main__':
#     #单个结构生成+可视化
#     print("=== 单个结构示例 ===")
#     sf_result = structureFactors(220, hkl=[[1,1,1]], AB=0, origin=1)
#     f_handle = sf_result[0]['f']
#     field = getLSField2(f_handle, t=0, lsType='tm', resPhi=64, normFlag=0)
#     print(f"体素形状: {field.shape}")
#     print(f"体积分数: {np.mean(field <= 0):.4f}")
#     plotIsosurface(field)

#     # 生成DPP子集
#     # dpp_structs = generate_dpp_subset(
#     #     dpp_path='final_dpp_sets_latent_idx_python.txt',
#     #     key_path='familyNameKey.csv',
#     #     res=42,
#     #     row_idx=0
#     # )
#     # print(f"生成DPP子集数量: {len(dpp_structs)}")

def generate_metaset_dataset(
    key_path="familyNameKey.csv",
    mat_path="sfs_012_noperm_final_flat_densfilt.mat",
    save_dir="./metaset_64_npy",
    res=64,
    n_samples_per_class=100,
    use_official_trange=True
):
    """
    【修正版】与原MATLAB METASET 1:1对齐的数据集生成
    - 默认关闭场归一化（normFlag=0）
    - 优先使用官方标定的tRange
    - 采样方式与原代码一致（去掉首尾端点）
    """
    import scipy.io as sio
    os.makedirs(save_dir, exist_ok=True)

    # 读取类别映射表
    key_df = pd.read_csv(key_path, header=None)
    key_names = key_df.iloc[1:, 1].tolist()
    n_total_classes = len(key_names)

    # 原MATLAB跳过的类别ID（1-based转0-based）
    skipid_1based = [112,130,157,189,200,211,236,270,283,302]
    skipid = [x-1 for x in skipid_1based]
    valid_class_ids = [i for i in range(n_total_classes) if i not in skipid]

    # 加载官方tRange（如果提供了mat文件）
    sf_functions = None
    if use_official_trange and os.path.exists(mat_path):
        sf_data = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        sf_functions = sf_data["sfFunctions"]
        print("已加载官方标定的tRange")

    all_voxels = []
    all_conditions = []

    for cls_idx in tqdm(valid_class_ids, desc="生成结构"):
        kk = key_names[cls_idx]
        ss = kk.split('-')
        spaceGroup = int(ss[0])
        origin = int(ss[1])
        AB = int(ss[2])
        HKL = [int(c) for c in ss[3]]
        lsType = ss[-1]

        # 获取水平集函数
        sf_res = structureFactors(spaceGroup, hkl=[HKL], AB=AB, origin=origin, verbose=0)
        f_handle = sf_res[0]['f']
        if f_handle is None:
            continue

        # ========== 关键修正1：关闭归一化，使用原始场值 ==========
        field_base = getLSField2(f_handle, t=0, lsType=lsType, resPhi=res, normFlag=0)
        f_min, f_max = float(field_base.min()), float(field_base.max())

        # ========== 关键修正2：使用官方tRange或原始场极值 ==========
        if sf_functions is not None:
            # 官方1-based索引
            t_range = sf_functions[cls_idx].tRange
            t0, t1 = float(t_range[0]), float(t_range[1])
        else:
            t0, t1 = f_min, f_max

        # ========== 关键修正3：采样方式对齐原代码（去掉首尾端点） ==========
        t_list = np.linspace(t0, t1, n_samples_per_class + 2)[1:-1]

        # ========== 构建条件向量：离散部分（前48维） ==========
        cond_base = np.zeros(50, dtype=np.float32)

        # [0~33] 空间群独热 34维（195号对应索引0）
        sg_index = spaceGroup - 195
        if 0 <= sg_index < 34:
            cond_base[sg_index] = 1.0

        # [34~36] 结构类型独热 3维（tm / tp / tt）
        type_map = {'tm': 0, 'tp': 1, 'tt': 2}
        type_idx = type_map.get(lsType, 0)
        cond_base[34 + type_idx] = 1.0

        # [37~40] 拓扑变体独热 4维（origin+AB组合编码）
        variant_idx = (origin - 1) + AB * 2
        cond_base[37 + variant_idx] = 1.0

        # [41~47] HKL独热 7维（分量取值0~6，出现过的分量置1）
        for val in HKL:
            if 0 <= val <= 6:
                cond_base[41 + val] = 1.0

        # ========== 遍历等值面，生成每个样本 ==========
        max_abs_f = max(abs(f_min), abs(f_max))
        for t in t_list:
            # 同样关闭归一化
            field = getLSField2(f_handle, t=t, lsType=lsType, resPhi=res, normFlag=0)
            voxel = (field <= 0).astype(np.bool_)
            volume_fraction = float(np.mean(voxel))
            t_normalized = t / max_abs_f  # 仅作为条件特征，不参与几何计算

            # 填充连续维度
            cond = cond_base.copy()
            cond[48] = volume_fraction   # 体积分数（连续）
            cond[49] = t_normalized      # 归一化等值面（仅条件用）

            all_voxels.append(voxel)
            all_conditions.append(cond)

    # 堆叠并保存
    all_voxels = np.stack(all_voxels, axis=0)
    all_conditions = np.stack(all_conditions, axis=0)

    np.save(os.path.join(save_dir, "voxels.npy"), all_voxels)
    np.save(os.path.join(save_dir, "geo_conditions_onehot.npy"), all_conditions)

    print(f"生成完成！总样本数: {all_voxels.shape[0]}")
    print(f"体素张量形状: {all_voxels.shape}")
    print(f"条件张量形状: {all_conditions.shape}")
    return all_voxels, all_conditions


# 运行生成
if __name__ == "__main__":
    generate_metaset_dataset(
        key_path="familyNameKey.csv",
        save_dir="./metaset_64_npy",
        res=64,
        n_samples_per_class=100
    )