#!/usr/bin/python
# -*- coding: utf-8 -*-
""" math.py
Last update: 27/10/2019

Math utilities
Usage:

"""
import numpy as np

def wavenumber_to_nm(wn):
    """ Wavenumber (cm^-1) to nm
    """
    return 1e7/wn

def mean_energy(df, state, level):
    filtered_df = df[(df['state'] ==state) & (df['j'] == level)]
    print(filtered_df)
    return filtered_df.mean(numeric_only=True)
