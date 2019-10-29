#!/usr/bin/python
# -*- coding: utf-8 -*-
""" geometry.py
Last update: 01/04/2017

Geometrical analysis of beams and coloidal or dryed UNCP measurement.
Usage:

"""

import numpy as np
import matplotlib.pylab as plt
# from mayavi import mlab
from scipy import integrate
import matplotlib.pyplot as plt
import logging
from .data import load_ion_states, load_ion_transitions
from .math import wavenumber_to_nm


class Ion(object):
    """ Light beam definitions, properties and visualization.
    """
    def __init__(self,):
        """
        Ion model
        """
        self.name = None


class Erbium(Ion):
    """ Light beam definitions, properties and visualization.
    """
    def __init__(self,):
        """
        Erbium model
        """
        self.name = 'erbium'
        self.ground = ('4I', '15/2')
        self.df_carnall = load_ion_states(ion_name=self.name)
        self.transitions = load_ion_transitions(ion_name=self.name)

    def mean_energy(self, init_state, final_state=None):
        """ The mean energy of a given state and J
        """
        df = self.df_carnall
        if isinstance(final_state, tuple):
            fstate, fj = final_state
        else:
            fstate, fj = self.ground
        istate, ij = init_state
        try:
            ie = df[(df['state']==istate) & (df['j']==ij)]
            fe = df[(df['state']==fstate) & (df['j']==fj)]
        except:
            raise(NameError)
        mean_energy = ie.mean(numeric_only=True)-fe.mean(numeric_only=True)
        std_energy = np.sqrt(ie.std(numeric_only=True)**2+fe.std(numeric_only=True)**2)
        return (mean_energy[0], std_energy[0])

    def available_states(self):
        """ The available states of the ion.
        """
        df = self.df_carnall
        return df.state.unique()

    def available_levels(self):
        """ The available state+j levels of the ion.
        """
        df = self.df_carnall
        return df.groupby(['state','j']).size().reset_index().rename(columns={0:'count'})

    def energy_levels(self, units='wavenumber'):
        """ The average energies of each state+j combination.
        """
        df = self.df_carnall
        calc_means = df.groupby(['state','j']).mean()
        if units == 'wavenumber':
            means = calc_means
        elif units == 'nm':
            means = wavenumber_to_nm(calc_means)
        return means
