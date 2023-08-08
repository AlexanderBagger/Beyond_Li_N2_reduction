#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 10:49:24 2020

@author: bagger
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from scipy.stats import norm, chi2
import numpy.linalg as linalg
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
from scipy.interpolate import splrep, splev
from scipy.optimize import minimize
from matplotlib.ticker import MultipleLocator
import csv
from numpy import genfromtxt
import pandas as pd
#from pandas.tools.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib.artist import setp
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
from os import path
import ase.db
from ase.db import connect
from matplotlib.legend_handler import HandlerTuple
from ase.io import write, read
from ase.visualize import view
### large data import
import re
r = re.compile("([a-zA-Z]+)([0-9]+)")


L=3; size1=28; size2=20; size3=24;A1=0.5;
s1=15; s2=15; s3=15; sdot=150; sdot2=150;
size1=28;size2=20; sdot=150

def seperate_string_number(string):
    previous_character = string[0]
    groups = []
    newword = string[0]
    for x, i in enumerate(string[1:]):
        if i.isalpha() and previous_character.isalpha():
            newword += i
        elif i.isnumeric() and previous_character.isnumeric():
            newword += i
        else:
            groups.append(newword)
            newword = i

        previous_character = i

        if x == len(string) - 2:
            groups.append(newword)
            newword = ''
    return groups

#############################################################
#### load in molecule data 
##################################################################
Mol=ase.db.connect('databases/molecules.db')
ENH3=Mol.get(formula='NH3').energy
EH2=Mol.get(formula='H2').energy
EN2=Mol.get(formula='N2').energy
EH2O=Mol.get(formula='H2O').energy

dict_bulk={}
dM=ase.db.connect('databases/bulk_spin.db')

# Here I generate some lists to save data
bulk_Mname=[]
bulk_Mname_count=[]
bulk_energy=[]
bulk_name=[]

bulk2_Mname=[]
bulk2_Mname_count=[]
bulk2_atom2=[]
bulk2_atom2_count=[]
bulk2_energy=[]
bulk2_name=[]

MxOyHz_name=[]
MxOyHz_Mname=[]
MxOyHz_count=[]
MxOyHz_energy=[]
MxOyHz_M=[]
MxOyHz_O=[]
MxOyHz_H=[]

MxHy_name=[]
MxHy_Mname=[]
MxHy_count=[]
MxHy_energy=[]
MxHy_M=[]
MxHy_H=[]

MxOy_name=[]
MxOy_Mname=[]
MxOy_count=[]
MxOy_energy=[]
MxOy_M=[]
MxOy_O=[]

MxNy_name=[]
MxNy_Mname=[]
MxNy_count=[]
MxNy_energy=[]
MxNy_M=[]
MxNy_N=[]

Mx_name=[]
Mx_Mname=[]
Mx_count=[]
Mx_energy=[]
Mx_M=[]

#############################################################
#### load in structure data 
##################################################################

for row in dM.select(relax='unitcell'):
    name=row.formula
    data=seperate_string_number(name)
    Atoms=read('databases/bulk_spin.db@id=%s' %row.id)[0]
    sym=Atoms.get_chemical_symbols()
    
# If M_x O_y H_z
    if (any('H' in string for string in data)
        and any('O' in string for string in data) and sym.count('H')>0 and sym.count('O')>0
        and name!='Hf'):
        #print('MxOyHz' ,name)
        MxOyHz_name.append(name)
        MxOyHz_Mname.append(sym[0])
        MxOyHz_count.append(row.natoms)
        MxOyHz_energy.append(row.energy)
        MxOyHz_M.append(row.natoms-sym.count('O')-sym.count('H'))
        MxOyHz_O.append(sym.count('O'))
        MxOyHz_H.append(sym.count('H'))
    
# If M_x H_z
    elif any('H' in string for string in data) and sym.count('O')==0 and sym.count('H')>0:
        #print('MxHy' ,name)
        MxHy_name.append(name)
        MxHy_Mname.append(sym[0])
        MxHy_count.append(row.natoms)
        MxHy_energy.append(row.energy)
        MxHy_M.append(row.natoms-sym.count('O')-sym.count('H'))
        MxHy_H.append(sym.count('H'))
# If M_x O_y
    elif sym.count('H')==0 and any('O' in string for string in data)==True and sym.count('O')>0:
        #print('MxOy' ,name)
        MxOy_name.append(name)
        MxOy_Mname.append(sym[0])
        MxOy_count.append(row.natoms)
        MxOy_energy.append(row.energy)
        MxOy_M.append(row.natoms-sym.count('O')-sym.count('H'))
        MxOy_O.append(sym.count('O'))

# If M_x N_y
    elif any('N' in string for string in data) and sym.count('N')>0 and sym.count('H')==0:
        #print('MxNy' ,name)
        MxNy_name.append(name)
        MxNy_Mname.append(sym[0])
        MxNy_count.append(row.natoms)
        MxNy_energy.append(row.energy)
        MxNy_M.append(row.natoms-sym.count('N')-sym.count('H'))
        MxNy_N.append(sym.count('N'))
# If M_x
    elif sym.count('H')==0 and sym.count('O')==0 and sym.count('N')==0:
        #print('Mx' ,name)
        Mx_name.append(name)
        Mx_Mname.append(sym[0])
        Mx_count.append(row.natoms)
        Mx_energy.append(row.energy)
        Mx_M.append(row.natoms-sym.count('O')-sym.count('H'))
        
    else:
        print('REST' ,name)


#############################################################
#### Create pandas data frames
##################################################################

d_MxOyHz = {
'MxOyHz_name': MxOyHz_name,
'Mname':MxOyHz_Mname,
'MxOyHz_count':MxOyHz_count,
'MxOyHz_energy':MxOyHz_energy,
'MxOyHz_M':MxOyHz_M,
'MxOyHz_O':MxOyHz_O,
'MxOyHz_H':MxOyHz_H}
df_MxOyHz = pd.DataFrame(data=d_MxOyHz)


d_MxHy = {
'MxHy_name': MxHy_name,
'Mname':MxHy_Mname,
'MxHy_count':MxHy_count,
'MxHy_energy':MxHy_energy,
'MxHy_M':MxHy_M,
'MxHy_H':MxHy_H}
df_MxHy = pd.DataFrame(data=d_MxHy)

d_MxOy = {
'MxOy_name': MxOy_name,
'Mname':MxOy_Mname,
'MxOy_count':MxOy_count,
'MxOy_energy':MxOy_energy,
'MxOy_M':MxOy_M,
'MxOy_O':MxOy_O}
df_MxOy = pd.DataFrame(data=d_MxOy)

d_MxNy = {
'MxNy_name': MxNy_name,
'Mname':MxNy_Mname,
'MxNy_count':MxNy_count,
'MxNy_energy':MxNy_energy,
'MxNy_M':MxNy_M,
'MxNy_N':MxNy_N}
df_MxNy = pd.DataFrame(data=d_MxNy)

d_Mx = {
'Mx_name': Mx_name,
'Mname':Mx_Mname,
'Mx_count':Mx_count,
'Mx_energy':Mx_energy,
'Mx_M':Mx_M}
df_Mx = pd.DataFrame(data=d_Mx)

# Creating big datadframe (merge the pandas frames)
df=pd.merge(df_Mx, df_MxNy, on="Mname", how="left")
df=pd.merge(df, df_MxOy, on="Mname", how="left")
df=pd.merge(df, df_MxHy, on="Mname", how="left")
df=pd.merge(df, df_MxOyHz, on="Mname", how="left")




#############################################################
#### Calculations of formation energies and reaction energies
##################################################################

# Calculate formation energies
df['Nitride Formation Energy']=(df['MxNy_energy']-df['MxNy_M']*df['Mx_energy']/df['Mx_count'] 
    -df['MxNy_N']*EN2*0.5)/df['MxNy_count']

df['Hydride Formation Energy']=(df['MxHy_energy']-df['MxHy_M']*df['Mx_energy']/df['Mx_count'] 
    -df['MxHy_H']*EH2*0.5)/df['MxHy_count']

df['Oxide Formation Energy']=(df['MxOy_energy']-df['MxOy_M']*df['Mx_energy']/df['Mx_count'] 
    -df['MxOy_O']*(EH2O-EH2))/df['MxOy_count']

df['MxOyHz Formation Energy']=(df['MxOyHz_energy']-df['MxOyHz_M']*df['Mx_energy']/df['Mx_count'] 
    -df['MxOyHz_O']*(EH2O-EH2)-df['MxOyHz_H']*EH2*0.5)/df['MxOyHz_count']


# calculating ammonia reaction energies:
df['MxOyHz Reaction']=(df['MxOyHz_energy']+df['MxNy_N']*ENH3 
    -df['MxNy_energy']-df['MxOyHz_O']*EH2O
    -(df['MxOyHz_H']+3*df['MxNy_N']-2*df['MxOyHz_O'])*EH2*0.5
    -(df['MxOyHz_M']-df['MxNy_M'])*(df['Mx_energy']/df['Mx_M']))/(df['MxNy_N'])

df['MxOy Reaction']=(df['MxOy_energy']+df['MxNy_N']*ENH3 
    -df['MxNy_energy']-df['MxOy_O']*EH2O
    -(3*df['MxNy_N']-2*df['MxOy_O'])*EH2*0.5
    -(df['MxOy_M']-df['MxNy_M'])*(df['Mx_energy']/df['Mx_M']))/(df['MxNy_N'])

df['MxHy Reaction']=(df['MxHy_energy']+df['MxNy_N']*ENH3 
    -df['MxNy_energy']
    -(3*df['MxNy_N']+df['MxHy_H'])*EH2*0.5
    -(df['MxHy_M']-df['MxNy_M'])*(df['Mx_energy']/df['Mx_M']))/(df['MxNy_N'])

df['Mx Reaction']=(df['Mx_energy']+df['MxNy_N']*ENH3 
    -df['MxNy_energy']
    -(3*df['MxNy_N'])*EH2*0.5
    -(df['Mx_M']-df['MxNy_M'])*(df['Mx_energy']/df['Mx_M']))/(df['MxNy_N'])


###############################
# Load N binding energies
###############################
dict_bulk={}

bcc_slab=ase.db.connect('databases/bcc_slab.db')
fcc_slab=ase.db.connect('databases/fcc_slab.db')
hcp_slab=ase.db.connect('databases/hcp_slab.db')

bcc_slab_N=ase.db.connect('databases/bcc_slab_N.db')
fcc_slab_N=ase.db.connect('databases/fcc_slab_N.db')
hcp_slab_N=ase.db.connect('databases/hcp_slab_N.db')

bcc_slab_N2=ase.db.connect('databases/bcc_slab_N2.db')
fcc_slab_N2=ase.db.connect('databases/fcc_slab_N2.db')
hcp_slab_N2=ase.db.connect('databases/hcp_slab_N2.db')

bcc_name=[]
bcc_energy=[]
bcc_name_N=[]
bcc_energy_N=[]
bcc_name_N2=[]
bcc_energy_N2=[]

fcc_name=[]
fcc_energy=[]
fcc_name_N=[]
fcc_energy_N=[]
fcc_name_N2=[]
fcc_energy_N2=[]

hcp_name=[]
hcp_energy=[]
hcp_name_N=[]
hcp_energy_N=[]
hcp_name_N2=[]
hcp_energy_N2=[]


for row in bcc_slab.select():
    name=row.formula
    data=seperate_string_number(name)
    Atoms=read('databases/bcc_slab.db@id=%s' %row.id)[0]
    sym=Atoms.get_chemical_symbols()
    bcc_name.append(sym[0])
    bcc_energy.append(row.energy)

for row in bcc_slab_N.select():
    name=row.formula
    data=seperate_string_number(name)
    Atoms=read('databases/bcc_slab_N.db@id=%s' %row.id)[0]
    sym=Atoms.get_chemical_symbols()
    bcc_name_N.append(sym[0])
    bcc_energy_N.append(row.energy)
    
for row in bcc_slab_N2.select():
    name=row.formula
    data=seperate_string_number(name)
    Atoms=read('databases/bcc_slab_N2.db@id=%s' %row.id)[0]
    sym=Atoms.get_chemical_symbols()
    bcc_name_N2.append(sym[0])
    bcc_energy_N2.append(row.energy)

for row in fcc_slab.select():
    name=row.formula
    data=seperate_string_number(name)
    Atoms=read('databases/fcc_slab.db@id=%s' %row.id)[0]
    sym=Atoms.get_chemical_symbols()
    fcc_name.append(name[0])
    fcc_energy.append(row.energy)
    print(name)

for row in fcc_slab_N.select():
    name=row.formula
    data=seperate_string_number(name)
    Atoms=read('databases/fcc_slab_N.db@id=%s' %row.id)[0]
    sym=Atoms.get_chemical_symbols()
    fcc_name_N.append(sym[0])
    fcc_energy_N.append(row.energy)
    
for row in fcc_slab_N2.select():
    name=row.formula
    data=seperate_string_number(name)
    Atoms=read('databases/fcc_slab_N2.db@id=%s' %row.id)[0]
    sym=Atoms.get_chemical_symbols()
    fcc_name_N2.append(sym[0])
    fcc_energy_N2.append(row.energy)
    
for row in hcp_slab.select():
        name=row.formula
        data=seperate_string_number(name)
        Atoms=read('databases/hcp_slab.db@id=%s' %row.id)[0]
        sym=Atoms.get_chemical_symbols()
        hcp_name.append(sym[0])
        hcp_energy.append(row.energy)

for row in hcp_slab_N.select():
        name=row.formula
        data=seperate_string_number(name)
        Atoms=read('databases/hcp_slab_N.db@id=%s' %row.id)[0]
        sym=Atoms.get_chemical_symbols()
        hcp_name_N.append(sym[0])
        hcp_energy_N.append(row.energy)
        
for row in hcp_slab_N2.select():
        name=row.formula
        data=seperate_string_number(name)
        Atoms=read('databases/hcp_slab_N2.db@id=%s' %row.id)[0]
        sym=Atoms.get_chemical_symbols()
        hcp_name_N2.append(sym[0])
        hcp_energy_N2.append(row.energy)

#############################################################
#### Make N pandas frames, and combine it with the rest
##################################################################

d_bcc_N = {
'Mname': bcc_name_N,
'N_energy':np.asarray(bcc_energy_N)-np.asarray(bcc_energy)-0.5*EN2}
df_bcc_N = pd.DataFrame(data=d_bcc_N)

d_fcc_N = {
'Mname': fcc_name_N,
'N_energy':np.asarray(fcc_energy_N)-np.asarray(fcc_energy)-0.5*EN2}
df_fcc_N = pd.DataFrame(data=d_fcc_N)

d_hcp_N = {
'Mname': hcp_name_N,
'N_energy':np.asarray(hcp_energy_N)-np.asarray(hcp_energy)-0.5*EN2}
df_hcp_N = pd.DataFrame(data=d_hcp_N)

df_N = df_bcc_N.append(df_fcc_N, ignore_index=True)
df_N = df_N.append(df_hcp_N, ignore_index=True)
df=pd.merge(df, df_N, on="Mname", how="left")

d_bcc_N2 = {
'Mname': bcc_name_N2,
'N2_energy':np.asarray(bcc_energy_N2)-np.asarray(bcc_energy)-EN2}
df_bcc_N2 = pd.DataFrame(data=d_bcc_N2)

d_fcc_N2 = {
'Mname': fcc_name_N2,
'N2_energy':np.asarray(fcc_energy_N2)-np.asarray(fcc_energy)-EN2}
df_fcc_N2 = pd.DataFrame(data=d_fcc_N2)

d_hcp_N2 = {
'Mname': hcp_name_N2,
'N2_energy':np.asarray(hcp_energy_N2)-np.asarray(hcp_energy)-EN2}
df_hcp_N2 = pd.DataFrame(data=d_hcp_N2)

df_N2 = df_bcc_N2.append(df_fcc_N2, ignore_index=True)
df_N2 = df_N2.append(df_hcp_N2, ignore_index=True)
df=pd.merge(df, df_N2, on="Mname", how="left")


#############################################################
#### Here we add a V SHE collumn and add the SHE potential for the elements
#### Here we add a dissocciation collumn and write this data
##################################################################

# initiate V SHE and Dissociation
df['V SHE']=-2.0


# # from standard electro potential (data_page)
df['V SHE'].loc[df['Mname']=='Rb']=-2.98
df['V SHE'].loc[df['Mname']=='Mg']=-2.372
df['V SHE'].loc[df['Mname']=='Al']=-1.662
df['V SHE'].loc[df['Mname']=='K']=-2.931
df['V SHE'].loc[df['Mname']=='Ca']=-2.868
df['V SHE'].loc[df['Mname']=='Sr']=-2.899
df['V SHE'].loc[df['Mname']=='B']=-1.79 # From ion state
df['V SHE'].loc[df['Mname']=='Be']=-1.847
df['V SHE'].loc[df['Mname']=='Na']=-2.71
df['V SHE'].loc[df['Mname']=='Li']=-3.04
df['V SHE'].loc[df['Mname']=='Cs']=-3.026
df['V SHE'].loc[df['Mname']=='Ba']=-2.912
df['V SHE'].loc[df['Mname']=='Ga']=-0.53
df['V SHE'].loc[df['Mname']=='In']=-0.34
df['V SHE'].loc[df['Mname']=='Tl']=-0.34
df['V SHE'].loc[df['Mname']=='Sc']=-2.077
df['V SHE'].loc[df['Mname']=='Y']=-2.372
df['V SHE'].loc[df['Mname']=='Ti']=-1.63
df['V SHE'].loc[df['Mname']=='Zr']=-1.45
df['V SHE'].loc[df['Mname']=='Hf']=-1.724
df['V SHE'].loc[df['Mname']=='V']=-1.13
df['V SHE'].loc[df['Mname']=='Nb']=-1.099
df['V SHE'].loc[df['Mname']=='Ta']=-0.6
df['V SHE'].loc[df['Mname']=='Cr']=-0.74
df['V SHE'].loc[df['Mname']=='Mo']=-0.15
df['V SHE'].loc[df['Mname']=='W']=-0.12
df['V SHE'].loc[df['Mname']=='Mn']=-1.185
df['V SHE'].loc[df['Mname']=='Re']=0.3
df['V SHE'].loc[df['Mname']=='Fe']=-0.44
df['V SHE'].loc[df['Mname']=='Ru']=0.455
df['V SHE'].loc[df['Mname']=='Os']=0.65
df['V SHE'].loc[df['Mname']=='Co']=-0.28
df['V SHE'].loc[df['Mname']=='Rh']=0.76
df['V SHE'].loc[df['Mname']=='Ir']=1.0
df['V SHE'].loc[df['Mname']=='Ni']=-0.25
df['V SHE'].loc[df['Mname']=='Pd']=0.915
df['V SHE'].loc[df['Mname']=='Pt']=1.188
df['V SHE'].loc[df['Mname']=='Cu']=0.337
df['V SHE'].loc[df['Mname']=='Ag']=0.7996
df['V SHE'].loc[df['Mname']=='Au']=1.52
df['V SHE'].loc[df['Mname']=='Zn']=-0.7618
df['V SHE'].loc[df['Mname']=='Cd']=-0.4
df['V SHE'].loc[df['Mname']=='Hg']=0.85



df['Dissociation']='False'
df['Dissociation'].loc[df['Mname']=='Li']='True'
df['Dissociation'].loc[df['Mname']=='Na']='False'
df['Dissociation'].loc[df['Mname']=='K']='False'
df['Dissociation'].loc[df['Mname']=='Rb']='False'
df['Dissociation'].loc[df['Mname']=='Cs']='False'
df['Dissociation'].loc[df['Mname']=='Be']='True'
df['Dissociation'].loc[df['Mname']=='Mg']='True'
df['Dissociation'].loc[df['Mname']=='Ca']='True'
df['Dissociation'].loc[df['Mname']=='Sr']='False'
df['Dissociation'].loc[df['Mname']=='Ba']='False'
df['Dissociation'].loc[df['Mname']=='Sr']='False'
df['Dissociation'].loc[df['Mname']=='Y']='True'
df['Dissociation'].loc[df['Mname']=='Ti']='True'
df['Dissociation'].loc[df['Mname']=='Zr']='True'
df['Dissociation'].loc[df['Mname']=='Hf']='True'
df['Dissociation'].loc[df['Mname']=='V']='True'
df['Dissociation'].loc[df['Mname']=='Nb']='True'
df['Dissociation'].loc[df['Mname']=='Ta']='True'
df['Dissociation'].loc[df['Mname']=='Cr']='True'
df['Dissociation'].loc[df['Mname']=='Mo']='True'
df['Dissociation'].loc[df['Mname']=='W']='True'
df['Dissociation'].loc[df['Mname']=='Mn']='True'
df['Dissociation'].loc[df['Mname']=='Re']='True'
df['Dissociation'].loc[df['Mname']=='Fe']='True'
df['Dissociation'].loc[df['Mname']=='Ru']='True'
df['Dissociation'].loc[df['Mname']=='Os']='False'
df['Dissociation'].loc[df['Mname']=='Co']='True'
df['Dissociation'].loc[df['Mname']=='Rh']='False'
df['Dissociation'].loc[df['Mname']=='Ir']='False'
df['Dissociation'].loc[df['Mname']=='Ni']='True'
df['Dissociation'].loc[df['Mname']=='Pd']='False'
df['Dissociation'].loc[df['Mname']=='Pt']='False'
df['Dissociation'].loc[df['Mname']=='Cu']='False'
df['Dissociation'].loc[df['Mname']=='Ag']='False'
df['Dissociation'].loc[df['Mname']=='Au']='False'
df['Dissociation'].loc[df['Mname']=='Zn']='True'
df['Dissociation'].loc[df['Mname']=='Cd']='False'
df['Dissociation'].loc[df['Mname']=='Hg']='False'
df['Dissociation'].loc[df['Mname']=='Al']='True'
df['Dissociation'].loc[df['Mname']=='B']='True'
df['Dissociation'].loc[df['Mname']=='Ga']='True'
df['Dissociation'].loc[df['Mname']=='In']='True'
df['Dissociation'].loc[df['Mname']=='Tl']='False'


#############################################################
#### Here we can view structures if we need
##################################################################
#atoms=read('../data/bulk_large.db@formula=Tl4N12')
#view(atoms)

#atoms=read('../data/bcc_slab_N.db@Na')
#view(atoms)

#atoms=read('../data/bcc_slab_N.db@Li')
#view(atoms)

#atoms=read('../data/hcp_slab_N.db@Mg')
#view(atoms)

#############################################################
#### Print "df" pandas dataframe for romain
##################################################################
df.to_csv('N2RR_data.csv')

#############################################################
#### Plotting periodic table, by first making a small frame and saving .csv
##################################################################
df_small=df[['Mname', 'Nitride Formation Energy']]
df_small.to_csv('ptable.csv', index=False, header=False)

# # https://github.com/arosen93/ptable_trends
from ptable_trends import ptable_plotter
ptable_plotter("ptable.csv", cmap="viridis", alpha=1.0, extended=False)


#############################################################
#### We drop some data that are strong outliers in the dataset
##################################################################
df = df.drop(df[df['Mname']=='B'].index)
df = df.drop(df[df['Mname']=='Nb'].index)
df = df.drop(df[df['Mname']=='Cr'].index)


#############################################################
#### plotting of figures
##################################################################
df.plot(x='V SHE', y='Nitride Formation Energy', kind='scatter')

fig = plt.figure(figsize=(14,7));
ax=fig.gca()
plt.locator_params(axis='x',nbins=5);plt.grid(True)
plt.xticks(fontsize = size3); plt.yticks(fontsize = size3)

for k in range(0,len(df['V SHE'].values)):
    plt.text(df['V SHE'].values[k], df['Nitride Formation Energy'].values[k], df['Mname'].values[k], fontsize=size1)

    if df['Dissociation'].values[k]=='True':
        p1=plt.scatter(df['V SHE'].values[k], df['Nitride Formation Energy'].values[k], c='b', s=sdot)
    elif df['Dissociation'].values[k]=='False':
        p2=plt.scatter(df['V SHE'].values[k], df['Nitride Formation Energy'].values[k], c='r', s=sdot)

plt.xlabel(r'Standard Reduction Potential',fontsize=size1)
plt.ylabel(r'Nitride Formation Energy [eV/Atom]',fontsize=size1)
plt.xlim([-3.2,1.55])
l = ax.legend([p1,p2],['N-N cleave phase','N-N coupling phase'],handler_map={tuple: HandlerTuple(ndivide=None)}, loc=4, fontsize=size1-8,markerscale=1.0)
#plt.savefig('Nitride_Formation_vs_SHE.png', dpi=400, bbox_inches='tight')
