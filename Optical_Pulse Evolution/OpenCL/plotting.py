__author__ = 'JoaoCosta'

import scipy as sp
import pylab as pl
import platform

# Save files Folder path
if (platform.system() == 'Windows'):
    fPath = 'Data_Files\\'
    iPath = 'Images\\'
else:
    fPath = 'Data_Files/'
    iPath = 'Images/'

def pulse_graph(P0, OC, X, T, evolution, scale):
    """
    This function plots the amplitude of the pulse's envelope over time on all the grid positions
    """
    xGrid, tGrid = sp.meshgrid(X, T)
    pl.figure("Pulse_P0=" + str(P0) + "_OC=" + str(OC))
    pl.title("Pulse propagation W/ " + "P0=" + str(P0) + ", OC=" + str(OC))
    pl.xlabel("x")
    pl.ylabel("t")
    pl.contourf(xGrid, tGrid, evolution, levels = scale)
    # pl.contourf(xGrid, tGrid, evolution)
    pl.colorbar()
    pl.savefig(iPath+"Pulse_P0_"+str(P0)+"_OC_"+str(OC)+".png")

def state_graph(P0, OC, X, T, state11, state22, state33):
    """
    This function plots the state densities over time on all the grid positions
    """
    xGrid, tGrid = sp.meshgrid(X, T)
    pl.figure("p11_P0=" + str(P0) + "_OC=" + str(OC))
    pl.title("State 11 evolution W/ " + "P0=" + str(P0) + ", OC=" + str(OC))
    pl.xlabel("x")
    pl.ylabel("t")
    pl.contourf(xGrid, tGrid, state11, sp.linspace(0.0, state11.max(), 30))
    pl.colorbar()
    pl.savefig(iPath+"p11_P0_"+str(P0)+"_OC_"+str(OC)+".png")

    pl.figure("p22_P0=" + str(P0) + "_OC=" + str(OC))
    pl.title("State 22 evolution W/ " + "P0=" + str(P0) + ", OC=" + str(OC))
    pl.xlabel("x")
    pl.ylabel("t")
    pl.contourf(xGrid, tGrid, state22, sp.linspace(0.0, state22.max(), 30))
    pl.colorbar()
    pl.savefig(iPath+"p22_P0_"+str(P0)+"_OC_"+str(OC)+".png")

    pl.figure("p33_P0=" + str(P0) + "_OC=" + str(OC))
    pl.title("State 33 evolution W/ " + "P0=" + str(P0) + ", OC=" + str(OC))
    pl.xlabel("x")
    pl.ylabel("t")
    pl.contourf(xGrid, tGrid, state33, sp.linspace(0.0, state33.max(), 30))
    pl.colorbar()
    pl.savefig(iPath+"p33_P0_"+str(P0)+"_OC_"+str(OC)+".png")