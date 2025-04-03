import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import casadi as ca
from constants import *

def EDMD(Y,U):

    #generate m random inputs, obtaining Yu

    #apply to the model and get m observables 3x1

    #consider a linear combination of observables and get matrix Yx

    #compute svd of [Yx;Yu] and [Yx+]

    #compute A,B

    return A,B