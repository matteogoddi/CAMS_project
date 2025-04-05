import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import numpy as np

N = 200
TIME = 8
M = 10 #MPC horizon
m = 250 #DMD horizon

