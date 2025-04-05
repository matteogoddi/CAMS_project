import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import numpy as np

N = 200
TIME = 5
M = 20 #MPC horizon
m = 350 #DMD horizon

