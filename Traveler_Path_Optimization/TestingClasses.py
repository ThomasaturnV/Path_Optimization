###
'''
Author: Thomas Joyce

Version: 0.21

Class: MATH 496T - Mathematics of Generative AI 

Description: Used for testing my classes.py...
'''

### ----- # Importations # ----- ###
# Numpy 
import numpy as np
# File Management
from tkinter import Tk
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import glob
import os
# Plotting and Statistics
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as tick
from scipy import stats
# Raster Manipulation
import rasterio
from pyproj import Transformer

# Importing Classes 
from Classes import LandScape
from Classes import Traveler


##### TESTING #####


### File Selection ###
#print('SelectData')
#root = Tk()
#Path = askopenfilename(title = "Select file", filetypes = (("txt files","*.txt"),("all files","*.*")))
#FilePath = os.path.split(os.path.abspath(Path))[0] # saves file path as a string
#FileName = os.path.split(os.path.abspath(Path))[1] # saves file name as a string
#root.destroy()

# for easier repeatability
Path = 'C:/Users/thoma/OneDrive/GitHub/Path_Optimization/Traveler_Path_Optimization/ExampleData/OptimalControlElevation-Pittsburgh_Bin40-Median.txt'
FilePath = os.path.split(os.path.abspath(Path))[0] # saves file path as a string
FileName = os.path.split(os.path.abspath(Path))[1] # saves file name as a string


os.chdir(FilePath) # navigates to directory file is stored within





Pittsburgh = LandScape(FileName, 'Pittsburgh')
Pittsburgh.SpeedFunction(np.ones((Pittsburgh.DataHeight, Pittsburgh.DataWidth)))



John = Traveler([1,1], {2:[18,18]}, Pittsburgh)

John.Travel(3, [18,18])





#Pittsburgh.GradientContourDiagram()

#Pittsburgh.VisualizingSpeedFunction()




# print(Pittsburgh.DataHeight)
# print(Pittsburgh.DataWidth)