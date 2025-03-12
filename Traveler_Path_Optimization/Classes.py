###
'''
Author: Thomas Joyce

Version: 

Class: MATH 496T - Mathematics of Generative AI 

Description: Central Repository for all my classes used in the development of my optimal control program

--> still not sure who should have the speed function and the favorability function
    - I feel the favorability should be on the traveler object right, I mean he is weighting his 
    decisions to see what step he is gonna take
    - I felt like the speed function could be determined by the landscape and then the traveler would have to 
    reference the terrain object to make his favorability function?Makes sense right, you survey the land before 
    you move, so the two objects would have to interact!

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
import matplotlib.ticker as tick
from scipy import stats
# Raster Manipulation
import rasterio
from pyproj import Transformer


### ----- # Classes # ----- ###
class Landscape:
    ''' 
    Description: Classs to represent the landscape the traveler will be interacting with. This class has the 
    following functionality:
        
        --> Reading Geotiff output file and organizing the data
        --> Accessing terrain data
        --> makes obstacles ( basically it will allow the user to specify a height range that classifies an obstacle)
            ex) all height below 200 m is a river (river obstacles have slower speed depending on depth of river)
            ex) all abs(gradient) larger than ...  is a cliff (cliffs will essentially be impassable, speed --> 0)
            etc ...
        --> plotting ...
        
        UPDATE THIS AS IT IS DEVELOPED!!!
    '''
    
    def __init__(self, DataFileName):
        ''' 
        Description: reads in the data from the geotiff-xyz output, and perhpas reshapes it into a 
        grid for easier indexing and everything (all data is the same shape)
        
        --> for that to happen though, I need sort of a "binned metadata" at the top of the file
        that has the data shape outlined
            
        Inputs:
            - DataFileName: 
        '''
        ### File Management ###
        # Opening File #
        File = open(DataFileName, 'r')
        
        # First line encodes header length #
        Hlen = int(File.readline().split(':')[1])
        
        # Reading Header #
        Partition = File.readline() # = '---------- Geotiff Conversion ----------'
        self.DataHeight = int(File.readline().split(':')[1])
        self.DataWidth = int(File.readline().split(':')[1])
        
        # Closing File #
        File.close()
        
        # Retrieving Data #
        Data = np.genfromtxt(DataFileName, dtype = float, names = True, delimiter = ',', skip_header = (Hlen - 1))
        ### ----- ###
        
        
        ### Data Management ###
        # XPosition, YPosition, ZPosition, FractionalSlope, XSlopeUnitVector, YSlopeUnitVector # 
        
        # Reshaping and saving data to class #
        self.XPositions = np.reshape(Data['XPosition'], (self.DataHeight, self.DataWidth))
        self.YPositions = np.reshape(Data['YPosition'], (self.DataHeight, self.DataWidth))
        self.ZPositions = np.reshape(Data['ZPosition'], (self.DataHeight, self.DataWidth))
        self.Slopes = np.reshape(Data['FractionalSlope'], (self.DataHeight, self.DataWidth))
        self.XunitV = np.reshape(Data['XSlopeUnitVector'], (self.DataHeight, self.DataWidth))
        self.YunitV = np.reshape(Data['YSlopeUnitVector'], (self.DataHeight, self.DataWidth))
        ### ----- ###
        
    ### END __init__
        
    
    def GradientContourDiagram(self):
        '''
        Description: I want this function to essentially take the gradient and plot a heat map of it and have the 
        elevation contours overlayed, perhaps I can take a linear range between each elevation point to make it smooth for contour plotting?
        
            
        Inputs:
            -
        '''
        
        
        
        # testing #
        plt.imshow(self.Slopes, cmap='terrain', interpolation='nearest')  
        plt.colorbar(label='Gradient Intensity (unitless)') 
        
        #contours = plt.contour(self.XPositions, self.YPositions, self.ZPositions, levels=15, colors='white', linewidths=1)
        contours = plt.contour(self.ZPositions, levels=3, cmap = 'coolwarm', linewidth = 0.5) #colors='white', linewidths=0.5)



        # Label the contours
        plt.clabel(contours, inline=True, fontsize=8, fmt="%.0f")
        
        
        # interesting results so far, it actually looks correct, but funy how pittsburgh has not too many sharp changes...
        # --> test out mt. rainer and grandcanyon to confirm if its working
        
        
    ### END GradientContourDiagram
### END Landscape




class Traveler:
    '''
    Description: first iteration of traveler class (can only move in N,E,S,W), perhpas I should copy this class
    and make another with a new name for each version?
        
    '''
    
    def __init__(self, DataFileName):
        ''' 
        Description:
            
        Inputs:
            - D 
            -  
        '''
    ### END __init__
    
    
    def PlanRoute(self):
        ''' 
        Description: this is the function that will plan the 5 steps out and weight each accordingly, make a user
        defined amount of how many steps out it needs to plan
            
        Inputs:
            -
        '''
    ### END PlanRoute
    
    
    def TakeStep(self):
        ''' 
        Description: weights the routes outputted by plan route and chooses the best one and takes a step
        updating its position and storing the movement in the file to keep trakc of what happened, perhpas I should 
        have an update file function, called by this guy?
            
        Inputs:
            -
        '''
    ### END TakeStep
    
### END Traveler
    
        
    
    
    
    
# ### TESTING ###
### File Selection ###
print('SelectData')
root = Tk()
Path = askopenfilename(title = "Select file", filetypes = (("txt files","*.txt"),("all files","*.*")))
FilePath = os.path.split(os.path.abspath(Path))[0] # saves file path as a string
FileName = os.path.split(os.path.abspath(Path))[1] # saves file name as a string
root.destroy()

os.chdir(FilePath) # navigates to directory file is stored within

Pittsburgh = Landscape(FileName)




Pittsburgh.GradientContourDiagram()

# print(Pittsburgh.DataHeight)
# print(Pittsburgh.DataWidth)
        
    