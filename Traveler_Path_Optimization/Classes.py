###
'''
Author: Thomas Joyce

Version: 0.02

Class: MATH 496T - Mathematics of Generative AI 

Description: Central Repository for all my classes used in the development of my optimal control program

--> still not sure who should have the speed function and the favorability function
    - I feel the favorability should be on the traveler object right, I mean he is weighting his 
    decisions to see what step he is gonna take
    - I felt like the speed function could be determined by the landscape and then the traveler would have to 
    reference the terrain object to make his favorability function?Makes sense right, you survey the land before 
    you move, so the two objects would have to interact!
    
    
for this entire thing lets index bu i and j, not the actual position value, let us just consider the pixel values


Eventually for the energy function --> https://www.outsideonline.com/health/training-performance/easy-hike-up-hills/?scope=anon

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


### ----- # Classes # ----- ###
class Landscape:
    ''' 
    Description: Classs to represent the landscape the traveler will be interacting with. This class has the 
    following functionality:
        
        --> Reading Geotiff output file and organizing the data | DONE: __init__
        --> Accessing terrain data | DONE: self objects
        --> makes obstacles ( basically it will allow the user to specify a height range that classifies an obstacle)
            ex) all height below 200 m is a river (river obstacles have slower speed depending on depth of river)
            ex) all abs(gradient) larger than ...  is a cliff (cliffs will essentially be impassable, speed --> 0)
            etc ...
        --> plotting ...
        
        UPDATE THIS AS IT IS DEVELOPED!!!
    '''
    
    def __init__(self, DataFileName):
        ''' 
        Description: Initializes an instance of the Landscape class based upon an input file. This function
        reads, and organizes the data into shaped arrays based upon original binned file specifications
        outlined in the header. The arrays can be accessed throughout the Landscape class as they are "self"
        accessible. The "self" varaibles determined through this function are outlined below.
            
        Inputs:
            - DataFileName: string, filename of data to be read in. File must be in "OptimalControl" Format.
            
        Returns: No variables returned, but "self" variables can be accessed throughout the landscape class.
            - DataHeight: interger, number representing the height (number of rows) of the point cloud
            - DataWidth: interger, number representing the width (number of columns) of the point cloud
            - XPositions: shapped array (DataHeight by DataWidth) of float values, representing the x positions in meters of each point
            - YPositions: shapped array (DataHeight by DataWidth) of float values, representing the y positions in meters of each point
            - ZPositions: shapped array (DataHeight by DataWidth) of float values, representing the z positions in meters of each point
            - Slopes:
            - XunitV:
            - YunitV:
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
        
        plt.figure('', figsize = (24, 24))
        
        # testing #
        plt.imshow(self.Slopes, cmap='terrain', interpolation='nearest')  
        plt.colorbar(label='Gradient Intensity (unitless)') 
        
        #contours = plt.contour(self.XPositions, self.YPositions, self.ZPositions, levels=15, colors='white', linewidths=1)
        #contours = plt.contour(self.ZPositions, levels=4, cmap = 'coolwarm', linewidth = 0.5) #colors='white', linewidths=0.5)
        #contours = plt.contour(self.Slopes, levels=[0.5, 1.0, 2, 4], cmap = 'coolwarm', linewidths=2)
        #Fcountours = plt.contourf(self.Slopes, levels=np.linspace(self.Slopes.min(), 1, 100), cmap='Blues')
        
        
        ''' essentially humans cannot climb a slope less than 45 degress or in my case 1 '''
        
        # Label the contours
        #plt.clabel(contours, inline=True, fontsize=8, fmt="%.0f")
        
        # Plot Formatting #
        plt.colorbar(label='Gradient Intensity (unitless)') 
        
        plt.title(f'Gradient Map')  
        plt.xlabel('X-Position')
        plt.ylabel('Y-Position')
        
        plt.tight_layout()
        
        # Saving Figure #
        #plt.savefig(f'Elevation-{Location}_Bin{BinFactor}-{BinMode}.png')
        
    ### END GradientContourDiagram
    
    
    def SpeedFunction(self, Obstacles, LimitingAngle=45):
        '''
        Description: I think speed step will take upon the roll of multiplicative factor for regions of decent gradient
        I think this function will essentially just determine how passable the terrain is
        
        let's consider for example regions of really high gradient (these will be essentially cliffs or walls), these can be 
        uncrossable after a certain point dependent on tan^-1(slope) will give us a limiting angle
        
        then the obstacle functions will give us our unpassable terrain or our obstacle objects
        
        then the speed function can factor the gradient into the delta z calculation at the speed step stage of the 
        pipeline
        
        here will essentially just assign speed limitation values to each point depending upon terrain obstacles and how steep the gradient is
            
        Inputs:
            - Obstacles: ______________ <-- what Im thinking right now is to have obstacles basically be a 2D
            matrix of the same shape as everything else assigning a speed multiplier to each point
            
            1 = speed normal
            0 = full stop (can't cross basically) for example a huge rock you cannot walk overtop of
            0.4 --> walking through a light speed slowing your movement by 60 %
        
        
        put a note in that the limiting angle must be smaller than 50 (at least for the version I have now)
        
        '''
        
        
        
        
        ''' this could be a wasted line... delete when idea is clear to make it more pythonic... '''
        # Initializing Speed Matrix # 
        Speed = np.zeros((self.DataHeight, self.DataWidth))
        
        # Implementing Obstacles into Speed Function #
        Speed = Obstacles
        
        # Discouraged Movement at Limiting Angle #
        Speed[(self.Slopes > np.arctan(np.radians(LimitingAngle)))] = 0.01 
        ''' Essentially we are introducing a small value to discourage travel at the limiting angle
        (default is 45 degrees) wherin the travller can still traverse the space (typically calmoring up with
        their hands and feet), but is highly discouraged to as it is unsafe (especially with mud, gravel, or snow),
        slow, and most likely out of the capabilities of most travelers. '''
        
        # Unable to be traveled due to steep gradient (>50 degrees angle) #
        Speed[(self.Slopes > np.arctan(np.radians(50)))] = 0
        ''' At any angle greater than 45 degrees you are at a tan^-1(>45) > 1, wherein you
        would be rising more than the run, essentially requiring climbing gear at this point '''
        
        
        
        
        
        
        ### --> what should be left is basically the speed multiplier values at each navigable point
        
        self.SpeedMatrix = Speed
    ### END SpeedFunction
    
### END Landscape




class Traveler:
    '''
    Description: first iteration of traveler class (can only move in N,E,S,W), perhpas I should copy this class
    and make another with a new name for each version?
        
    '''
    
    def __init__(self, StartingPosition, EndingPosition):
        ''' 
        Description: we initilaze the initial and final poitions here, takes the xand y positions to begin with
        maybe we can take either the indeces or the actual meter position
            
        Inputs:
            - 
            -  
        '''
    ### END __init__
    
    
    def PlanRoute(self, StartingPosition, WeighingSteps=5):
        ''' 
        Description: this is the function that will plan the 5 steps out and weight each accordingly, make a user
        defined amount of how many steps out it needs to plan.
        
        calls plan step which will update the hypothetical (call it planning) position and store the cost of that step
        then we do plan step again from our new spot, do this WeighingSteps amount of times
        DO IT FOR EACH CARDINAL DIRECTION!
        
        it should basically return 4 paths, with 1st step taken to get on that path, the cost of each path will be returned
        as well and will essentially take into account each step in the path
        
        
            
        Inputs: 
            - Position: list of intergers, representing the indeces of the position in x and y (ex: [column index (x), row index(y)])
            of the traveler at the start of the plan
            - WeighingSteps: interger, number of steps to be weighed before a decision is made, defualt is 5
        '''
        
        
        
    ### END PlanRoute
    
    
    def SpeedStep(self, StartingPosition, EndingPosition, LandScape):
        '''
        Description: essentially this function will do the delta Z calculation, time that 
        by the gradient and determine the speed of that step
            
        
        Inputs:
            - StartingPosition:
            - EndingPosition:
                
        Returns:
            - Speed: 

        '''
        
        x0, y0 = StartingPosition[0], StartingPosition[1]
        xf,yf = EndingPosition[0], EndingPosition[1]
        
        ElevationChange = (LandScape.ZPositions[yf][xf] - LandScape.ZPositions[y0][x0])
        Gradient = LandScape.Slopes[yf][xf]
        SpeedMultiplier = LandScape.SpeedMatrix[yf][xf]
        
        
        
        ### so for the actual speed calculation I feel like what I should do is this...
        # --> basically I am going to use an exponetial function? 
        # --> Let's say for a negative ElevationChange then I have an exponential where
        # speed = e^(Gradient / Speed Multiplier) - 1 < slower at low gradients, steeper ones and your quicker, running down hill
        # For a positive elevation change then I have a decaying exponential where
        # speed = e^-(Gradient / Speed Multiplier) < faster at low gradients, steeper ones and your slower, running uphill
        
        # as this equation stands now if my gradient is 1 (45 degrees) and my speed multiplier is 1,
        # then you have a speed evaulated at e^-1 --> 0.36...
        
        # does this feel right? or maybe I should just do it as a line? ending to 0.01 or whatever at my limiting angle
        # the lines might be a better choice
        
        
        
    ### END SpeedStep
    
    
    def PlanStep(self, Position, LandScape):
        '''
        Description: I want this function to plan what step to take, essentially we would be doing the take step function 
        but updating some sort of planning position essentially determines which of the 4 directions is best
        and stores the updated position and the cost of that step
        
        I feel like you move:
            faster when going -deltaZ (higher to lower), unless its super steep
            slower when going +deltaZ (lower to higher), unless its super steep
            normal at 0 = deltaZ (even terrain)
            
        so maybe we need to have a Speedstep function that basically takes my landscape object's speed function and 
        multiplies the speed by an additional factor depending upon the deltaZ of the move to account for this
        obviously my speed function will be already dependent upon the gradient 
        
        This speed step function is good too becuase we can eventually factor in mass and other things too!
        which would pull on the self parameter
        
        
        Inputs:
            - Position: list of intergers, representing the indeces of the position in x and y (ex: [column index (x), row index(y)])
            
            
        
        '''
        
        # Unpacking Traveler Position Values #
        x, y = Position[0], Position[1]
        
        # Northward Route (up) #
        NorthSpeed = self.SpeedStep(Position, [x, (y + 1)], LandScape)
        
        
        # ok so we can access the position we are at through the indeces, we can also access neighboring positions as well
        # use the speed matrix and the elevation change * gradient to determine a speed value, relative to the current
        # position to assemble an idea of what step to take
        
        # then we actually take the step updating the new position and saving the "speed value" we took
        
        # evaluate which speed it biggest, then return the speed of this planned step and the direction "north", etc
        
    ### END PlanStep
    
    
    def TakeStep(self):
        ''' 
        Description: weighs the routes outputted by plan route and chooses the best one and takes a step
        updating its position and storing the movement in the file to keep trakc of what happened, perhaps I should 
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




#Pittsburgh.GradientContourDiagram()




'''  '''
### this is good for testing, keep this lil nuggest of code, for seeing how passable the terrain is... 
Pittsburgh.SpeedFunction(np.ones((Pittsburgh.DataHeight, Pittsburgh.DataWidth)), 45)

cmap = mcolors.ListedColormap(['red', 'yellow', 'green'])
boundaries = [0,0.0099,0.0101, 1]
norm = mcolors.BoundaryNorm(boundaries, cmap.N)

plt.imshow(Pittsburgh.SpeedMatrix, cmap = cmap, norm = norm)

plt.colorbar(label='passability') 
''' '''






# print(Pittsburgh.DataHeight)
# print(Pittsburgh.DataWidth)




'''
I want a select input function with the tkinter code in it, maybe a mode, folder or file

then import my classes and run landscape

'''

        
    