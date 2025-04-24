###
'''
Author: Thomas Joyce

Version: 0.23

Class: MATH 496T - Mathematics of Generative AI 

Description: Central Repository for all my classes used in the development of my optimal control program

--> still not sure who should have the speed function and the favorability function
    - I feel the favorability should be on the traveler object right, I mean he is weighting his 
    decisions to see what step he is gonna take
    - I felt like the speed function could be determined by the landscape and then the traveler would have to 
    reference the terrain object to make his favorability function?Makes sense right, you survey the land before 
    you move, so the two objects would have to interact!
    
    
for this entire thing lets index be i and j, not the actual position value, let us just consider the pixel values


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
class LandScape:
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
    
    def __init__(self, DataFileName, Location):
        ''' 
        Description: Initializes an instance of the Landscape class based upon an input file. This function
        reads, and organizes the data into shaped arrays based upon original binned file specifications
        outlined in the header. The arrays can be accessed throughout the Landscape class as they are "self"
        accessible. The "self" varaibles determined through this function are outlined below.
            
        Inputs:
            - DataFileName: string, filename of data to be read in. File must be in "OptimalControl" Format.
            - Location: string, Name of locatio being referenced for the terrain (closest city/landmark)
            
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
        
        self.Location = Location
        
        ### File Management ###
        # Opening File #
        File = open(DataFileName, 'r')
        
        # First line encodes header length #
        Hlen = int(File.readline().split(':')[1])
        
        # Reading Header #
        Partition = File.readline() # = '---------- Geotiff Conversion ----------'
        self.DataHeight = int(File.readline().split(':')[1]) # = 'Bin Height: ___' 
        self.DataWidth = int(File.readline().split(':')[1]) # = 'Bin Width: ___'
        self.BinMode = str(File.readline().split(':')[1][1:-2]) # = 'Bin Mode Used: ___'
        self.BinFactor = int(File.readline().split(':')[1]) # = 'Bin Factor Used: ___'
        
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
        
        plt.figure('Gradient', figsize = (20, 20))
        
        # testing #
        plt.imshow(self.Slopes, cmap='binary', interpolation='nearest')  
        cbar = plt.colorbar(label='Gradient Value [tan(angle of slope)]') 
        cbar.ax.tick_params(labelsize=20)  # Change the font size of the tick labels
        cbar.set_label('Gradient Value [tan(angle of slope)]', fontsize=25)  # Change label font size
        
        #contours = plt.contour(self.XPositions, self.YPositions, self.ZPositions, levels=15, colors='white', linewidths=1)
        #contours = plt.contour(self.ZPositions, levels=4, cmap = 'coolwarm', linewidth = 0.5) #colors='white', linewidths=0.5)
        #contours = plt.contour(self.Slopes, levels=[0.5, 1.0, 2, 4], cmap = 'coolwarm', linewidths=2)
        #Fcountours = plt.contourf(self.Slopes, levels=np.linspace(self.Slopes.min(), 1, 100), cmap='Blues')
        
        
        #contours = plt.contour(self.Slopes, levels=[1.0, 1.19175359259421], cmap = 'coolwarm', linewidths=2, fontsize = 30)
        '''
        tan(45) > 1.0
        tan(50) > 1.19175359259421 '''
        
        
        ''' essentially humans cannot climb a slope less than 45 degress or in my case 1 '''
        
        # Label the contours
        #plt.clabel(contours, inline=True, fontsize=8, fmt="%.0f")
        
        # Plot Formatting #
        #plt.colorbar(label='Gradient Intensity (unitless)') 
        
        plt.title(f'Gradient Intensity Map of {self.Location}', fontsize = 45)  
        plt.xlabel('X-Position [pixels]', fontsize = 40)
        plt.ylabel('Y-Position [pixels]', fontsize = 40)
        
        plt.tick_params(axis='x', labelsize=20)
        plt.tick_params(axis='y', labelsize=20)
        
        plt.tight_layout()
        
        # Saving Figure #
        plt.savefig(f'GradientIntensityMap_{self.Location}_Bin{self.BinFactor}-{self.BinMode}.png')
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
        Speed[(self.Slopes > np.arctan(np.radians(LimitingAngle)))] *= 0.1 
        ''' Essentially we are introducing a small value to discourage travel at the limiting angle
        (default is 45 degrees) wherein the traveler can still traverse the space (typically calmoring up with
        their hands and feet), but is highly discouraged to as it is unsafe (especially with mud, gravel, or snow),
        slow, and most likely out of the capabilities of most travelers. '''
        
        # Unable to be traveled due to steep gradient (>50 degrees angle) #
        Speed[(self.Slopes > np.arctan(np.radians(50)))] *= 0
        ''' At any angle greater than 45 degrees you are at a tan^-1(>45) > 1, wherein you
        would be rising more than the run, essentially requiring climbing gear at this point '''
        
        
        Speed = Speed * (np.exp(-1 * self.Slopes))
        
        ''' 
        Wether the travler moves uphil or downhill, their speed will exponentially decay with the gradient
        models by an e^-x function. Going uphill is slower and harder, while going down hill is slower becuase 
        the trvaler will have to control their speed to not slip and fall. The favorability function does weight
        downward movements more heavily however.
        '''
        
        
        
        ### --> what should be left is basically the speed multiplier values at each navigable point
        
        self.SpeedMatrix = Speed
    ### END SpeedFunction
    
    
    def VisualizingSpeedFunction(self):
        ''' '''
        
        plt.figure('Speed', figsize = (20, 20))
        
        self.SpeedFunction(np.ones((self.DataHeight, self.DataWidth)), 45)

        #cmap = mcolors.ListedColormap(['red', 'yellow', 'green'])
        #boundaries = [0,0.0099,0.0101, 1]
        #norm = mcolors.BoundaryNorm(boundaries, cmap.N)

        #plt.imshow(self.SpeedMatrix, cmap = cmap, norm = norm)
        
        ### combining speed functional form into everything ###
        plt.imshow(self.SpeedMatrix, cmap = 'plasma') # cool to see what is not passable, magma to see gradient based structure,, 
        cbar = plt.colorbar(label='Ratio of Max Speed')
        cbar.ax.tick_params(labelsize=20)
        cbar.set_label('Ratio of Max Speed', fontsize=25)
        ### ''''''''''''''''''''''''''''''''''''''''''''''' ###
        

        #cbar = plt.colorbar(label='Speed Multiplier') 
        #cbar.ax.tick_params(labelsize=0)  # Change the font size of the tick labels
        #cbar.set_label('Speed Multiplier (Red = 1, Yellow = 0.01, Green = 1)', fontsize=25)  # Change label font size
        
        plt.title(f'Speed Function of {self.Location}', fontsize = 45)  
        plt.xlabel('X-Position [pixels]', fontsize = 40)
        plt.ylabel('Y-Position [pixels]', fontsize = 40)
        
        plt.tick_params(axis='x', labelsize=20)
        plt.tick_params(axis='y', labelsize=20)
        
        plt.tight_layout()
        
        # Saving Figure #
        plt.savefig(f'LandscapeSpeedFunction_{self.Location}_Bin{self.BinFactor}-{self.BinMode}.png')
    ### END VisualizingSpeedFunction
### END Landscape




class Traveler:
    '''
    Description: first iteration of traveler class (can only move in N,E,S,W), perhpas I should copy this class
    and make another with a new name for each version?
        
    '''
    
    def __init__(self, StartingPosition, Nodes, LandScape, C):
        ''' 
        Description: we initilaze the initial and final poitions here, takes the xand y positions to begin with
        maybe we can take either the indeces or the actual meter position
            
        Inputs:
            - StartingPosition: list of floats/intergers, Starting Position of the traveler in meters or position indeces [x_0, y_0]. 
            - Nodes: dictionary, containing node positions and their weights for points of interest. Format looks like:
            Nodes = {weight1: [x_1, y_1], weight2: [x_2, y_2], etc...}, where the positions are the indeces. 
            - LandScape: 
        '''
        
        ### Temporary Variables ###
        self.C = np.zeros((LandScape.DataHeight, LandScape.DataWidth))
        self.C += C
        self.CValue = C
        
        # ----- #
        
        self.RandomPositions = []
        
        # Initializing Self Parameters #
        self.Nodes = Nodes
        
        self.LandScape = LandScape
        
        # Position adjustment (if given in meters) #
        if StartingPosition < [self.LandScape.DataHeight, self.LandScape.DataWidth]: # Given as position indeces
            self.Position = StartingPosition 
            # NOTE: This could be inaccurate if a location is selected near the gloabl origin... 
        else: # Given as meter location
            self.Position = [ int(np.where(self.LandScape.XPositions == StartingPosition[0])[0]), int(np.where(self.LandScape.XPositions == StartingPosition[0])[1])]
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
        
        note we need the speed step evaluation for initial move
        
            
        Inputs: 
            - Position: list of intergers, representing the indeces of the position in x and y (ex: [column index (x), row index(y)])
            of the traveler at the start of the plan
            - WeighingSteps: interger, number of steps to be weighed before a decision is made, defualt is 5
        '''
        # Unpacking Traveler Position Values #
        x0, y0 = StartingPosition[0], StartingPosition[1]
        
        Step = 1 # step taken by traveler, set to one to account for predefined routes below
        NorthTotalFavorability, EastTotalFavorability, SouthTotalFavorability, WestTotalFavorability = 0, 0, 0, 0 # Evaluating Criterion (value to be maximized)
        
        ''' note we could do the below code in a single function honestly '''
        
        # Planning Northward Route (up starting move) #
        NorthTotalFavorability += self.Favorability(StartingPosition, [x0, (y0 - 1)]) # accounts for initial predefined move
        PlanPosition = [x0, (y0 - 1)] # represents the first step (Step = 1)
        
        while Step <= WeighingSteps:
            PlanFavor, PlanPosition = self.PlanStep(PlanPosition)
            Step += 1
            NorthTotalFavorability += PlanFavor
        ###
        Step = 1
        
        # Planning Eastward Route (right starting move) #
        EastTotalFavorability += self.Favorability(StartingPosition, [(x0 + 1), y0]) # accounts for initial predefined move
        PlanPosition = [(x0 + 1), y0] # represents the first step (Step = 1)
        while Step <= WeighingSteps:
            PlanFavor, PlanPosition = self.PlanStep(PlanPosition)
            Step += 1
            EastTotalFavorability += PlanFavor
        ###
        Step = 1
        
        # Planning Southward Route (down starting move) #
        SouthTotalFavorability += self.Favorability(StartingPosition, [x0, (y0 + 1)]) # accounts for initial predefined move
        PlanPosition = [x0, (y0 + 1)] # represents the first step (Step = 1)
        while Step <= WeighingSteps:
            PlanFavor, PlanPosition = self.PlanStep(PlanPosition)
            Step += 1
            SouthTotalFavorability += PlanFavor
        ###
        Step = 1
        
        # Planning Westward Route (left starting move) #
        WestTotalFavorability += self.Favorability(StartingPosition, [(x0 - 1), y0]) # accounts for initial predefined move
        PlanPosition = [(x0 - 1), y0] # represents the first step (Step = 1)
        while Step <= WeighingSteps:
            PlanFavor, PlanPosition = self.PlanStep(PlanPosition)
            Step += 1
            WestTotalFavorability += PlanFavor
        ###
        Step = 1
        
        ''' NOTE: we need a condition to check for ties ---> propogate out one more step
        maybe we can do this by casting data into a set and if the set is less than 4 long
        then maybe we propogate out everything by one more step?'''
        # Evaluating Optimal Route #
        UpdatedRoutes = [[x0, (y0 - 1)], [(x0 + 1), y0], [x0, (y0 + 1)], [(x0 - 1), y0]] # [North, East, South, West] first move 
        PlannedTotalFavorabilities = [NorthTotalFavorability, EastTotalFavorability, SouthTotalFavorability, WestTotalFavorability]
        
        MostFavorable = max(PlannedTotalFavorabilities)
        
        OptimalRoute = UpdatedRoutes[PlannedTotalFavorabilities.index(MostFavorable)]
        
        return MostFavorable, OptimalRoute
    ### END PlanRoute
    
    
    def PlanStep(self, Position):
        '''
        Description: I want this function to plan what step to take, essentially we would be doing the take step function 
        but updating some sort of planning position essentially determines which of the 4 directions is best
        and stores the updated position and the cost of that step
        
        I feel like you move:
            faster when going -deltaZ (higher to lower), unless its super steep
            slower when going +deltaZ (lower to higher), unless its super steep
            normal at 0 = deltaZ (even terrain)
        
        
        NOTE: we can have a check here for for bounds, essentially if x = 0 or Landscape.Datawidth --> bound --> must consider only 3 paths
                                                                      y = 0 or Landscape.DataHeight --> bound --> must consider only 3 paths          
        
        --> define a matrix where it is: [ [topleft, top, top, top, top, ..., topright] 
                                           [left, 0, 0, 0, 0, ..., right]
                                           .
                                           .
                                           .
                                           [bottomleft, bottom, bottom, bottom, ..., bottomright] ]
        
        --> then we should have a dictionary with dic = {'topleft': ['East', South], 'top': ['East', 'South', 'West'], ... etc}
        --> basically if the position on the matrix is nonzero then we use that key in the dictionary and pass it through some if statements where
        if 'North' in dic[key] for example, do north plan speed...
        
        then we add all values in a list, and the direction in the list too, then we find max, then we find the direction corresponding to entry
        and pass that off as "bound check"
        
        
        Inputs:
            - Position: list of intergers, representing the indeces of the position in x and y (ex: [column index (x), row index(y)])
            
            
        
        '''
        
        
        # Unpacking Traveler Position Values #
        x, y = Position[0], Position[1]
        
        # Northward Route (up) #
        NorthPlanFavor = self.Favorability(Position, [x, (y - 1)])
        
        # Eastward Route (right) #
        EastPlanFavor = self.Favorability(Position, [(x + 1), y])
        
        # Southward Route (down) #
        SouthPlanFavor = self.Favorability(Position, [x, (y + 1)])
        
        # Westward Route (left) #
        WestPlanFavor = self.Favorability(Position, [(x - 1), y])
        
        
        ''' note: if two steps are equally viable we need an if condition to check for this 
        if this does happen tho, lets just utilize the self and find the step that is closest to the destination '''
        
        
        # Evaluating Best Step #
        UpdatedPositions = [[x, (y - 1)], [(x + 1), y], [x, (y + 1)], [(x - 1), y]] # [North, East, South, West] Positions 
        PlannedFavorability = [NorthPlanFavor, EastPlanFavor, SouthPlanFavor, WestPlanFavor]
        
        MostFavorable = max(PlannedFavorability)
        
        OptimalPosition = UpdatedPositions[PlannedFavorability.index(MostFavorable)]
        
        return MostFavorable, OptimalPosition
    ### END PlanStep
    
    
    def Displacement(self, r_i, r_f):
        '''
        Description: Determines the displacement between the r_i and r_f position for the favorability function
            
        Inputs:
            - r_i: list of intergers ([x_i, y_i]), current position in the form of matrix indeces
            - r_f: list of intergers ([x_f, y_f]), position of desired node (or destination) in the form of matrix indeces
            
        Returns:
            - Displacement: float, value determining the net displacemet between the initial and final positions
        '''
        
        # Small factor for numerical stability (when r_i = r_f) #
        Epsilon = 1e-2
        
        # X-Direction Displacement #
        Delta_X = r_f[0] - r_i[0]
        
        # Y-Direction Displacement #
        Delta_Y = r_f[1] - r_i[1]
        
        # Determing Displacement #
        Displacement = np.sqrt( (Delta_X ** 2) + (Delta_Y ** 2) ) + Epsilon
        
        #Displacement = Epsilon + abs(Delta_X) + abs(Delta_Y)
        
        return Displacement
    ### END Displacement
        
    
    def Favorability(self, StartingPosition, EndingPosition):
        '''
        Description:
            
        Inputs:
            - C:
            - Speed:
            - StartingPosition: list of intergers, representing the indeces of the starting position in x and y (ex: [column index (x), row index(y)])
            - EndingPosition: list of intergers, representing the indeces of the ending position in x and y (ex: [column index (x), row index(y)])
            - Nodes: dictionary, containing node positions and their weights for points of interest. Format looks like:
            Nodes = {weight1: [x_1, y_1], weight2: [x_2, y_2], etc...}, where the positions are the indeces. 
            - Landscape:
                
        Returns:
            F:
        
                
        
        NOTES:
            - if need be later on we can define a speed variable: Speed = MASSTERM * OTHERPHYSICSTERMS * self.LandScape.SpeedMatrix[Xf][Yf]
        '''
        
        # Unpacking Traveler Position Values #
        Xi, Yi = StartingPosition[0], StartingPosition[1]
        Xf, Yf = EndingPosition[0], EndingPosition[1]
        
        # Node Weighting #
        NodeWeightingTerm = 0
        for weight, node in self.Nodes.items(): # determines the effect of multiple nodes and their weights
            NodeWeightingTerm += weight * (self.Displacement(EndingPosition, node) ** -3)
        
        
        # Elevation Dependent Favorability #
        ElevationChange = (self.LandScape.ZPositions[Yf][Xf] - self.LandScape.ZPositions[Yi][Xi])
        
        if ElevationChange > 0: # DeltaZ = + (uphill)
            F = self.C[Yf][Xf] * self.LandScape.SpeedMatrix[Yf][Xf] * NodeWeightingTerm
        else: # DeltaZ = - (downhill)
            F = 1.1 * self.C[Yf][Xf] * self.LandScape.SpeedMatrix[Yf][Xf] * NodeWeightingTerm
            #F = self.C[Yf][Xf] * self.LandScape.SpeedMatrix[Yf][Xf] * NodeWeightingTerm
            
        
        return F
    ### END Favorability
    
    
    def MovementFile(self, StartPosition, EndPosition, Favorability, Mode):
        '''
        Description: Bare bones for now just storing end position and favorability, later on I want it to store the following:
            start position [x_i,y_i] | end position [x_i+1, y_i+1] | movement direction: north south etc | favorability of movement | speed of movement | time taken to move

        Inputs:
            - Position:
            - Favorability:
            - Mode: 
        

        '''
        
        
        '''
        Useful code from another project:
            
        np.savetxt(OutputName, AngMom, fmt = "%15.3f"*4, comments = '#', 
                   header = "{:15s}{:15s}{:15s}{:15s}".format('t', 'L_tot_x', 'L_tot_y', 'L_tot_z'))
        '''
        
        #Speed = self.LandScape.SpeedMatrix[EndPosition[0]][EndPosition[1]] # note I just realized we may need to check this [yf,xf]???
        ### using the end position relative to the start position we cna find the movement direction, we can use the speed, self.maxspeed, and the position difference to find the time taken to move
        
        if Mode == 'Initialize':
            FileI = open('TravelFile.txt', 'w')
            FileI.write('# Start Position | End Position | Favorability of Route # \n')
            FileI.close()
            
        if Mode == 'Update':
            FileU = open('TravelFile.txt', 'a')
            FileU.write(f'{StartPosition} | {EndPosition} | {Favorability} \n')
            print(f'{StartPosition} | {EndPosition} | {Favorability} \n')
            FileU.close()
    ### END MovementFile
    

    def TakeStep(self, WeighingSteps=5):
        ''' 
        Description: weighs the routes outputted by plan route and chooses the best one and takes a step
        updating its position and storing the movement in the file to keep trakc of what happened, perhaps I should 
        have an update file function, called by this guy?
        
        the thing is right now this needs to be depedent on the favorability function!!!
            
        Inputs:
            -
        '''
        
        
        '''NOTE: I can return the start position and store it, then maybe when plan route is called it doesn't consider this position?'''
        # Storing The position Before Step #
        StartPosition = self.Position # Before Plan Route is activated and used
        
        MostFavorable, self.Position = self.PlanRoute(self.Position, WeighingSteps)
        
        self.MovementFile(StartPosition, self.Position, MostFavorable, 'Update')
    ### END TakeStep
    
    
    def RandomCBox(self, Position, Mode, n=3):
        '''
        Note: we need to implement a way to take away the randomness after you leave the random box,
        
        thinking simple if check on position and see if its "n" away from the self.RandomPositions
        
        look into how to delete things from a lit and not break a for loop or whatever
        '''
        
        [X_c, Y_c] = Position # center values of random box
    
        BoxRadius = n // 2 # radius of the bounding box (half the size of the box accounting for odd number)
        
        if Mode == 'Random':
            self.RandomPositions.append(Position)
            
            # Box Bounding Indeces (correcting for points near of at the edge of the field)
            X_LeftEdge = max(0, X_c - BoxRadius) 
            X_RightEdge = min((self.LandScape.DataWidth - 1), X_c + BoxRadius) 

            Y_TopEdge = max(0, Y_c - BoxRadius) 
            Y_BottomEdge = min((self.LandScape.DataHeight - 1), Y_c + BoxRadius)
            
            # Calculate the actual dimensions of the bounding box
            BoxHeight = Y_BottomEdge - Y_TopEdge + 1
            BoxWidth = X_RightEdge - X_LeftEdge + 1

            # Generate random values with the correct shape
            RandomC = np.random.uniform(0.01, (self.CValue - (self.CValue/n)), (BoxHeight, BoxWidth))

            # Creating Bounding Box (of random values)
            self.C[Y_TopEdge:(Y_BottomEdge + 1), X_LeftEdge:(X_RightEdge + 1)] = RandomC # reversed becuase (0,0) point is at the top left
        ###
            
        if Mode == 'Normalize':
            if self.C[Y_c, X_c] == self.CValue: # point at which it leaves the random noise
                if self.RandomPositions != []:
                
                    # Finding Closest Node to current Location (the node that was reached)
                    MinDisplacement = self.Displacement([0,0], [self.LandScape.DataHeight, self.LandScape.DataWidth])
                    for Position in self.RandomPositions:
                        Displacement = self.Displacement(self.Position, Position)
                        if Displacement <= MinDisplacement:
                            MinDisplacement = Displacement
                            NodePosition = Position
                
                    # Box Bounding Indeces (correcting for points near of at the edge of the field)
                    X_LeftEdge = max(0, NodePosition[0] - BoxRadius) 
                    X_RightEdge = min((self.LandScape.DataWidth - 1), NodePosition[0] + BoxRadius) 

                    Y_TopEdge = max(0, NodePosition[1] - BoxRadius) 
                    Y_BottomEdge = min((self.LandScape.DataHeight - 1), NodePosition[1] + BoxRadius)
                
                        
                    # Normalizing C noise values (no that traveler has left) #
                    self.C[Y_TopEdge:(Y_BottomEdge + 1), X_LeftEdge:(X_RightEdge + 1)] = self.CValue # reversed becuase (0,0) point is at the top left
                    
                
                    # Removing Box of Random Points #
                    self.RandomPositions.remove(NodePosition)
                ###
            ###
        ###
    ### END RandomCBox


    def BoundingBox(self, Position, n=5):
        ''' 
        Description: this function will take the current position and create a bounding box of size n x n around the position
        to for tracking if the traveler is stuck. This creates a matrix of size (DataHeight, DataWidth) with zeros everywhere except 
        for ones in a region n x n around the Position variable. The region of ones is a bounding box, representing a region
        where the traveler could get stuck within, best size to use if 5.  

        Inputs:
            - n: interger, size of the bounding box (n x n) to be created around the traveler. Must be odd value
            - Position: list of intergers, current position of the traveler in position indeces [x_i, y_i]
        
        '''
        TravelerBounds = np.zeros((self.LandScape.DataHeight, self.LandScape.DataWidth))

        [X_c, Y_c] = Position # center values of bounding box

        BoxRadius = n // 2 # radius of the bounding box (half the size of the box accounting for odd number)

        # Box Bounding Indeces (correcting for points near of at the edge of the field)
        X_LeftEdge = max(0, X_c - BoxRadius) 
        X_RightEdge = min((self.LandScape.DataWidth - 1), X_c + BoxRadius) 

        Y_TopEdge = max(0, Y_c - BoxRadius) 
        Y_BottomEdge = min((self.LandScape.DataHeight - 1), Y_c + BoxRadius)

        # Creating Bounding Box (of ones)
        TravelerBounds[Y_TopEdge:(Y_BottomEdge + 1), X_LeftEdge:(X_RightEdge + 1)] = 1 # reversed becuase (0,0) point is at the top left

        return TravelerBounds
    ### END BoundingBox

    
    def UpdateNodes(self, Weight, Position, Mode):
        '''
        
        '''
        # Adding a Node #
        if Mode == 'Add':
            self.Nodes[Weight] = Position
            return
    
        
        if Mode == 'Add-NoDuplicate': # Halves the position value if you have the same position in the Nodes dictionary
            n = 2
            W = []
            Pos = []
            for w, pos in self.Nodes.items():
                W.append(w)
                Pos.append(pos)
                
            AdjustedPosition = Position
            while AdjustedPosition in Pos:
                AdjustedPositionX = ((self.Position[0] + Position[0]) // n)
                AdjustedPositionY = ((self.Position[1] + Position[1]) // n)
                
                AdjustedPosition = [AdjustedPositionX, AdjustedPositionY]
                n += 1
            
            self.Nodes[Weight] = AdjustedPosition
            return
                
                
                
            # for w, pos in self.Nodes.items():
            #     if pos == Position:
            #         PositionX = ((self.Position[0] + Position[0]) // n)
            #         PositionY = ((self.Position[1] + Position[1]) // n)
            #         self.Nodes[Weight] = [PositionX, PositionY]
            #         return
                
            #     else:
            #         [PositionX, PositionY] = Position
            
            # self.Nodes[Weight] = [PositionX, PositionY]
            # return
                    
                
        # Deleting a Node #
        elif Mode == 'Delete-Weight':
            del self.Nodes[Weight]
            return
            
        
        elif Mode == 'Delete-Position':
            for w, Pos in self.Nodes.items():
                if Pos == Position:
                    del self.Nodes[w]
                    return
    ### END UpdateNodes


    def Travel(self, Destination, Iterations = 5000, WeighingSteps=5):
        '''
        Description: we are gonna use this guy as the method that actually initiates all of the steps and moves the traveler
            
        Inputs:
            - Destination: list of floats/intergers, Final (ending) Position of the traveler in meters or position indeces [x_f, y_f]
            - WeighingSteps:
        '''
    
        # Position adjustment (if given in meters) #
        if Destination < [self.LandScape.DataWidth, self.LandScape.DataHeight]: # Given as position indeces
            self.Destination = Destination 
        # NOTE: This could be inaccurate if a location is selected near the gloabl origin... 
        else: # Given as meter location
            self.Destination = [ int(np.where(self.LandScape.XPositions == Destination[0])[0]), int(np.where(self.LandScape.XPositions == Destination[0])[1])]
    

        # Determining Destination Weight #
        NodeBounds = np.zeros((self.LandScape.DataHeight, self.LandScape.DataWidth))
        for Weight, Position in self.Nodes.items():
            if Position == self.Destination:
                DestinationWeight = Weight
            else:
                NodeBounds += self.BoundingBox(Position)
    
    
    
    
    
        i = 1 # temporary variable used in testing to see if things get stuck...
        Multiple = 1 # used for weight multiplication on temporary nodes 
        
        self.MovementFile(0, 0, 0, 'Initialize')

        # Initializing the bounding box (for traveler getting stuck)
        TravelerBounds = self.BoundingBox(self.Position) # NOTE: we are currently excluding the radius for testing
        # we will use a vairable MinimaRadius for the "n" in bounding box, for now I will define it:
        MinimaRadius = 5
        
        ### Traveling Loop ###
        while (self.Position != self.Destination) and (i <= (Iterations + 1)):
            self.TakeStep(WeighingSteps)

            
            # Deleting Node (that is not Destination) whenever it is reached (within a 5 x 5 square radius) #
            if NodeBounds[self.Position[1]][self.Position[0]] >= 1:
                
                # Finding Closest Node to current Location (the node that was reached)
                MinDisplacement = self.Displacement([0,0], [self.LandScape.DataHeight, self.LandScape.DataWidth])
                for Weight, Position in self.Nodes.items():
                    Displacement = self.Displacement(self.Position, Position)
                    if Displacement <= MinDisplacement:
                        MinDisplacement = Displacement
                        NodePosition = Position
                    ###
                ###
                
                # Deleting the accomplished Node #
                self.UpdateNodes(0, NodePosition, 'Delete-Position')
                NodeBounds -= self.BoundingBox(NodePosition)
            ###
            
            # Updating Traveler Bounds #
            if i % ((MinimaRadius ** 2) // 2) == 0:
                TravelerBounds = self.BoundingBox(self.Position) # NOTE: we are currently excluding the radius for testing
            ###
            
            
            # If the traveler is stuck within a local minima (bounding box) #
            if i % (MinimaRadius ** 2) == 0:
                if TravelerBounds[self.Position[1]][self.Position[0]] == 1:
                    
                    self.RandomCBox(self.Position, 'Random')
                    print('Injecting Random Noise')
                    print(self.C[self.Position[1]][self.Position[0]])
                ###
            else:
                self.RandomCBox(self.Position, 'Normalize')
            ###
                    
            ''' What I want here is t create a box of radius 5, with some favorability multipliers
                    centered around the current position, where the weights increase in the direction of the nearest node
                    these weights would instantly disappear once the traveler leaves the box 
                    
                    --> easiest way to do this would maybe be to multiple the speed matrix by this stuff
                    but I don't really want to mess with the physical meaning of the speed matrix
                    
                    ----> maybe the C value (the constant for numerical stability can be augmented into a matrix where whatever the user defines
                    (lets say C = 3) then its just a matrix of a bunch of 3s. However whenever this condition is reached it will change the c values around
                    the traveler temporarily. So whenever the traveler reaches the point where C = 3 again, it will renormalize that box to a bunch of 
                    3s again. 
                    
                    Let's say we have the node directly to the right:
                        
                        [...3 [ 0.5, 1.0, 1.5, 2.0, 2.5 ]   3...
                         ...3 [ 0.5, 1.0, 1.5, 2.0, 2.5 ]   3...
                         ...3 [ 0.5, 1.0, 1.5, 2.0, 2.5 ]   3...
                         ...3 [ 0.5, 1.0, 1.5, 2.0, 2.5 ]   3...
                         ...3 [ 0.5, 1.0, 1.5, 2.0, 2.5 ]   3...]
                    '''
                    
                    
                    #Multiple += 1
                    #self.UpdateNodes((DestinationWeight * Multiple), [((self.Position[0] + self.Destination[0]) // 2),  ((self.Position[1] + self.Destination[1]) // 2)], 'Add')
                    #self.UpdateNodes((DestinationWeight * Multiple), [((self.Position[0] + self.Destination[0]) // 2),  ((self.Position[1] + self.Destination[1]) // 2)], 'Add-NoDuplicate')
                    
                    #''' Creates a temporary Node halfway between the destination and the current position, to get traveler unstuck'''
                    #NodeBounds += self.BoundingBox(self.Nodes[DestinationWeight * Multiple]) # I want this to remain 5, so we are using defualt value
                ###
                #print('Instuting Temporary Node')
                #print(self.Nodes)
            
            
            ''' Temporarily Disabled ---
            # If the traveler is stuck within a local minima (bounding box) #
            # if i % (MinimaRadius ** 2) == 0:
            #     if TravelerBounds[self.Position[1]][self.Position[0]] == 1:
            #         Multiple += 1
            #         self.UpdateNodes((DestinationWeight * Multiple), [((self.Position[0] + self.Destination[0]) // 2),  ((self.Position[1] + self.Destination[1]) // 2)], 'Add')
            #         #self.UpdateNodes((DestinationWeight * Multiple), [((self.Position[0] + self.Destination[0]) // 2),  ((self.Position[1] + self.Destination[1]) // 2)], 'Add-NoDuplicate')
                    
            #         # Creates a temporary Node halfway between the destination and the current position, to get traveler unstuck #
            #         NodeBounds += self.BoundingBox(self.Nodes[DestinationWeight * Multiple]) # I want this to remain 5, so we are using defualt value
            #     ###
            #     print('Instuting Temporary Node')
            #     print(self.Nodes)
            ###
            '''
            
            
            # Updating Iteration Counter #
            i += 1
        ###
        
        if self.Position == self.Destination:
            print('Destination Reached!')
        else:
            print('i reached max value | Too many Iterations')
    
    
    ### END Travel
    
### END Traveler
    

      





        
    