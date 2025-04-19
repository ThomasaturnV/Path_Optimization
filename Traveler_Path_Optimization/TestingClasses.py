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
Path = 'C:/Users/thoma/OneDrive/GitHub/Path_Optimization/Traveler_Path_Optimization/ExampleData/OptimalControlElevation-Pittsburgh_Bin10-Median.txt'
FilePath = os.path.split(os.path.abspath(Path))[0] # saves file path as a string
FileName = os.path.split(os.path.abspath(Path))[1] # saves file name as a string


os.chdir(FilePath) # navigates to directory file is stored within





Pittsburgh = LandScape(FileName, 'Pittsburgh')
Pittsburgh.SpeedFunction(np.ones((Pittsburgh.DataHeight, Pittsburgh.DataWidth)))

print('Done with Speed!')


# Testing position Dependence#
#Jim = Traveler([499957.17080, 4538813.59472], {1e13:[500510.05981, 4538762.21902]}, Pittsburgh)

#Jim.Travel(10, [500510.05981, 4538762.21902])

#print(Jim.Position)

#print(Jim.Destination)

# NOTE: Jim is getting stuck!!!, he is oscillating about the x point, but not going up to his y point
# good note is the position option does work whenever you use 



# liam is gonna use waypoints, to help lead him to the goal
#Liam = Traveler([1,1], {100: [70, 84], 50: [35,35], 10: [18,18]}, Pittsburgh)
### OK now we need to remove a waypoint once its reached
#Liam.Travel([70,84], 10)

#print(Liam.Position)


Leroy = Traveler([1,1], {100: [150, 130]}, Pittsburgh)
Leroy.Travel([150,130], 40)
print(Leroy.Position)




'''
4/18/25
Vanishing node works whenever detination isn't too far away however my iterative node forces it to get stuck so
i need to look into this. I think we need a condition where the same position is selected as a half node, we just
give it a burst of favorability in the direction of the destination'



'''







'''
vanishing node doesn't seem to work:
    
--> try on non binned data set where eveything is a bit smoother and local minmia are less likely
--> maybe just add some random noise if the same position is selected as a temporary node. So for example if 200: [42, 49] and 300: [42, 49]
exists then maybe we just default to adding some random noise (a small bounding box of random multiplier values to mix everything up
or maybe just force a movement to the pixel in the direction of the node
or a small bounding box with values increasing in direction of node?


'''


# John did good! #
#John = Traveler([1,1], {5:[80,80]}, Pittsburgh)

#John.Travel([80,83], 10)

#print(John.Position)


### I think I'm hitting issues with floating point precision


### maybe we implement a user defined pixel box where if the traveler tstays within the box for length^2 iterations,
### we do something. Of which I cant say Im sure... (maybe is what we can do is increase the nearest node weight?)
### or we can create a whole new node? one that is half the distance between current traveler position and destination node?


### rescaling units

### disappearing node

#Pittsburgh.GradientContourDiagram()

#Pittsburgh.VisualizingSpeedFunction()




# print(Pittsburgh.DataHeight)
# print(Pittsburgh.DataWidth)



def DataAnalysis(LandScape): # temporary function to make a plot of the path
    ''' 
    '''
    
    plt.figure('Path', figsize = (20, 20))
    
    SpeedFunction = LandScape.SpeedMatrix
    
    ### Finding the travel file ###
    Path = 'C:/Users/thoma/OneDrive/GitHub/Path_Optimization/Traveler_Path_Optimization/ExampleData/TravelFile.txt'
    FilePath = os.path.split(os.path.abspath(Path))[0] # saves file path as a string
    FileName = os.path.split(os.path.abspath(Path))[1] # saves file name as a string


    os.chdir(FilePath) # navigates to directory file is stored within
    
    
    #### reading file
    file = open(FileName, 'r')
    file.readline() # header
    
    EndingPosition = []
    Favorability = []
    for line in file.readlines():
        Line = line.split('|')
        EndingPosition.append( [ int(Line[1][2:-2].split(',')[0]), int(Line[1][2:-2].split(',')[1]) ] )
        Favorability.append(float(Line[2]))
        
    
    #print()
    #print(EndingPosition)
    
    np.array(EndingPosition)
    np.array(Favorability)

    # Determine the size of the matrix (adjust as needed)
    matrix_size = (LandScape.DataHeight, LandScape.DataWidth)  # Example size (rows, columns)

    # Create a zero matrix
    Movements = np.zeros(matrix_size)

    # Assign values to the specified positions
    for (y, x), favor in zip(EndingPosition, Favorability):
        Movements[y, x] = 2

    #Movements = Movements * 1e-4
    
    
    SpeedFunction += Movements
    
    
    
        
        
    
    
    
    
    ####
    
    
    
    #cmap = mcolors.ListedColormap(['red', 'yellow', 'green'])
    #boundaries = [0,0.0099,0.0101, 1]
    #norm = mcolors.BoundaryNorm(boundaries, cmap.N)

    #plt.imshow(self.SpeedMatrix, cmap = cmap, norm = norm)
    
    ### combining speed functional form into everything ###
    plt.imshow(SpeedFunction, cmap = 'magma') # cool to see what is not passable, magma to see gradient based structure,, 
    cbar = plt.colorbar(label='Ratio of Max Speed')
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label('Ratio of Max Speed', fontsize=25)
    ###  ###

    #cbar = plt.colorbar(label='Speed Multiplier') 
    #cbar.ax.tick_params(labelsize=0)  # Change the font size of the tick labels
    #cbar.set_label('Speed Multiplier (Red = 1, Yellow = 0.01, Green = 1)', fontsize=25)  # Change label font size
    
    plt.title(f'Path Overlayed Upon The Speed Function of {LandScape.Location}', fontsize = 45)  
    plt.xlabel('X-Position [pixels]', fontsize = 40)
    plt.ylabel('Y-Position [pixels]', fontsize = 40)
    
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)
    
    plt.tight_layout()
    
    # Saving Figure #
    plt.savefig(f'Path_{LandScape.Location}.png')
    
### END DataAnalysis

DataAnalysis(Pittsburgh)
    


