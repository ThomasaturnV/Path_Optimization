###
'''
Author: Thomas Joyce

Class: MATH 496T - Mathematics of Generative AI 


Description: Program derives the x, y, z values of a given terrain map taken from the 
USGS publically available data set. See "DataAquisition" for information on how to acquire a USGS GeoTiff file. 

A GeoTiff file format comes with 3 components:
    - Raster Data: A pixel grid of elevation avlues
    - Metadata: Metadata information regarding raster array height and width values, 
    coordinate reference system (CRS), pixel coordinate transforms, top left corner latitude and longitude,
    number of data layers or bands, etc...
    - Bands: bands of data (1 = elevation, 2 = slope, 3 = aspect), The count variable in the metadata specifies 
    how many bands are present, typically there is only one (elevation) available. 

Data products can be available for download from the USGS here: https://apps.nationalmap.gov/downloader/#/

Put the data access instructions in a README, plus a few example files...
---> Elevation Products (3D Elevation Program Products and Services)
---> 1/3 arc-second DEM
---> GeoTIFF format
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


### ----- # Functions # ----- ###
def LonLat_to_XY(Longitudes, Latitudes, CRS):
    '''
    Description:
    
    Inputs: 
        - Longitudes: 
        - Latitudes:
        - CRS:
    
    Returns:
        - X:
        - Y:
        - EPSG_UTM:
    '''
    
    
    CenterLongitude = np.mean(Longitudes) # Determines the center longitude value
    UTMZone = int((CenterLongitude + 180) / 6) + 1 # Determines relevant UTM zone from centerpoint longitude value
    EPSG_UTM = f"EPSG:{32600 + UTMZone}"  # EPSG UTM Zone Computed using an f string

    # Create a transformer from source CRS to UTM #
    # Essentially a one line conversion object
    # from a global longitude, latitude coordinate system into a 
    # projected flat plane of x,y values (x = east, y = north)
    Transformation = Transformer.from_crs(CRS, EPSG_UTM, always_xy=True)

    # Convert coordinates
    X, Y = Transformation.transform(Longitudes, Latitudes)

    return X, Y, EPSG_UTM
### END LonLat_to_XY


def Gradient(Elevation):
    ''' 
    Description:
        
    Inputs:
        - Elevation
    
    Returns:
        - Slope:
        - X_unitV
        - Y_unitY
    '''
    
    # Finding Elevation Gradients #
    dz_dx, dz_dy = np.gradient(Elevation) # gradient of the elvation matrix

    # Computing slope as a fraction (rise/run) #
    Slope = np.sqrt(dz_dx**2 + dz_dy**2)

    # Aspect (basically the direction) in terms of unitvectors #
    magnitude = np.sqrt(dz_dx**2 + dz_dy**2) + 1e-10 # small epsilon for numerical stability (no data points)
    X_unitV = (-dz_dx / magnitude)
    Y_unitV= (-dz_dy / magnitude)

    return Slope, X_unitV, Y_unitV
### END Gradient


def DataRetrieve(BinMode, BinFactor):
    '''
    Description:
        
    Inputs:
        - 
    
    Returns:
        - X:
        - Y:
        - Z:
        - Slope:
        - X_unitV:
        - Y_unitV:
        '''
    
    ### File Selecttion ###
    print('SelectData')
    root = Tk()
    Path = askopenfilename(title = "Select file", filetypes = (("tif files","*.tif"),("all files","*.*")))
    FilePath = os.path.split(os.path.abspath(Path))[0] # saves file path as a string
    FileName = os.path.split(os.path.abspath(Path))[1] # saves file name as a string
    root.destroy()

    os.chdir(FilePath) # navigates to directory file is stored within


    ### Opening and Retrieving data ###
    File = rasterio.open(FileName)
    
    # Retrieving Meta Data Information #
    Bands = File.count # Determines the number of bands from the metadata
    CRS = File.crs # Determines CRS (coordinate reference system)
    '''
    There are two relevant CRS systems:
        Geographic: where points are expressed as a global longitude and latitude coordinate
        Projected: where points are converted to a flat plane on Earth and expressed in meters
    '''
    TransformMatrix = File.transform # Affine Matrix (description below in docstring)
    '''
    | a b c |  The transformation matrix is used to map pixels to
    | d e f |  geographic coordinates
    | 0 0 1 |
    
    a: Pixel size in X direction (longitude space)
    b: X skew direction (0 if image is unrotated)
    c: Top Left X coordinate (corresponds to very first point)
    d: Y skew direction (0 if image is unrotated)
    e: Negative pixel size in the Y direction (latitude space), 
    the value is negative since images are read from top to bottom
    f: Top Left Y coordinate (corresponds to the very first point)
    '''
    NAN = File.nodata # value expressed for points not having data available (usually -999999.0)
    
    print('----- File MetaData ----- ')
    print(File.meta)
    print('----- ----- ----- ')
    
    # Retrieving Data #
    # Band 1 (Elevation)
    Elevation = File.read(1) # reads band 1
    
    if BinMode == 'Sampling': # Samples every BinFactor points in both x and y
        Indeces_X = np.arange(0, File.width, (BinFactor - 1)) # List of Sample Points
        Indeces_Y = np.arange(0, File.height, (BinFactor - 1)) # List of Sample Points
        
        # Initializing Binned List #
        Elevation_Binned = []
    
        # Iterating through Elevation Data and Binning #
        for index_y in range(0, len(Elevation)): # Iterating through each row of data 
            if index_y in Indeces_Y: # If row index is within binned indeces 
                Bin_row = [] # initializing binned row variable
                for index_x in range(0, len(Elevation[index_y])): # iterating through column of data
                    if index_x in Indeces_X: # If column is within binned indeces
                        Bin_row.append(Elevation[index_y][index_x]) # Saving Binned point
                Elevation_Binned.append(Bin_row) # appending to binned elevation list
        ###
        Elevation_Binned = np.array(Elevation_Binned) # Initializing object as numpy array
        Z = Elevation_Binned
    ###
    
    ##### ----- #####
    
    if BinMode == 'Median': # Takes a median of every square (side length = BinFactor) set of points
        Indeces_X = np.arange(0, File.width, (BinFactor - 1)) # List of Center Points
        Indeces_Y = np.arange(0, File.height, (BinFactor - 1)) # List of Center Points
            
        # Initializing Binned List #
        Elevation_Binned = []
        
        for index_y in range(0, len(Elevation)): # Iterating through each row of data
            if index_y in Indeces_Y: # If row index is within binned indeces 
                Bin_row = [] # initializing binned row variable
                for index_x in range(0, len(Elevation[index_y])): # Iterating through column of data
                    if index_x in Indeces_X: # If column is within binned indeces
                        MedianPoints = [] # initializing median points list (points that will need to be passed in for median combining)
                        for row in Elevation[index_y - (BinFactor - 1): index_y + (BinFactor + 1)]: # iterating through each row of binned column segment
                            MedianPoints.append(row[index_x - (BinFactor - 1) : index_x + (BinFactor + 1)]) # points to be considered from each row for median combining
                            
                        Bin_row.append(np.median(MedianPoints)) # Saving binned point
                Elevation_Binned.append(Bin_row) # appending to binned elevation list
        ###
    
        Elevation_Binned = np.array(Elevation_Binned) # initilaizing object as numpy array
        Z = Elevation_Binned
    ###
    
    # Updating all NAN points to zero (adjustments for nonsense data carried through calculations)
    Z[np.where(np.isnan(Z))] = 0
    
    # Acquiring row and column values where data is defined
    Rows, Columns = np.where(Z != 0)
    
    # Image Correction #
    #Z = Z[1:, 1:] # eliminate top tow and leftmost row, nonsense data
    
    
    
    
    
    Lons, Lats = rasterio.transform.xy(TransformMatrix, Rows, Columns) # Converts each point into a longitude and latitiude measurement
    
    
    X, Y, EPSG_UTM = LonLat_to_XY(Lons, Lats, CRS) # cartesian x and y values associated with each point
    
    
    # Band 2 (Slope)
    if Bands >= 2:
        SlopeAngle = File.read(2)[Rows, Columns] # reads band 2 (slope is measured in degrees)
        Slope = np.tan(np.radians(SlopeAngle)) # Converts slope to fraction
    
    else: # Band 2 data unavailable
        Slope, X_unitV, Y_unitV = Gradient(Elevation)
        Slope = Slope[Rows, Columns] # Determining Slope at pixels with data
    ###
    
    # Band 3 (Aspect)
    if Bands >= 3:
        AspectAngle = File.read(3)[Rows, Columns] # reads band 3 (direction measured in degrees)
        ''' Aspect is measured in 0 to 360 degrees (0 = North, 90 = East, 180 = South, 270 = West) '''
        X_unitV = -np.cos(np.radians(AspectAngle)) # Converting to East (x-direction) unit vector
        Y_unitV = -np.sin(np.radians(AspectAngle)) # Converting to North (y-direction) unit vector
    
    else: # Band 3 data unavailable
        X_unitV = X_unitV[Rows, Columns]
        Y_unitV = Y_unitV[Rows, Columns]
    ###

    return X, Y, Z, Slope, X_unitV, Y_unitV
### END DataRetrieve
    
    
def OutputTXT(X, Y, Z, Slope, X_unitV, Y_unitV, OutputFileName):
    '''
    Description:
    
    Inputs:
        - X: 
        - Y:
        - Z:
        - Slope:
        - X_unitV:
        - Y_unitV:
        - OutputFileName:
    
    Returns:
        - 
    '''
    
    File = open(OutputFileName, 'w')
    # Wiring Python Readable Header #
    #File.write('# X-Position (m) | Y-Position (m) | Z-Position (m) | Fractional Slope | Slope Unit Vector (X) | Slope Unit Vector (Y) #')
    
    for index in range(0, len(X)): # Iterating through each point and writing data to file
        File.write((f"{X[index]:.5f}, {Y[index]:.5f}, {Z[index]:.5f}, {Slope[index]:.5f}, {X_unitV[index]:.5f}, {Y_unitV[index]:.5f} \n"))
        
    File.close()
### END OutputTXT
    
    
X, Y, Z, Slope, X_unitV, Y_unitV = DataRetrieve('Median', 40)

print('length X: ' + str(len(X)) + ' | Length Y: ' + str(len(Y)) + ' | Length Z: ' + str(len(Z) * len(Z[0])))
print(f'length Slope: {len(Slope)} | length X_unitV: {len(X_unitV)} | length Y_unitV: {len(Y_unitV)}')

print('Data Retrieved... Writing to a file...')

OutputTXT(X, Y, Z.flatten(), Slope, X_unitV, Y_unitV, "Out.txt")

print("Done!")










# Visualize the Elevation Map (Terrain)
plt.figure(figsize=(10, 8))  
plt.imshow(Z, cmap='terrain', interpolation='nearest')  
plt.colorbar(label='Elevation (m)')  
plt.title('Elevation Map (Terrain)')  
plt.xlabel('X (UTM)')
plt.ylabel('Y (UTM)')
plt.show() 






