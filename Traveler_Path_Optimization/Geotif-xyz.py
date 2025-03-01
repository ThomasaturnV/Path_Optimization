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
    
    #Delta_x = TransformMatrix.a  # X pixel size
    #Delta_y = abs(TransformMatrix.e)  # Y pixel size (absolute value bdue to negativity)
    
    #Delta_x = np.median(np.diff(X))  # Mean distance between consecutive points in the x direction (in meters)
    #Delta_y = np.median(np.diff(Y))  # Mean distance between consecutive points in the y direction (in meters)
    
    #print('Delta_x: ' + str(Delta_x))
    #print('Delta_y: ' + str(Delta_y))

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


def DataRetrieve(bin_factor):
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
    
    ''' binning --> works''' '''
    #trying out the binning on the elevation:
    Indeces_X = np.arange(0, File.width, bin_factor-1)
    Indeces_Y = np.arange(0, File.height, bin_factor-1)
        
    Elevation_Binned = []
    
    for index_y in range(0, len(Elevation)): # iterating through each row
        if index_y in Indeces_Y: # if row index is within binned indeces 
            e_row = []
            for index_x in range(0, len(Elevation[index_y])): # iterating through column
                if index_x in Indeces_X:
                    e_row.append(Elevation[index_y][index_x])                    #print(Elevation[index_y - (bin_factor - 1) : index_y + (bin_factor - 1)][index_x - (bin_factor - 1) : index_x + (bin_factor - 1)])
            Elevation_Binned.append(e_row)
    ###
    
    Elevation_Binned = np.array(Elevation_Binned)
    
    Elevation = Elevation_Binned
    
    
    print(Elevation_Binned)
            
                
        
        
    ''' ''' '''
    
    ''' Binning --> median '''
    
    Indeces_X = np.arange(0, File.width, bin_factor-1)
    Indeces_Y = np.arange(0, File.height, bin_factor-1)
        
    Elevation_Binned = []
    
    for index_y in range(0, len(Elevation)): # iterating through each row
        if index_y in Indeces_Y: # if row index is within binned indeces 
            e_row = []
            for index_x in range(0, len(Elevation[index_y])): # iterating through column
                if index_x in Indeces_X:
                    # median filter:
                    MedianPoints = []
                    for row in Elevation[index_y - (bin_factor - 1): index_y + (bin_factor + 1)]: # iterating through each row of binned segment
                        MedianPoints.append(row[index_x - (bin_factor - 1) : index_x + (bin_factor + 1)]) # points to be considered from each row
                        
                    
                    e_row.append(np.median(MedianPoints))
            Elevation_Binned.append(e_row)
    ###
    
    Elevation_Binned = np.array(Elevation_Binned)
    
    Elevation = Elevation_Binned
    
    
    # Fixing nanas up early and seeing what happens
    Z = Elevation
    
    # fixing NANs
    Z[np.where(np.isnan(Z))] = 0
    
    
    #print(Elevation_Binned)
    
    
    ''' '''
    


    
    
    
    #Rows, Columns = np.where(Elevation != NAN) # acquiring values where elvation is defined
    Rows, Columns = np.where(Elevation != 0)
    Z = Z[1:, 1:] # eliminate top tow and leftmost row, nonsense data
    
    #print('Rows:')
    #print(Rows)
    
    #Rows = Rows.reshape(-1, File.width)  # formatting in 2D square grid
    #Columns = Columns.reshape(File.height, -1) # formatting into 2d square grid
    
    '''
    Rows:
    [    0     0     0 ... 10811 10811 10811] # value of each row correpsonding to each point

    Cols:
    [    0     1     2 ... 10809 10810 10811] # value of each column corresponding to each point
    
    
    '''
    
    
    

    
    Lons, Lats = rasterio.transform.xy(TransformMatrix, Rows, Columns) # Converts each point into a longitude and latitiude measurement
    #Z = Elevation[Rows, Columns] # list of Z values (in meters)
    
    print('lons')
    print(Lons)
    
    print('Lats')
    print(Lats)
    
    
    '''  
    Z = Elevation
    
    # fixing NANs
    Z[np.where(np.isnan(Z))] = 0
    
    print()
    print()
    
    
    '''
    
    
    
    X, Y, EPSG_UTM = LonLat_to_XY(Lons, Lats, CRS) # cartesian x and y values associated with each point
    
    print()
    print('X:')
    print(X)
    print('Y:')
    print(Y)
    print('Z:')
    print(Z)
    print()
    print(np.shape(X))
    print(np.shape(Y))
    print(np.shape(Z))
    print()
    
    #X = X.reshape(-1, File.width)  # formatting in 2D square grid
    #Y = Y.reshape(File.height, -1) # formatting into 2d square grid
    
    # Band 2 (Slope)
    if Bands >= 2:
        SlopeAngle = File.read(2)[Rows, Columns] # reads band 2 (slope is measured in degrees)
        Slope = np.tan(np.radians(SlopeAngle)) # Converts slope to fraction
    
    else: # Band 2 data unavailable
        Slope, X_unitV, Y_unitV = Gradient(X, Y, Elevation, TransformMatrix)
        Slope = Slope[Rows, Columns] # Determining Slope at pixels with data
    ###
    
    # Band 3 (Aspect)
    if Bands >= 3:
        AspectAngle = File.read(3)[Rows, Columns] # reads band 3 (direction measured in degrees)
        ''' Aspect is measured in 0 to 360 degrees (0 = North, 90 = East, 180 = South, 270 = West) '''
        X_unitV = -np.cos(np.radians(AspectAngle))
        Y_unitV = -np.sin(np.radians(AspectAngle))
    
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
    
    for index in range(0, len(X)):
        File.write((f"{X[index]:.5f}, {Y[index]:.5f}, {Z[index]:.5f}, {Slope[index]:.5f}, {X_unitV[index]:.5f}, {Y_unitV[index]:.5f} \n"))
        
    File.close()
### END OutputTXT
    
    
X, Y, Z, Slope, X_unitV, Y_unitV = DataRetrieve(40)

print('length X: ' + str(len(X)) + ' | Length Y: ' + str(len(Y)) + ' | Length Z: ' + str(len(Z)))

print('Data Retrieved... Writing to a file...')

OutputTXT(X, Y, Z.flatten(), Slope, X_unitV, Y_unitV, "Out.txt")

print("Done!")












# Quick display #

# Visualize the Elevation Map (Terrain)
plt.figure(figsize=(10, 8))  
plt.imshow(Z, cmap='terrain', interpolation='nearest')  
plt.colorbar(label='Elevation (m)')  
plt.title('Elevation Map (Terrain)')  
plt.xlabel('X (UTM)')
plt.ylabel('Y (UTM)')
plt.show()  







