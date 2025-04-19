###
'''
Author: Thomas Joyce

Version: 1.1

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
    Description: Converts the longitudes and latitudes of the raw Geotiff data intoa projected cartesian axis of X (east)
    and Y (north)
    
    Inputs: 
        - Longitudes: list of floats, list of longitude float points obtained from the Geotiff file (raw data) to be converted
        - Latitudes: list of floats, list of latitudes float points obtained from the Geotiff file (raw data) to be converted
        - CRS: string, coordinate reference system (geographic or projected), often comes in the form of an EPSG code    
    Returns:
        - X: list of floats, range of x points as a 1-dimensional list
        - Y: list of floats, range of y points as a 1-dimensional list
        - EPSG_UTM: string, EPSG zone used for longitude identification. This code is needed to compute the spherical geometry
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


def Gradient(X, Y, Elevation):
    ''' 
    Description: Computes the gradient (slope) of the cloud point data at each point and returns the magnitude and 
    direction of the gradient at each defined point
        
    Inputs:
        - Elevation: array (2 x 2) of floats, range of z points as a 2-dimensional array of square structure
    
    Returns:
        - Slope: list of floats, range of fractional slope values associated with each data point
        - X_unitV: list of floats, range of x-directional unit vectors associated with the direction of slope at each point
        - Y_unitV: list of floats, range of y-directional unit vectors associated with the direction of slope at each point
    '''
    
    # Reshaping X and Y Data #
    Height, Width = Elevation.shape
    
    X_2d = np.reshape(X, (Height, Width))
    Y_2d = np.reshape(Y, (Height, Width))
    
    
    # Computing Absolute Differences Between Points #
    dx = np.diff(X_2d, axis=1)  # Differences along x-axis (vertical column differences)
    dy = np.diff(Y_2d, axis=0)  # Differences along y-axis (horizontal row differences)
    
    # Pad to match the original shape of Elevation
    dx = np.pad(dx, ((0, 0), (0, 1)), mode='edge')  # Pad right side (last column) to match shape of Elevation
    dy = np.pad(dy, ((0, 1), (0, 0)), mode='edge')  # Pad bottom side (last row) to match shape of Elevation
    # NOTE: very last point is "simulated" however it is one point int the entire grid, located in a bottom right corner 
    
    # Finding Elevation Gradients #
    dz_x, dz_y = np.gradient(Elevation, edge_order=1) # gradient of the elvation matrix
    
    dz_dx = ((dz_x * 2) / dx) / np.median(dx) # x non-uniform seperation correction
    dz_dy = ((dz_y * 2) / dy) / np.median(dy) # y non-uniform seperation correction
    '''
    Due to the raster point being non-uniformly spaced (since we are projecting of a plane unto a sphere)
    then we must use the np.gradient function to find the z-seperation in two dimensions:
        
        dz = ( z[i+1] - z[1] ) / 2  , for both dimensions
        
    Then we must weight this by the seperation in both x and y (dx and dy) by the correction factor above
    giving us:
        
        dz/dx = ( z[i+1] - z[i] ) / ( x[i+1] - x[i] )
        dz/dy = ( z[i+1] - z[i] ) / ( y[i+1] - y[i] )
        
    The additional division of the median spacing (np.median(dx or dy) renormalizes the points to the 
    true fractional slope, where if tan^-1(Slope) was taken it would give the angle of the land at that 
    particular point. This way we can accurately model the terrain. 
    '''

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
    Description: Retrieves the data encoded in the raster image (Geotiff file) and returns the 
    6 elements needed for documentation of the cloud point map (positions, slope, and slope direction)
        
    Inputs:
        - BinMode: string, variable description which bin mode is used when compressing the data
            'Sampling' takes every BinFactor point and discards the rest
            'Median' takes the median over a grid of side length BinFactor and saves the point as a median of the local terrain
        - BinFactor: interger, value describing the number of pixels (data points) in each bin
    
    Returns:
        - X: list of floats, range of x points as a 1-dimensional list
        - Y: list of floats, range of y points as a 1-dimensional list
        - Z: array (2 x 2) of floats, range of z points as a 2-dimensional array of square structure
        - Slope: list of floats, range of fractional slope values associated with each data point
        - X_unitV: list of floats, range of x-directional unit vectors associated with the direction of slope at each point
        - Y_unitV: list of floats, range of y-directional unit vectors associated with the direction of slope at each point
        '''
    
    ### File Selection ###
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
        Elevation_Binned = [] # going to be an array soon
    
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
    
    elif BinMode == 'Median': # Takes a median of every square (side length = BinFactor) set of points
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
        
    ##### ----- #####
    
    else:
        Z = Elevation
    ###
    
    # Updating all NAN points to zero (adjustments for nonsense data carried through calculations)
    Z[np.where(np.isnan(Z))] = 0
    
    # Image Correction #
    Z = Z[1:, 1:] # eliminate top tow and leftmost row, nonsense data
    
    # Acquiring row and column values where data is defined
    Rows, Columns = np.where(Z != 0)
    
    # Longitude, Latitiude to X,Y (meters) Conversion #
    Lons, Lats = rasterio.transform.xy(TransformMatrix, Rows, Columns) # Converts each point into a longitude and latitiude measurement
    X, Y, EPSG_UTM = LonLat_to_XY(Lons, Lats, CRS) # cartesian x and y values associated with each point
    
    
    # Band 2 (Slope)
    if Bands >= 2:
        SlopeAngle = File.read(2)[Rows, Columns] # reads band 2 (slope is measured in degrees)
        Slope = np.tan(np.radians(SlopeAngle)) # Converts slope to fraction
    
    else: # Band 2 data unavailable
        Slope, X_unitV, Y_unitV = Gradient(X, Y, Z)
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

    return X, Y, Z, Slope, X_unitV, Y_unitV, File.meta
### END DataRetrieve
    

def MakeHeader(Metadata, BinFactor, BinMode, Z_width, Z_height):
    '''
    Description: Intakes the original file meta data and custom header variables to write a master header string 
    to be written to the file. 
    
    Inputs:
        - Metadata: dictionary, encoded JSON file metadata
        - BinFactor: interger, value describing the number of pixels (data points) in each bin
        - BinMode: string, variable description which bin mode is used when compressing the data, see DataRetrieve for more details
        - Z_width: interger, width of the new binned elevation array
        - Z_height: interger, height of the new binned elevation array
    
    Returns: 
        - Header: string, large string to be writen as the custom header for the output file
    '''
    
    # Initilaizing Output Variable #
    Header = ''
    
    # Determining Length of Dictionary #
    MetaLength = len(Metadata)
    
    # Adding each component of custom header #
    Header += f'# Header Length: {MetaLength + 9} \n' # +8 to account for custom header my additions
    Header += '# ---------- Geotiff Conversion ---------- # \n'
    Header += f'# Bin Height: {Z_height} \n'
    Header += f'# Bin Width: {Z_width} \n'
    Header += f'# Bin Mode Used: {BinMode} \n'
    Header += f'# Bin Factor Used: {BinFactor} \n'
    Header += '# Units: X-Position = m , Y-Position = m , Z-Position = m , Fractional Slope = unitless, Slope Unit Vector = unitless , Slope Unit Vector = unitless \n'
    Header += '# ---------- Original File Metadata ---------- # \n'
    
    # Building metadata header based upon dictionary entries #
    MetaHeader = ''
    
    for key, value in Metadata.items():
        if key != 'transform': # excluding multiline transform matrix ( it is no longer needed after this step)
            MetaHeaderLine = f'# {key}:{value} \n'
            MetaHeader += MetaHeaderLine
        
    Header += MetaHeader
        
    Header += '# ---------- END HEADER ---------- # \n'
    
    return Header
### END ReadMetaData


    
def OutputFile(X, Y, Z, Slope, X_unitV, Y_unitV, Location, BinFactor, BinMode, OutputMode, Metadata):
    '''
    Description: Saves the converted Geotiff data to a txt file for easier reading and manipulation from other
    programs
    
    Inputs:
        - X: list of floats, range of x points as a 1-dimensional list
        - Y: list of floats, range of y points as a 1-dimensional list
        - Z: array (2 x 2) of floats, range of z points as a 2-dimensional array of square structure
        - Slope: list of floats, range of fractional slope values associated with each data point
        - X_unitV: list of floats, range of x-directional unit vectors associated with the direction of slope at each point
        - Y_unitV: list of floats, range of y-directional unit vectors associated with the direction of slope at each point
        - Location: string, name of location being displayed (nearest city / landmark)
        - BinFactor: interger, value describing the number of pixels (data points) in each bin
        - BinMode: string, variable description which bin mode is used when compressing the data, see DataRetrieve for more details
        - OutputMode: string, name of ouput mode being considered for the output file
            'OptimalControl' places a metadata header at the beginning of the file along with column designators. The data takes the form:
                ex) X-Position (m) | Y-Position (m) | Z-Position (m) | Fractional Slope | Slope Unit Vector (X) | Slope Unit Vector (Y)
            '3DPrint' takes the augmented x,y,z values into an ASCII file for importation into meshlab
                ex) X-Position (m) | Y-Position (m) | Z-Position (m)
        - Metadata: dictionary, encoded JSON file metadata
    '''
    
    # Flattening Z array and determining shape #
    Z_width, Z_height = np.shape(Z)
    Z = Z.flatten()
    
    
    if OutputMode.lower() == 'optimalcontrol':
        File = open(f'{OutputMode}Elevation-{Location}_Bin{BinFactor}-{BinMode}.txt', 'w')
        
        # Writing Custom Header #
        Header = MakeHeader(Metadata, BinFactor, BinMode, Z_width, Z_height)
        File.write(Header)
        
        # Writing numpy readable title (for column names) #
        File.write('# XPosition, YPosition, ZPosition, FractionalSlope, XSlopeUnitVector, YSlopeUnitVector # \n')
    
        for index in range(0, len(X)): # Iterating through each point and writing data to file
            File.write((f"{X[index]:.5f}, {Y[index]:.5f}, {Z[index]:.5f}, {Slope[index]:.5f}, {X_unitV[index]:.5f}, {Y_unitV[index]:.5f} \n")) # 5 decimal float values
            print('Iteration: ' + str(index + 1) + ' / ' + str(len(X)))
        ###
    ###
    
    if OutputMode.lower() == '3dprint':
        File = open(f'{OutputMode}Elevation-{Location}_Bin{BinFactor}-{BinMode}.asc', 'w')
        
        for index in range(0, len(X)): # Iterating through each point and writing data to file
            File.write((f"{X[index]:.5f}, {Y[index]:.5f}, {Z[index]:.5f} \n")) # 5 decimal float values
        ###
    ###
            
        
    File.close()
### END OutputTXT
    

def PlotElevation(Z, BinMode, BinFactor, Location):
    ''' 
    Description: Plots and saves a figure of the elevation of a given location for easier visualization
    
    Inputs:
        - Z: array (2 x 2) of floats, range of z points as a 2-dimensional array of square structure
        - BinMode: string, variable description which bin mode is used when compressing the data, see DataRetrieve for more details
        - BinFactor: interger, value describing the number of pixels (data points) in each bin
        - Location: string, name of location being displayed (nearest city / landmark)
    '''
    
    # Figure #
    plt.figure(figsize=(8, 8))
    
    # Plotting #
    plt.imshow(Z, cmap='terrain', interpolation='nearest')  
    
    # Plot Formatting #
    plt.colorbar(label='Elevation (m)')  
    
    plt.title(f'Elevation Map of {Location}')  
    plt.xlabel('X-Position')
    plt.ylabel('Y-Position')
    
    plt.tight_layout()
    
    # Saving Figure #
    plt.savefig(f'Elevation-{Location}_Bin{BinFactor}-{BinMode}.png')
### END PlotElevation




### ----- # Main Function # ----- ###
def MAIN():
    '''
    Description: Executes the program by retrieving the Geotiff data and formatting it correctly for an
    output txt file. PLEASE FILL OUT USER INPUTS HERE!
    '''
    
    ############### User Inputs ###############
    BinFactor = 4 # interger, value describing the number of pixels (data points) in each bin
    BinMode = 'Median' # string, variable description which bin mode is used when compressing the data, see DataRetrieve for more details
    Location = 'Pittsburgh' # string, name of location being displayed (nearest city / landmark)
    OutputMode = 'OptimalControl' # string, name of ouput mode being considered for the output file (see OutputFile for more details)
    ############### ----------- ###############

    # Quick correction if you forget to adjust BinFactor #
    if BinMode.lower() == 'none': 
        BinFactor = 1

    X, Y, Z, Slope, X_unitV, Y_unitV, Metadata = DataRetrieve(BinMode, BinFactor)

    # testing #
    #print('length X: ' + str(len(X)) + ' | Length Y: ' + str(len(Y)) + ' | Length Z: ' + str(len(Z) * len(Z[0])))
    #print(f'length Slope: {len(Slope)} | length X_unitV: {len(X_unitV)} | length Y_unitV: {len(Y_unitV)}')

    #print('Data Retrieved... Writing to a file...')

    OutputFile(X, Y, Z, Slope, X_unitV, Y_unitV, Location, BinFactor, BinMode, OutputMode, Metadata)
    
    PlotElevation(Z, BinMode, BinFactor, Location)
    
    

    #print("Done!")
    
### END MAIN


### ----- # Execution # ----- ###
MAIN()
#if __name__ == '__MAIN__':
#    MAIN()