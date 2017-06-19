import numpy as np
import cv2

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

def color_obstacle(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] < rgb_thresh[0]) \
                & (img[:,:,1] < rgb_thresh[1]) \
                & (img[:,:,2] < rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

def color_rock(img):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])

    threshold_low = (100, 100, 20)
    threshold_high = (255, 255, 80)

    thresh = (img[:,:,0] > threshold_low[0]) & (img[:,:,0] < threshold_high[0])  \
                & (img[:,:,1] > threshold_low[1]) & (img[:,:,1] < threshold_high[1]) \
                & (img[:,:,2] > threshold_low[2]) & (img[:,:,2] < threshold_high[2])

    # Index the array of zeros with the boolean array and set to 1
    color_select[thresh] = 1

    # Return the binary image
    return color_select

# Define a function to convert from image coords to rover coords
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))
                            
    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result  
    return xpix_rotated, ypix_rotated

def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result  
    return xpix_translated, ypix_translated


# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    
    return warped


# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # TODO: 
    # NOTE: camera image is coming to you in Rover.img
    # 1) Define source and destination points for perspective transform
    image = Rover.img
    dst_size =5
    bottom_offset =6
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32([[image.shape[1]/2 - dst_size, image.shape[0] - bottom_offset],
                              [image.shape[1]/2 + dst_size, image.shape[0] - bottom_offset],
                              [image.shape[1]/2 + dst_size, image.shape[0] - 2*dst_size - bottom_offset],
                              [image.shape[1]/2 - dst_size, image.shape[0] - 2*dst_size - bottom_offset],
                             ])

    # 2) Apply perspective transform
    long_view = perspect_transform(image, source, destination)
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    threshold_terrain  = color_thresh(long_view)
    threshold_obstacle = color_obstacle(long_view)
    threshold_rock = color_rock(long_view)
    
    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
        # Example: Rover.vision_image[:,:,0] = obstacle color-thresholded binary image
        #          Rover.vision_image[:,:,1] = rock_sample color-thresholded binary image
        #          Rover.vision_image[:,:,2] = navigable terrain color-thresholded binary image

    #Rover.terrain = threshold_terrain
    Rover.vision_image[:,:,2] = threshold_terrain * 255
    Rover.vision_image[:,:,1] = threshold_rock * 255
    Rover.vision_image[:,:,0] = threshold_obstacle * 255
    
    # 5) Convert map image pixel values to rover-centric coords
    terrain_xpix, terrain_ypix = rover_coords(threshold_terrain)
    obstacle_xpix, obstacle_ypix = rover_coords(threshold_obstacle)
    rock_xpix, rock_ypix = rover_coords(threshold_rock)
    
    # 6) Convert rover-centric pixel values to world coordinates
    scale = 10    
    worldmap = np.zeros((200, 200))

    # Get navigable pixel positions in world coords
    rover_xpos = Rover.pos[1]
    rover_ypos = Rover.pos[0]
    rover_yaw = Rover.yaw

    # Get navigable pixel positions of terrian in world coords
    terrain_x_world, terrain_y_world = pix_to_world(terrain_xpix, terrain_ypix, rover_xpos, 
                                                    rover_ypos, rover_yaw, 
                                                    worldmap.shape[0], scale)

    # Get navigable pixel positions of obstacles in world coords
    obstacle_x_world, obstacle_y_world = pix_to_world(obstacle_xpix, obstacle_ypix, rover_xpos, 
                                                      rover_ypos, rover_yaw, 
                                                      worldmap.shape[0], scale)

    # Get navigable pixel positions of rocks in world coords
    rock_x_world, rock_y_world = pix_to_world(rock_xpix, rock_ypix, rover_xpos, 
                                              rover_ypos, rover_yaw, 
                                              worldmap.shape[0], scale)



    # 7) Update Rover worldmap (to be displayed on right side of screen)
        # Example: Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
        #          Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
        #          Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1

    # added to improve fidelity
    if Rover.roll < 1.0 or Rover.roll > 359:
        if Rover.pitch < 2.0 or Rover.pitch > 358:
            Rover.worldmap[obstacle_x_world, obstacle_y_world, 0] +=255
            Rover.worldmap[rock_x_world, rock_y_world, 1] +=255
            Rover.worldmap[terrain_x_world, terrain_y_world, 2] +=255
    
    # 8) Convert rover-centric pixel positions to polar coordinates
    # Update Rover pixel distances and angles
        # Rover.nav_dists = rover_centric_pixel_distances
        # Rover.nav_angles = rover_centric_angles
    #Rover.nav_dists, Rover.nav_angles = to_polar_coords(terrain_xpix, terrain_ypix)
    terrain_dist, terrain_angles = to_polar_coords(terrain_xpix, terrain_ypix)
    rocks_dist, rocks_angles = to_polar_coords(rock_xpix, rock_ypix)
    
    Rover.nav_dists = terrain_dist
    Rover.nav_angles = terrain_angles

    Rover.rocks_dists = rocks_dist
    Rover.rocks_angles = rocks_angles
    
    
    return Rover