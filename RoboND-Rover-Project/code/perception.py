import numpy as np
import cv2

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh_l=(0, 0, 0), rgb_thresh_u =(255,255,255)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    above_thresh = (img[:,:,0] >= rgb_thresh_l[0]) & (img[:,:,0] <= rgb_thresh_u[0]) & \
    (img[:,:,1] >= rgb_thresh_l[1]) & (img[:,:,1] <= rgb_thresh_u[1]) & \
    (img[:,:,2] >= rgb_thresh_l[2]) & (img[:,:,2] <= rgb_thresh_u[2])
    color_select[above_thresh] = 1
    return color_select

# Define a function to convert from image pixel values to rover-centric pixel
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
    bottom_offset = 6
    dst_size = 5
    image = Rover.img
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32([[image.shape[1]/2 - dst_size, image.shape[0] - bottom_offset],
                  [image.shape[1]/2 + dst_size, image.shape[0] - bottom_offset],
                  [image.shape[1]/2 + dst_size, image.shape[0] - 2*dst_size - bottom_offset],
                  [image.shape[1]/2 - dst_size, image.shape[0] - 2*dst_size - bottom_offset],
                  ])

    # 2) Apply perspective transform
    warped_img = perspect_transform(Rover.img, source, destination)

    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    rgb_thresh_terrain_l = (160,160,160)
    rgb_thresh_rocksample_l = (50,110,0)
    rgb_thresh_rocksample_u = (250,250,75)
    rgb_thresh_obstacle_l = (0,0,0) # <-------------fix rgb values!
    rgb_thresh_obstacle_u = (200,100,200) # <-------------fix rgb values!

    terrain_img = color_thresh(warped_img, rgb_thresh_terrain_l)
    rocksample_img = color_thresh(warped_img, rgb_thresh_rocksample_l, rgb_thresh_rocksample_u)
    obstacle_img = color_thresh(warped_img, rgb_thresh_obstacle_l, rgb_thresh_obstacle_u)

    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
    Rover.vision_image[:,:,0] = obstacle_img
    Rover.vision_image[:,:,1] = rocksample_img
    Rover.vision_image[:,:,2] = terrain_img

    # 5) Convert map image pixel values to rover-centric pixel
    x_pixel_t, y_pixel_t = rover_coords(terrain_img)
    x_pixel_r, y_pixel_r = rover_coords(rocksample_img)
    x_pixel_o, y_pixel_o = rover_coords(obstacle_img)

    # 6) Convert rover-centric pixel values to world coordinates
    x_world_t, y_world_t = pix_to_world(x_pixel_t, y_pixel_t, Rover.pos[0], Rover.pos[1], Rover.yaw, Rover.world_size, Rover.scale)
    x_world_r, y_world_r = pix_to_world(x_pixel_r, y_pixel_r, Rover.pos[0], Rover.pos[1], Rover.yaw, Rover.world_size, Rover.scale)
    x_world_o, y_world_o = pix_to_world(x_pixel_o, y_pixel_o, Rover.pos[0], Rover.pos[1], Rover.yaw, Rover.world_size, Rover.scale)

    # 7) Update Rover worldmap (to be displayed on right side of screen)
    Rover.worldmap[y_world_o, x_world_o, 0] += 1
    Rover.worldmap[y_world_r, x_world_r, 1] += 1
    Rover.worldmap[y_world_t, x_world_t, 2] += 1

    # 8) Convert rover-centric pixel positions to polar coordinates
    Rover.nav_dists, Rover.nav_angles = to_polar_coords(x_pixel_t,y_pixel_t)


    return Rover
