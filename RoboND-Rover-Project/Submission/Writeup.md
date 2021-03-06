
### October 14, 2017
### Submitted by Ashim Neupane
   





##  Search and Sample Return Project


## Notebook Analysis
### 1. Run the functions provided in the notebook on test images (first with the test data provided, next on data you have recorded). Add/modify functions to allow for color selection of obstacles and rock samples.


- color_thresh() function was modified such that it now takes three parameter, i.e. image, lower and upper threshold for identification of rocksample and obstacle.


```python
    #Lower bound and upper boundRGB threshold values
    rgb_thresh_terrain_l = (160,160,160)   #lower bound values for navigable terrain
    rgb_thresh_rocksample_l = (50,110,0)   #lower bound values for rock sample
    rgb_thresh_rocksample_u = (250,250,75) #upper bound values for rock sample
    rgb_thresh_obstacle_l = (0,0,0)        #lower bound values for obstacles
    rgb_thresh_obstacle_u = (200,100,200)  #upper bound values for obstacle
    
    
def color_thresh(img, rgb_thresh_l=(0, 0, 0), rgb_thresh_u =(255,255,255)): #rbg_thresh_u parameter was added
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    #following lines were modified to extract vlaues that lie within upper & lower threshold values
    above_thresh = (img[:,:,0] >= rgb_thresh_l[0]) & (img[:,:,0] <= rgb_thresh_u[0]) & \
    (img[:,:,1] >= rgb_thresh_l[1]) & (img[:,:,1] <= rgb_thresh_u[1]) & \
    (img[:,:,2] >= rgb_thresh_l[2]) & (img[:,:,2] <= rgb_thresh_u[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select
```

### 2. Populate the process_image() function with the appropriate analysis steps to map pixels identifying navigable terrain, obstacles and rock samples into a worldmap. Run process_image() on your test data using the moviepy functions provided to create video output of your result.

- Following lines wered added inside process_image() function.


```python
# 1) Define source and destination points for perspective transform
    source = np.float32([[13.3064,140.855],[118.468,96.9839],[199.758,96.9839],[302.339,140.855]])
    destination = np.float32([[150,155], [150,145], [160,145 ], [160,155]]) 
    
    # 2) Apply perspective transform
    warped_img = perspect_transform(img,source,destination)
        
    rgb_thresh_terrain_l = (160,160,160)#RGB lower bound values for navigable terrain
    rgb_thresh_rocksample_l = (50,110,0)#lower bound values for rock sample
    rgb_thresh_rocksample_u = (250,250,75)#upper bound values for rock sample
    rgb_thresh_obstacle_l = (0,0,0)#lower bound values for obstacles
    rgb_thresh_obstacle_u = (200,100,200) #upper bound values for obstacle

    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    terrain_img = color_thresh(warped_img, rgb_thresh_terrain_l) * 255
    rocksample_img = color_thresh(warped_img, rgb_thresh_rocksample_l, rgb_thresh_rocksample_u) * 255
    obstacle_img = (1 - np.float32(color_thresh(warped_img, rgb_thresh_terrain_l)))*255
    
    # 4) Convert thresholded image pixel values to rover-centric coords
    x_pixel_t, y_pixel_t = rover_coords(terrain_img)
    x_pixel_r, y_pixel_r = rover_coords(rocksample_img)
    x_pixel_o, y_pixel_o = rover_coords(obstacle_img)
    
    
    # 5) Convert rover-centric pixel values to world coords
    x_world_t, y_world_t = pix_to_world(x_pixel_t, y_pixel_t, data.xpos[data.count], data.ypos[data.count], data.yaw[data.count], data.world_size, data.scale)
    x_world_r, y_world_r = pix_to_world(x_pixel_r, y_pixel_r, data.xpos[data.count], data.ypos[data.count], data.yaw[data.count], data.world_size, data.scale)
    x_world_o, y_world_o = pix_to_world(x_pixel_o, y_pixel_o, data.xpos[data.count], data.ypos[data.count], data.yaw[data.count], data.world_size, data.scale)

    # 6) Update worldmap (to be displayed on right side of screen)
    data.worldmap[y_world_o, x_world_o, 0] += 1
    data.worldmap[y_world_r, x_world_r, 1] += 1
    data.worldmap[y_world_t, x_world_t, 2] += 1
```

- Here, after obtaining binary images that identify each of navigable terrain, rock sample and obstacle, the pixel in the images were converted to rover-centric pixel coordinates using rover_coords() function. x_pixel & y_pixel coordinates for each of the three cases further translated to coordinates in top-down view frame, i.e. world coordinates using pix_to_world() function. These world coordinates were fed to data.worldmap to display them in the video.
- Please find Rover_Project_Test_Notebook.ipynb and test_mapping.mp4 output video  in the directory containing this file:                                                                                                                             


## Autonomous Navigation and Mapping

### 1. Fill in the perception_step() (at the bottom of the perception.py script) and decision_step() (in decision.py) functions in the autonomous mapping scripts and an explanation is provided in the writeup of how and why these functions were modified as they were.



- Within perception_step() function, camera images were transformed to top-down view using perpect_transform() function.
- Next, color_thresh() function was utilized thrice to produce three binary images, each showing naviagble terrain, rock sample and obstacles only.
- Rover.vision_image was updated with these binary_images so that they could be displayed in the video.
- rover_coords() function was used to obtain rover-centric pixel coordinates from the binary images.
- Using pix_to_world() function, rover-centric pixel coordinates were converted to world coordinates in order to build map and identify rover's position in the world map. It also helped to identify position of rock samples in world map.
- Finally, rover-centric pixel coordinates, which were initially in cartesian coordinates, were finally converted to rover-centric polar coordinates using to_polar_coords() to eventually calculate rover steering angle.


### 2. Launching in autonomous mode your rover can navigate and map autonomously. Explain your results and how you might improve them in your writeup.



- The simulation in autonomous mode was conducted with following settings:
- Screen Resolution: 1366 x 768, Graphics quality: Good, Windowed = True, FPS: ~15
- During the autonomous mode, the rover was able to map more than 45% of the actual navigable terrain in 7 out 10 trials.
- Fidelity in these trials ranged between 50% to 60%. 
- No rocks were ever picked by the rover. It is because no changes have been made in decision.py that helps rover to pick golden rocks.
- A couple of times, the rover was stuck in a loop two of the widest regions of the map.
- A couple of time, the rover got stuck on the obstacles. There is a lot of room for obstacle avoidance here. 
- The rover correctly identifies rock samples. However, it does not yet steer towards rock and pick them.
- Hence, much can be done on steer and picking of rocks.
- For this project, I  followed the instructions that were provided to simply pass the project. I was unable to work on decision.py, however, I hope to keep working on this project when time permits to improve the overall perception and decision making for the autonomous navigation of the rover. 
- If I were to pursue this project further I would implement a proportional controller to slow the rover down in the event that it sees a rock. And, to steer it towards the rock, I'd pass angles values of pixles of rock's image in polar coordinates to Rover.nav_angles.
- I'd would work on obstacle avoidance too.
- Please find drive_rover.py, supporting_functions.py, decision.py and perception.py in the directory containig this file.
