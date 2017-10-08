# Import some packages from matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
# Uncomment the next line for use in a Jupyter notebook
#%matplotlib inline

# Define the filename, read and plot the image
filename = '/home/ashim/Downloads/sample.jpg'
image = mpimg.imread(filename)
plt.imshow(image)
