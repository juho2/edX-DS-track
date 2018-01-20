import os
import pandas as pd

from scipy import misc
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap

from assignment4 import Plot2D, Plot3D

# Look pretty...
# matplotlib.style.use('ggplot')
plt.style.use('ggplot')

# TODO: Start by creating a regular old, plain, "vanilla"
# python list. You can call it 'samples'.
samples = []
colors = []
# TODO: Write a for-loop that iterates over the images in the
# Module4/Datasets/ALOI/32/ folder, appending each of them to
# your list. Each .PNG image should first be loaded into a
# temporary NDArray, just as shown in the Feature
# Representation reading.
path = r'.\Datasets\ALOI\32'
for file in os.listdir(path):
    img = misc.imread(os.path.join(path, file))
    samples.append(img.flatten())
    colors.append('b')

path = r'.\Datasets\ALOI\32i'
for file in os.listdir(path):
    img = misc.imread(os.path.join(path, file))
    samples.append(img.flatten())
    colors.append('r')
    
df = pd.DataFrame(samples)
iso = Isomap(n_neighbors=6, n_components=3, n_jobs=-1)
T = iso.fit_transform(df)

fig = plt.figure()
ax = fig.add_subplot(221)
ax.set_color_cycle(colors)
ax.scatter(T[:,0],T[:,1], c=colors)
ax = fig.add_subplot(222)
ax.set_color_cycle(colors)
ax.scatter(T[:,0],T[:,2], c=colors)
ax = fig.add_subplot(223)
ax.set_color_cycle(colors)
ax.scatter(T[:,1],T[:,1], c=colors)
ax = fig.add_subplot(224)
ax.set_color_cycle(colors)
ax.scatter(T[:,1],T[:,2], c=colors)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.set_color_cycle(colors)
ax.scatter(T[:,0],T[:,1],T[:,2], c=colors, alpha=0.7)

# Optional: Resample the image down by a factor of two if you
# have a slower computer. You can also convert the image from
# 0-255  to  0.0-1.0  if you'd like, but that will have no
# effect on the algorithm's results.

# TODO: Once you're done answering the first three questions,
# right before you converted your list to a dataframe, add in
# additional code which also appends to your list the images
# in the Module4/Datasets/ALOI/32_i directory. Re-run your
# assignment and answer the final question below.
#
# .. your code here .. 


#
# TODO: Convert the list to a dataframe
#
# .. your code here .. 



#
# TODO: Implement Isomap here. Reduce the dataframe df down
# to three components, using K=6 for your neighborhood size
#
# .. your code here .. 



#
# TODO: Create a 2D Scatter plot to graph your manifold. You
# can use either 'o' or '.' as your marker. Graph the first two
# isomap components
#
# .. your code here .. 




#
# TODO: Create a 3D Scatter plot to graph your manifold. You
# can use either 'o' or '.' as your marker:
#
# .. your code here .. 



plt.show()

