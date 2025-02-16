Include a README file describing how to run your code, the methods chosen,
results, and observations.

## How to run the code:
First install the following via conda or pip(with a virtual environment setup):
'''
conda install -c conda-forge opencv
conda install matplotlib
conda install numpy
'''

Then simply run code via "python3 part1.py" or "python3 part2.py"
Note that the folders images_part1 and images_part2 contain the relevant images.
- "coins.jpeg" is the input for part1, the outputs are "a_canny.jpeg" and "segmented_output.jpeg"
- "first.jpeg" and "second.jpeg" are the inputs for part2; the output is "output.jpeg"


## Methods Chosen
### part1
a. Canny Edge Detector
b. K means and Watershed Algorithm
c. Function call(see code)

### part2
The standard image stitching algorithm is used.
Steps:
- Basic Image Processing
- Feature Detection and Computing Descriptors
- Feature Matching
- Computing Homography
- Transform first image into plane of second image
- Blend the images to make it as seemless as possible


## Results and Observations
In part1 watershed performs the best and even though the coins are sticking together, it is able to differentiate between them and count the number of coins.
Canny edge detector needs it's thresholds to be chosen carefully in order for it to work on certain images.
K means in part1 can help separate the foreground from the background, but suffers when used on the coins as the coins are quite similar.
The choice of k also remains a question.

In part2 the output is almost seamless except for a few black lines as the intersection points of the two images.


For more detail refer to the report