# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of several steps. The pipeline was consolidated into one function called "pipeline_draw_lane_lines()". I used the supplied helper functions for convience instead of calling the API functions directly. First, I conformed the image format. If a path to an image was passed in, it was converted to a numpy.ndarray using opencv. If it was already a numpy.ndarray, the function would continue working on it. The image was converted to grayscale, then I gaussian blur was applied. Canny transform was then applied. A mask was applied via region of interest method to only focus on the area we care about (the front of the road we see.) Vertice positions were supplied for the ROI. This was a hyperparameter since it was guess and test. Hough transforms were defined next. The generated lines were then composited back with the original image using the weighted_img() function.

How I modified the draw lines. I checked the slope to see if it was positive or negative. Positive slopes is for the right line, negative slopes are for the left line. The line segments were grouped according to their side. Use the numpy.polyfit to get the minimal least square error to determine the coefficients for a 1 degree equation, thus assuming it is of the form y = mx + b. Solving for the coefficients meant solving for 'm' (slope) and 'b' (coefficient.) These values were saved per image over time because it is a video. The coefficients would only get saved if there was nothing in the list or if the slope was between acceptable values (also an arbitrary hyperparameter set.)

Using numpy.convolution (https://stackoverflow.com/questions/20036663/understanding-numpys-convolve), averaged out the numbers over time for the slope and intercept. The last number was used for the value of the slope or intercept at that moment  for the image being processed. Now that we have the slope and intercept, we start to draw a new line using the fact that img.shape[0] is the y-axis and that the origin is at the top right. This means going positive on y-axis goes down on the image visually. So starting at y = img.shape[0] starts at the bottom of the image. I was able to figure out x by manipulating y = mx + b => x = (y - b) / m, where m is the slope and b is the intercept, which we averaged out already with convolution. This was done for x1 and y1. For x2 and y2, I just took a percentage down from the img.shape[0] for the y, and solve the x with the same formula x = (y - b) / m.

Now I have all the information I need, just need to call cv2.line() twice, once for the left line and once for the right line. This edits function edits the image directly.

### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when the image size is different.

Another shortcoming could be if the line sizes are different, or  if they are not straight.

The values generated were received by tweaking hyper-parameters. Another image source will produce different results.


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to be able to account for lines that are not straight.

Another potential improvement could be to have to set hyperparameters for specific situations.
