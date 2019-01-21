## CarND-P2 : Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistorted_chessboard.png "undistorted_chessboard"
[image2]: ./output_images/undistorted_images.png "Road Transformed"
[image3]: ./output_images/combo_threshold_images.png "Binary Example"
[image4]: ./output_images/perspective_transformation_images.png "Warp Example"
[image5]: ./output_images/lane_finding_images.png "Fit Visual"
[image6]: ./output_images/final_images.png "Output"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  All the codes follow the steps above and are included in "./P2.ipynb". There are comments before each step, so it will be easy to locate the codes for a specific one.

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.
 
I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

---

### Pipeline (tested images)
To show the performance of my pipeline in different conditions, I provide results of all tested_images for each step.

#### 1. Provide an example of a distortion-corrected image.

Using the camera calibration and distortion coefficients obtained from the previous step, I apply the undistortion operation to all test images, the results are: 
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
Six types of thresholding functions are tested:
* Sobelx and Sobely function: first convert the image into grayscale using `cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)` function. Then calculate the image gradient along the x axis or the y axis using `cv2.Sobel` function. After this, take the absolute value of the gradient and scale the gradient to 8 bit(0-255). Finally apply a upper and lower thresholding to obtain the binary image.
* Sobel magnitude and direction function: First obtain the sobelx and sobely gradient. Then calculate the magnitude and direction of the vector formed by (sobelx, sobely). Finally apply an upper and lower thresholding to obtain the binary image.
* HSL H channel and S channel function: First convert the image into HSL color space using `cv2.cvtColor(img,cv2.COLOR_RGB2HLS)`. Then for the H channel or S channel, apply thresholding to obtain the corresponding binary image.

The observations I get from the tests are: 
* As the lanes are nearly vertical, the Sobelx transform performs better than the Sobely transform. The Sobely transform detects many fake horizontal lane lines and will disturb the following steps. 
* The combined magnitude and direction thresholding function has similar performance to the Sobelx thresholding function. Both of them can obtain good results after careful parameter tuning.
* The HSL S channel thresholding function can work well on most test images except when there is large area of shadow on the image. However, the shadow can be properly filtered by the HSL H thresholding function. 
* The HSL functions can detect color better while the Sobel gradient functions are good at finding edges. They can be combined to get more complete lane detection results.

Based on the above observations, my final choice is to integrate Sobelx function and HSL H and S function into the final combination function: thresholded_binary = Sobelx OR (HSL_h AND HSL_s). 
The thresholded binary image I get is like the following. The combination function can clearly mark the left and right lanes on the images.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The `cv2.getPerspectiveTransform()` function takes the source (`src`) and destination (`dst`) points as inputs and output the transformation matrix M.  I chose to select the source and destination points in the following manner:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 210, 719      | 320, 719        | 
| 597, 450      | 320, 0      |
| 683, 450     | 960, 0      |
| 1108,719      | 960, 719        |

Then the inverse matrix of M, Minv is calculated using `numpy.linalg.inv`. Minv will be used in Step 6 to warp the detected lane boundaries back onto the original image.

After M and Minv are calculated, I apply a perspective transform to all test binary images obtained from the previous step using `cv2.warpPerspective`. My perspective transform is working as expected because the straight lines in the second and third images are transformed into vertical lines in the bird-eye view. 

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_videos/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Unfortunately, my current pipeline failed in the challenge_video and the harder_challenge_video. 

* **Shortcomings:**

(1) In the challenge_video, two black color lines exist on the road and the algorithms detect them as the lane lines uncorrectly.

(2) In the harder_challenge_video, the light condition changes rapidly; the curvature of the lane is large; and the bushes close to the road disturb the pipeline severely

* **Improvements:**

(1) More work should be put into tuning the color and gradient thresholding function to reduce disturbances as much as possible. 

(2) The sliding window search used in the current pipeline need to be modified to adapt to larger curvature of the lane. 

(3) The current sanity check only involves checking that the left and right lanes are roughly parallel. More sanity checks, like the difference in the line fit between frames should be considered.
