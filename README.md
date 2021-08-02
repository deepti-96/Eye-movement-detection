# EYE MOVEMENT DETECTION

## Introduction
The technique of monitoring where we look, also known as our point of gaze, is known as eye tracking. Let us begin by examining the eye.

## Analyzing the eye
Before we dive into the intricacies of image processing, let us look at the eye and consider the many options.
An eye may be seen in the image below. The eye is made up of three major components:
• Pupil - the dark circle in the center. <br /> 
• Iris – the larger circle, which might be colored differently for various people. <br /> 
• Sclera - mainly white. <br /> 
![image](https://user-images.githubusercontent.com/72935128/127892959-4d38878a-bdf8-4670-8def-8841a7ecc68b.png)
We use the eye-moving video as a starting point. Then we will consider how to follow the movement afterwards. <br /> 

We import the libraries OpenCV and NumPy, load the movie “eye recording.flv,” and then loop over the frames of the video, processing image by picture.<br /> 
```
import cv2
import NumPy as np
cap = cv2.VideoCapture("eye_recording.flv")
while True:
ret, frame = cap.read()
if ret is False:
break

```
Let us pick a Roi immediately (region of interest). In this approach, we are just detecting the pupil, iris, and sclera, leaving out features like eyelashes and the region around the eye.
```
roi = frame [269: 795, 537: 1416]
rows, cols, _ = roi.shape
gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)

```
Now we can focus on determining the best method for motion detection.
Let us look at all the different directions that the eye may go (in the image below) and see what components are common and unusual among them all.
![image](https://user-images.githubusercontent.com/72935128/127893447-d9a40b56-8719-4279-83e9-59274114f8c3.png)

### What can we conclude from this picture?
Starting on the left, we can observe that the sclera covers the side of the eye opposite the pupil and iris. When the eye is facing straight ahead, the sclera on the left and right sides are evenly distributed.

## Motion detection
We might utilize a variety of methods to identify it, such as concentrating on the sclera, iris, or pupil.
We will take the simplest route possible, which is usually the best option anyhow.
Let us just concentrate on the pupil. We can see that the pupil is always darker than the rest of the eye by changing the image to grayscale format. Regardless of the person's sclera color or where the eye is gazing.
To extract only the pupil, first convert to grayscale and then locate the threshold.
```
gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)
_, threshold = cv2.threshold(gray_roi, 3, 255, cv2.THRESH_BINARY_INV)

```

The contours are obtained from the threshold. And then we just eliminate all the noise by choosing the element with the largest area (which is meant to represent the pupil) and skipping the rest.
```
_, contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted (contours, key=lambda x: cv2.contourArea(x), reverse=True)
for cnt in contours:
(x, y, w, h) = cv2.boundingRect(cnt)
#cv2.drawContours(roi, [cnt], -1, (0, 0, 255), 3)
cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 2)
cv2.line(roi, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 2)
cv2.line(roi, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 2)
break

```
Finally, everything is displayed on the screen.
```
cv2.imshow("Threshold", threshold)
cv2.imshow("gray roi", gray_roi)
cv2.imshow("Roi", roi)
key = cv2.waitKey(30)
if key == 27:
break
cv2.destroyAllWindows()

```
![image](https://user-images.githubusercontent.com/72935128/127894609-ee612526-0294-445b-b405-9b8d0a330c95.png)


