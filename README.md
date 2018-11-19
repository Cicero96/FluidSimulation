# Assignment_3
# 3D Scene Editor [![Build Status](https://travis-ci.com/NYUGraphics/assignment3-Cicero96.svg?token=sEEq1xREVPvVpNvLmKHT&branch=master)](https://travis-ci.com/NYUGraphics/assignment3-Cicero96)

## Overview
In this assignment, all tasks are completed. I will give a brief introduction about the 3D editor in this part, and task related details will be provided in later sections.

First of all, this assignment is very similar to the assignment 2. The only difference is that the scene now is 3D. To support 3D scenes, we should first enable depth test so that elements can be correctly displayed.

    glEnable(GL_DEPTH_TEST);

Also, we have to clear the depth buffer every time.

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

Similar to assignment 2, each element has its own position, translation matrix. Therefore, we can wrap each element and they can have their own VAO and VBO. And then, everything can be calculated on GPU side, and there is no need to change its vertexs' positions.

There are two ways to draw the element, with or without element buffer. In this assignment, element buffer is used. Now we have everything we need to draw an element, and the remaining question is how to draw it. This is related to the coordinate systems. Generally speaking, there are a total of 5 different coordinate systems that are important:

1. Object Space
2. World Space
3. Camera Space
4. Clip Space (Canonical View Volume)
5. Screen Space

The idea is to translate the coordinates in object space to screen space. To achieve this, we need several matrixs. The first matrix is model matrix that translates the coordinates from object space to world space. Each element should have its own model matrix because model matrix will contain the element's scale, rotation information. And this is unique to every element. The rest are camera matrix and projection matrix. These are the same for all the elements. Camera matrix is related to camera's position and projection has two types, i.e. orthographic projection and perspective projection. The last step is to translate everything into screen space and openGL does this for us.

By now, we have everything we need to draw an element and aslo know how to draw an element. Before we go for the details, I would like to present all the keyboard keys used and the operations associated to them.

1. I: Switch into INSERT mode. In INSERT mode, user can add primitives.
2. O: Switch into TRANSLATION mode. In TRANSLATION mode, user can edit the selected element.
3. P: Switch into DELETE mode. In DELETE mode, user can delete the selected element.
4. 1: In INSERT mode, it will add an unit cube.
5. 2: In INSERT mode, it will add the bumpy cube and the bumpy cube is scaled into a unit cube.
6. 3: In INSERT mode, it will add the bunny and the bunny is scaled into a unit cube.
7. 4 & 5: In TRANSLATION mode, it will move the selected element by 0.1 along the x or -x axis.
8. 6 & 7: In TRANSLATION mode, it will move the selected element by 0.1 along the y or -y axis.
9. 8 & 9: In TRANSLATION mode, it will move the selected element by 0.1 along the z or -z axis.
10. Z & X: In TRANSLATION mode, it will scale up or down the selected element by 25%.
11. C & V: In TRANSLATION mode, it will rotate the selected element by 10 degree counter clockwise or clockwise.
12. W & S: Move the camera in the trackball. Move up or down by 10 degree.
13. A & D: Move the camera in the trackball. Move left or right by 10 degree.
14. R & F: Change the radius of the trackball. Increase or decrease the radius by 0.5.
15. Q: Switch between orthographic view and perspective view. Default is perspective view.
16. E: Switch between phong shading -> flat shading -> wireframe. Default is phong shading.
17. T: Add a control point for the bezier curve which is owned by the selected element.
18. Y: Switch between all the control points of the bezier curve.
19. U: Change the translation mode. Will affect the behavior of key 4-9. 4-9 will translate the position of the selected element by default. If switched, it will translate the position of the selected control point of bezier curve.
20. B: Export the current scene in SVG format.
21. \- & +: Scale up or down the scene by 25%.
22. Space: Play Animation.


## 1.1 Scene Editor
In this assignment, we should support add, edit and delete operations for three kinds of primitives, which are unit cube, bumpy cube and bunny. Let's start from the easiest one, add a primitive.

To add an element, we first should load everything we need. For unit cube, we will create all the coordinates and the element buffer by ourself. For bumpy cube and bunny, we will read what we need from the OFF file. Each element is wrapped by the class *Element*. To draw it, we should construct all the mentioned matrixs.

Frist of all, the model matrix. To begin with, we should scale the bumpy cube and the bunny into a unit cube. To achieve this, we record the max and min value of x-axis, y-axis and z-axis coordinates seperately. With these values, we can find a rectangle that contains the element. And we can scale this rectangle into a unit cube. When scale or rotate the selected element, we should first move it to the origin, do the translation, and move it back. These should already be covered by the previous assignments.

Then, the camera matrix. First, we need the camera position and camera will always face the origin. With this information, we will construct three vector that build the camera coordinate system. The first vector is *cameraDirection*. Its direction is opposite to the gaze direction and the gaze direction is camera position to the origin. The second vector is *cameraRight* and this is more complicated since we implemented the trackball which causes some cornor cases. Let's split the world into two parts by the plane XoY. In +z-axis part, the view up vector should be *(0, 1, 0)*, and in -z-axis part, the view up vector shuold be *(0, -1, 0)*. And, if the camera looks from the top or the bottom, since the view up and the *cameraDirection* are parallel, therefore we cannot direct use cross multiply. For these two cases, we will use camera's projected position in XoZ plane. Once we computed the *cameraRight*, we can calculated the *cameraUp* by a cross multiply. With these three vectors and camera position, we can get the camera matrix according to the lecture materials.

Finally, the projection matrix. This space is also called clip space, because in this space we will decide what can be seen. For the orthographic projection, all coordinates in camera space are linearly mapped to Normalized Device Coordinates(NDC). We just need to scale a rectangular volume to a cube, then move it to the origin. And this is very similar to what we did for the bumpy cube and bunny to scale them into a unit cube.

For the perspective projection, we actually are mapping points in a truncated pyramid frustum to a cube. The range of x-coordinate from [l, r] to [-1, 1], the y-coordinate from [b, t] to [-1, 1] and the z-coordinate from [-n, -f] to [-1, 1]. With l, r, b, t, n and f, we can use this matrix as perspective matrix.

<img src="https://github.com/NYUGraphics/assignment3-Cicero96/raw/master/result/perspective.png" width="200" height="200" align=center/>

And now, the element can be correctly showed in the screen.

<img src="https://github.com/NYUGraphics/assignment3-Cicero96/raw/master/result/1.0.png" width="200" height="200" align=left/>
<img src="https://github.com/NYUGraphics/assignment3-Cicero96/raw/master/result/1.1.png" width="200" height="200" align=center/>
<img src="https://github.com/NYUGraphics/assignment3-Cicero96/raw/master/result/1.2.png" width="200" height="200" align=right/>
<img src="https://github.com/NYUGraphics/assignment3-Cicero96/raw/master/result/1.3.png" width="200" height="200" align=left/>

After adding elements, now we should support edit elements. The key problem is how to select an element precisely. When user clicks the screen, we should transform this 2D coordinates in screen space to 3D coordinates in world space. First of all, we can translate this 2D coordinates to the point in the near plane. After this, we get a 3D point in camera view. If we multiply the inverse of camera matrix, then we translate it into the world space. And now, with the camera position and the click point, we can reuse the code in assignment 1 to check intersection with each element and find the nearest element. Once selected, we can move it, scale it and rotate it. Also we can delete element.

To distinguish between the selected element and other elements, the selected one has red color.

<img src="https://github.com/NYUGraphics/assignment3-Cicero96/raw/master/result/1.4.png" width="200" height="200" align=left/>

## 1.2 Object Control
In this assignment, translation part is already covered in 1.1. So I will focus on the shading part. The default shading is phong shading. To achieve phong shading, we should calculate all faces' normal and average it to the vertexs. For flat shading, I draw triangle by triangle and each triangle's color is determined by their vertexs. Each vertexs now will use the same normal which is this triangle's normal. For wireframe, I only draw the line loop.

<img src="https://github.com/NYUGraphics/assignment3-Cicero96/raw/master/result/2.0.png" width="200" height="200" align=left/>
<img src="https://github.com/NYUGraphics/assignment3-Cicero96/raw/master/result/2.1.png" width="200" height="200" align=center/>
<img src="https://github.com/NYUGraphics/assignment3-Cicero96/raw/master/result/2.2.png" width="200" height="200" align=right/>

## 1.3 Camera Control
In this assignment, to achieve the translation of camera, I implemented the trackball. For the trackball, I need three parameters, *radius, theta and phi*. *Radius* is the distance between camera and origin. *Theta* is the angle between camera and the +z-axis. *Phi* is the angle between camera and XoZ plane. Now camera's coordinates can be calculated like this:

    y = radius * sin((double)phi / 180 * pi);
    x = radius * cos((double)phi / 180 * pi) * sin((double)theta / 180 * pi);
    z = radius * cos((double)phi / 180 * pi) * cos((double)theta / 180 * pi);

For perspective view and orthographic view, we have to calculate different projection matrix as mentioned in the 1.1. Also, a matrix is maintained for the scene view. User can zoom up or down the window, also the size of window can be changed. And the aspect ratio is adapted so that the element will not be distorted.

<img src="https://github.com/NYUGraphics/assignment3-Cicero96/raw/master/result/3.0.png" width="200" height="200" align=left/>
<img src="https://github.com/NYUGraphics/assignment3-Cicero96/raw/master/result/3.1.png" width="200" height="200" align=center/>
<img src="https://github.com/NYUGraphics/assignment3-Cicero96/raw/master/result/3.2.png" width="200" height="200" align=right/>

<img src="https://github.com/NYUGraphics/assignment3-Cicero96/raw/master/result/3.3.png" width="200" height="200" align=left/>
<img src="https://github.com/NYUGraphics/assignment3-Cicero96/raw/master/result/3.4.png" width="200" height="200" align=center/>

<img src="https://github.com/NYUGraphics/assignment3-Cicero96/raw/master/result/3.5.png" width="200" height="200" align=left/>

## Optional Tasks
## 1.4 Animation Mode
First of all, each element can have a bezier curve attached to it. User can press *T* to add a control point and use 4-9 to change the position of selected control point. These parts are coverd in Overview section. With control points, the curve will be evaluated using De Casteljau’s algorithm. When user presses space, it will start the animation. I have 100 points on curve, therefore I will update element's position by 0.05s.

<img src="https://github.com/NYUGraphics/assignment3-Cicero96/raw/master/result/4.0.png" width="200" height="200" align=left/>
<img src="https://github.com/NYUGraphics/assignment3-Cicero96/raw/master/result/4.1.gif" width="200" height="200" align=left/>

## 1.5 Export in SVG format
For this assignment, first we should use all the matrixs to translate all the coordinates from object space into screen space. And then, we should check whether a triangle should be drawed. To check this, I cast a ray from camera to the barycenter of the triangle. If no other triangles are intersected, we should draw this triangle. To determine the color of this triangle, we should determine the color of three vertexs. Once this is done, I can reused the code in previous assignment to export a svg.

<img src="https://github.com/NYUGraphics/assignment3-Cicero96/raw/master/result/5.0.png" width="200" height="200" align=left/>

## 1.6 Trackball
Already mentioned in 1.3
<img src="https://github.com/NYUGraphics/assignment3-Cicero96/raw/master/result/6.0.gif" width="200" height="200" align=left/>