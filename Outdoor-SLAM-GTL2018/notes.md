02-08 meeting:
==============

work done during last week:
---------------------------

* first version of feature extraction, two variants in c++ and python
+ ros expermientation


split of work:
--------------

for next monday:
=> we need a block diagram for the tasks we need to accomplish
=> messages definitions

- state estimation of the husky with kalman filter (no map)
*- predict part of the kalman (motion prediction)
*- feature extraction
*- feature viz
- map estimation (and then data association)



messages:
---------
- list of features
- map description (full state + full covariance is too big !)

we need visualization for these message


next meeting:
-------------

9:45 next monday (02-12)


02-12 meeting:
==============

Work done: Visualization of the features as markers in rviz. Some glitch are still present but it may be fixed soon.
Also we need to split the data flow to have both features messages and marker output.
The first version of message definition are made, we still need to check for other eventual useful fields.

initialisation:
---------------
start with an empty map, and add feature in the map as you sees it
=> ignore the radius but store it separately for matching (use low pass filter (moyenn glissante exponentielle) as kalman filter simplification on it)

X matrix = map
Z observations

init map with feature position combiened wth robot estimated position

P = matrix of correlation matrix (variance covariance)
        si features indep, bloc diagonal matrix

R = matrix containing the uncertainties on the measurment
so put Pl = JPt-1T^-1+R (J is the jacobian) take the robot position inverytzintoes into account
we can use Pl = Q for the first version

state estimation:
-----------------

common pitfall: needs to linearize the jacobian of f function but not f itelf
P = AtPA + BtQuB + Q
A jacobian of F
Qu incertainty of odom
Q incertainty of thr model


next tasks:
-----------
- landmark adding
- dynamic
- data association
these three steps needs to be done together

___________________
Remark:
force numpy mat type in python

02-19 meeting:
==============

Last week
---------
We successfully finished designing and implementing the feature extraction algorithm. One problem remaining, however, is that we are publishing visualization messages. We should instead publish features, and currently Thibaut is integrating his definition of the feature type into our project.

This week
---------
We will attempt to gain a better understanding of the Kalman Filter and the EKF/SLAM algorithm. I (Kevin) will begin by implementing the Kalman Filter and will meet with Thibaut to get some help. We plan to meet together on Wednesday, and will work on this throughout the week.

02-26 meeting:
==============

Last week
---------

We experimented Kalman filters by doing the autonomous robotic homework. It consist in performing SLAM, on a turtlebot using visual markers. (which avoid the data association problem)

This week:
----------

We now need to transpose the work done on the turtlebot to the Husky, it implies, find the proper functions and compute the jacobians.
We will also need to perform the data association, although it will probably be hard to implement it for the end of the week.
