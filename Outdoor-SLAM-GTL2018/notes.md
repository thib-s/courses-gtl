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

!
force numpy mat type in python