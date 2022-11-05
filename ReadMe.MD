# Control Methods on a Kinova robot

### Description
Main purpose of this repository is to get familiar with the [raisim simulation engine](https://raisim.com/index.html). A variety of control methods is tested on the [kinova 6 link manipulator](https://www.kinovarobotics.com/product/gen2-robots) and visualized using the Raisim engine.


### Running project
* Python version used: 3.10
* The raisim build libraries in the [raisim_build](raisim_build) directory was compile by an M1 apple laptop. You may need to re-build them for your own computer. Instructions on that can be found on the official [raisim website](https://raisim.com/sections/Installation.html).
* The [raisim_build](raisim_build/lib) directory contains the raisim libraries and python binding files. Make sure the [python binding directory](raisim_build/lib/python3.10/site-packages) is added to your python path. I always prefer to create a virtual ennvironemnt and then add an extras.pth file with the binding directory written in there.
* If you are using a smart IDE like one by Jetbrains, you can add the python binding path to your environment's path to enable indexing of the bindings.
* Another point

### Project Highlights
* Using the Raisim simulationo engine for visualization
* Using Pytorch automatic differentiation to solve for the Jacobian of the manipulator
*

### Whole Body Obstacle avoidance