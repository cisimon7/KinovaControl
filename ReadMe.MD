# Using an LQR to control a Kinova robot

### Description
Main purpose of this repository is to get familiar with the [raisim simulation engine](https://raisim.com/index.html). An LQR is used to control a kinova 6 link manipulator and visualozed using the Raisim engine.


### Running project
* Python version used: 3.10
* The [raisim_build](raisim_build/lib) directory contains the raisim libraries and python binding files. Make sure the [python binding directory](raisim_build/lib/python3.10/site-packages) is added to your python path. I always prefer to create a virtual ennvironemnt and then add an extras.pth file with the binding directory written in there.
* If you are using a smart IDE like one by Jetbrains, you can add the python binding path to your environment's path to enable indexing of the bindings.
* Another point