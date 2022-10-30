import os
import time
import numpy as np
import raisimpy as raisim


world = raisim.World()
world.setTimeStep(0.001)

kinova_urdf_file = os.path.dirname(os.path.abspath(__file__)) + "/rsc/kinova/urdf/kinova.urdf"
ground = world.addGround()
kinova = world.addArticulatedSystem(kinova_urdf_file)
kinova.setName("kinova")

joint_nomical_config = np.array([0.0, 2.76, -1.57, 0.0, 2.0, 0.0])
joint_velocity_target = np.zeros(kinova.getDOF())

joint_Pgain = np.array([40.0, 40.0, 40.0, 15.0, 15.0, 15.0])
joint_Dgain = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])

kinova.setGeneralizedCoordinate(joint_nomical_config)
kinova.setGeneralizedForce(np.zeros(kinova.getDOF()))

kinova.setPdGains(joint_Pgain, joint_Dgain)
kinova.setPdTarget(joint_nomical_config, joint_velocity_target)

server = raisim.RaisimServer(world)
server.launchServer(8080)
server.focusOn(kinova)

for i in range(500_000):
    time.sleep(0.001)
    server.integrateWorldThreadSafe()

server.killServer()
