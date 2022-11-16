import torch as th
import numpy as np
from Kinova6LinksModel import Kinova6Links
from torch.utils.benchmark import Timer
from BenchMarker import *

if __name__ == '__main__':
    config = th.rand(6)
    vel = th.rand(6)
    manipulator = Kinova6Links()

    duration, value = time_it(
        lambda: manipulator.pos_vel(config, vel)
    )
    print(f"Duration:\n {duration}\nValue:\n {value}\n")

    duration, value = time_it(
        lambda: (
            th.vstack(manipulator.forward_kinematics(config)),
            manipulator.jacobian(config) @ vel
        )
    )
    print(f"Duration:\n {duration}\nValue:\n {value}")
