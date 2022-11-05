from typing import List, Tuple
from Transformations import *
from functools import reduce
from torch import Tensor
import torch as th


class Kinova6Links:
    def __init__(self, config=Tensor([0 for _ in range(6)])):
        self.config = config
        self.joints = None

    @staticmethod
    def __forward_kinematics(config: Tensor) -> Tuple[Tensor, List[Tensor]]:
        # A pure function without any side effects that can be differentiated

        q1, q2, q3, q4, q5, q6 = config

        T01: Tensor = TTz(Tensor([0.333])) @ TRz(q1)
        T12: Tensor = TRy(q2)
        T23: Tensor = TTz(Tensor([0.316])) @ TRz(q3)
        T34: Tensor = TTx(Tensor([0.0825])) @ TRy(-q4)
        T45: Tensor = TTx(-Tensor([0.0825])) @ TTz(Tensor([0.384])) @ TRz(q5)
        T56: Tensor = TTz(Tensor([0.088])) @ TRx(Tensor([th.pi])) @ TTz(Tensor([0.107])) @ TRz(q6)

        links: List[Tensor] = [T01, T12, T23, T34, T45, T56]
        FK: Tensor = T01 @ T12 @ T23 @ T34 @ T45 @ T56

        return FK, links

    def __jacobian_forward_kinematics(self, config: Tensor):
        jacobian = th.autograd.functional.jacobian(
            lambda tensor: self.__forward_kinematics(tensor)[0],
            config
        ).transpose(dim0=0, dim1=2)

        return jacobian

    def __hessian_forward_kinematics(self, config: Tensor):
        hessian = th.autograd.functional.hessian(
            lambda tensor: self.__forward_kinematics(tensor)[0],
            config
        ).transpose(dim0=0, dim1=2)

        return hessian

    def forward_kinematics(self, config: Tensor = None, decompose=True):
        config = self.config if config is None else config
        assert config.shape == th.Size([6]), f"Kinova Manipulator has only 7 joints, not {config.shape}"
        FK, links = self.__forward_kinematics(config)
        self.joints = [reduce(th.matmul, links[:i]) for i in range(1, 7)]

        if not decompose:
            return FK

        R, P = T2RP(FK)
        euler_angles = R2Euler(R)

        return th.round(P, decimals=4), th.round(euler_angles, decimals=4)

    def jacobian(self, config: Tensor = None):
        config = self.config if config is None else config
        assert config.shape == th.Size([6]), f"Kinova Manipulator has only 7 joints, not {config.shape}"

        FK = self.forward_kinematics(config, decompose=False)
        R, _ = T2RP(FK)

        dT_dConfig = self.__jacobian_forward_kinematics(config)
        dRs, dPs = zip(*[T2RP(dT) for dT in dT_dConfig]) # TODO(Look into using vmap here)
        Jvs = th.hstack(dPs)
        Jws = th.hstack([
            inverse_skew(dR @ th.transpose(R, dim0=0, dim1=1))
            for dR in dRs
        ])  # TODO(Broadcasting should be able to work here)

        return th.vstack([Jvs, Jws])

    def hessian(self, config: Tensor = None):
        config = self.config if config is None else config
        assert config.shape == th.Size([6]), f"Kinova Manipulator has only 7 joints, not {config.shape}"

        FK = self.forward_kinematics(config, decompose=False)
        R, P = T2RP(FK)

        ddT_dConfig = self.__hessian_forward_kinematics(config)
        ddRs, ddPs = zip(*[T2RP(dT) for dT in ddT_dConfig])
        Hvs = th.hstack(ddPs)
        Hws = th.hstack([
            inverse_skew(dR @ th.transpose(R, dim0=0, dim1=1))
            for dR in ddRs
        ])

        return th.vstack([Hvs, Hws])

    def inverse_kinematics(self):
        pass

    def forward_dynamics(self):
        pass

    def inverse_dynamics(self):
        pass


if __name__ == '__main__':
    manipulator = Kinova6Links()
    print(manipulator.forward_kinematics(decompose=False).round(decimals=4))
    print(manipulator.jacobian(th.rand(6)).round(decimals=4))
