from BenchMarker import *
from functools import reduce, partial
from typing import List, Tuple
from Transformations import *
from torch import Tensor
import torch as th
import functorch


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

    def __jacobian_fk(self, config: Tensor):
        jacobian = th.autograd.functional.jacobian(
            lambda tensor: self.__forward_kinematics(tensor)[0],
            config
        ).transpose(dim0=0, dim1=2).transpose(dim0=1, dim1=2)

        return jacobian

    def jacobian_product_fk(self, config: Tensor, config_vel: Tensor):
        fk_v = th.autograd.functional.jvp(
            lambda tensor: self.__forward_kinematics(tensor)[0],
            config,
            config_vel
        )

        return fk_v

    def __hessian_fk(self, config: Tensor):
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

        dT_dConfig = self.__jacobian_fk(config)
        dRs, dPs = zip(*[T2RP(dT) for dT in dT_dConfig])  # TODO(Look into using vmap here)
        Jvs = th.hstack(dPs)

        # Unsafe because doesn't check for skew-symmetry, but faster because of vmap and unchecking,
        # hence best to use in real time application
        Jws = (functorch
               .vmap(partial(inverse_skew, check=False))(th.stack(dRs) @ th.transpose(R, dim0=0, dim1=1))
               .squeeze(dim=2)
               .transpose(dim0=0, dim1=1))

        # This is safer as it checks the skew-symmetry, best to use this in test case
        # Jws = th.hstack([
        #     inverse_skew(dR @ th.transpose(R, dim0=0, dim1=1))
        #     for dR in dRs
        # ])

        return th.vstack([Jvs, Jws])

    def hessian(self, config: Tensor = None):
        config = self.config if config is None else config
        assert config.shape == th.Size([6]), f"Kinova Manipulator has only 7 joints, not {config.shape}"

        FK = self.forward_kinematics(config, decompose=False)
        R, P = T2RP(FK)

        ddT_dConfig = self.__hessian_fk(config)
        ddRs, ddPs = zip(*[T2RP(dT) for dT in ddT_dConfig])
        Hvs = th.hstack(ddPs)
        Hws = th.hstack([
            inverse_skew(dR @ th.transpose(R, dim0=0, dim1=1))
            for dR in ddRs
        ])

        return th.vstack([Hvs, Hws])

    def pos_vel(self, config: Tensor, config_vel: Tensor):
        """
        Taking advantage of Pytorch jvp to compute velocities faster. Might need to show equations prooving this
        is allowed, or simply test case to proove this
        :return the position and velocity of the end-effector for a given configuration and configuration velocity
        """
        FK, Vs = self.jacobian_product_fk(config, config_vel)
        R, P = T2RP(FK)
        theta = R2Euler(R)
        Vr, Vt = T2RP(Vs)

        return th.vstack([P, theta]), th.vstack([Vt, inverse_skew(Vr @ th.transpose(R, dim0=0, dim1=1))])

    def inverse_kinematics(self):
        pass

    def forward_dynamics(self):
        pass

    def inverse_dynamics(self):
        pass


if __name__ == '__main__':
    config = th.rand(6)
    vel = th.rand(6)
    manipulator = Kinova6Links()

    duration, value = time_it(
        lambda: manipulator.pos_vel(config, vel)
    )
    print(f"Duration: {duration}\nValue: {value}\n")

    duration, value = time_it(
        lambda: (manipulator.forward_kinematics(config), manipulator.jacobian(config) @ vel)
    )
    print(f"Duration: {duration}\nValue: {value}")
