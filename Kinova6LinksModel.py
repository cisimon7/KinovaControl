from functools import partial
from Transformations import *
from torch import Tensor
import functorch as fth
import torch as th


class Kinova6Links:
    def __init__(self, config: Tensor = Tensor([0 for _ in range(6)])):
        self.config = config
        self.dof = 6
        self.masses = [th.tensor(0) for _ in range(6)]
        self.inertias = [th.eye(3) for _ in range(6)]
        self.lengths = [0.2755, 0.4100, 0.2073, 0.1038, 0.1038, 0.1600]

        T01 = lambda b1_, q1_: TTz(th.tensor(b1_)) @ TRz(q1_)
        T12 = lambda b2_, q2_: TTz(th.tensor(b2_)) @ TRy(q2_)
        T23 = lambda b3_, q3_: TTz(th.tensor(b3_)) @ TRy(q3_)
        T34 = lambda b4_, q4_: TTz(th.tensor(b4_)) @ TRz(q4_)
        T45 = lambda b5_, q5_: TTz(th.tensor(b5_)) @ TRy(q5_)
        T56 = lambda b6_, q6_: TTz(th.tensor(b6_)) @ TRz(q6_)

        self.ETS = [T01, T12, T23, T34, T45, T56]  # Elementary Transform sequence for the KINOVA robot

    def __forward_kinematics_end(self, config: Tensor, end=6) -> Tensor:
        FK = th.eye(4)
        for i in range(end):
            FK = FK @ self.ETS[i](self.lengths[i], config[i])

        return FK

    def __forward_kinematics_center(self, config: Tensor, center=6) -> Tensor:
        FK = th.eye(4)
        for i in range(center - 1):
            FK = FK @ self.ETS[i](self.lengths[i], config[i])

        FK = FK @ self.ETS[center - 1](0.5 * self.lengths[center - 1], config[center - 1])

        return FK

    def __jacobian_fk(self, config: Tensor, fk_fun=None):
        forward_kinematics_fun = self.__forward_kinematics_end if fk_fun is None else fk_fun
        jacobian = th.autograd.functional.jacobian(
            lambda tensor: forward_kinematics_fun(tensor),
            config
        ).permute(2, 1, 0)

        return jacobian

    def jacobian_product_fk(self, config: Tensor, config_vel: Tensor, fk_fun=None):
        forward_kinematics_fun = self.__forward_kinematics_end if fk_fun is None else fk_fun
        fk_v = th.autograd.functional.jvp(
            lambda tensor: forward_kinematics_fun(tensor),
            config,
            config_vel
        )

        return fk_v

    def __hessian_fk(self, config: Tensor):
        hessian = th.autograd.functional.hessian(
            lambda tensor: self.__forward_kinematics_end(tensor),
            config
        ).transpose(dim0=0, dim1=2)

        return hessian

    def forward_kinematics(self, config: Tensor = None, decompose=True):
        config = self.config if config is None else config
        assert config.shape == th.Size([6]), f"Kinova Manipulator has only 6 joints, not {config.shape}"
        FK = self.__forward_kinematics_end(config)

        if not decompose:
            return FK

        R, P = T2RP(FK)
        euler_angles = R2Euler(R)

        return th.round(P, decimals=4), th.round(euler_angles, decimals=4)

    def jacobian(self, config: Tensor = None):
        config = self.config if config is None else config
        assert config.shape == th.Size([6]), f"Kinova Manipulator has only 7 joints, not {config.shape}"

        FK = self.__forward_kinematics_end(config)
        R, _ = T2RP(FK)

        dT_dConfig = self.__jacobian_fk(config)
        dRs, dPs = zip(*[T2RP(dT) for dT in dT_dConfig])  # TODO(Look into using vmap here)
        Jvs = th.hstack(dPs)  # TODO(th.hstack may create a new tensor which makes derivative not possible)

        # Unsafe because doesn't check for skew-symmetry, but faster because of vmap and unchecking,
        # hence best to use in real time application
        # Jws = fth.vmap(partial(inverse_skew, check=False))(
        #     th.stack(dRs) @ th.transpose(R, dim0=0, dim1=1)
        # ).squeeze(dim=2).transpose(dim0=0, dim1=1)

        # This is safer as it checks the skew-symmetry, best to use this in test case
        Jws = th.hstack([
            inverse_skew(dR @ th.transpose(R, dim0=0, dim1=1))
            for dR in dRs
        ])

        return th.vstack([Jvs, Jws])

    def jacobian_centers(self, config: Tensor):
        FK_centers = [partial(self.__forward_kinematics_end(), level=i) for i in range(self.dof - 1)]

        J_cis, FK_cis = 0, 0
        R_cis = FK_cis

        return J_cis, R_cis

    def hessian(self, config: Tensor = None):
        config = self.config if config is None else config
        assert config.shape == th.Size([6]), f"Kinova Manipulator has only 6 joints, not {config.shape}"

        FK = self.__forward_kinematics_end(config)
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

    def mass_matrix(self, config: Tensor):
        def link_mass_matrix(m_i, J_i, R_i, I_i):
            J_i.transpose(dim0=0, dim1=1) @ th.vstack([
                th.hstack([m_i * th.eye(3), th.zeros((3, 3))]),
                th.hstack([th.zeros((3, 3)), R_i.transpose(dim0=0, dim1=1) @ I_i @ R_i])
            ]) @ J_i

        J_cs, Rs = self.jacobian_centers(config)

        return fth.vmap(link_mass_matrix)(
            th.vstack(self.masses), J_cs, Rs, th.vstack(self.inertias)
        ).sum(dim=0)

    def corriolis_centrifugal(self, config: Tensor, M: Tensor = None):
        M = self.mass_matrix(config) if M is None else M
        corriolis = 0
        centrifugal = 0
        return corriolis, centrifugal

    def kinetic_energy(self, config: Tensor, config_vel: Tensor):
        M = self.mass_matrix(config)
        return 0.5 * config_vel.transpose(0, 1) @ M @ config_vel

    def potential_energy(self, config: Tensor):
        def link_potential(m, g, z):
            return m * g * z

        z_cs = 0
        return fth.vmap(link_potential)(
            th.vstack(self.masses), th.vstack([th.tensor(9.81) for _ in range(6)]), z_cs
        ).sum(0)

    def forward_dynamics(self, config: Tensor, config_vel: Tensor, config_toq: Tensor, wrench: Tensor = None):
        pass

    def inverse_dynamics(self, config: Tensor, config_vel: Tensor, config_acc: Tensor, wrench: Tensor = None):
        wrench = th.zeros(6) if wrench is None else wrench  # Force/Torque applied at end effector
        J_ee = self.jacobian(config)  # Jacobian at end effector
        M = self.mass_matrix(config)  # Robot mass matrix
        corriolis, centrifugal = self.corriolis_centrifugal(config, M)
        gravity_term = 0

        return (
                (M @ config_acc) +  # M(q) * q_dd
                (centrifugal @ config_vel @ config_vel) -
                (config_vel.transpose(dim0=0, dim1=1) @ corriolis @ config_vel) +
                gravity_term +  # G(q)
                (J_ee @ wrench)  # J(q) * F_tip
        )
