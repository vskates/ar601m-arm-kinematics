"""The three required public operations; all joint angles are radians."""

from task2_fk import FK as _FK
from task3_ik import IK as _IK
from task5_jacobian import DifferentialKinematics as _J


def FK(q, robot):
    return _FK(q, robot)


def IK(T, robot):
    return _IK(T, robot)


def DifferentialKinematics(q, robot):
    return _J(q, robot)
