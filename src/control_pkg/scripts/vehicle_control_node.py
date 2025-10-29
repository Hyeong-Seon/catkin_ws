#!/usr/bin/env python3
"""MORAI 차량 제어 노드 스켈레톤."""

import rospy
from morai_msgs.msg import CtrlCmd, EgoVehicleStatus


class VehicleControlNode:
    """PID/Pure Pursuit 등을 결합할 제어 노드 뼈대."""

    def __init__(self) -> None:
        self.cmd_pub = rospy.Publisher("/ctrl_cmd", CtrlCmd, queue_size=1)
        self.status_sub = rospy.Subscriber(
            "/morai/status", EgoVehicleStatus, self.status_cb, queue_size=1
        )

        rospy.loginfo("[control] vehicle control node initialized.")

    def status_cb(self, msg: EgoVehicleStatus) -> None:
        """상태 토픽 콜백 - 향후 제어 알고리즘 입력으로 사용."""
        # TODO: 속도/조향 PID 및 경로 추종 알고리즘 연동
        rospy.logdebug(
            "status: speed=%.2f, steering=%.2f", msg.velocity.x, msg.ctrl_cmd.steering
        )

    def spin(self) -> None:
        rospy.spin()


def main() -> None:
    rospy.init_node("vehicle_control_node")
    VehicleControlNode().spin()


if __name__ == "__main__":
    main()
