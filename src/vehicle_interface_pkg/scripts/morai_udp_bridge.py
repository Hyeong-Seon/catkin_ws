#!/usr/bin/env python3
"""MORAI UDP <-> ROS 브릿지 스켈레톤."""

import socket
import struct
from typing import Optional

import rospy
from morai_msgs.msg import CtrlCmd, EgoVehicleStatus


class MoraiUDPBridge:
    """EgoVehicleStatus 수신 및 CtrlCmd 송신을 담당."""

    def __init__(self) -> None:
        self.recv_ip = rospy.get_param("~status_ip", "0.0.0.0")
        self.recv_port = rospy.get_param("~status_port", 9091)
        self.send_ip = rospy.get_param("~cmd_ip", "127.0.0.1")
        self.send_port = rospy.get_param("~cmd_port", 9092)

        self.recv_sock = self._create_socket(self.recv_ip, self.recv_port, bind=True)
        self.send_sock = self._create_socket(self.send_ip, self.send_port, bind=False)

        self.status_pub = rospy.Publisher("/morai/status", EgoVehicleStatus, queue_size=1)
        self.cmd_sub = rospy.Subscriber("/ctrl_cmd", CtrlCmd, self.cmd_cb, queue_size=1)
        rospy.Timer(rospy.Duration(0.01), self.poll_udp)

        rospy.loginfo(
            "[interface] UDP bridge listening %s:%d, sending -> %s:%d",
            self.recv_ip,
            self.recv_port,
            self.send_ip,
            self.send_port,
        )

    @staticmethod
    def _create_socket(ip: str, port: int, bind: bool) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if bind:
            sock.bind((ip, port))
        return sock

    def poll_udp(self, _: rospy.TimerEvent) -> None:
        """UDP 수신 데이터를 ROS 메시지로 변환 (프로토콜 파싱은 TODO)."""
        self.recv_sock.settimeout(0.0)
        try:
            data, _ = self.recv_sock.recvfrom(1024)
        except BlockingIOError:
            return

        status = self.parse_status(data)
        if status is not None:
            self.status_pub.publish(status)

    def parse_status(self, data: bytes) -> Optional[EgoVehicleStatus]:
        """UDP 패킷 -> EgoVehicleStatus 변환 (필요한 포맷으로 구현)."""
        # TODO: 실제 MORAI UDP 프로토콜로 파싱 로직 작성
        rospy.logdebug("[interface] raw status packet len=%d", len(data))
        return None

    def cmd_cb(self, msg: CtrlCmd) -> None:
        """ROS CtrlCmd를 UDP 패킷으로 직렬화."""
        packet = self.serialize_cmd(msg)
        self.send_sock.sendto(packet, (self.send_ip, self.send_port))

    def serialize_cmd(self, msg: CtrlCmd) -> bytes:
        """CtrlCmd -> UDP 바이너리 (임시 예시: steering, accel, brake)."""
        rospy.logdebug(
            "[interface] send cmd steer=%.3f accel=%.3f brake=%.3f",
            msg.steering,
            msg.accel,
            msg.brake,
        )
        return struct.pack("<fff", msg.steering, msg.accel, msg.brake)

    def close(self) -> None:
        self.recv_sock.close()
        self.send_sock.close()


def main() -> None:
    rospy.init_node("morai_udp_bridge")
    bridge = MoraiUDPBridge()
    rospy.on_shutdown(bridge.close)
    rospy.spin()


if __name__ == "__main__":
    main()
