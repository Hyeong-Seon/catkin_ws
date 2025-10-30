#!/usr/bin/env python3
"""Pure Pursuit + 속도 제어를 수행하는 MORAI 차량 제어 노드."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import rospy
from morai_msgs.msg import CtrlCmd, EgoVehicleStatus
from nav_msgs.msg import Path
from std_msgs.msg import Float32, Float32MultiArray, String


@dataclass
class TargetSpeedState:
    """종방향 제어를 위한 상태 변수."""

    integrator: float = 0.0
    prev_error: float = 0.0
    prev_time: Optional[rospy.Time] = None


class VehicleControlNode:
    """Pure Pursuit 기반 조향 + 속도 프로파일 제어."""

    def __init__(self) -> None:
        # 파라미터 로드
        self.wheel_base = rospy.get_param("~wheel_base", 3.0)  # m
        self.lookahead_base = rospy.get_param("~lookahead_base", 4.0)
        self.lookahead_gain = rospy.get_param("~lookahead_gain", 0.4)
        self.lookahead_min = rospy.get_param("~lookahead_min", 3.5)
        self.lookahead_max = rospy.get_param("~lookahead_max", 15.0)
        self.cruise_speed_kph = rospy.get_param("~cruise_speed_kph", 30.0)
        self.speed_kp = rospy.get_param("~speed_kp", 0.6)
        self.speed_ki = rospy.get_param("~speed_ki", 0.05)
        self.speed_kd = rospy.get_param("~speed_kd", 0.0)
        self.integrator_limit = rospy.get_param("~integrator_limit", 1.0)
        self.max_accel = rospy.get_param("~max_accel", 1.5)
        self.max_brake = rospy.get_param("~max_brake", 3.5)
        self.stop_hold_time = rospy.Duration.from_sec(
            float(rospy.get_param("~stop_hold_time", 1.0))
        )

        # 내부 상태
        self.path_points: List[Tuple[float, float]] = []
        self.speed_limit_mps = self._kph_to_mps(self.cruise_speed_kph)
        self.traffic_light_state = "unknown"
        self.stop_line_detected = False
        self.stop_line_timestamp = rospy.Time(0)
        self.obstacle_detected = False
        self.lane_offset = 0.0
        self.speed_state = TargetSpeedState()

        # 퍼블리셔 / 서브스크라이버 구성
        self.cmd_pub = rospy.Publisher("/ctrl_cmd", CtrlCmd, queue_size=1)
        rospy.Subscriber("/morai/status", EgoVehicleStatus, self.status_cb, queue_size=1)
        rospy.Subscriber("/reference_path", Path, self.path_cb, queue_size=1)
        rospy.Subscriber("/lane/center_offset", Float32, self.lane_offset_cb, queue_size=1)
        rospy.Subscriber("/perception/speed_limit", Float32, self.speed_limit_cb, queue_size=1)
        rospy.Subscriber(
            "/perception/traffic_light_state", String, self.traffic_light_cb, queue_size=1
        )
        rospy.Subscriber(
            "/perception/stop_line", Float32MultiArray, self.stop_line_cb, queue_size=1
        )
        rospy.Subscriber(
            "/perception/obstacles_2d", Float32MultiArray, self.obstacle_cb, queue_size=1
        )

        rospy.loginfo("[control] vehicle control node ready (Pure Pursuit).")

    # ----- 콜백들 -----

    def path_cb(self, msg: Path) -> None:
        self.path_points = [(pose.pose.position.x, pose.pose.position.y) for pose in msg.poses]
        rospy.loginfo_once("[control] reference path received: %d points.", len(self.path_points))

    def speed_limit_cb(self, msg: Float32) -> None:
        value = float(msg.data)
        if value <= 0:
            return
        self.speed_limit_mps = min(self._kph_to_mps(value), self._kph_to_mps(self.cruise_speed_kph))

    def traffic_light_cb(self, msg: String) -> None:
        self.traffic_light_state = msg.data.lower()

    def stop_line_cb(self, msg: Float32MultiArray) -> None:
        has_line = bool(msg.data)
        if has_line:
            self.stop_line_detected = True
            self.stop_line_timestamp = rospy.Time.now()
        else:
            # 잔존 타임아웃 후 해제
            if rospy.Time.now() - self.stop_line_timestamp > self.stop_hold_time:
                self.stop_line_detected = False

    def obstacle_cb(self, msg: Float32MultiArray) -> None:
        # 간단히 탐지 여부만 확인 (향후 거리 추정으로 확장 가능)
        self.obstacle_detected = bool(msg.data)

    def lane_offset_cb(self, msg: Float32) -> None:
        self.lane_offset = float(msg.data)

    # ----- 메인 제어 -----

    def status_cb(self, status: EgoVehicleStatus) -> None:
        if not self.path_points:
            rospy.logwarn_throttle(5.0, "[control] waiting for reference path.")
            return

        vehicle_pos = (status.position.x, status.position.y)
        yaw = float(status.heading)
        speed = float(status.velocity.x)

        target_point = self._select_lookahead_point(vehicle_pos, yaw, speed)
        if target_point is None:
            rospy.logwarn_throttle(2.0, "[control] no valid lookahead point.")
            return

        steering_cmd = self._pure_pursuit_steering(vehicle_pos, yaw, target_point)
        steering_cmd = np.clip(steering_cmd, -math.radians(35.0), math.radians(35.0))

        target_speed = self._compute_target_speed(speed)
        accel_cmd, brake_cmd = self._longitudinal_control(speed, target_speed)

        cmd = CtrlCmd()
        cmd.longlCmdType = 1  # accel/brake 제어 모드
        cmd.steering = float(steering_cmd)
        cmd.accel = float(accel_cmd)
        cmd.brake = float(brake_cmd)
        cmd.velocity = target_speed
        cmd.acceleration = accel_cmd - brake_cmd

        self.cmd_pub.publish(cmd)

    # ----- Pure Pursuit -----

    def _select_lookahead_point(
        self, vehicle_pos: Tuple[float, float], yaw: float, speed: float
    ) -> Optional[Tuple[float, float]]:
        lookahead = self.lookahead_base + self.lookahead_gain * abs(speed)
        lookahead = max(min(lookahead, self.lookahead_max), self.lookahead_min)

        points = np.asarray(self.path_points, dtype=np.float64)
        diff = points - np.asarray(vehicle_pos)
        distances = np.hypot(diff[:, 0], diff[:, 1])

        min_idx = int(np.argmin(distances))
        dist_min = distances[min_idx]
        if dist_min > 50.0:
            rospy.logwarn_throttle(5.0, "[control] vehicle far from reference path (%.2f m).", dist_min)

        for idx in range(min_idx, len(points)):
            if distances[idx] >= lookahead:
                return points[idx][0], points[idx][1]
        return points[-1][0], points[-1][1]

    def _pure_pursuit_steering(
        self, vehicle_pos: Tuple[float, float], yaw: float, target_point: Tuple[float, float]
    ) -> float:
        dx = target_point[0] - vehicle_pos[0]
        dy = target_point[1] - vehicle_pos[1]

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        # 차량 좌표계로 변환
        local_x = cos_yaw * dx + sin_yaw * dy
        local_y = -sin_yaw * dx + cos_yaw * dy

        # 차선 중심 오프셋을 라디안 각도로 약하게 보정
        offset_gain = rospy.get_param("~lane_offset_gain", 0.0)
        local_y += self.lane_offset * offset_gain

        lookahead = math.hypot(local_x, local_y)
        if lookahead < 1e-3:
            return 0.0
        alpha = math.atan2(local_y, local_x)
        steering = math.atan2(2.0 * self.wheel_base * math.sin(alpha), lookahead)
        return steering

    # ----- 종방향 제어 -----

    def _compute_target_speed(self, current_speed: float) -> float:
        cruise = self._kph_to_mps(self.cruise_speed_kph)
        limit = min(self.speed_limit_mps, cruise)
        target = limit

        # 사고 예방을 위한 기본 장애물 감속
        if self.obstacle_detected:
            target = min(target, self._kph_to_mps(15.0))

        # 신호 및 정지선 로직
        if self.traffic_light_state in ("red", "yellow"):
            target = min(target, self._kph_to_mps(5.0))

        if self.traffic_light_state in ("red", "yellow") and self.stop_line_detected:
            target = 0.0

        return max(target, 0.0)

    def _longitudinal_control(self, current_speed: float, target_speed: float) -> Tuple[float, float]:
        now = rospy.Time.now()
        state = self.speed_state

        error = target_speed - current_speed
        dt = 0.0
        if state.prev_time is not None:
            dt = (now - state.prev_time).to_sec()
        state.prev_time = now

        if dt > 0.0:
            state.integrator += error * dt
            state.integrator = float(
                np.clip(state.integrator, -self.integrator_limit, self.integrator_limit)
            )

        derivative = 0.0
        if dt > 1e-3:
            derivative = (error - state.prev_error) / dt
        state.prev_error = error

        control = (
            self.speed_kp * error + self.speed_ki * state.integrator + self.speed_kd * derivative
        )

        if control >= 0.0:
            accel = float(np.clip(control, 0.0, self.max_accel))
            brake = 0.0
        else:
            accel = 0.0
            brake = float(np.clip(-control, 0.0, self.max_brake))

        # 목표 속도가 0 부근이면 브레이크를 강제
        if target_speed < 0.1 and current_speed < 0.2:
            accel = 0.0
            brake = max(brake, 0.5)

        return accel, brake

    @staticmethod
    def _kph_to_mps(value: float) -> float:
        return float(value) / 3.6

    def spin(self) -> None:
        rospy.spin()


def main() -> None:
    rospy.init_node("vehicle_control_node")
    VehicleControlNode().spin()


if __name__ == "__main__":
    main()
