# minsungk02 Catkin Workspace

This repository contains a ROS-based autonomous driving stack organized as a Catkin workspace. It includes packages for vehicle control, planning, perception, and system bring-up, along with an external directory for contest-specific resources.

## Repository Structure

- `src/` – Catkin packages and the workspace-level `CMakeLists.txt`.
  - `autonomy_bringup/` – Launch files and configuration to start the integrated stack.
  - `control_pkg/` – Controllers and related utilities for vehicle actuation.
  - `perception_pkg/` – Lidar, camera, and lane detection modules for perception tasks.
  - `planning_pkg/` – Path planning utilities and behavior planning logic.
  - `vehicle_interface_pkg/` – Interfaces to vehicle hardware or simulation middleware.
  - `MORAI-ROS_morai_msgs/` – Message definitions provided by MORAI for simulation integration.
- `external/2025-kookmin-contest/` – Reference implementations and documentation used for the 2025 Kookmin contest.
- `.catkin_workspace` – Marker file that enables the directory to be treated as a Catkin workspace.

## Getting Started

1. **Install ROS**: Ensure that the ROS distribution compatible with your target environment is installed (e.g., ROS Noetic on Ubuntu 20.04).
2. **Clone the repository**:
   ```bash
   git clone https://github.com/minsungk02/catkin_ws.git
   cd catkin_ws
   ```
3. **Install dependencies**:
   ```bash
   rosdep install --from-paths src --ignore-src -r -y
   ```
4. **Build the workspace**:
   ```bash
   catkin_make
   ```
5. **Source the workspace**:
   ```bash
   source devel/setup.bash
   ```

## Development Tips

- Keep generated build artifacts (e.g., `build/` and `devel/`) out of version control. The provided `.gitignore` already excludes them.
- Store sensitive configuration (API keys, credentials) outside the repository.
- Use feature branches when adding significant functionality and submit pull requests for review before merging into the main branch.

## Contributing

1. Fork the repository on GitHub.
2. Create a feature branch: `git checkout -b feature/my-feature`.
3. Commit your changes with descriptive messages.
4. Push the branch and open a pull request describing your updates.

## License

If you plan to open source the project, add an appropriate license file (e.g., MIT, Apache 2.0) so others know how they can use the code.
