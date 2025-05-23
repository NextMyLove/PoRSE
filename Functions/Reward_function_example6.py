@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor,
    drawer_grasp_pos: torch.Tensor,
    cabinet_dof_pos: torch.Tensor,
    cabinet_dof_vel: torch.Tensor,
    franka_dof_pos: torch.Tensor,
    franka_dof_vel: torch.Tensor,
    franka_dof_lower_limits: torch.Tensor,
    franka_dof_upper_limits: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Temperature parameters for reward components
    distance_temp: float = 0.1
    door_open_temp: float = 0.5
    velocity_temp: float = 0.01
    control_temp: float = 0.01

    # Reward for reducing the distance between the gripper and the drawer handle
    distance = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)
    distance_reward = torch.exp(-distance / distance_temp)

    # Reward for opening the cabinet door (positive dof position)
    door_open_reward = torch.exp(cabinet_dof_pos[:, 3] / door_open_temp)

    # Reward for the cabinet door moving in the positive direction (positive dof velocity)
    velocity_reward = torch.exp(cabinet_dof_vel[:, 3] / velocity_temp)

    # Penalty for excessive control effort (joint velocity)
    control_penalty = torch.norm(franka_dof_vel, dim=-1) * control_temp

    # Total reward
    reward = distance_reward + door_open_reward + velocity_reward - control_penalty

    # Dictionary of individual reward components
    reward_dict = {
        "distance_reward": distance_reward,
        "door_open_reward": door_open_reward,
        "velocity_reward": velocity_reward,
        "control_penalty": control_penalty,
    }

    return reward, reward_dict
