@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor,
    drawer_grasp_pos: torch.Tensor,
    cabinet_dof_pos: torch.Tensor,
    cabinet_dof_vel: torch.Tensor,
    franka_dof_vel: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Temperature parameters for reward shaping
    distance_temp: float = 0.1
    progress_temp: float = 1.0
    velocity_temp: float = 0.01

    # Reward for reducing the distance between the hand and the drawer handle
    distance = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    distance_reward = torch.exp(-distance_temp * distance)

    # Reward for increasing the cabinet door opening angle
    # Assuming cabinet_dof_pos[:, 3] represents the door angle
    door_angle = cabinet_dof_pos[:, 3]
    progress_reward = torch.exp(progress_temp * door_angle)

    # Penalty for high velocities to encourage smooth movement
    velocity_penalty = torch.norm(cabinet_dof_vel[:, 3], dim=-1) + torch.norm(franka_dof_vel, dim=-1)
    velocity_penalty = torch.exp(-velocity_temp * velocity_penalty)

    # Total reward is a combination of the above components
    total_reward = distance_reward + progress_reward + velocity_penalty

    # Individual reward components for debugging
    reward_dict = {
        "distance_reward": distance_reward,
        "progress_reward": progress_reward,
        "velocity_penalty": velocity_penalty,
    }

    return total_reward, reward_dict
