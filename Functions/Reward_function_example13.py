@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor, 
    drawer_grasp_pos: torch.Tensor, 
    cabinet_dof_pos: torch.Tensor, 
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Temperature parameters for reward shaping
    distance_temp: float = 0.05  # Reduced temperature for higher sensitivity
    velocity_temp: float = 0.1   # Increased temperature to reduce scale
    progress_temp: float = 1.0   # Temperature for progress reward
    time_penalty_temp: float = 0.01  # Temperature for time penalty

    # Distance between the Franka hand and the drawer handle
    distance = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    distance_reward = torch.exp(-distance / distance_temp)

    # Velocity of the cabinet door (encouraging movement)
    velocity_reward = torch.abs(cabinet_dof_vel[:, 3])  # Assuming the 4th DOF is the cabinet door
    velocity_reward = torch.exp(-velocity_reward / velocity_temp)

    # Progress reward: encourage increasing the cabinet door's position
    progress_reward = cabinet_dof_pos[:, 3]  # Assuming the 4th DOF is the cabinet door
    progress_reward = torch.exp(progress_reward / progress_temp)

    # Time penalty: discourage long episodes
    time_penalty = -torch.ones_like(cabinet_dof_pos[:, 3])  # Constant penalty per timestep
    time_penalty = time_penalty * time_penalty_temp

    # Total reward is a weighted sum of the components
    total_reward = distance_reward + velocity_reward + progress_reward + time_penalty

    # Individual reward components for debugging
    reward_dict = {
        "distance_reward": distance_reward,
        "velocity_reward": velocity_reward,
        "progress_reward": progress_reward,
        "time_penalty": time_penalty
    }

    return total_reward, reward_dict
