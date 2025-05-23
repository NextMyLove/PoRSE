@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor, 
    drawer_grasp_pos: torch.Tensor, 
    cabinet_dof_pos: torch.Tensor, 
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Temperature parameters for reward shaping
    distance_temp: float = 0.05  # Increased sensitivity for distance reward
    velocity_temp: float = 0.1   # Adjusted for better velocity reward
    progress_temp: float = 0.2   # Adjusted for better intermediate progress
    time_penalty_temp: float = 0.1  # Increased scale for time penalty

    # Distance between the Franka hand and the drawer handle
    distance = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    distance_reward = torch.exp(-distance / distance_temp) * 2.0  # Increased scale

    # Velocity of the cabinet door (encouraging movement)
    velocity_reward = torch.abs(cabinet_dof_vel[:, 3])  # Assuming the 4th DOF is the cabinet door
    velocity_reward = torch.exp(velocity_reward / velocity_temp)  # Reward higher velocities

    # Progress reward for partially opening the cabinet door
    progress_reward = cabinet_dof_pos[:, 3]  # Assuming the 4th DOF is the cabinet door
    progress_reward = torch.clamp(progress_reward, min=0.0, max=1.0)  # Normalize to [0, 1]
    progress_reward = torch.exp(progress_reward / progress_temp) * 1.5  # Increased scale

    # Time penalty to encourage faster task completion
    time_penalty = torch.ones_like(cabinet_dof_pos[:, 3]) * time_penalty_temp

    # Total reward is a weighted sum of the components
    total_reward = distance_reward + velocity_reward + progress_reward - time_penalty

    # Individual reward components for debugging
    reward_dict = {
        "distance_reward": distance_reward,
        "velocity_reward": velocity_reward,
        "progress_reward": progress_reward,
        "time_penalty": time_penalty
    }

    return total_reward, reward_dict
