@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor, 
    drawer_grasp_pos: torch.Tensor, 
    cabinet_dof_pos: torch.Tensor, 
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Temperature parameters for reward shaping
    distance_temp: float = 0.1
    velocity_temp: float = 0.01
    success_temp: float = 1.0

    # Distance between the Franka hand and the drawer handle
    distance = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    distance_reward = torch.exp(-distance / distance_temp)

    # Velocity of the cabinet door (encouraging movement)
    velocity_reward = torch.abs(cabinet_dof_vel[:, 3])  # Assuming the 4th DOF is the cabinet door
    velocity_reward = torch.exp(-velocity_reward / velocity_temp)

    # Success reward when the cabinet door is fully open (assuming fully open is at position 1.0)
    success_threshold: float = 0.95
    success_reward = torch.where(cabinet_dof_pos[:, 3] >= success_threshold, torch.tensor(1.0, device=cabinet_dof_pos.device), torch.tensor(0.0, device=cabinet_dof_pos.device))
    success_reward = success_reward * success_temp

    # Total reward is a weighted sum of the components
    total_reward = distance_reward + velocity_reward + success_reward

    # Individual reward components for debugging
    reward_dict = {
        "distance_reward": distance_reward,
        "velocity_reward": velocity_reward,
        "success_reward": success_reward
    }

    return total_reward, reward_dict
