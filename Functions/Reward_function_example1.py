@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor, 
    drawer_grasp_pos: torch.Tensor, 
    cabinet_dof_pos: torch.Tensor, 
    cabinet_dof_vel: torch.Tensor, 
    franka_dof_pos: torch.Tensor, 
    franka_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Temperature parameters for reward components
    temp_distance: float = 0.1
    temp_velocity: float = 0.01
    temp_door_open: float = 1.0
    
    # Distance between the franka hand and the drawer handle
    distance = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    distance_reward = torch.exp(-distance / temp_distance)
    
    # Velocity of the cabinet door (encouraging movement)
    velocity_reward = torch.exp(-torch.abs(cabinet_dof_vel[:, 3]) / temp_velocity)
    
    # Position of the cabinet door (encouraging opening)
    door_open_reward = torch.exp(cabinet_dof_pos[:, 3] / temp_door_open)
    
    # Total reward is a weighted sum of the components
    total_reward = distance_reward + velocity_reward + door_open_reward
    
    # Individual reward components for debugging
    reward_dict = {
        "distance_reward": distance_reward,
        "velocity_reward": velocity_reward,
        "door_open_reward": door_open_reward
    }
    
    return total_reward, reward_dict
