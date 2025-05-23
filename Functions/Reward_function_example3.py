@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor, 
    drawer_grasp_pos: torch.Tensor, 
    cabinet_dof_pos: torch.Tensor, 
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Temperature parameters for reward components
    pos_temp: float = 0.1
    vel_temp: float = 0.01
    dof_temp: float = 0.1
    
    # Reward for minimizing the distance between the hand and the drawer handle
    pos_diff = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    pos_reward = torch.exp(-pos_diff / pos_temp)
    
    # Reward for opening the cabinet door (maximize the cabinet DOF position)
    dof_reward = torch.exp(cabinet_dof_pos[:, 3] / dof_temp)
    
    # Reward for minimizing the velocity of the cabinet door (encourage smooth opening)
    vel_reward = torch.exp(-torch.abs(cabinet_dof_vel[:, 3]) / vel_temp)
    
    # Total reward is a weighted sum of the individual rewards
    total_reward = 0.5 * pos_reward + 0.3 * dof_reward + 0.2 * vel_reward
    
    # Dictionary of individual reward components for debugging
    reward_dict = {
        "pos_reward": pos_reward,
        "dof_reward": dof_reward,
        "vel_reward": vel_reward
    }
    
    return total_reward, reward_dict
