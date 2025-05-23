@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor, 
    drawer_grasp_pos: torch.Tensor, 
    cabinet_dof_pos: torch.Tensor, 
    franka_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Temperature parameters for reward components
    distance_temp: float = 0.1
    angle_temp: float = 1.0
    action_penalty_temp: float = 0.01

    # Distance between the hand and the drawer handle
    distance_to_target = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    distance_reward = torch.exp(-distance_temp * distance_to_target)

    # Cabinet door angle (assuming the door angle is the 3rd DOF)
    cabinet_door_angle = cabinet_dof_pos[:, 3]
    angle_reward = torch.exp(angle_temp * cabinet_door_angle)

    # Penalty for large actions (using the velocity of the Franka DOFs)
    action_penalty = torch.norm(franka_dof_vel, dim=-1)
    action_penalty_reward = torch.exp(-action_penalty_temp * action_penalty)

    # Total reward is a weighted sum of the components
    total_reward = distance_reward + angle_reward + action_penalty_reward

    # Individual reward components for debugging
    reward_components = {
        "distance_reward": distance_reward,
        "angle_reward": angle_reward,
        "action_penalty_reward": action_penalty_reward
    }

    return total_reward, reward_components
