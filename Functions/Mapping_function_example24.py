    from typing import Tuple, List
    def mapping_function(self) -> Tuple[List[torch.Tensor], List[bool], List[str]]:
        # Analysis of existing mapping variables:
        # 1. hand_to_handle_distance: 
        #    - Values remain nearly identical throughout training (Min: 0.33, Max: 0.39, Mean: 0.37).
        #    - This variable does not effectively capture task mapping dynamics, as it is weakly correlated with the task goal.
        #    - The agent's success rate improves significantly despite minimal changes in this variable, indicating redundancy.
        # 2. cabinet_door_joint_position: 
        #    - Values show some variation (Min: 0.08, Max: 0.12, Mean: 0.10).
        #    - This variable is moderately correlated with the task goal but does not fully capture the agent's interaction with the door.
        #    - It is insufficient as a standalone mapping variable.
        # 3. cabinet_door_force: 
        #    - Values show moderate variation (Min: 0.12, Max: 0.23, Mean: 0.17).
        #    - This variable is strongly correlated with the task goal, as it reflects the agent's active engagement with the door.
        #    - It is a useful mapping variable but should be combined with other sensitive indicators.
    
        # Proposed improvements:
        # 1. Remove `hand_to_handle_distance` as it is redundant and does not contribute to task mapping.
        # 2. Replace `cabinet_door_joint_position` with a more sensitive variable: the angle of the cabinet door relative to its fully open position.
        #    - This directly measures the door's openness and is strongly correlated with the task goal.
        # 3. Retain `cabinet_door_force` as it reflects the agent's active interaction with the door.
        # 4. Add a new mapping variable: the distance between the hand and the door handle after the door starts moving.
        #    - This ensures the agent maintains proper interaction with the door throughout the task.
    
        # Extract relevant variables from the observation buffer
        hand_pos = self.rigid_body_states[:, self.hand_handle][:, 0:3]
        drawer_handle_pos = self.rigid_body_states[:, self.drawer_handle][:, 0:3]
        cabinet_door_pos = self.obs_buf[:, -2]  # Cabinet door joint position
        cabinet_door_force = self.obs_buf[:, -1]  # Cabinet door joint velocity
    
        # Compute the angle of the cabinet door relative to its fully open position
        cabinet_door_angle = cabinet_door_pos  # Assuming this is already normalized to the open position
    
        # Compute the distance between the hand and the door handle
        hand_to_handle_distance = torch.norm(hand_pos - drawer_handle_pos, dim=-1)
    
        # Define mapping variables
        mapping_vars = [cabinet_door_angle, cabinet_door_force, hand_to_handle_distance]
        
        # Define mapping directions:
        # 1. The cabinet door angle should increase (True).
        # 2. The force applied to the door should increase (True).
        # 3. The distance between the hand and the handle should decrease (False).
        mapping_directions = [True, True, False]
        
        # Define mapping variable names
        mapping_vars_name = ["cabinet_door_angle", "cabinet_door_force", "hand_to_handle_distance"]
        
        return mapping_vars, mapping_directions, mapping_vars_name
