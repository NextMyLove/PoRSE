    from typing import Tuple, List
    def mapping_function(self) -> Tuple[List[torch.Tensor], List[bool], List[str]]:
        # Analysis of existing mapping variables:
        # 1. hand_to_handle_distance: 
        #    - Values remain nearly identical throughout training (Min: 0.33, Max: 0.39, Mean: 0.37).
        #    - This variable does not effectively capture task mapping dynamics and is weakly correlated with the task goal.
        #    - Suggestion: Replace with a more sensitive variable that reflects the agent's interaction with the door handle.
        # 2. cabinet_door_joint_position:
        #    - Values show some variation (Min: 0.08, Max: 0.12, Mean: 0.10), but the range is narrow.
        #    - This variable is somewhat correlated with the task goal but lacks sensitivity.
        #    - Suggestion: Retain but combine with other variables to better measure door openness.
        # 3. cabinet_door_force:
        #    - Values vary moderately (Min: 0.12, Max: 0.23, Mean: 0.17) and are correlated with task mapping.
        #    - This variable is effective in measuring the agent's active engagement with the door.
        #    - Suggestion: Retain and refine to better capture the force dynamics.
    
        # Proposed improvements:
        # 1. Replace hand_to_handle_distance with hand_to_door_distance, which measures the distance between the hand and the door surface.
        #    This ensures the agent is correctly positioned to interact with the door.
        # 2. Retain cabinet_door_joint_position but add a threshold to ensure the door is sufficiently open.
        # 3. Refine cabinet_door_force to measure the cumulative force applied by the agent to the door.
    
        # Extract relevant variables from the observation buffer
        hand_pos = self.rigid_body_states[:, self.hand_handle][:, 0:3]
        door_pos = self.rigid_body_states[:, self.drawer_handle][:, 0:3]
        cabinet_door_pos = self.obs_buf[:, -2]  # Cabinet door joint position
        cabinet_door_force = self.obs_buf[:, -1]  # Cabinet door joint velocity
    
        # Compute the distance between the hand and the door surface
        hand_to_door_distance = torch.norm(hand_pos - door_pos, dim=-1)
    
        # Compute the cumulative force applied by the agent (approximated by the integral of the door joint velocity)
        cumulative_door_force = torch.cumsum(torch.abs(cabinet_door_force), dim=-1)
    
        # Define mapping variables
        mapping_vars = [hand_to_door_distance, cabinet_door_pos, cumulative_door_force]
        
        # Define mapping directions:
        # 1. The distance between the hand and the door should decrease (False).
        # 2. The cabinet door joint position should increase (True).
        # 3. The cumulative force applied to the door should increase (True).
        mapping_directions = [False, True, True]
        
        # Define mapping variable names
        mapping_vars_name = ["hand_to_door_distance", "cabinet_door_joint_position", "cumulative_door_force"]
        
        return mapping_vars, mapping_directions, mapping_vars_name
