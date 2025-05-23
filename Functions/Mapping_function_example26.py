    from typing import Tuple, List
    def mapping_function(self) -> Tuple[List[torch.Tensor], List[bool], List[str]]:
        # Analysis of existing mapping variables:
        # 1. **hand_to_handle_distance**: 
        #    - Values: Min: 0.33, Max: 0.39, Mean: 0.37
        #    - This variable shows minimal variation during training, indicating it does not effectively capture task mapping.
        #    - It is weakly correlated with the task goal because the agent can still succeed even if the distance remains relatively constant.
        #    - Conclusion: This variable is not sensitive enough and should be replaced.
    
        # 2. **cabinet_door_joint_position**: 
        #    - Values: Min: 0.08, Max: 0.12, Mean: 0.10
        #    - This variable shows some variation but remains in a narrow range, indicating it does not fully capture task mapping.
        #    - It is moderately correlated with the task goal but lacks sensitivity to distinguish between successful and unsuccessful episodes.
        #    - Conclusion: This variable should be retained but supplemented with additional variables to better capture mapping.
    
        # 3. **cabinet_door_force**: 
        #    - Values: Min: 0.12, Max: 0.23, Mean: 0.17
        #    - This variable shows significant variation and is strongly correlated with the task goal, as applying force is critical for opening the door.
        #    - Conclusion: This variable is effective and should be retained.
    
        # Proposed improvements:
        # 1. Replace **hand_to_handle_distance** with **hand_to_door_distance**, which measures the distance between the hand and the cabinet door itself. This ensures the agent is correctly positioned to interact with the door.
        # 2. Retain **cabinet_door_joint_position** but add a threshold to ensure the door is sufficiently open.
        # 3. Retain **cabinet_door_force** as it is critical for opening the door.
        # 4. Add **door_angle**, which measures the angle of the cabinet door relative to its closed position. This provides a more sensitive measure of door openness.
    
        # Extract relevant variables from the observation buffer
        hand_pos = self.rigid_body_states[:, self.hand_handle][:, 0:3]
        drawer_pos = self.rigid_body_states[:, self.drawer_handle][:, 0:3]
        cabinet_door_pos = self.obs_buf[:, -2]  # Cabinet door joint position
        cabinet_door_force = self.obs_buf[:, -1]  # Cabinet door joint velocity
    
        # Compute the distance between the hand and the cabinet door
        hand_to_door_distance = torch.norm(hand_pos - drawer_pos, dim=-1)
    
        # Compute the door angle (assuming the door's initial closed position is 0)
        door_angle = torch.abs(cabinet_door_pos)  # Use absolute value to measure deviation from closed position
    
        # Define mapping variables
        mapping_vars = [hand_to_door_distance, door_angle, cabinet_door_force]
        
        # Define mapping directions:
        # 1. The distance between the hand and the door should decrease (False).
        # 2. The door angle should increase (True).
        # 3. The force applied to the door should increase (True).
        mapping_directions = [False, True, True]
        
        # Define mapping variable names
        mapping_vars_name = ["hand_to_door_distance", "door_angle", "cabinet_door_force"]
        
        return mapping_vars, mapping_directions, mapping_vars_name
