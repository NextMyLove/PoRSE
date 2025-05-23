    from typing import Tuple, List
    def mapping_function(self) -> Tuple[List[torch.Tensor], List[bool], List[str]]:
        # Analysis of existing mapping variables:
        # 1. **hand_to_handle_distance**: 
        #    - Values: Min: 0.33, Max: 0.39, Mean: 0.37
        #    - Analysis: The distance remains nearly constant throughout training, indicating it does not effectively capture task mapping. This variable is weakly correlated with the task goal of opening the cabinet door.
        #    - Conclusion: This variable should be replaced with a more sensitive alternative.
    
        # 2. **cabinet_door_joint_position**: 
        #    - Values: Min: 0.08, Max: 0.12, Mean: 0.10
        #    - Analysis: The joint position shows minimal variation, suggesting it is not a strong indicator of task mapping. It does not reflect the agent's active interaction with the door.
        #    - Conclusion: This variable should be replaced with a more dynamic measure of door opening mapping.
    
        # 3. **cabinet_door_force**: 
        #    - Values: Min: 0.12, Max: 0.23, Mean: 0.17
        #    - Analysis: The force applied to the door shows some variation but does not strongly correlate with success rates. It may not be a reliable indicator of task mapping.
        #    - Conclusion: This variable should be replaced with a more meaningful measure of agent interaction.
    
        # Proposed improvements:
        # 1. **Replace hand_to_handle_distance with hand_to_door_distance**: Measures the distance between the hand and the cabinet door itself, not just the handle. This ensures the agent is correctly positioned to interact with the door.
        # 2. **Replace cabinet_door_joint_position with door_opening_angle**: Measures the angle of the door relative to its closed position, providing a more dynamic indicator of door opening mapping.
        # 3. **Replace cabinet_door_force with hand_force_on_door**: Measures the force applied by the hand directly to the door, ensuring the agent is actively engaging with the door.
    
        # Extract relevant variables from the observation buffer
        hand_pos = self.rigid_body_states[:, self.hand_handle][:, 0:3]
        door_pos = self.rigid_body_states[:, self.drawer_handle][:, 0:3]
        door_rot = self.rigid_body_states[:, self.drawer_handle][:, 3:7]
    
        # Compute the distance between the hand and the cabinet door
        hand_to_door_distance = torch.norm(hand_pos - door_pos, dim=-1)
    
        # Compute the door opening angle (assuming the door rotates around the z-axis)
        door_opening_angle = torch.atan2(door_rot[:, 1], door_rot[:, 0])  # Simplified angle calculation
    
        # Compute the force applied by the hand to the door (approximated by the velocity of the hand)
        hand_force_on_door = torch.norm(self.rigid_body_states[:, self.hand_handle][:, 7:10], dim=-1)
    
        # Define mapping variables
        mapping_vars = [hand_to_door_distance, door_opening_angle, hand_force_on_door]
        
        # Define mapping directions:
        # 1. The distance between the hand and the door should decrease (False).
        # 2. The door opening angle should increase (True).
        # 3. The force applied to the door should increase (True).
        mapping_directions = [False, True, True]
        
        # Define mapping variable names
        mapping_vars_name = ["hand_to_door_distance", "door_opening_angle", "hand_force_on_door"]
        
        return mapping_vars, mapping_directions, mapping_vars_name
