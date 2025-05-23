    from typing import Tuple, List
    def mapping_function(self) -> Tuple[List[torch.Tensor], List[bool], List[str]]:
        # Analysis of the existing mapping variable:
        # 1. The mapping variable `cabinet_door_joint_position` shows some variation during training (Min: 0.04, Max: 0.29, Mean: 0.16).
        #    However, the success rates are consistently near zero (Max: 0.37, Mean: 0.02, Min: 0.00), indicating that this variable alone is not sufficient to capture task mapping.
        # 2. The task involves opening the cabinet door, which requires the agent to interact with the door handle and apply force to move it.
        #    The current mapping variable does not account for the agent's interaction with the door handle or the distance between the hand and the handle.
    
        # Proposed improvements:
        # 1. Add a mapping variable for the distance between the hand and the cabinet door handle. This ensures the agent is correctly positioned to interact with the door.
        # 2. Keep the cabinet door joint position as a mapping variable but add a threshold to ensure the door is sufficiently open.
        # 3. Add a mapping variable for the force applied by the agent to the door handle, which is critical for opening the door.
    
        # Extract relevant variables from the observation buffer
        hand_pos = self.rigid_body_states[:, self.hand_handle][:, 0:3]
        drawer_handle_pos = self.rigid_body_states[:, self.drawer_handle][:, 0:3]
        cabinet_door_pos = self.obs_buf[:, -2]  # Cabinet door joint position
    
        # Compute the distance between the hand and the drawer handle
        hand_to_handle_distance = torch.norm(hand_pos - drawer_handle_pos, dim=-1)
    
        # Compute the force applied by the agent (approximated by the velocity of the cabinet door joint)
        cabinet_door_force = self.obs_buf[:, -1]  # Cabinet door joint velocity
    
        # Define mapping variables
        mapping_vars = [hand_to_handle_distance, cabinet_door_pos, cabinet_door_force]
        
        # Define mapping directions:
        # 1. The distance between the hand and the handle should decrease (False).
        # 2. The cabinet door joint position should increase (True).
        # 3. The force applied to the door should increase (True).
        mapping_directions = [False, True, True]
        
        # Define mapping variable names
        mapping_vars_name = ["hand_to_handle_distance", "cabinet_door_joint_position", "cabinet_door_force"]
        
        return mapping_vars, mapping_directions, mapping_vars_name
