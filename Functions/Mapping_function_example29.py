    from typing import Tuple, List
    def mapping_function(self) -> Tuple[List[torch.Tensor], List[bool], List[str]]:
        # Analysis of existing mapping variables:
        # 1. hand_to_handle_distance: 
        #    - Values range from 0.33 to 0.39, with a mean of 0.37. The variation is minimal, indicating that this variable does not effectively capture task mapping.
        #    - The task goal is to open the cabinet door, and while positioning the hand near the handle is important, this variable alone is not strongly correlated with the task objective.
        #    - Recommendation: Replace this variable with a more sensitive indicator of hand-handle interaction, such as the alignment of the hand with the handle.
    
        # 2. cabinet_door_joint_position:
        #    - Values range from 0.08 to 0.12, with a mean of 0.10. The variation is minimal, and the values are consistently low, indicating that the door is not being opened sufficiently.
        #    - This variable is strongly correlated with the task objective but fails to capture mapping effectively due to its low sensitivity.
        #    - Recommendation: Retain this variable but adjust its sensitivity by scaling or adding a threshold to ensure the door is sufficiently open.
    
        # 3. cabinet_door_force:
        #    - Values range from 0.12 to 0.23, with a mean of 0.17. This variable shows moderate variation and is somewhat correlated with task mapping.
        #    - However, it does not significantly contribute to task performance, as the success rates are inconsistent despite changes in this variable.
        #    - Recommendation: Replace this variable with a more direct indicator of task mapping, such as the angular velocity of the cabinet door.
    
        # Proposed improvements:
        # 1. Replace hand_to_handle_distance with hand_handle_alignment, which measures the alignment of the hand with the handle.
        # 2. Retain cabinet_door_joint_position but scale it to ensure it captures sufficient door opening.
        # 3. Replace cabinet_door_force with cabinet_door_angular_velocity, which directly measures the door's movement.
    
        # Extract relevant variables from the observation buffer
        hand_pos = self.rigid_body_states[:, self.hand_handle][:, 0:3]
        drawer_handle_pos = self.rigid_body_states[:, self.drawer_handle][:, 0:3]
        cabinet_door_pos = self.obs_buf[:, -2]  # Cabinet door joint position
        cabinet_door_vel = self.obs_buf[:, -1]  # Cabinet door joint velocity
    
        # Compute the alignment of the hand with the handle (dot product of hand and handle vectors)
        hand_handle_vector = drawer_handle_pos - hand_pos
        hand_handle_alignment = torch.sum(hand_handle_vector * hand_handle_vector, dim=-1)
    
        # Scale the cabinet door joint position to ensure it captures sufficient door opening
        scaled_cabinet_door_pos = cabinet_door_pos * 10.0
    
        # Define mapping variables
        mapping_vars = [hand_handle_alignment, scaled_cabinet_door_pos, cabinet_door_vel]
        
        # Define mapping directions:
        # 1. The alignment of the hand with the handle should increase (True).
        # 2. The scaled cabinet door joint position should increase (True).
        # 3. The cabinet door angular velocity should increase (True).
        mapping_directions = [True, True, True]
        
        # Define mapping variable names
        mapping_vars_name = ["hand_handle_alignment", "scaled_cabinet_door_pos", "cabinet_door_angular_velocity"]
        
        return mapping_vars, mapping_directions, mapping_vars_name
