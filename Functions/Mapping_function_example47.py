    from typing import Tuple, List
    def mapping_function(self) -> Tuple[List[torch.Tensor], List[bool], List[str]]:
        # Analysis of existing mapping variables:
        # 1. hand_handle_distance:
        #    - Values range from 0.37 to 0.38, with a mean of 0.38. The variation is minimal, indicating that this variable does not effectively capture task mapping.
        #    - The task goal is to open the cabinet door, and while the distance between the hand and the handle is important, this variable alone is not strongly correlated with the task objective.
        #    - Recommendation: Replace this variable with a more sensitive indicator of hand-handle interaction, such as the alignment of the hand with the handle.
    
        # 2. scaled_cabinet_door_pos:
        #    - Values range from 0.54 to 0.81, with a mean of 0.60. This variable shows moderate variation and is somewhat correlated with task mapping.
        #    - However, it does not significantly contribute to task performance, as the success rates are inconsistent despite changes in this variable.
        #    - Recommendation: Retain this variable but adjust its sensitivity by scaling or adding a threshold to ensure the door is sufficiently open.
    
        # 3. cabinet_door_angular_displacement:
        #    - Values range from 0.05 to 0.08, with a mean of 0.06. This variable shows minimal variation and is not strongly correlated with task mapping.
        #    - The task goal is to open the cabinet door, and while angular displacement is important, this variable alone does not effectively capture the task mapping dynamics.
        #    - Recommendation: Replace this variable with a more direct indicator of task mapping, such as the angular velocity of the cabinet door.
    
        # Proposed improvements:
        # 1. Replace hand_handle_distance with hand_handle_alignment, which measures the alignment of the hand with the handle.
        # 2. Retain scaled_cabinet_door_pos but scale it to ensure it captures sufficient door opening.
        # 3. Replace cabinet_door_angular_displacement with cabinet_door_angular_velocity, which directly measures the door's movement.
    
        # Extract relevant variables from the observation buffer
        hand_pos = self.rigid_body_states[:, self.hand_handle][:, 0:3]
        drawer_handle_pos = self.rigid_body_states[:, self.drawer_handle][:, 0:3]
        cabinet_door_pos = self.obs_buf[:, -2]  # Cabinet door joint position
        cabinet_door_vel = self.obs_buf[:, -1]  # Cabinet door joint velocity
    
        # Compute the alignment of the hand with the handle
        hand_handle_alignment = torch.sum((hand_pos - drawer_handle_pos) ** 2, dim=-1)
    
        # Scale the cabinet door joint position to ensure it captures sufficient door opening
        scaled_cabinet_door_pos = cabinet_door_pos * 10.0
    
        # Compute the angular velocity of the cabinet door
        cabinet_door_angular_velocity = torch.abs(cabinet_door_vel)
    
        # Define mapping variables
        mapping_vars = [hand_handle_alignment, scaled_cabinet_door_pos, cabinet_door_angular_velocity]
        
        # Define mapping directions:
        # 1. The alignment of the hand with the handle should increase (True).
        # 2. The scaled cabinet door joint position should increase (True).
        # 3. The cabinet door angular velocity should increase (True).
        mapping_directions = [True, True, True]
        
        # Define mapping variable names
        mapping_vars_name = ["hand_handle_alignment", "scaled_cabinet_door_pos", "cabinet_door_angular_velocity"]
        
        return mapping_vars, mapping_directions, mapping_vars_name
