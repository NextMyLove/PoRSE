    from typing import Tuple, List
    def mapping_function(self) -> Tuple[List[torch.Tensor], List[bool], List[str]]:
        # Analysis of existing mapping variables:
        # 1. hand_handle_alignment:
        #    - Values range from 0.01 to 0.04, with a mean of 0.01. The variation is minimal, indicating that this variable does not effectively capture task mapping.
        #    - The task goal is to open the cabinet door, and while alignment of the hand with the handle is important, this variable alone is not strongly correlated with the task objective.
        #    - Recommendation: Replace this variable with a more sensitive indicator of hand-handle interaction, such as the distance between the hand and the handle.
    
        # 2. scaled_cabinet_door_pos:
        #    - Values range from 0.89 to 1.55, with a mean of 1.18. This variable shows moderate variation and is somewhat correlated with task mapping.
        #    - However, it does not significantly contribute to task performance, as the success rates are inconsistent despite changes in this variable.
        #    - Recommendation: Retain this variable but adjust its sensitivity by scaling or adding a threshold to ensure the door is sufficiently open.
    
        # 3. cabinet_door_angular_velocity:
        #    - Values range from 0.10 to 0.18, with a mean of 0.15. This variable shows moderate variation and is somewhat correlated with task mapping.
        #    - However, it does not significantly contribute to task performance, as the success rates are inconsistent despite changes in this variable.
        #    - Recommendation: Replace this variable with a more direct indicator of task mapping, such as the angular displacement of the cabinet door.
    
        # Extract relevant variables from the observation buffer
        hand_pos = self.rigid_body_states[:, self.hand_handle][:, 0:3]
        drawer_handle_pos = self.rigid_body_states[:, self.drawer_handle][:, 0:3]
        cabinet_door_pos = self.obs_buf[:, -2]  # Cabinet door joint position
        cabinet_door_vel = self.obs_buf[:, -1]  # Cabinet door joint velocity
    
        # Compute the distance between the hand and the handle
        hand_handle_distance = torch.norm(drawer_handle_pos - hand_pos, dim=-1)
    
        # Scale the cabinet door joint position to ensure it captures sufficient door opening
        scaled_cabinet_door_pos = cabinet_door_pos * 10.0
    
        # Compute the angular displacement of the cabinet door
        cabinet_door_angular_displacement = torch.abs(cabinet_door_pos)
    
        # Define mapping variables
        mapping_vars = [hand_handle_distance, scaled_cabinet_door_pos, cabinet_door_angular_displacement]
        
        # Define mapping directions:
        # 1. The distance between the hand and the handle should decrease (False).
        # 2. The scaled cabinet door joint position should increase (True).
        # 3. The cabinet door angular displacement should increase (True).
        mapping_directions = [False, True, True]
        
        # Define mapping variable names
        mapping_vars_name = ["hand_handle_distance", "scaled_cabinet_door_pos", "cabinet_door_angular_displacement"]
        
        return mapping_vars, mapping_directions, mapping_vars_name
