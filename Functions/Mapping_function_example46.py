    from typing import Tuple, List
    def mapping_function(self) -> Tuple[List[torch.Tensor], List[bool], List[str]]:
        # Analysis of existing mapping variables:
        # 1. hand_handle_distance:
        #    - Values remain nearly identical throughout training (Min: 0.37, Max: 0.38, Mean: 0.38).
        #    - This variable does not effectively capture task mapping dynamics, as it shows minimal variation.
        #    - The task goal is to open the cabinet door, and while the hand-handle distance is important, this variable alone is not strongly correlated with the task objective.
        #    - Recommendation: Replace this variable with a more sensitive indicator of hand-handle interaction, such as the alignment of the hand with the handle.
    
        # 2. scaled_cabinet_door_pos:
        #    - Values show moderate variation (Min: 0.54, Max: 0.81, Mean: 0.60).
        #    - This variable is somewhat correlated with task mapping, as it reflects the position of the cabinet door.
        #    - However, it does not significantly contribute to task performance, as the success rates are inconsistent despite changes in this variable.
        #    - Recommendation: Retain this variable but adjust its sensitivity by scaling or adding a threshold to ensure the door is sufficiently open.
    
        # 3. cabinet_door_angular_displacement:
        #    - Values remain nearly identical throughout training (Min: 0.05, Max: 0.08, Mean: 0.06).
        #    - This variable does not effectively capture task mapping dynamics, as it shows minimal variation.
        #    - The task goal is to open the cabinet door, and while the angular displacement is important, this variable alone is not strongly correlated with the task objective.
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
        # 1. The alignment of the hand with the handle should decrease (False).
        # 2. The scaled cabinet door joint position should increase (True).
        # 3. The cabinet door angular velocity should increase (True).
        mapping_directions = [False, True, True]
        
        # Define mapping variable names
        mapping_vars_name = ["hand_handle_alignment", "scaled_cabinet_door_pos", "cabinet_door_angular_velocity"]
        
        return mapping_vars, mapping_directions, mapping_vars_name
