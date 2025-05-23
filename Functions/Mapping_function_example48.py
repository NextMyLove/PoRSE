    from typing import Tuple, List
    def mapping_function(self) -> Tuple[List[torch.Tensor], List[bool], List[str]]:
        # Analysis of existing mapping variables:
        # 1. hand_handle_distance:
        #    - Values remain nearly identical throughout training (0.37 to 0.38).
        #    - This variable fails to effectively capture task mapping dynamics.
        #    - The task goal is to open the cabinet door, and while hand-handle distance is important, it does not strongly correlate with the task objective.
        #    - Recommendation: Replace this variable with a more sensitive indicator of hand-handle interaction, such as the force applied by the hand on the handle.
    
        # 2. scaled_cabinet_door_pos:
        #    - Values range from 0.54 to 0.81, with a mean of 0.60. This variable shows moderate variation and is somewhat correlated with task mapping.
        #    - However, it does not significantly contribute to task performance, as the success rates are inconsistent despite changes in this variable.
        #    - Recommendation: Retain this variable but adjust its sensitivity by scaling or adding a threshold to ensure the door is sufficiently open.
    
        # 3. cabinet_door_angular_displacement:
        #    - Values remain nearly identical throughout training (0.05 to 0.08).
        #    - This variable fails to effectively capture task mapping dynamics.
        #    - The task goal is to open the cabinet door, and while angular displacement is important, it does not strongly correlate with the task objective.
        #    - Recommendation: Replace this variable with a more direct indicator of task mapping, such as the angular velocity of the cabinet door.
    
        # Extract relevant variables from the observation buffer
        hand_pos = self.rigid_body_states[:, self.hand_handle][:, 0:3]
        drawer_handle_pos = self.rigid_body_states[:, self.drawer_handle][:, 0:3]
        cabinet_door_pos = self.obs_buf[:, -2]  # Cabinet door joint position
        cabinet_door_vel = self.obs_buf[:, -1]  # Cabinet door joint velocity
    
        # Compute the force applied by the hand on the handle
        hand_handle_force = torch.norm(self.rigid_body_states[:, self.hand_handle][:, 7:10], dim=-1)
    
        # Scale the cabinet door joint position to ensure it captures sufficient door opening
        scaled_cabinet_door_pos = cabinet_door_pos * 10.0
    
        # Compute the angular velocity of the cabinet door
        cabinet_door_angular_velocity = torch.abs(cabinet_door_vel)
    
        # Define mapping variables
        mapping_vars = [hand_handle_force, scaled_cabinet_door_pos, cabinet_door_angular_velocity]
        
        # Define mapping directions:
        # 1. The force applied by the hand on the handle should increase (True).
        # 2. The scaled cabinet door joint position should increase (True).
        # 3. The cabinet door angular velocity should increase (True).
        mapping_directions = [True, True, True]
        
        # Define mapping variable names
        mapping_vars_name = ["hand_handle_force", "scaled_cabinet_door_pos", "cabinet_door_angular_velocity"]
        
        return mapping_vars, mapping_directions, mapping_vars_name
