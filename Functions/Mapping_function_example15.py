    from typing import Tuple, List
    def mapping_function(self) -> Tuple[List[torch.Tensor], List[bool], List[str]]:
        # Analysis of the current mapping function:
        # The current mapping variable is `cabinet_door_joint_position`, which measures the joint position of the cabinet door.
        # From the training data:
        # - The `cabinet_door_joint_position` values range from 0.04 to 0.29, with a mean of 0.16.
        # - The `success_rate` is consistently near zero (max 0.37, mean 0.02), indicating that the current mapping variable is not effectively capturing task mapping.
        # - The joint position alone does not strongly correlate with the task objective (opening the cabinet door) because it does not account for the interaction between the robot hand and the door.
    
        # Optimized mapping function:
        # To better solve the task, we need to introduce a more sensitive and task-relevant mapping variable:
        # 1. **Distance between the robot hand and the cabinet door handle**: This ensures the robot is approaching the door handle, which is a prerequisite for opening the door.
        # 2. **Angle of the cabinet door**: This directly measures how much the door has been opened, which is the ultimate goal.
    
        # Compute the distance between the robot hand and the cabinet door handle
        hand_pos = self.rigid_body_states[:, self.hand_handle][:, 0:3]
        drawer_handle_pos = self.rigid_body_states[:, self.drawer_handle][:, 0:3]
        hand_to_handle_distance = torch.norm(hand_pos - drawer_handle_pos, dim=-1)
    
        # Extract the cabinet door's joint angle (directly from the observation buffer)
        cabinet_door_angle = self.obs_buf[:, -2]  # Assuming the last two elements are cabinet_dof_pos and cabinet_dof_vel
    
        # Mapping variables:
        # - `hand_to_handle_distance`: Should decrease as the robot approaches the handle.
        # - `cabinet_door_angle`: Should increase as the door opens.
        mapping_vars = [hand_to_handle_distance, cabinet_door_angle]
        mapping_directions = [False, True]  # Decrease distance, increase angle
        mapping_vars_name = ["hand_to_handle_distance", "cabinet_door_angle"]
    
        return mapping_vars, mapping_directions, mapping_vars_name
