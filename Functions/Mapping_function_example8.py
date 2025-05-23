    from typing import Tuple, List
    def mapping_function(self) -> Tuple[List[torch.Tensor], List[bool], List[str]]:
        # The task is to open the cabinet door, which is a single-stage task.
        # The mapping variable is the position of the cabinet door's joint (dof_pos).
        # As the door opens, the joint position increases.
        
        # Extract the cabinet door's joint position from the observation buffer
        cabinet_door_pos = self.obs_buf[:, -2]  # Assuming the last two elements are cabinet_dof_pos and cabinet_dof_vel
        
        # The mapping variable is the cabinet door's joint position
        mapping_vars = [cabinet_door_pos]
        
        # The mapping direction is True because the door should open (joint position increases)
        mapping_directions = [True]
        
        # The name of the mapping variable
        mapping_vars_name = ["cabinet_door_joint_position"]
        
        return mapping_vars, mapping_directions, mapping_vars_name
