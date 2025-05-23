    from typing import Tuple, List
    def mapping_function(self) -> Tuple[List[torch.Tensor], List[bool], List[str]]:
        # The task is to open the cabinet door, which is a single-stage task.
        # The mapping can be measured by the position of the cabinet door's joint.
        # The cabinet door's joint position is stored in self.cabinet_dof_pos[:, 3].
        
        # Mapping variable: cabinet door joint position
        mapping_vars = [self.cabinet_dof_pos[:, 3]]
        
        # Mapping direction: True (increase) because the door needs to open
        mapping_directions = [True]
        
        # Mapping variable name
        mapping_vars_name = ["cabinet_door_joint_position"]
        
        return mapping_vars, mapping_directions, mapping_vars_name
