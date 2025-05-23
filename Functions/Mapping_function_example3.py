    from typing import Tuple, List
    def mapping_function(self) -> Tuple[List[torch.Tensor], List[bool], List[str]]:
        # The task is to open the cabinet door. The relevant mapping variable is the position of the cabinet door.
        # The cabinet door position is stored in self.cabinet_dof_pos[:, 3].
        
        # Mapping variable: cabinet door position
        mapping_vars = [self.cabinet_dof_pos[:, 3]]
        
        # Mapping direction: True means the variable should increase (door opening)
        mapping_directions = [True]
        
        # Mapping variable name
        mapping_vars_name = ["cabinet_door_position"]
        
        return mapping_vars, mapping_directions, mapping_vars_name
