    from typing import Tuple, List
    def mapping_function(self) -> Tuple[List[torch.Tensor], List[bool], List[str]]:
        # The task is to open the cabinet door. The relevant variable for mapping is the cabinet door's position.
        # We assume that the cabinet door's position is represented by `self.cabinet_dof_pos[:, 3]`.
        
        # Mapping variable: cabinet door position
        mapping_vars = [self.cabinet_dof_pos[:, 3]]
        
        # Mapping direction: True if the door position should increase (i.e., the door should open)
        mapping_directions = [True]
        
        # Name of the mapping variable
        mapping_vars_name = ["cabinet_door_position"]
        
        return mapping_vars, mapping_directions, mapping_vars_name
