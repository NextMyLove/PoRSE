    from typing import Tuple, List
    def mapping_function(self) -> Tuple[List[torch.Tensor], List[bool], List[str]]:
        # Stage 1: Grasp the cabinet door handle
        # Mapping is measured by the distance between the hand and the drawer handle
        to_target = self.drawer_grasp_pos - self.franka_grasp_pos
        distance_to_handle = torch.norm(to_target, dim=-1)
        
        # Stage 2: Open the cabinet door
        # Mapping is measured by the cabinet door's position (dof_pos)
        cabinet_door_pos = self.cabinet_dof_pos[:, 3]
        
        # Mapping variables and their directions
        mapping_vars = [distance_to_handle, cabinet_door_pos]
        mapping_directions = [False, True]  # Decrease distance, increase door position
        mapping_vars_name = ["distance_to_handle", "cabinet_door_pos"]
        
        return mapping_vars, mapping_directions, mapping_vars_name
