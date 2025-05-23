    from typing import Tuple, List
    def mapping_function(self) -> Tuple[List[torch.Tensor], List[bool], List[str]]:
        # Stage 1: Mapping is measured by the distance between the hand and the cabinet door handle.
        hand_to_drawer_distance = torch.norm(self.drawer_grasp_pos - self.franka_grasp_pos, dim=-1)
        
        # Stage 2: Mapping is measured by the cabinet door's degree of openness.
        cabinet_door_openness = self.cabinet_dof_pos[:, 3]
        
        mapping_vars = [hand_to_drawer_distance, cabinet_door_openness]
        mapping_directions = [False, True]  # Decrease distance, increase openness
        mapping_vars_name = ["hand_to_drawer_distance", "cabinet_door_openness"]
        
        return mapping_vars, mapping_directions, mapping_vars_name
