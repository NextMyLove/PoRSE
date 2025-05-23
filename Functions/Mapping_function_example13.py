    from typing import Tuple, List
    def mapping_function(self) -> Tuple[List[torch.Tensor], List[bool], List[str]]:
        # Stage 1: Distance between the hand and the cabinet door handle
        hand_pos = self.rigid_body_states[:, self.hand_handle][:, 0:3]
        handle_pos = self.rigid_body_states[:, self.drawer_handle][:, 0:3]
        hand_to_handle_distance = torch.norm(handle_pos - hand_pos, dim=-1)
        
        # Stage 2: Angle of the cabinet door (joint position)
        cabinet_door_pos = self.cabinet_dof_pos[:, 3]  # Assuming index 3 corresponds to the door joint
        
        # Stage 3: Distance between the hand and a target position after opening
        # Define a target position (e.g., slightly away from the door)
        target_pos = torch.tensor([0.5, 0.0, 0.5], device=self.device).unsqueeze(0)  # Example target
        hand_to_target_distance = torch.norm(target_pos - hand_pos, dim=-1)
        
        # Mapping variables
        mapping_vars = [hand_to_handle_distance, cabinet_door_pos, hand_to_target_distance]
        
        # Mapping directions:
        # Stage 1: Decrease distance to handle (False)
        # Stage 2: Increase door angle (True)
        # Stage 3: Decrease distance to target (False)
        mapping_directions = [False, True, False]
        
        # Names of mapping variables
        mapping_vars_name = ["hand_to_handle_distance", "cabinet_door_angle", "hand_to_target_distance"]
        
        return mapping_vars, mapping_directions, mapping_vars_name
