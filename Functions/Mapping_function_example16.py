    from typing import Tuple, List
    def mapping_function(self) -> Tuple[List[torch.Tensor], List[bool], List[str]]:
        # Stage 1: Distance to the cabinet door handle
        hand_pos = self.rigid_body_states[:, self.hand_handle][:, 0:3]
        door_handle_pos = self.rigid_body_states[:, self.drawer_handle][:, 0:3]
        distance_to_door = torch.norm(hand_pos - door_handle_pos, dim=-1)
    
        # Stage 2: Grasp success (distance between fingers and door handle)
        lfinger_pos = self.rigid_body_states[:, self.lfinger_handle][:, 0:3]
        rfinger_pos = self.rigid_body_states[:, self.rfinger_handle][:, 0:3]
        grasp_distance = torch.norm((lfinger_pos + rfinger_pos) / 2 - door_handle_pos, dim=-1)
    
        # Stage 3: Door opening (joint position of the cabinet door)
        cabinet_door_pos = self.obs_buf[:, -2]  # Assuming the last two elements are cabinet_dof_pos and cabinet_dof_vel
    
        # Mapping variables for each stage
        mapping_vars = [distance_to_door, grasp_distance, cabinet_door_pos]
        
        # Mapping directions:
        # Stage 1: Decrease distance to door
        # Stage 2: Decrease grasp distance
        # Stage 3: Increase door joint position
        mapping_directions = [False, False, True]
        
        # Names of the mapping variables
        mapping_vars_name = ["distance_to_door", "grasp_distance", "cabinet_door_joint_position"]
        
        return mapping_vars, mapping_directions, mapping_vars_name
