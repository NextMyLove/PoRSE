    from typing import Tuple, List
    def mapping_function(self) -> Tuple[List[torch.Tensor], List[bool], List[str]]:
        # Analysis of the existing mapping variable:
        # 1. The `cabinet_door_joint_position` variable does show some variation during training (min: 0.04, max: 0.29, mean: 0.16).
        #    However, the success rate remains consistently low (max: 0.37, mean: 0.02, min: 0.00), indicating that this variable alone is not sufficient to guide the agent to solve the task effectively.
        # 2. The `cabinet_door_joint_position` variable is correlated with the task goal (opening the cabinet door), but it fails to capture the dynamics of the agent's interaction with the door, such as grasping and pulling.
    
        # Improved mapping function:
        # To better solve the task, we need to introduce additional mapping variables that capture the agent's interaction with the cabinet door. Specifically:
        # - The distance between the hand and the cabinet door handle (to ensure the agent is close enough to grasp the handle).
        # - The alignment of the hand with the cabinet door handle (to ensure the agent is properly positioned to grasp the handle).
        # - The cabinet door joint position (to track the door's opening mapping).
    
        # Calculate the distance between the hand and the cabinet door handle
        hand_pos = self.rigid_body_states[:, self.hand_handle][:, 0:3]
        drawer_handle_pos = self.rigid_body_states[:, self.drawer_handle][:, 0:3]
        hand_to_handle_distance = torch.norm(hand_pos - drawer_handle_pos, dim=-1)
    
        # Calculate the alignment of the hand with the cabinet door handle
        hand_rot = self.rigid_body_states[:, self.hand_handle][:, 3:7]
        drawer_handle_rot = self.rigid_body_states[:, self.drawer_handle][:, 3:7]
        hand_handle_alignment = torch.sum(hand_rot * drawer_handle_rot, dim=-1)
    
        # Extract the cabinet door's joint position from the observation buffer
        cabinet_door_pos = self.obs_buf[:, -2]
    
        # The mapping variables are:
        # 1. The distance between the hand and the cabinet door handle (should decrease).
        # 2. The alignment of the hand with the cabinet door handle (should increase).
        # 3. The cabinet door joint position (should increase).
        mapping_vars = [hand_to_handle_distance, hand_handle_alignment, cabinet_door_pos]
    
        # The mapping directions are:
        # 1. Decrease the distance between the hand and the cabinet door handle.
        # 2. Increase the alignment of the hand with the cabinet door handle.
        # 3. Increase the cabinet door joint position.
        mapping_directions = [False, True, True]
    
        # The names of the mapping variables
        mapping_vars_name = ["hand_to_handle_distance", "hand_handle_alignment", "cabinet_door_joint_position"]
    
        return mapping_vars, mapping_directions, mapping_vars_name
