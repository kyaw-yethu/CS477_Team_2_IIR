# Testing Findings — Thanapat

## Fixes Applied
1. **Wrong Gemini model name** — `gemini-3-flash-preview` does not exist, changed to `gemini-2.5-flash` in `perception_node.py`
2. **`pose_to_joints` missing** — method exists in Ye Thu's `assignment_2/move_joint.py` but workspace was loading the wrong version; fixed by rebuilding from correct source

## Bugs Found
3. **Scout detection returns empty** — Gemini returns no detections from scout pose when objects spawn late; robot gives up immediately
4. **No grasp success detection** — robot doesn't check if gripper actually grabbed something; proceeds to place even when holding air
5. **Gripper orientation wrong** — finger hits object sideways instead of gripping; object falls off table (main area to improve)
6. **Placement position inaccurate** — even successful grasps may be placed in wrong spot
7. **Gazebo controller timing (WSL2)** — spawn_entity times out on WSL2; requires manual workaround to spawn robot

## Next Steps
- Fix gripper orientation (assigned to Thanapat + Ye Thu)
- Add grasp success detection
- Add retry logic for scout detection
