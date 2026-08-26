1. Change the parts to row1_kit, row2_kit, row3_kit, ... --> Done
2. Default pose for a specific step --> Done
3. Removal of parts / tools from pegboard based on not only handed over part --> Done
4. Right Side GearStand first --> Done

5. Referring expression output template and answer and then highlight as well

5. Pegboard parts/tools bounding boxes adjustment --> Dante
6. bounding boxes around tools/parts (more transparent) --> Dante
7. TCP mismatch / Robot Mismatch (Unsolved)
8. Take many pictures from different angles for each step --> VLM feeding --> Not needed

9. check the referring expressions.csv (s) --> Dante



python3 workholding_study.py   --session-name target_setup   --mode hybrid   --target-navigation move   --target-poses-file task_graph/workholding_targets.json

python main_with_robot.py --simulation --vlm-test-auto-deliver
python3 task_graph/gearbox_task_graph.py \
  --vlm-model Qwen/Qwen3-VL-8B-Instruct


#Resume function based on json
#Check the logic by typing? 
#Real Robot Test


python main_with_robot.py --simulation --vlm-test-auto-deliver


python robot_control_server.py --simulation