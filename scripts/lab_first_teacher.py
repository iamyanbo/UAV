"""Process the first actual clip into the canonical target cache on CPU.

Keeps the single GPU available to the currently collecting physical flights.
"""
from pathlib import Path
import torch
from photo_goal.mission_storage import configure
from photo_goal.mission_contracts import city_config
from photo_goal.mission_resources import Resources,RunWindow
from photo_goal.mission_bootstrap import targets
root=Path('/mnt/hdd2/yanbocheng/photo-goal-native')
configure(root);torch.set_num_threads(2)
output=root/'data/visual-bootstrap'
targets(root,output/'clips.json',output,Resources(root,city_config(),'cpu'),RunWindow(1),device='cpu',limit=1)
print('First recorded clip processed with the actual released V-JEPA 2 and MobileNet weights.',flush=True)
