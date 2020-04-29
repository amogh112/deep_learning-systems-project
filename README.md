# deep_learning-systems-project
Project repository

## How to run:

#### In debug mode
python mtasks_train.py --model resnet18 --dataset taskonomy --debug

### To prune with a strategy from config/prune/global_l1unstructured_40.json
python mtasks_train.py --model resnet18 --dataset taskonomy --debug --prune global_l1unstructured_40