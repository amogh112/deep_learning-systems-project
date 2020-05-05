CUDA_VISIBLE_DEVICES=0 python mtasks_train.py --model resnet18 --dataset taskonomy --customize_class --class_to_train A --class_to_test A
CUDA_VISIBLE_DEVICES=0 python mtasks_train.py --model resnet18 --dataset taskonomy --customize_class --class_to_train r --class_to_test r
CUDA_VISIBLE_DEVICES=0 python mtasks_train.py --model resnet18 --dataset taskonomy --customize_class --class_to_train n  --class_to_test n 

