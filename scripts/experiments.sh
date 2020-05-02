CUDA_VISIBLE_DEVICES=0 python mtasks_train.py --model resnet18 --dataset taskonomy --prune global_l1unstructured_40 --resume --resume_path /home/ag4202/backup/MAINMODEL3train_resnet-18_taskonomy_2020-04-30_05\:38\:21_13717b27_trainset_None_testset_None_lambda_0_seed_42_lrs_140_200/savecheckpoint/checkpoint_latest.pth.tar

CUDA_VISIBLE_DEVICES=0 python mtasks_train.py --model resnet18 --dataset taskonomy --prune global_random_40 --resume --resume_path /home/ag4202/backup/MAINMODEL3train_resnet-18_taskonomy_2020-04-30_05\:38\:21_13717b27_trainset_None_testset_None_lambda_0_seed_42_lrs_140_200/savecheckpoint/checkpoint_latest.pth.tar

CUDA_VISIBLE_DEVICES=0 python mtasks_train.py --model resnet18 --dataset taskonomy --prune local_l1unstructured_40 --resume --resume_path /home/ag4202/backup/MAINMODEL3train_resnet-18_taskonomy_2020-04-30_05\:38\:21_13717b27_trainset_None_testset_None_lambda_0_seed_42_lrs_140_200/savecheckpoint/checkpoint_latest.pth.tar

CUDA_VISIBLE_DEVICES=0 python mtasks_train.py --model resnet18 --dataset taskonomy --prune local_l2structured_40 --resume --resume_path /home/ag4202/backup/MAINMODEL3train_resnet-18_taskonomy_2020-04-30_05\:38\:21_13717b27_trainset_None_testset_None_lambda_0_seed_42_lrs_140_200/savecheckpoint/checkpoint_latest.pth.tar

CUDA_VISIBLE_DEVICES=0 python mtasks_train.py --model resnet18 --dataset taskonomy --prune local_random_40 --resume --resume_path /home/ag4202/backup/MAINMODEL3train_resnet-18_taskonomy_2020-04-30_05\:38\:21_13717b27_trainset_None_testset_None_lambda_0_seed_42_lrs_140_200/savecheckpoint/checkpoint_latest.pth.tar

