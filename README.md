# Deep Learning System Performance Project

Project repository

## How to run?

python mtasks_train.py --model resnet18 --dataset taskonomy

#### In debug mode (Helpful to set the batch size)
python mtasks_train.py --model resnet18 --dataset taskonomy --debug

### To prune with a strategy from config/prune/global_l1unstructured_40.json
python mtasks_train.py --model resnet18 --dataset taskonomy --debug --prune global_l1unstructured_40

### To resume using a checkpoint model
python mtasks_train.py --dataset taskonomy --model resnet18 --resume --resume_path <model_path eg - /home/ag4202/backup/saved/train_resnet-18_taskonomy_2020-04-29_05:57:14_fe0e7bc6_trainset_None_testset_None_lambda_0_seed_42_lrs_140_200_MAIN_MODEL_Arn/savecheckpoint/checkpoint_latest.pth.tar>

First check the GPU usage in train and val using --debug flag

### To run tensorboard
Go to the backup folder and cd in teh experiment directory. Run tensorboard there
eg - 

cd /home/ag4202/backup/saved/train_resnet-18_taskonomy_2020-04-29_05:57:14_fe0e7bc6_trainset_None_testset_None_lambda_0_seed_42_lrs_140_200_MAIN_MODEL_Arn/

tensorboard --logdir ./runs --host=0.0.0.0 --port=8080

Make sure port is open in GCP

In GCP Console, pull the drawer and go to Networks/VPC network. Create a firewall rule. (Ingress , tcp: 8080, IP range 0.0.0.0/0)

Then open in browser <instance IP>
:8080
