# Current code for Deep Survival Machines.

## How to run:
1. __Clone conda environment:__ `conda env create -f environment.yml`
2. __Activate conda environment:__ `conda activate DeepSurvivalMachines`
3. __Run:__ `python train.py --path /home/zrashidi/OrganAI/OrganAI/ --hidden_size 50,100 --mlp_type 3 --dataset SEER --num_experts 8 --x_fold 5 --distribution Weibull --lr 1e-3 --num_iter 1000 --batch_size 25 --alpha 1 --threshold 1e-4 --lambda_ 1e-2 --ELBO --seed 0`
- __NOTE:__ Change the `--path` argument to your local repository
- __NOTE:__ To switch between different versions of SEER dataset, refer to part of the code after `elif args.dataset == 'SEER` in `train.py`
