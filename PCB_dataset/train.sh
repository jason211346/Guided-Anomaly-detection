# data regroup


for seed in 2 211 1212
do 
    
    CUDA_VISIBLE_DEVICES=0 python train_CEdue.py --output_dir /root/notebooks/nfs/work/jason.chen/DUE/regroup_output_model/CEdue --seed $seed
    
done



for seed in 2 211 1212
do 
    ckpt=/root/notebooks/nfs/work/jason.chen/DUE/regroup_output_model/CEdue/$seed/ckpt.pt
    
    CUDA_VISIBLE_DEVICES=0 python train_CEDUE_stage2.py --output_dir /root/notebooks/nfs/work/jason.chen/DUE/regroup_output_model/CEdue_s2 --checkpoint_path $ckpt --seed $seed
    
done