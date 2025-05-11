# Train with TC_branch, V_branch, and F_branch on T2VQA-DB
nohup python main.py --batch_size=8 --num_epochs=100 --lr=1e-5 --backbone=swinv2 --weight_decay=1e-4 --flow --wandb --ada_voter --F_branch --V_branch >log 2>&1 &
# Train on LGVQ
nohup python main.py --batch_size=8 --num_epochs=100 --lr=1e-5 --backbone=swinv2 --weight_decay=1e-4 --flow --wandb --ada_voter --F_branch --V_branch --data_name=LGVQ --data_path=/data0/xxy/data/LGVQ/videos --anno_file=/data0/xxy/data/LGVQ/MOS.txt >log_lgvq 2>&1 &