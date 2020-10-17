layer=$1
dataset=cifar100
tmodel_name=resnet${1}_cifar
smodel_name=resnet${1}_cifar
aim=${smodel_name}_${layer}_baseline
save_dir=/cache/code/output/${dataset}/${tmodel_name}/${aim}/
data_dir=/cache/dataset/
seed=1
python mox.py others -s='s3://bucket-auto/Zhangjunlei/dataset/' -t=${data_dir}
CUDA_VISIBLE_DEVICES=1 python train.py --stage RES_NMT \
                       --baseline_epochs 300 \
                       --cutout_length 0 \
                       --procedure RES_NMT \
                       --save_dir ${save_dir} \
                       --smodel_name ${smodel_name} \
                       --tmodel_name ${tmodel_name} \
                       --dataset ${dataset} \
                       --data_dir ${data_dir} \
                       --seed ${seed} \
                       --learning_rate 0.1 \
                       --batch_size 128 \
                       --aim ${aim} \
                       --start_epoch 0 \
                       --alpha 0.9 \
                       --weight_decay 5e-4 \
                       --kd_type none \
                       --dis_weight 1e-3 \
                       --lr_sch cosine \

                       
                       
                       
