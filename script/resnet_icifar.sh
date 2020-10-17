layer=$1
dataset=cifar100
tmodel_name=resnet${1}_cifar
smodel_name=resnet${1}_cifar_dirac
seed=$4
procedure=$3
aim=Correct_cnn_${dataset}_${smodel_name}_${layer}_$1_${procedure}_SEED_${seed}_DC_${2}_keepdownsampling
save_dir=/cache/code/output/${dataset}/${tmodel_name}/${aim}/
data_dir=/cache/dataset/

dc=$2

###############

echo ${procedure}
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python train_dirac.py --stage CNN_NMT \
                       --baseline_epochs 200 \
                       --cutout_length 0 \
                       --procedure ${procedure} \
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
                       --kd_type margin \
                       --dis_weight 0. \
                       --lr_sch step \
                       --dc ${dc} \
                       --model ${model_dir} \

                       
                       
                       
