#!/bin/bash
#SBATCH -o /seu_nvme/home/huangjie/230248634/SceneDetection/slurm/%J.out            # 脚本执行的输出将被保存在当 %J.out文件下，%j表示作
#SBATCH -e /seu_nvme/home/huangjie/230248634/SceneDetection/slurm/%J.err            # 脚本执行的错误日志将被保存在当 %J.err文件下，%j表
#SBATCH --partition=gpu_v100     # 作业提交的指定队列/分区为normal_test；
#SBATCH --job-name=CL     # 作业在调度系统中的作业名为myFirstJob;
#SBATCH --nodes=1            # 申请节点数为1,如果作业不能跨节点(MPI)运行, 申请的节点数应不超过1;
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3    # 一张卡默认配置3个CPU核心(根据卡数自行调整);
#SBATCH --gres=gpu:1         # 申请一块GPU卡;
module load anaconda3
#module load cuda-12.4
source activate MAGE



python main.py