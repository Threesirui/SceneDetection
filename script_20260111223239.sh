#!/bin/bash

#SBATCH --job-name=vscode
#SBATCH --partition=jupyter
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1   # 一张卡默认配置3个CPU核心(根据卡数自行调整);
#SBATCH --gres=shard:1
#SBATCH -o %j.out
#SBATCH -e %j.err


#生成随机端口
port=$((RANDOM % (65535 - 2048) + 1024))

# 检查端口是否被占用
while [[ $(ss -tuln | grep ":$port ") ]]; do
    port=$((port + 1))
done

#查看启动节点
node=$(cat /etc/hosts |grep -m 1 `hostname -s` |awk '{print $1}')
host=$(hostname -s)
echo "随机未被占用的端口号： /$host/$port"



#输出启动用户
user=$(whoami)

CODE_SERVER_DATAROOT="$(pwd)/code-server"
mkdir -p "$CODE_SERVER_DATAROOT/extensions"

if [ -z "/seu_nvme/home/huangjie/230248634/SceneDetection" ]; then
  echo "work_dir is empty"
  nohup /seu_share/apps/code-server-4.91.1/bin/code-server \
  --port $port \
  --host "0.0.0.0" \
  --user-data-dir "$CODE_SERVER_DATAROOT" \
  --log debug "$CODE_SERVER_DATAROOT" \
  --config /seu_share/OGSP/vscode/config.yaml \
  --disable-telemetry \
  --ignore-last-opened > out_$SLURM_JOBID.log 2>&1 &
else
  echo "work_dir is not empty"
  nohup /seu_share/apps/code-server-4.91.1/bin/code-server \
  --port $port \
  --host "0.0.0.0" \
  --user-data-dir "/seu_nvme/home/huangjie/230248634/SceneDetection" \
  --log debug "/seu_nvme/home/huangjie/230248634/SceneDetection" \
  --config /seu_share/OGSP/vscode/config.yaml \
  --disable-telemetry \
  --ignore-last-opened > out_$SLURM_JOBID.log 2>&1 &
fi



#获取code server url
url="$(cat out_$SLURM_JOBID.log |grep '0.0.0.0' |tail -n1 |grep -oE 'http://[^ ]+' | awk '{print $1}')"
while true
do
  if [[ "$url" == "" ]];then
    sleep 2
    url="$(cat out_$SLURM_JOBID.log |grep '0.0.0.0' |tail -n1 |grep -oE 'http://[^ ]+' | awk '{print $1}')"
  else
    break
  fi
done

# 生成ogsp所需信息
echo -e "{'hostname':'$host','port':'$port','webUrl':'$url'}" > ogsp_conn_$SLURM_JOBID.json
sleep infinity