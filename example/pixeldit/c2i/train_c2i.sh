#!/bin/bash

NUM_NODES=1
NUM_GPUS=8
MASTER_ADDR=localhost
MASTER_PORT=29500
NODE_RANK=0
CONFIG_FILE="configs/pix256_xl.yaml"
CKPT_PATH=""
C2I_ROOT=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$C2I_ROOT/.." && pwd)
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --num-nodes)
            NUM_NODES="$2"
            shift 2
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --master-addr)
            MASTER_ADDR="$2"
            shift 2
            ;;
        --master-port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --node-rank)
            NODE_RANK="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --ckpt-path)
            CKPT_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Starting distributed training with torchrun..."
if [[ "$CONFIG_FILE" != /* ]]; then
    CONFIG_FILE="$C2I_ROOT/$CONFIG_FILE"
fi

echo "Config: $CONFIG_FILE"
echo "Nodes: $NUM_NODES, GPUs per node: $NUM_GPUS"
echo "Master: $MASTER_ADDR:$MASTER_PORT, Node rank: $NODE_RANK"

CMD=(torchrun
    --nnodes="$NUM_NODES"
    --nproc_per_node="$NUM_GPUS"
    --master_addr="$MASTER_ADDR"
    --master_port="$MASTER_PORT"
    --node_rank="$NODE_RANK"
    "$C2I_ROOT/main.py" fit
    -c "$CONFIG_FILE"
    --trainer.num_nodes="$NUM_NODES"
    --trainer.devices="$NUM_GPUS")

if [[ -n "$CKPT_PATH" ]]; then
    CMD+=("--ckpt_path=$CKPT_PATH")
fi

"${CMD[@]}"