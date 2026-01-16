#!/bin/bash

# AF Simulator 启动脚本
# 使用方法: ./bash.sh [可选参数]

# 设置默认参数
GENERATOR=${GENERATOR:-1}
NUM_SERVER=${NUM_SERVER:-1}
NUM_BATCH=${NUM_BATCH:-2}
BATCH_SIZE=${BATCH_SIZE:-16}
USE_LENGTH_LIMIT=${USE_LENGTH_LIMIT:-""}
BATCH_MAX_LENGTH=${BATCH_MAX_LENGTH:-65536}

NEXT_TOKEN_PROB=${NEXT_TOKEN_PROB:-0.99}
GEN_PROB=${GEN_PROB:-0.001}
RATE=${RATE:-1}
BASIC_NUM=${BASIC_NUM:-80}
GEN_REQ_PER_CYC=${GEN_REQ_PER_CYC:-1}
TOTAL_REQUEST=${TOTAL_REQUEST:-80}

MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-512}
MAXIMAL_GENERATION=${MAXIMAL_GENERATION:-800}

REQUEST_MODE=${REQUEST_MODE:-"geometric_input_output"}
LI=${LI:-100}
LO=${LO:-50}
P=${P:-0.99}
Q=${Q:-0.99}

NUM_FFN=${NUM_FFN:-1}

ALPHA_A=${ALPHA_A:-0.03}
ALPHA_T=${ALPHA_T:-0.001}
ALPHA_F=${ALPHA_F:-3.5}
BETA_A=${BETA_A:-5.0}
BETA_T=${BETA_T:-0.0}
BETA_F=${BETA_F:-8.0}

OUT_PREFIX=${OUT_PREFIX:-""}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --generator)
            GENERATOR="$2"
            shift 2
            ;;
        --num-server)
            NUM_SERVER="$2"
            shift 2
            ;;
        --num-batch)
            NUM_BATCH="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --use-length-limit)
            USE_LENGTH_LIMIT="--use_length_limit"
            shift
            ;;
        --batch-max-length)
            BATCH_MAX_LENGTH="$2"
            shift 2
            ;;
        --next-token-prob)
            NEXT_TOKEN_PROB="$2"
            shift 2
            ;;
        --gen-prob)
            GEN_PROB="$2"
            shift 2
            ;;
        --rate)
            RATE="$2"
            shift 2
            ;;
        --basic-num)
            BASIC_NUM="$2"
            shift 2
            ;;
        --gen-req-per-cyc)
            GEN_REQ_PER_CYC="$2"
            shift 2
            ;;
        --total-request)
            TOTAL_REQUEST="$2"
            shift 2
            ;;
        --max-prompt-len)
            MAX_PROMPT_LEN="$2"
            shift 2
            ;;
        --maximal-generation)
            MAXIMAL_GENERATION="$2"
            shift 2
            ;;
        --request-mode)
            REQUEST_MODE="$2"
            shift 2
            ;;
        --LI)
            LI="$2"
            shift 2
            ;;
        --LO)
            LO="$2"
            shift 2
            ;;
        --p)
            P="$2"
            shift 2
            ;;
        --q)
            Q="$2"
            shift 2
            ;;
        --num-FFN)
            NUM_FFN="$2"
            shift 2
            ;;
        --alpha-A)
            ALPHA_A="$2"
            shift 2
            ;;
        --alpha-T)
            ALPHA_T="$2"
            shift 2
            ;;
        --alpha-F)
            ALPHA_F="$2"
            shift 2
            ;;
        --beta-A)
            BETA_A="$2"
            shift 2
            ;;
        --beta-T)
            BETA_T="$2"
            shift 2
            ;;
        --beta-F)
            BETA_F="$2"
            shift 2
            ;;
        --out-prefix)
            OUT_PREFIX="$2"
            shift 2
            ;;
        -h|--help)
            echo "AF Simulator 启动脚本"
            echo ""
            echo "使用方法: ./bash.sh [选项]"
            echo ""
            echo "选项:"
            echo "  --generator INT              生成器类型 (0=uniform, 1=random-uniform, 2=geometry, 3=poisson) [默认: 1]"
            echo "  --num-server INT              服务器数量 [默认: 1]"
            echo "  --num-batch INT               每个服务器内的批次数量 [默认: 2]"
            echo "  --batch-size INT              每个批次内的请求数量 [默认: 16]"
            echo "  --use-length-limit            使用最大长度限制"
            echo "  --batch-max-length INT        批次内最大允许的token数 [默认: 65536]"
            echo "  --next-token-prob FLOAT       pipeline中下一个token的概率 [默认: 0.95]"
            echo "  --gen-prob FLOAT              生成下一个token的概率 [默认: 0.001]"
            echo "  --rate INT                    生成token的周期频率 [默认: 1]"
            echo "  --basic-num INT               第一个周期生成的请求数 [默认: 80]"
            echo "  --gen-req-per-cyc INT         每个周期生成的请求数 [默认: 1]"
            echo "  --total-request INT           停止实验前生成的总请求数 [默认: 80]"
            echo "  --max-prompt-len INT           生成请求的最大prompt长度 [默认: 4096]"
            echo "  --maximal-generation INT       最大生成数 [默认: 80]"
            echo "  --request-mode STRING          请求生成模式 (default/identical/geometric_output/geometric_input_output) [默认: default]"
            echo "  --LI INT                      输入长度（identical模式） [默认: 100]"
            echo "  --LO INT                      输出长度（identical模式） [默认: 50]"
            echo "  --p FLOAT                     geometric分布概率（输出） [默认: 0.5]"
            echo "  --q FLOAT                     geometric分布概率（输入，geometric_input_output模式） [默认: 0.5]"
            echo "  --num-FFN INT                 FFN worker数量 [默认: 1]"
            echo "  --alpha-A FLOAT               Alpha A参数 [默认: 0.1]"
            echo "  --alpha-T FLOAT               Alpha T参数 [默认: 0.001]"
            echo "  --alpha-F FLOAT               Alpha F参数 [默认: 0.1]"
            echo "  --beta-A FLOAT                Beta A参数 [默认: 512.0]"
            echo "  --beta-T FLOAT                Beta T参数 [默认: 16.0]"
            echo "  --beta-F FLOAT                Beta F参数 [默认: 512.0]"
            echo "  --out-prefix STRING           输出文件前缀 [默认: \"\"]"
            echo "  -h, --help                    显示此帮助信息"
            echo ""
            echo "示例:"
            echo "  ./bash.sh --generator 1 --num-server 2 --batch-size 32 --total-request 100"
            echo "  ./bash.sh --use-length-limit --batch-max-length 32768"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 构建命令
CMD="python3 main.py"
CMD="$CMD --generator $GENERATOR"
CMD="$CMD --num_server $NUM_SERVER"
CMD="$CMD --num_batch $NUM_BATCH"
CMD="$CMD --batch_size $BATCH_SIZE"
if [ -n "$USE_LENGTH_LIMIT" ]; then
    CMD="$CMD $USE_LENGTH_LIMIT"
fi
CMD="$CMD --batch_max_length $BATCH_MAX_LENGTH"
CMD="$CMD --next_token_prob $NEXT_TOKEN_PROB"
CMD="$CMD --gen_prob $GEN_PROB"
CMD="$CMD --rate $RATE"
CMD="$CMD --basic_num $BASIC_NUM"
CMD="$CMD --gen_req_per_cyc $GEN_REQ_PER_CYC"
CMD="$CMD --total_request $TOTAL_REQUEST"
CMD="$CMD --max_prompt_len $MAX_PROMPT_LEN"
CMD="$CMD --maximal_generation $MAXIMAL_GENERATION"
CMD="$CMD --request_mode $REQUEST_MODE"
CMD="$CMD --LI $LI"
CMD="$CMD --LO $LO"
CMD="$CMD --p $P"
CMD="$CMD --q $Q"
CMD="$CMD --num_FFN $NUM_FFN"
CMD="$CMD --alpha_A $ALPHA_A"
CMD="$CMD --alpha_T $ALPHA_T"
CMD="$CMD --alpha_F $ALPHA_F"
CMD="$CMD --beta_A $BETA_A"
CMD="$CMD --beta_T $BETA_T"
CMD="$CMD --beta_F $BETA_F"
if [ -n "$OUT_PREFIX" ]; then
    CMD="$CMD --out_prefix $OUT_PREFIX"
fi

# 显示执行的命令
echo "=========================================="
echo "执行命令:"
echo "$CMD"
echo "=========================================="
echo ""

# 执行命令
$CMD

