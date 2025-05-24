cd $INFERENCE_DIR

if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ] || [ -z "$5" ]; then
    echo "Usage: $0 <temp> <k> <IN_PATH> <OUT_PATH> <MODEL>"
    exit 1
fi

temp=$1
k=$2
IN_PATH=$3
OUT_PATH=$4
MODEL=$5

bash scripts/run_func.sh $temp $k $IN_PATH $OUT_PATH $MODEL

cd $MAIN_DIR