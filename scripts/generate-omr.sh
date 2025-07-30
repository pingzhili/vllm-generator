URLS=$(printf "http://localhost:800%d " {0..7})

python -m vllm_generator generate \
    --input /root/open-math-reasoning/sample_10.parquet \
    --output /root/open-math-reasoning/sample_10_results_repeat_.parquet \
    --model-urls $URLS \
    --num-workers 16 \
    --batch-size 128