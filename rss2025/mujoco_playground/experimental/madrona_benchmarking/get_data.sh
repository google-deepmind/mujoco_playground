# Main Paper Figure. Use img_size = 64 data from Bottlenecks Figure.
for img_size in 128 256 512
do
    for env in "CartpoleBalance" "PandaPickCubeCartesian"
    do
        for ((run=0; run<5; run++)); do
            python benchmark.py \
                --num-envs 1024 --img-size $img_size --env-name $env \
                --measurement-mode 1 --bottleneck-mode
        done
    done
done

# Benchmark against Maniskill3 tables (state-based)
for num_envs in 4 16 32 64 128 256 512 1024 2048 4096 8192 16384
do
    for ((run=0; run<5; run++)); do
        python benchmark.py \
            --num-envs $num_envs --img-size 0 --env-name CartpoleBalance \
            --measurement-mode 0
    done
done

# Benchmark against Maniskill3 tables (pixel-based)
for img_size in 80 128 224 256
do
    for num_envs in 128 256 512 1024
    do
        for ((run=0; run<5; run++)); do
            python benchmark.py \
                --num-envs $num_envs --img-size $img_size --env-name CartpoleBalance \
                --measurement-mode 1
        done
    done
done

# Bottlenecks Figure
for env in "CartpoleBalance" "PandaPickCubeCartesian"
do
    for mode in 0 1 2
    do
        for ((run=0; run<5; run++)); do
            python benchmark.py \
                --num-envs 1024 --env-name $env --img-size 64 --measurement-mode $mode \
                --bottleneck-mode
        done
    done
done

