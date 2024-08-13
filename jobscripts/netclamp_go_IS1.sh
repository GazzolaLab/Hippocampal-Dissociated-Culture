export DATA_PREFIX=./datasets

mpirun -n 1 network-clamp go  \
       -c Network_Clamp_IS1_gid_835350.yaml \
       --template-paths templates --dt 0.01 \
       -p IS1 -g 835350  -t 5000 \
       --recording-profile "Network clamp exc synaptic" \
       --dataset-prefix $DATA_PREFIX \
       --config-prefix config \
       --arena-id A --stimulus-id Diag \
       --input-features-path $DATA_PREFIX/Full_Scale/Full_Scale_CA1_input_features.h5 \
       --input-features-namespaces 'Constant Selectivity' \
       --results-path results/netclamp
