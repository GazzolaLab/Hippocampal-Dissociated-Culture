export DATA_PREFIX=./datasets

mpirun -n 1 network-clamp go  \
       -c Network_Clamp_IS3_gid_840420.yaml \
       --template-paths templates --dt 0.01 \
       -p IS3 -g 840420  -t 5000 \
       --recording-profile "Network clamp exc synaptic" \
       --dataset-prefix $DATA_PREFIX \
       --config-prefix config \
       --arena-id A --stimulus-id Diag \
       --input-features-path $DATA_PREFIX/Full_Scale/Full_Scale_CA1_input_features.h5 \
       --input-features-namespaces 'Constant Selectivity' \
       --results-path results/netclamp
