#!/bin/bash

arg_fn='/mnt/atlas/WRI/args.in'
out_fn='/mnt/atlas/WRI/gpw_strata.count_esri.54008.pq'
n_parallel=48

n_lines=$(cat $arg_fn | wc -l)
n_chunk=$(( $n_lines / $n_parallel ))

for i0 in $(seq 0 $n_chunk $n_lines); do
    i1=$(($i0 + n_chunk))
    i1=$([[ "$i1" -lt "$n_lines" ]] && echo $i1 || echo $n_lines)
    echo python gpw_strata.assessment.py $arg_fn $i0 $i1 $out_fn
done