#!/bin/bash

## Run NEAT in profiling mode
# Save name of results subfolder in the name variable
declare -a name=($(python3 -c "import sys; sys.path.append('..'); import inputs; print(inputs.name)"))
# Run NEAT in profiling mode
python3 -m yep -v -- main.py
google-pprof --svg main.py main.py.prof > ../results/${name}/profiler_result.svg

## Run NEAT normally
# python3 main.py