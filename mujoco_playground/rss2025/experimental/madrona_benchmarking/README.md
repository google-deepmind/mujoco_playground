A section for studying the performance of [Madrona MJX](https://github.com/shacklettbp/madrona_mjx) for the CartpoleBalance and PandaPickCubeCartesian envs from Mujoco Playground.

#### Data Collection
`benchmark.py`: Instantiates the selected env + configuration and creates / benchmarks an unroll as specified in the CLI arguments. Each run appends a row to data/madrona_mjx.csv.
`get_data.sh`: Runs multiple trials of benchmark.py with different arguments. Collects all data required for the two analysis scripts except for PPO training curves, which are loaded from ../data.

#### Data Analysis
`make_plots.py`: Analyzes pixels Cartpole performance with random actions as a function of image size and num_envs. Produces various plots under figures/
`print_tables.py`: Analyzes the bottlenecks in PPO training of Cartpole and PandaPickCubeCartesian. Prints a table to stdout.
