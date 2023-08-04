# Neural Abstractions

This repository serves as the artifact for the paper "On the Trade-off between Precision and Efficiency of Neural Abstraction". We use Docker to create a reproducible environment for the experiments. The Docker image is based on Ubuntu 22.04. Unfortunately, we require on SpaceEx for part of the experiments which users must register for and download themselves. We include instructions on how to do this below. All other dependencies are installed automatically during the Docker build process.

## System Requirements

* Ideally, users are running Linux as the host OS.
* Many of experiments rely on process-based parallelism, and so the machine running the experiments should have at least 6 cores. We only use 4 at any time, but the experiments in the paper are ran on a machine with 8 cores. Using fewer than 4 cores will break reproducibility.
* We do not know the exact memory requirements, but the machine used to run the experiments had 16GB of RAM, so we recommend at least this much.
* The host machine will also need 15GB of free disk space to store the Docker image and results.
* M1 and M2 Macbooks (and other ARM64 architechtures) are not supported. This is due to an architecture incompatibility preventing the building of DReal ([see here](https://github.com/dreal/dreal4/issues/295)).

## Repository Cloning

Clone the repository (qest-na.zip).

```console
git clone https://github.com/aleccedwards/qest-na
```

The root directory should contain various python files, scripts, a dockerfile, and subdirectories named `experiments`, `results` and `cegis`. The `experiments` directory contains the experiments to be run, the `results` directory contains the results of the experiments, and the `cegis` directory contains more python code. The `experiments` and `results` directories are mounted to the Docker container, so that the results can be easily accessed from the host machine.

## Install SpaceEx

A registration is required to install SpaceEx, though it is available for free. It can be installed into the Docker container as follows.
On your host machine, go to <http://spaceex.imag.fr/download-6>, and download the SpaceEx command line executable v0.9.8f (64 bit Linux), under the SpaceEx Analysis Core. This version is required for the Docker container. This README assumes you have a 64 bit architecture; we have not tested the results on other architectures.
Extract the corresponding archive to some location on your host machine. Then move the spaceex_exe folder to the project root directory. The folder should be named `spaceex_exe`. If you are struggling to extract the archive, you might need to go down a level in the directory structure.

## Docker Installation

Docker can be installed by visiting <https://docs.docker.com/get-docker/>. Note that building the image may take a while due to downloading and building the dependencies, including PyTorch, Dreal and Flow*. Within the project root directory, run:

```console
# docker build -t qest-na:latest .
```

The build time is around 10 minutes with good internet connection. Also, depending on how Docker is installed, you may need to run the Docker commands with `sudo`.

We mount the results directory to the host machine so that the results can be easily accessed from the host. Start the container with:

```console
# docker run --name na -v $PWD/results:/neural-abstraction/results/ -v $PWD/experiments:/neural-abstraction/experiments/ -it qest-na:latest bash 
```

You are now inside the container. Move to the project directory.

```console
cd /neural-abstraction
```

You are now able to run the program. The settings for a program are determined using a .yaml config file, the default location for which is `./config.yaml`. The used config file can be changed using the `-c` command line option.

## Running Experiments

### Main Results

We provide scripts to automate the running of experiments. All scripts should be run from the project root directory.
For the results in the main table (and corresponding time table), the script `experiments.py` should be used. This takes two arguments: `-m`, which takes a list of models to run, and `-t`, which takes a list of templates. The models correspond to benchmarks, and the options are "nl1 nl2 watertank tank4d tank6d node1". The parameter "all" can be passed instead to run over all models.
The templates correspond to different abstract models, with options "pwc pwa nl". Again, "all" can be passed to run over all templates.

Due to the long runtime of experiments, we do not recommend running all results here. A suitable subset of results can be obtained by running the following commands:

```console
python3 experiments.py -m nl1 nl2 watertank -t all
```

**NOTE: Do NOT try to run more than one experiment or script or command at once. Due to the use of process-based parallelism and a (hacky) fix to terminate SpaceEx and Flow\* processes, running multiple experiments at once will cause the processes to terminate prematurely.**

To facilitate reviewers, we include the approximate expected runtimes of each experiment in the table below. These are based on the machine used to run the experiments, which had 8 cores and 16GB of RAM. We encourage reviewers to repeat whichever combination of experiments they wish, but we recommend running at least the experiments in the table below. Our recommended experiments, being nl1, nl2 and watertank over all templates, should take around 3 hours to run.

|                           |   Total Time (s)      |
|:--------------------------|----------------------:|
| ('Non-Lipschitz1', 'pwa') |              530      |
| ('Non-Lipschitz1', 'pwc') |              110      |
| ('Non-Lipschitz1', 'sig') |             1200      |
| ('Non-Lipschitz2', 'pwa') |              1000     |
| ('Non-Lipschitz2', 'pwc') |              250      |
| ('Non-Lipschitz2', 'sig') |             4950      |
| ('Water-tank', 'pwa')     |              740      |
| ('Water-tank', 'pwc')     |               50      |
| ('Water-tank', 'sig')     |              200      |
| ('Water-tank-4d', 'pwa')  |              230      |
| ('Water-tank-4d', 'pwc')  |             3850      |
| ('Water-tank-4d', 'sig')  |             1500      |
| ('Water-tank-6d', 'pwa')  |             8400      |
| ('NODE1', 'pwa')          |              210      |
| ('NODE1', 'pwc')          |              810      |
| ('NODE1', 'sig')          |             1300      |

### Figures

The figures can be reproduced using the script:
  
```console
./abstraction_figures.sh
```

The script should take less than 10 minutes to run.
This will generate 7 figures in the folder 'experiments/nl2', four pdfs of the model abstractions and three postscript (ps) files of the flowpipes.

### Error Refinement

The error refinement results are presented in the appendix. Reviewers may also repeat these results if they wish, but we do not consider them a significant part to be reproduced (though we expect them to be reproducable). They can be reproduced using the script:
  
```console
./error_refine.sh
```

The script should take around 1.5 hours to run.
Please note: various failures within SpaceEx may appear during the error refinement experiments. These are not fatal, are handled by the program, and are part of the error refinement experiments.

### Tables

Once the experiments have been run, the tables can be generated using the script:
  
```console
python3 qest_res.py
```

This will produce three files in the results folder: "main_tab.tex", "time_tab.tex" and "error_check.tex".
If the relevant experiments have not been run (e.g., the error refinement experiments), the script will use the backup files in the results folder. These are the results from the machine used to run the experiments. The script will also print to the console any results that timed out. These results are not included in the tables. The tables are formatted differently to the paper, but they contain the same information.

Finally, we note that the results obtained by reviewers may not be identical to ours. This is due to several reasons. First is the use of process-based parallelism, which can lead to non-deterministic behaviour. Secondly, it is known that identical random seeds are not reproducible across different machines in [PyTorch](https://pytorch.org/docs/stable/notes/randomness.html).
However, the results should be similar, and the conclusions should be the same.

### Flow* Neural ODEs

The baseline Flow* result Section 5.2 of reachability on a neural ODE can be reproduced using the script:
  
```console
./node-flowstar.sh
```
