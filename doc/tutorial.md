# Muskits

## Recipes using Muskits

You can find the recipes in `egs` which follows the structure of ESPnet2 style:

```
muskit/  # Python modules of muskit
egs/     # corresponding recipes
```

The usage of recipes is **almost the same** as that of ESPnet2.


1. Change directory to the base directory

    ```bash
    # e.g.
    cd egs/kiritan/svs1/
    ```
    `kiritan` is one of the Japanese singing voice database. The KIRITAN data should be downloaded from https://zunko.jp/kiridev/login.php 
    You can perform any other recipes as the same way. e.g. `csd`, `natsume`, and etc.

    Keep in mind that all scripts should be ran at the level of `egs/*/svs1`.
    
    ```bash
    # Doesn't work
    cd egs/kiritan/
    ./asr1/run.sh
    ./asr1/scripts/<some-script>.sh
    
    # Doesn't work
    cd egs/kiritan//svs1/local/
    ./data.sh
    
    # Work
    cd egs/kiritan//svs1
    ./run.sh
    ./scripts/<some-script>.sh
    ```
    
1. Change the configuration
    Describing the directory structure as follows:
    
    ```
    egs2/kiritan/svs1/
     - conf/      # Configuration files for training, inference, etc.
     - scripts/   # Bash utilities of muskit
     - pyscripts/ # Python utilities of muskit
     - utils/     # From Kaldi utilities
     - db.sh      # The directory path of each corpora
     - path.sh    # Setup script for environment variables
     - cmd.sh     # Configuration for your backend of job scheduler
     - run.sh     # Entry point
     - svs.sh     # Invoked by run.sh
    ```

    - You need to modify `db.sh` for specifying your corpus before executing `run.sh`. For example, when you touch the recipe of `egs/kiritan`, you need to change the paths of `KIRITAN`in `db.sh`.
    - Some corpora can be freely obtained from the WEB and they are written as "downloads/" at the initial state. You can also change them to your corpus path if it's already downloaded.
    - `path.sh` is used to set up the environment for `run.sh`. Note that the Python interpreter used for ESPnet is not the current Python of your terminal, but it's the Python which was installed at `tools/`. Thus you need to source `path.sh` to use this Python.
        ```bash
        . path.sh
        python
        ```
    - `cmd.sh` is used for specifying the backend of the job scheduler. If you don't have such a system in your local machine environment, you don't need to change anything about this file. See [Using Job scheduling system](./parallelization.md)

1. Run `run.sh`

    ```bash
    ./run.sh
    ```

    `run.sh` is an example script, which we often call as "recipe", to run all stages related to DNN experiments; data-preparation, training, and evaluation.

## See training status

### Show the log file

```bash
% tail -f exp/*_train_*/train.log
[host] 2020-04-05 16:34:54,278 (trainer:192) INFO: 2/40epoch started. Estimated time to finish: 7 minutes and 58.63 seconds
[host] 2020-04-05 16:34:56,315 (trainer:453) INFO: 2epoch:train:1-10batch: iter_time=0.006, forward_time=0.076, loss=50.873, los
s_att=35.801, loss_ctc=65.945, acc=0.471, backward_time=0.072, optim_step_time=0.006, lr_0=1.000, train_time=0.203
[host] 2020-04-05 16:34:58,046 (trainer:453) INFO: 2epoch:train:11-20batch: iter_time=4.280e-05, forward_time=0.068, loss=44.369
, loss_att=28.776, loss_ctc=59.962, acc=0.506, backward_time=0.055, optim_step_time=0.006, lr_0=1.000, train_time=0.173
```

### Show the training status in a image file

```bash
# Accuracy plot
# (eog is Eye of GNOME Image Viewer)
eog exp/*_train_*/images/acc.img
# Attention plot
eog exp/*_train_*/att_ws/<sample-id>/<param-name>.img
```

### Use tensorboard

```sh
tensorboard --logdir exp/*_train_*/tensorboard/
```

# Instruction for run.sh
## How to parse command-line arguments in shell scripts?

All shell scripts in muskit depend on [utils/parse_options.sh](https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/utils/parse_options.sh) to parase command line arguments.

e.g. If the script has `ngpu` option

```sh
#!/usr/bin/env bash
# run.sh
ngpu=1
. utils/parse_options.sh
echo ${ngpu}
```

Then you can change the value as follows:

```sh
$ ./run.sh --ngpu 2
echo 2
```

You can also show the help message:

```sh
./run.sh --help
```

## Start from a specified stage and stop at a specified stage
The procedures in `run.sh` can be divided into some stages, e.g. data preparation, training, and evaluation. You can specify the starting stage and the stopping stage.

```sh
./run.sh --stage 2 --stop-stage 6
```

There are also some altenative otpions to skip specified stages:

```sh
run.sh --skip_data_prep true  # Skip data preparation stages.
run.sh --skip_train true      # Skip training stages.
run.sh --skip_eval true       # Skip decoding and evaluation stages.
run.sh --skip_upload false    # Enable packing and uploading stages.
```

Note that `skip_upload` is true by default. Please change it to false when uploading your model.

## Change the configuration for training
Please keep in mind that `run.sh` is a wrapper script of several tools including DNN training command. 
You need to do one of the following two ways to change the training configuration. 

```sh
# Give a configuration file
./run.sh --svs_train_config conf/train_svs.yaml
# Give arguments to "muskit/bin/svs_train.py" directly
./run.sh --svs_args "--foo arg --bar arg2"
```

See [Change the configuration for training](./muskit_training_option.md) for more detail about the usage of training tools.


## Change the number of parallel jobs

```sh
./run.sh --nj 10             # Chnage the number of parallels for data preparation stages.
./run.sh --inference_nj 10   # Chnage the number of parallels for inference jobs.
```

We also support submitting jobs to multiple hosts to accelerate your experiment: See [Using Job scheduling system](./parallelization.md)


## Use specified experiment directory for evaluation

If you already have trained a model, you may wonder how to give it to run.sh when you'll evaluate it later.
By default the directory name is determined according to given options, `svs_args`, or etc.
You can overwrite it by `--svs_exp`.

```sh
# For SVS recipe
./run.sh --skip_data_prep true --skip_train true --svs_exp <your_svs_exp_directory>
```

