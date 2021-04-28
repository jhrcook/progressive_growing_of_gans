#!/bin/sh

# Run the training of the Progressive Growing of GAN on O2 using GPUs.

#SBATCH -c 2
#SBATCH -t 1-15:50
#SBATCH --mem 24G
#SBATCH -p gpu
#SBATCH --gres=gpu:teslaV100:4
#SBATCH -o logs/%j.log
#SBATCH -e logs/%j.log

module load gcc conda2


# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/n/app/conda2/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/n/app/conda2/etc/profile.d/conda.sh" ]; then
        "/n/app/conda2/etc/profile.d/conda.sh"
    else
        export PATH="/n/app/conda2/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<


conda activate pggan

echo "Python information:"
python3 --version
which python3
echo "==========================="

python3 train.py
