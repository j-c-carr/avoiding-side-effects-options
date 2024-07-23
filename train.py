"""
Main entry point for starting a training job.
"""

import os
import sys
import argparse
import subprocess
import shutil
import torch
import random
import logging
import logging.config

import numpy as np

env_types = [
    'test',
    'append-still',
    'append-still-easy',
    'append-spawn',
    'prune-still',
    'prune-spawn',
    'navigate',
]

parser = argparse.ArgumentParser(description="""
    Run agent training using proximal policy optimization.

    This will set up the data/log directories, optionally install any needed
    dependencies, start tensorboard, configure loggers, and start the actual
    training loop. If the data directory already exists, it will prompt for
    whether the existing data should be overwritten or appended. The latter
    allows for training to be restarted if interrupted.
    """)
parser.add_argument('--log_dir', type=str, default='logs/tmp', help='Logging directory'),
parser.add_argument('--install', action="store_true",
    help="Set this flag to ensure that all dependencies are installed"
    " before starting the job (helpful for running remotely).")
parser.add_argument('--shutdown', action="store_true",
    help="Shut down the system when the job is complete"
    "(helpful for running remotely).")
parser.add_argument('--port', default=0, type=int,
    help="Port on which to run tensorboard.")
parser.add_argument('--impact-penalty', default=0.0, type=float)
parser.add_argument('--env-type', choices=env_types)
parser.add_argument('--algo', default='ppo', choices=('ppo', 'aup', 'aup-p', 'naive'))
parser.add_argument('--config', default='ppo', choices=('ppo', 'aup', 'aupp', 'naive'))
parser.add_argument('--z', default=1.0)
parser.add_argument('--n_envs', type=int, default=8)
parser.add_argument('--wandb', action="store_true", help="Enable wandb")
parser.add_argument('--project', type=str, default=None, help="Wandb project name")
parser.add_argument('--name', type=str, default=None, help="Name of the run")
parser.add_argument('--seed', type=int, default=0, help="Seed")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def spawn_loader(child, parent, n_levels, seed):
    # When training in spawn environments, we first pre-train in the static
    # environments for a couple million time steps. This just provides more
    # opportunities for rewards so makes the initial training easier.
    from safelife.file_finder import SafeLifeLevelIterator
    from safelife.safelife_env import SafeLifeEnv

    loader1 = SafeLifeLevelIterator('safelife/levels/random/{}-easy'.format(child), seed=seed)
    loader2 = SafeLifeLevelIterator(parent, distinct_levels=n_levels, total_levels=-1, seed=seed)
    t0 = 2.0e6
    while True:
        if SafeLifeEnv.global_counter.num_steps < 0:
            yield next(loader1)
        else:
            yield next(loader2)


def spawn_loader_aup(child, parent, n_levels, seed):
    from safelife.file_finder import SafeLifeLevelIterator
    from safelife.safelife_env import SafeLifeEnv

    loader1 = SafeLifeLevelIterator('safelife/levels/random/{}-easy'.format(child), seed=seed)
    loader2 = SafeLifeLevelIterator(parent, distinct_levels=n_levels, total_levels=-1, seed=seed)

    trand = 0
    tstill = 1.0e6
    tspawn = 3.0e6
    while True:
        count = SafeLifeEnv.global_counter.num_steps
        if count >= trand and count < tstill:
            p = np.random.randint(2, size=1)[0]
            yield next(loader1) if p else next(loader2)

        elif count >= tstill and count < tspawn:
            yield next(loader1)
        
        elif count >= tspawn:
            yield next(loader2)
        else:
            raise ValueError


def main(args):
    try:
        from training.env_factory import linear_schedule, safelife_env_factory
        from safelife.file_finder import SafeLifeLevelIterator
        from tensorboardX import SummaryWriter
        """ Conditions
        Loops:
            run_env_types: the environment strings that we will test each
                algorithm for,
            batches (int): the number of sets of N levels that we want to
                average over in the end. Default is 5 batches of 8 levels
            n_levels (int): the number of discrete levels in each batch
        """

        run_env_types = [
                'append-still-easy',
                #'append-still',
                #'append-spawn',
                #'prune-still'
                ]

        n_levels = 8

        if args.algo == 'naive':
            penalty = 1.0
        else:
            penalty = 0.0

        # Loop over environment conditions
        for run_env_type in run_env_types:
            #log_dir = 'training_results/{}/{}/'.format(args.algo, run_env_type)
            print('-------------------------------------------')
            print('Running {} in {}'.format(
                args.algo,
                run_env_type))
            print('-------------------------------------------')

            # Setup the directories
            safety_dir = os.path.realpath(os.path.join(__file__, '../'))
            active_job_file = os.path.join(safety_dir, 'active_job.txt')
            sys.path.insert(1, safety_dir)  # ensure current directory is on the path
            os.chdir(safety_dir)

            # If the run name isn't supplied, get it from 'active_job.txt'
            # This is basically just used to restart after crashes.
            log_dir = os.path.realpath(args.log_dir)
            job_name = os.path.split(log_dir)[1]

            if os.path.exists(args.log_dir) and args.log_dir is not None:
                print(f"WARNGING: overriding {args.log_dir}")

            os.makedirs(log_dir, exist_ok=True)
            logfile = os.path.join(log_dir, 'training.log')

            # Get the environment type from the job name if not otherwise supplied
            assert run_env_type in env_types

            # Setup logging
            if not os.path.exists(logfile):
                open(logfile, 'w').close()  # write an empty file
            logging.config.dictConfig({
                'version': 1,
                'disable_existing_loggers': False,
                'formatters': {
                    'simple': {
                        'format': '{levelname:8s} {message}',
                        'style': '{',
                        },
                    'dated': {
                        'format': '{asctime} {levelname} ({filename}:{lineno}) {message}',
                        'style': '{',
                        'datefmt': '%Y-%m-%d %H:%M:%S',
                        },
                    },
                'handlers': {
                    'console': {
                        'class': 'logging.StreamHandler',
                        'level': 'INFO',
                        'stream': 'ext://sys.stdout',
                        'formatter': 'simple',
                        },
                    'logfile': {
                        'class': 'logging.FileHandler',
                        'level': 'INFO',
                        'formatter': 'dated',
                        'filename': logfile,
                        }
                    },
                'loggers': {
                    'training': {
                        'level': 'INFO',
                        'propagate': False,
                        'handlers': ['console', 'logfile'],
                        },
                    'safelife': {
                        'level': 'INFO',
                        'propagate': False,
                        'handlers': ['console', 'logfile'],
                        }
                    },
                'root': {
                    'level': 'WARNING',
                    'handlers': ['console', 'logfile'],
                    }
                })

            # By making the build lib the same as the base folder, the extension
            # should just get built into the source directory.
            subprocess.run([
                "python3", os.path.join(safety_dir, "setup.py"),
                "build_ext", "--build-lib", safety_dir
                ])

            # start configuring the testing suite per run
            if run_env_type == 'append-still':
                t_penalty = [1.0e6, 2.0e6]
                t_performance = [1.0e6, 2.0e6]
            elif run_env_type == 'append-still-easy':
                t_penalty = [1.0e6, 2.0e6]
                t_performance = [1.0e6, 2.0e6]
            elif run_env_type == 'append-spawn':
                t_penalty = [2.0e6, 3.5e6]
                t_performance = [1.0e6, 2.0e6]
            elif run_env_type == 'prune-still':
                t_penalty = [0.5e6, 1.5e6]
                t_performance = [0.5e6, 1.5e6]

            training_levels = 'safelife/levels/random/{}.yaml'.format(run_env_type)

            if 'spawn' in run_env_type:
                if args.algo in ['aup', 'aup-p']:
                    level_iterator = spawn_loader_aup(
                            'append-still',
                            training_levels,
                            n_levels,
                            seed=args.seed)
                else:
                    level_iterator = spawn_loader(
                        'append-still',
                        training_levels,
                        n_levels,
                        seed=args.seed)
                test_levels = 'benchmarks/v1.0/append-spawn.npz'
            else:
                level_iterator = SafeLifeLevelIterator(
                        training_levels,
                        distinct_levels=n_levels,
                        total_levels=-1,
                        seed=args.seed)
                if run_env_type == 'append-still-easy' or run_env_type == 'append-still':
                    test_levels = 'benchmarks/v1.0/append-still.npz'
                elif run_env_type == 'prune-still':
                    test_levels = 'benchmarks/v1.0/prune-still.npz'

            # Use either tensorboard or wandb for training
            if args.wandb:
                from training.wandb_summary_writer import WandbSummaryWriter
                assert args.project is not None, "Must include a project name"
                summary_writer = WandbSummaryWriter(project=args.project, name=args.name, log_dir=log_dir)
            else:
                summary_writer = SummaryWriter(log_dir)

            training_envs = safelife_env_factory(
                    logdir=log_dir, summary_writer=summary_writer, num_envs=args.n_envs,
                    impact_penalty=linear_schedule(t_penalty, [0, penalty]),
                    min_performance=linear_schedule(t_performance, [0.01, 0.3]),
                    level_iterator=level_iterator,
                    )
            testing_envs = safelife_env_factory(
                    logdir=log_dir, summary_writer=summary_writer, num_envs=args.n_envs, testing=True,
                    level_iterator=SafeLifeLevelIterator(
                        test_levels, distinct_levels=n_levels, total_levels=-1, seed=args.seed)
                    )

            aux_train_steps = 1e6
            if run_env_type == 'append-still':
                aup_train_steps = 5e6
                ppo_train_steps = 6e6
            else:
                aup_train_steps = 4e6
                ppo_train_steps = 5e6

            if args.algo in ['aup', 'aup-p']:
                from training.models import SafeLifePolicyNetwork
                from training.aux_training_ppo import PPO_AUX
                from training.aup_training_ppo import PPO_AUP

                obs_shape = training_envs[0].observation_space.shape
                train_model_aux = SafeLifePolicyNetwork(obs_shape)
                train_model_aup = SafeLifePolicyNetwork(obs_shape)

                if args.algo == 'aup-p':
                    aup_p = True
                else:
                    aup_p = False

                aux_model = PPO_AUX(
                        train_model_aux, run_env_type,
                        training_envs=training_envs,
                        testing_envs=None,
                        z_dim=int(args.z),
                        n_rfn=1,
                        buf_size=100e3,
                        vae_epochs=50,
                        random_projection=aup_p,
                        aux_train_steps=aux_train_steps,
                        logdir=log_dir,
                        summary_writer=summary_writer,
                )
                aux_model.train()

                aup_model = PPO_AUP(
                        train_model_aup, aux_model, run_env_type,
                        training_envs=training_envs,
                        testing_envs=None,
                        z_dim=int(args.z),
                        logdir=log_dir,
                        aup_train_steps=aup_train_steps,
                        summary_writer=summary_writer)
                aup_model.train()

            elif args.algo in ['ppo', 'naive']:
                from training.models import SafeLifePolicyNetwork
                from training.ppo import PPO

                obs_shape = training_envs[0].observation_space.shape
                model = SafeLifePolicyNetwork(obs_shape)

                algo = PPO(
                        model,
                        training_envs=training_envs,
                        testing_envs=testing_envs,
                        logdir=log_dir,
                        train_steps=ppo_train_steps,
                        summary_writer=summary_writer)
                algo.train()

            elif args.algo == 'dqn':
                from training.models import SafeLifeQNetwork
                from training.dqn import DQN
                dqn_train_steps = 6.0e6

                obs_shape = training_envs[0].observation_space.shape
                train_model = SafeLifeQNetwork(obs_shape)
                target_model = SafeLifeQNetwork(obs_shape)
                algo = DQN(
                    train_model, target_model,
                    training_envs=training_envs,
                    testing_envs=testing_envs,
                    logdir=log_dir, summary_writer=summary_writer)
                algo.train(dqn_train_steps)


    except Exception:
        logging.exception("Ran into an unexpected error. Aborting training.")
    finally:
        if os.path.exists(active_job_file):
            os.remove(active_job_file)
        if args.shutdown:
            # Shutdown in 3 minutes.
            # Enough time to recover if it crashed at the start.
            subprocess.run("sudo shutdown +3".split())
            print("Shutdown commenced. Exiting to bash.")
            subprocess.run(["bash", "-il"])

        if args.wandb:
            import wandb
            wandb.finish()

if __name__ == "__main__":
    args = parser.parse_args()

    set_seed(args.seed)
    main(args)
