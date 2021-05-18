import numpy as np
import torch
import gym
gym.logger.set_level(40)
import QDgym
import argparse
from sklearn.neighbors import KDTree
import multiprocessing as multiprocessing
from functools import partial
import os

from utils import *
from networks import Actor, Critic
from variational_operators import VariationalOperator
from vectorized_env import ParallelEnv


class LoadFromFile(argparse.Action):
	def __call__ (self, parser, namespace, values, option_string = None):
		with values as f:
			parser.parse_args([s.strip("\n") for s in f.readlines()], namespace)


def make_env(env_id):
    env = gym.make(env_id)
    return env


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--config_file', type=open, action=LoadFromFile)    # Config file to load args (Typically you would only specifiy this arg)
	parser.add_argument("--env", default="QDAntBulletEnv-v0")       	# Environment name (only QDgym envs will run)
	parser.add_argument("--seed", default=0, type=int)              	# Seed
	parser.add_argument("--save_path", default=".")                 	# Path where to save results
	##########################################################################################################
	########################## QD PARAMS #####################################################################
	##########################################################################################################
	parser.add_argument("--dim_map", default=4, type=int)			# Dimentionality of behaviour space
	parser.add_argument("--n_niches", default=1296, type=int)		# nr of niches/cells of behaviour
	parser.add_argument("--n_species", default=1, type=int)			# nr of species/cells in species archive (The species archive is disabled in the GECCO paper by setting n_species=1. See readme for details)
	parser.add_argument("--max_evals", default=1e6, type=int)		# nr of evaluations (I) 
	parser.add_argument("--mutation_op", default=None)			# Mutation operator to use (Set to None in GECCO paper)
	parser.add_argument("--crossover_op", default="iso_dd")			# Crossover operator to use (Set to iso_dd aka directional variation in GECCO paper which uses mutation and crossover in one)
	parser.add_argument("--min_genotype", default=False)            	# Minimum value a gene in the genotype can take (if False no limit) (Set to False in GECCO paper)
	parser.add_argument("--max_genotype", default=False)            	# Maximum value a gene in the genotype can take (if False no limit) (Set to False in GECCO paper)
	parser.add_argument("--mutation_rate", default=0.05, type=float)        # Probablity of a gene to be mutated (Not used in GECCO paper. iso_dd mutates all genes unconditionally)
	parser.add_argument("--crossover_rate", default=0.75, type=float)       # Probablity of genotypes being crossed over (Not used in GECCO paper. iso_dd crosses all genes unconditionally)
	parser.add_argument("--eta_m", default=5.0, type=float)                 # Parameter for polynomaial mutation (Not used in GECCO paper)
	parser.add_argument("--eta_c", default=10.0, type=float)                # Parameter for Simulated Binary Crossover (Not used in GECCO paper)
	parser.add_argument("--sigma", default=0.2, type=float)            	# Sandard deviation for gaussian muatation (Not used in GECCO paper)
	parser.add_argument("--iso_sigma", default=0.01, type=float)            # Gaussian parameter in iso_dd/directional variation (sigma_1)
	parser.add_argument("--line_sigma", default=0.2, type=float)            # Line parameter in iso_dd/directional variation (sigma_2)
	parser.add_argument("--max_uniform", default=0.1, type=float)           # Max mutation for uniform muatation (Not used in GECCO paper)
	parser.add_argument("--cvt_samples", default=100000, type=int)	        # Nr. of samples to use when approximating archive cell-centroid locations
	parser.add_argument("--eval_batch_size", default=100, type=int)	        # Batch size for parallel evaluation of policies (b)
	parser.add_argument("--random_init", default=500, type=int)		# Number of random evaluations to inililise (G)
	parser.add_argument("--init_batch_size", default=100, type=int)	        # Batch size for parallel evaluation during random init (b)
	parser.add_argument("--save_period", default=10000, type=int)	        # How many evaluations between saving archives
	parser.add_argument("--num_cpu", default=32, type=int) 			# Nr. of CPUs to use in parallel evaluation
	parser.add_argument("--num_cpu_var", default=32, type=int) 		# Nr. of CPUs to use in parallel variation
	parser.add_argument("--use_cached_cvt", action="store_true") 	        # Use cached centroids for creating archive if avalable
	parser.add_argument("--not_discard_dead", action="store_true") 	        # Don't discard solutions that does not survive the entire simulation (Set to not dicard in GECCO paper)
	parser.add_argument("--neurons_list", default="128 128", type=str) 	# List of neurons in actor network layers. Network will be of form [neurons_list + [action dim]]
	#########################################################################################################
	######################### RL PARAMS #####################################################################
	#########################################################################################################
	parser.add_argument("--train_batch_size", default=256, type=int)        # Batch size for both actors and critic (N)
	parser.add_argument("--discount", default=0.99)                         # Discount factor for critic (gamma)
	parser.add_argument("--tau", default=0.005, type=float)                 # Target networks update rate (tau)
	parser.add_argument("--policy_noise", default=0.2)                      # Noise added to target during critic update (sigma_p)
	parser.add_argument("--noise_clip", default=0.5)                        # Range to clip target noise (c)
	parser.add_argument("--policy_freq", default=2, type=int)               # Frequency of delayed actor updates (d)
	parser.add_argument('--nr_of_steps_crit', default=300, type=int)	# Nr of. training steps for critic traning (n_crit)
	parser.add_argument('--nr_of_steps_act', default=10, type=int)		# Nr of. training steps for PG varaiation (n_grad)
	parser.add_argument("--proportion_evo", default=0.5, type=float)	# Proportion of batch to use GA variation (n_evo = proportion_evo * b. Set to 0.5 in GECCO paper)
	parser.add_argument("--normalise", action="store_true")			# Use layer norm (Not used in GECCO paper)
	parser.add_argument("--affine", action="store_true")			# Use affine transormation with layer norm (Not used in GECCO paper)
	parser.add_argument("--gradient_op", action="store_true")		# Use PG variation 
	parser.add_argument("--lr", default=0.001, type=float)			# Learning rate PG variation

	args = parser.parse_args()
	args.neurons_list = [int(x) for x in args.neurons_list.split()]
	for arg in vars(args):
		print(f"{arg}: {getattr(args, arg)}")

	num_cores = multiprocessing.cpu_count()
	print("-"*80)
	print(f"Number of found cores: {num_cores}")
	print("-"*80)

	print("-"*80)	
	print(f"Algorithm: PGA-MAP-Elites, Env: {args.env}, Seed: {args.seed}")
	print("-"*80)
	file_name = f"PGA-MAP-Elites_{args.env}_{args.seed}_{args.dim_map}"
	
	# make folders
	if not os.path.exists(f"{args.save_path}"):
		os.mkdir(f"{args.save_path}")
		os.mkdir(f"{args.save_path}/models/")
	if not os.path.exists(f"{args.save_path}/models"):
		os.mkdir(f"{args.save_path}/models/")

	# File for saving progress
	log_file=open(f"{args.save_path}/progress_{file_name}.dat", 'w')
	# File for saving info about each actor (parent etc.)
	actors_file=open(f"{args.save_path}/actors_{file_name}.dat", 'w')
	#set seeds
	torch.manual_seed(args.seed * int(1e6))
	np.random.seed(args.seed * int(1e6))
	# Get env info to initilise networks
	temp_env = gym.make(args.env)
	state_dim = temp_env.observation_space.shape[0]
	action_dim = temp_env.action_space.shape[0] 
	max_action = float(temp_env.action_space.high[0])
	temp_env.close()
	# Setup functions to launch each paralell evaluation environment
	make_fns = [partial(make_env, args.env) for _ in range(args.num_cpu)]
	# Function that creates new actor
	actor_fn = partial(Actor, 
				state_dim,
				action_dim,
				max_action,
				args.neurons_list,
				normalise=args.normalise,
				affine=args.affine)
	# Function that create the replay buffer
	replay_fn = partial(ReplayBuffer, state_dim, action_dim)
	# Function to create critic
	critic_fn = partial(Critic,
					state_dim,
					action_dim,
					max_action,
					discount=args.discount,
					tau=args.tau,
					policy_noise=args.policy_noise * max_action,
					noise_clip=args.noise_clip * max_action,
					policy_freq=args.policy_freq)
	# Start parallel simulations
	envs = ParallelEnv(make_fns,
					replay_fn,
					critic_fn, 
					args.nr_of_steps_crit,
					args.train_batch_size,
					args.nr_of_steps_act,
					args.random_init,
					args.seed)
	# Initilise the variation operator
	variational_op = VariationalOperator(actor_fn = actor_fn,
										num_cpu = args.num_cpu_var,
										gradient_op = args.gradient_op,
										crossover_op = args.crossover_op,
										mutation_op = args.mutation_op,
										learning_rate = args.lr, 
										max_gene = args.max_genotype,
										min_gene = args.min_genotype,
										mutation_rate = args.mutation_rate,
										crossover_rate = args.crossover_rate,
										eta_m = args.eta_m,
										eta_c = args.eta_c,
										sigma = args.sigma,
										max_uniform = args.max_uniform,
										iso_sigma = args.iso_sigma,
										line_sigma =args.line_sigma)
            
	# Compute CVT for main and species archive 
	c = cvt(args.n_niches, args.dim_map, args.cvt_samples, args.use_cached_cvt)
	sc = cvt(args.n_species, args.dim_map, args.cvt_samples, args.use_cached_cvt) 
	# k-nn for achive addition. The nearest centroid is found by this by setting k=1.
	kdt = KDTree(c, leaf_size=30, metric='euclidean') # main k-nn
	s_kdt = KDTree(sc, leaf_size=30, metric='euclidean') # species k-nn

	archive = {} # init archive (empty)
	s_archive = {} # init species archive (empty)
	n_evals = 0 # number of evaluations since the beginning
	b_evals = 0 # number evaluation since the last dump
	max_fit = -float("inf") # track max fit for extra evaluations to check robustness

	# Enter MAP-Elites loop
	while (n_evals < args.max_evals):
		print(f"Number of solutions: {len(archive)}")
		print(f"Number of species: {len(s_archive)}")
		to_evaluate = []
		if n_evals < args.random_init:
			print("Random Loop")
			for i in range(0, args.init_batch_size):
				to_evaluate += [actor_fn()]
		
		else: 
			print("Selection/Variation Loop")
			# Sync critic training
			critic, actors, states, train_time = envs.get_critic()
			to_evaluate += actors
			# Selection and Variation
			to_evaluate += variational_op(archive,
										  args.eval_batch_size - len(actors),
										  args.proportion_evo, 
										  critic=critic,
										  states=states,
										  train_batch_size=args.train_batch_size,
										  nr_of_steps_act=args.nr_of_steps_act)
		# evaluation of the fitness and BD for new batch
		solutions = envs.eval_policy(to_evaluate)
		n_evals += len(to_evaluate)
		b_evals += len(to_evaluate)
		print(f"[{n_evals}/{int(args.max_evals)}]")
		# Add to archive
		for idx, solution in enumerate(solutions):
			# Check if alive (or if we ignore if robot died or not)
			if solution[2] or args.not_discard_dead:
				# Initate individual
				s = Individual(to_evaluate[idx], solution[1], solution[0])
				added_main = add_to_archive(s, s.desc, archive, kdt)
				added_species = add_to_archive(s, s.desc, s_archive , s_kdt, main=False)
				if added_main:
					actors_file.write("{} {} {} {} {} {} {} {} {} {}\n".format(n_evals,
																			s.x.id,
																			s.fitness,
																			str(s.desc).strip("[]"),
																			str(s.centroid).strip("()"),
																			s.x.parent_1_id,
																			s.x.parent_2_id,
																			s.x.type,
																			s.x.novel,
																			s.x.delta_f))
					actors_file.flush()	
		# Send new species archive to critic training
		envs.update_archive(s_archive)
		# save the state of the archive
		if b_evals >= args.save_period and args.save_period != -1:
			print(f"[{n_evals}/{int(args.max_evals)}]", end=" ", flush=True)
			save_archive(archive, n_evals, file_name, args.save_path)
			b_evals = 0
        # log the progrees needed to calulate metrics
		if log_file != None:
			fit_list = np.array([x.fitness for x in archive.values()])
			max_actor_id = archive[max(archive, key=lambda desc: archive[desc].fitness)].x.id # track best behaviour for potential further eval later
			if fit_list.max() > max_fit:
				print("Evaluate max actor 10 times")
				max_fit = fit_list.max()
				max_actor_desc = max(archive, key=lambda desc: archive[desc].fitness)
				sol = envs.eval_policy([archive[max_actor_desc].x for _ in range(10)], eval_mode=True)
				ave_fit = sum([s[0] for s in sol])/10
				ave_desc = sum([s[1] for s in sol])/10

			log_file.write("{} {} {} {} {} {} {} {} {} {} {}\n".format(n_evals,
																len(archive.keys()),
																fit_list.max(),
																np.sum(fit_list), #for QD-score
																np.mean(fit_list),
																np.median(fit_list),
																np.percentile(fit_list, 5),
																np.percentile(fit_list, 95),
																ave_fit,
																str(ave_desc).strip("[]"),
																max_actor_id))
			print(f"Max fitness: {fit_list.max()}")
			print(f"Mean fit: {np.mean(fit_list)}")
			log_file.flush()
	# Save the final state of archive
	save_archive(archive, n_evals, file_name, args.save_path, save_models=True)
	# End paralell processes
	critic, replay_buffer, env_rng_states = envs.close() 
	critic.save(f"{args.save_path}/models/{file_name}_critic_" + str(n_evals))
	variational_op.close()
