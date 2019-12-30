import sys, os
sys.path.append(os.path.realpath('../..'))
import inspect

# Importing the libraries
import os
import numpy as np
import gym
from gym import wrappers
import pybullet_envs
import time
import multiprocessing as mp
from multiprocessing import Process, Pipe
import argparse
import math
#Utils
from pybRL.utils.logger import DataLog
from pybRL.utils.make_train_plots import make_train_plots_ars

#Registering new environments
from gym.envs.registration import registry, register, make, spec

#Stoch 2 Test imports
import pybullet as p 
import numpy as np
PI = math.pi
# Setting the Hyper Parameters
import math
PI = math.pi
class HyperParameters():
    """
    This class is basically a struct that contains all the hyperparameters that you want to tune
    """
    def __init__(self,forward_reward_cap = 1,stairs = False, action_dim = 10 , normal = True,gait = 'trot' ,msg = '', nb_steps=10000, episode_length=1000, learning_rate=0.02, nb_directions=16, nb_best_directions=8, noise=0.03, seed=1, env_name='HalfCheetahBulletEnv-v0', energy_weight = 0.2):
        self.nb_steps = nb_steps
        self.episode_length = episode_length
        self.learning_rate = learning_rate
        self.nb_directions = nb_directions
        self.nb_best_directions = nb_best_directions
        assert self.nb_best_directions <= self.nb_directions
        self.noise = noise
        self.seed = seed
        self.env_name = env_name
        self.energy_weight = energy_weight
        self.normal = normal
        self.msg = msg
        self.forward_reward_cap = forward_reward_cap
        self.gait = gait
        self.action_dim = action_dim
        self.stairs = stairs
    def to_text(self, path):
        res_str = ''
        res_str = res_str + 'learning_rate: ' + str(self.learning_rate) + '\n'
        res_str = res_str + 'noise: ' + str(self.noise) + '\n'
        if(self.stairs):
          res_str = res_str + 'env_name: ' + str(self.env_name) + 'with stairs \n'
        else:
          res_str = res_str + 'env_name: ' + str(self.env_name)
        res_str = res_str + 'episode_length: ' + str(self.episode_length) + '\n'
        res_str = res_str + 'direction ratio: '+ str(self.nb_directions/ self.nb_best_directions) + '\n'
        res_str = res_str + 'Normal initialization: '+ str(self.normal) + '\n'
        res_str = res_str + 'Gait: '+ str(self.gait) + '\n'
        res_str = res_str + 'Spline polynomial degree: '+ str(self.action_dim) + '\n'
        res_str = res_str + self.msg + '\n'
        fileobj = open(path, 'w')
        fileobj.write(res_str)
        fileobj.close()

# Multiprocess Exploring the policy on one specific direction and over one episode

_RESET = 1
_CLOSE = 2
_EXPLORE = 3


def ExploreWorker(rank, childPipe, envname, args):
  env = gym.make(envname)
  nb_inputs = env.observation_space.sample().shape[0]
  normalizer = Normalizer(nb_inputs)
  observation_n = env.reset()
  n = 0
  while True:
    n += 1
    try:
      # Only block for short times to have keyboard exceptions be raised.
      if not childPipe.poll(0.001):
        continue
      message, payload = childPipe.recv()
    except (EOFError, KeyboardInterrupt):
      break
    if message == _RESET:
      observation_n = env.reset()
      childPipe.send(["reset ok"])
      continue
    if message == _EXPLORE:
      #normalizer = payload[0] #use our local normalizer
      policy = payload[1]
      hp = payload[2]
      direction = payload[3]
      delta = payload[4]
      state = env.reset()
      done = False
      num_plays = 0.
      sum_rewards = 0
      while num_plays < hp.episode_length:
        # normalizer.observe(state)
        # state = normalizer.normalize(state)
        action = policy.evaluate(state, delta, direction, hp)
        state, reward, done, _ = env.step(action)
        # reward = max(min(reward, 1), -1)
        sum_rewards += reward
        num_plays += 1
      # print('rewards: ',sum_rewards)
      childPipe.send([sum_rewards, num_plays])
      continue
    if message == _CLOSE:
      childPipe.send(["close ok"])
      break
  childPipe.close()


# Normalizing the states


class Normalizer():

  def __init__(self, nb_inputs):
    self.n = np.zeros(nb_inputs)
    self.mean = np.zeros(nb_inputs)
    self.mean_diff = np.zeros(nb_inputs)
    self.var = np.zeros(nb_inputs)

  def observe(self, x):
    self.n += 1.
    last_mean = self.mean.copy()
    self.mean += (x - self.mean) / self.n
    self.mean_diff += (x - last_mean) * (x - self.mean)
    self.var = (self.mean_diff / self.n).clip(min=1e-2)

  def normalize(self, inputs):
    obs_mean = self.mean
    obs_std = np.sqrt(self.var)
    return (inputs - obs_mean) / obs_std


# Building the AI


class Policy():

  def __init__(self, input_size, output_size, env_name, normal, args):
    try:
      self.theta = np.load(args.policy)
    except:
      if(normal):
        self.theta = np.random.randn(output_size, input_size)
      else:
        self.theta = np.zeros((output_size, input_size))
        self.theta[:2,:] = 0.5
        self.theta[:,:2] = 0.5
    self.env_name = env_name
    print("Starting policy theta=", self.theta)

  def evaluate(self, input, delta, direction, hp):
    if direction is None:
      return np.clip(self.theta.dot(input), -1.0, 1.0)
    elif direction == "positive":
      return np.clip((self.theta + hp.noise * delta).dot(input), -1.0, 1.0)
    else:
      return np.clip((self.theta - hp.noise * delta).dot(input), -1.0, 1.0)

  def sample_deltas(self):
    return [np.random.randn(*self.theta.shape) for _ in range(hp.nb_directions)]

  def update(self, rollouts, sigma_r, args):
    step = np.zeros(self.theta.shape)
    for r_pos, r_neg, direction in rollouts:
      step += (r_pos - r_neg) * direction 
    self.theta += hp.learning_rate / (hp.nb_best_directions * sigma_r) * step
    timestr = time.strftime("%Y%m%d-%H%M%S")


# Exploring the policy on one specific direction and over one episode


def explore(env, policy, direction, delta, hp):
  nb_inputs = env.observation_space.sample().shape[0]
  normalizer = Normalizer(nb_inputs)
  state = env.reset()
  done = False
  num_plays = 0.
  sum_rewards = 0
  while num_plays < hp.episode_length:
    # normalizer.observe(state)
    # state = normalizer.normalize(state)
    action = policy.evaluate(state, delta, direction, hp)
    # print("action : ", action)
    state, reward, done, _ = env.step(action)
    # print("reward: ", reward)
    # reward = max(min(reward, 1), -1)
    sum_rewards += reward
    num_plays += 1
  # print("sum rewards: ", sum_rewards)
  return sum_rewards


# Training the AI


def train(env, policy, normalizer, hp, parentPipes, args):
  logger = DataLog()
  total_steps = 0
  best_return = -99999999
  if os.path.isdir(args.logdir) == False:
    os.mkdir(args.logdir)
  previous_dir = os.getcwd()
  os.chdir(args.logdir)
  if os.path.isdir('iterations') == False: os.mkdir('iterations')
  if os.path.isdir('logs') ==False: os.mkdir('logs')
  hp.to_text('hyperparameters')
  
  for step in range(hp.nb_steps):

    # Initializing the perturbations deltas and the positive/negative rewards
    deltas = policy.sample_deltas()
    positive_rewards = [0] * hp.nb_directions
    negative_rewards = [0] * hp.nb_directions
    if(parentPipes):
      process_count = len(parentPipes)
    if parentPipes:
      p = 0
      while(p < hp.nb_directions):
        temp_p = p
        n_left = hp.nb_directions - p #Number of processes required to complete the search
        for k in range(min([process_count, n_left])):
          parentPipe = parentPipes[k]
          parentPipe.send([_EXPLORE, [normalizer, policy, hp, "positive", deltas[temp_p]]])
          temp_p = temp_p+1
        temp_p = p
        for k in range(min([process_count, n_left])):
          positive_rewards[temp_p], step_count = parentPipes[k].recv()
          total_steps = total_steps + step_count
          temp_p = temp_p+1
        temp_p = p

        for k in range(min([process_count, n_left])):
          parentPipe = parentPipes[k]
          parentPipe.send([_EXPLORE, [normalizer, policy, hp, "negative", deltas[temp_p]]])
          temp_p = temp_p+1
        temp_p = p

        for k in range(min([process_count, n_left])):
          negative_rewards[temp_p], step_count = parentPipes[k].recv()
          total_steps = total_steps + step_count
          temp_p = temp_p+1
        p = p + process_count

        # print('mp step has worked, ', p)
        print('total steps till now: ', total_steps, 'Processes done: ', p)
        
    else:
      # Getting the positive rewards in the positive directions
      for k in range(hp.nb_directions):
        positive_rewards[k] = explore(env, policy, "positive", deltas[k], hp)

      # Getting the negative rewards in the negative/opposite directions
      for k in range(hp.nb_directions):
        negative_rewards[k] = explore(env, policy, "negative", deltas[k], hp)

    # Sorting the rollouts by the max(r_pos, r_neg) and selecting the best directions
    scores = {
        k: max(r_pos, r_neg)
        for k, (r_pos, r_neg) in enumerate(zip(positive_rewards, negative_rewards))
    }
    order = sorted(scores.keys(), key=lambda x: -scores[x])[:int(hp.nb_best_directions)]
    rollouts = [(positive_rewards[k], negative_rewards[k], deltas[k]) for k in order]
   
    # Gathering all the positive/negative rewards to compute the standard deviation of these rewards
    all_rewards = np.array([x[0] for x in rollouts] + [x[1] for x in rollouts])
    sigma_r = all_rewards.std() # Standard deviation of only rewards in the best directions is what it should be
    # Updating our policy
    policy.update(rollouts, sigma_r, args)

    # Printing the final reward of the policy after the update
    reward_evaluation = explore(env, policy, None, None, hp)
    logger.log_kv('steps', total_steps)
    logger.log_kv('return', reward_evaluation)
    if(reward_evaluation > best_return):
        best_policy = policy.theta
        best_return = reward_evaluation
        np.save("iterations/best_policy.npy",best_policy )
    print('Step:', step, 'Reward:', reward_evaluation)
    policy_path = "iterations/" + "policy_"+str(step)
    np.save(policy_path, policy.theta)
    logger.save_log('logs/')
    make_train_plots_ars(log = logger.log, keys=['steps', 'return'], save_loc='logs/')


# Running the main code


def mkdir(base, name):
  path = os.path.join(base, name)
  if not os.path.exists(path):
    os.makedirs(path)
  return path


if __name__ == "__main__":  
  # mp.freeze_support()
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--env', help='Gym environment name', type=str, default='MinitaurTrottingEnv-v0')
  parser.add_argument('--seed', help='RNG seed', type=int, default=1234123)
  parser.add_argument('--render', help='OpenGL Visualizer', type=int, default=0)
  parser.add_argument('--movie', help='rgb_array gym movie', type=int, default=0)
  parser.add_argument('--steps', help='Number of steps', type=int, default=10000)
  parser.add_argument('--policy', help='Starting policy file (npy)', type=str, default='')
  parser.add_argument(
      '--logdir', help='Directory root to log policy files (npy)', type=str, default='.')
  parser.add_argument('--mp', help='Enable multiprocessing', type=int, default=1)
  #these you have to set
  parser.add_argument('--lr', help='learning rate', type=float, default=0.2)
  parser.add_argument('--noise', help='noise hyperparameter', type=float, default=0.03)
  parser.add_argument('--episode_length', help='length of each episode', type=float, default=10)
  parser.add_argument('--normal', help='length of each episode', type=int, default=1)
  parser.add_argument('--gait', help='type of gait you want (Only in Stoch2 normal env', type=str, default='trot')
  parser.add_argument('--energy_weight', help='reward shaping, amount to penalise the energy', type=float, default=0.2)
  parser.add_argument('--msg', help='msg to save in a text file', type=str, default='')
  parser.add_argument('--forward_reward_cap', help='Forward reward cap used in training', type=float, default=10000)
  parser.add_argument('--distance_weight', help='The weight to be given to distance moved by robot', type=float, default=1.0)
  parser.add_argument('--stairs', help='add stairs to the bezier environment', type=int, default=0)
  parser.add_argument('--action_dim', help='degree of the spline polynomial used in the training', type=int, default=20)

  args = parser.parse_args()
  walk = [0, PI, PI/2, 3*PI/2]
  canter = [0, PI, 0, PI]
  bound = [0, 0, PI, PI]
  trot = [0, PI, PI , 0]
  custom_phase = [0, PI, PI+0.1 , 0.1]
  phase = 0
  if(args.gait == "trot"):
    phase = trot
  elif(args.gait == "canter"):
    phase = canter
  elif(args.gait == "bound"):
    phase = bound
  elif(args.gait == "walk"):
    phase = walk    
  elif(args.gait == "custom_phase1"):
    phase = custom_phase
  elif(args.gait == "trot+turn"):
    phase = trot
  # #Custom environments that you want to use ----------------------------------------------------------------------------------------
  register(id='Stoch2-v0',entry_point='pybRL.envs.stoch2_gym_bullet_env_bezier:Stoch2Env', kwargs = {'gait' : args.gait, 'phase': phase, 'action_dim': args.action_dim, 'stairs': args.stairs} )
  # #---------------------------------------------------------------------------------------------------------------------------------

  hp = HyperParameters()
  hp.msg = args.msg
  hp.env_name = args.env
  env = gym.make(hp.env_name)
  hp.seed = args.seed
  hp.nb_steps = args.steps
  hp.learning_rate = args.lr
  hp.noise = args.noise
  hp.episode_length = args.episode_length
  hp.energy_weight = args.energy_weight
  hp.forward_reward_cap = args.forward_reward_cap
  print(env.observation_space.sample())
  hp.nb_directions = int(env.observation_space.sample().shape[0] * env.action_space.sample().shape[0])
  hp.nb_best_directions = int(hp.nb_directions / 2)
  hp.normal = args.normal
  hp.gait = args.gait
  hp.action_dim = args.action_dim
  hp.stairs = args.stairs
  # print('number directions: ', hp.nb_directions)
  # print('number best directions: ', hp.nb_best_directions)
  # exit()
  # print(hp.nb_best_directions)
  print("seed = ", hp.seed)
  np.random.seed(hp.seed)
  max_processes = 10

  parentPipes = None
  if args.mp:
    num_processes = min([hp.nb_directions, max_processes])
    print('processes: ',num_processes)
    processes = []
    childPipes = []
    parentPipes = []

    for pr in range(num_processes):
      parentPipe, childPipe = Pipe()
      parentPipes.append(parentPipe)
      childPipes.append(childPipe)

    for rank in range(num_processes):
      p = mp.Process(target=ExploreWorker, args=(rank, childPipes[rank], hp.env_name, args))
      p.start()
      processes.append(p)

  # env = stoch2_gym_env.StochBulletEnv(render = False, gait = 'trot')
  nb_inputs = env.observation_space.sample().shape[0]
  nb_outputs = env.action_space.sample().shape[0]
  policy = Policy(nb_inputs, nb_outputs, hp.env_name, hp.normal, args)
  normalizer = Normalizer(nb_inputs)

  print("start training")
  train(env, policy, normalizer, hp, parentPipes, args)

  if args.mp:
    for parentPipe in parentPipes:
      parentPipe.send([_CLOSE, "pay2"])

    for p in processes:
      p.join()

  # --------------------------------------------------------------------------------
  # STOCH2 Test
  # env = sv.StochBulletEnv(render = True, gait = 'trot')
  # env = gym.make('Stoch2-v1')
  # hp = HyperParameters()
  # nb_inputs = env.observation_space.shape[0]
  # nb_outputs = env.action_space.shape[0]
  # policy = Policy(nb_inputs, nb_outputs, hp.env_name, None)
  # normalizer = Normalizer(nb_inputs)

  # deltas = policy.sample_deltas()
  # state = env.reset()
  # i = 0
  # hp.noise = 0.2
  # sum_rewards = 0
  # while i <1000:
  #   normalizer.observe(state)
  #   # print('state before: ', state)
  #   state = normalizer.normalize(state)
  #   # print('state after: ', state)
  #   action = policy.evaluate(state, deltas[0], 'positive', hp)
  #   # print(action)
  #   state, reward, done ,info = env.step(np.clip(action, -1, 1))
  #   sum_rewards = sum_rewards + reward
  #   if(done):
  #     print('terminated')
  #     break
  #   i = i + 1
  
  # print('total reward: ', sum_rewards)