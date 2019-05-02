#!/usr/bin/env python3
import os
import numpy as np
import h5py as h5
import argparse
import gym
import gym_minigrid
import time
import torch
from torch_ac.utils.penv import ParallelEnv

import utils

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--episodes", type=int, default=100,
                    help="number of episodes of evaluation (default: 100)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--procs", type=int, default=16,
                    help="number of processes (default: 16)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="action with highest probability is selected")
parser.add_argument("--worst-episodes-to-show", type=int, default=10,
                    help="The number of worse episodes to show")
args = parser.parse_args()

# Set seed for all randomness sources

utils.seed(args.seed)

# Generate environment

envs = []
for i in range(args.procs):
    env = gym.make(args.env)
    env.seed(args.seed + 10000*i)
    envs.append(env)
env = ParallelEnv(envs)

# define some sampling helpers
outputs = {}
outputs['obss'] = []
outputs['rewards'] = []
outputs['dones'] = []

def get_layers(tree):
    children = list(tree.children())
    if len(children) > 0:
        return [get_layers(child) for child in children]
    else:
        return tree
    
def flatten(aList):
    t = []
    for i in aList:
        if not isinstance(i, list):
             t.append(i)
        else:
             t.extend(flatten(i))
    return t

class Extractor(object):
    def __init__(self, name):
        self.name = name
        outputs[self.name]=[]

    def extract(self, module, input, output):
        if 'LSTM' in str(self.name):
            outputs[self.name].append([
                output[0].detach().cpu().numpy(),
                output[1].detach().cpu().numpy()
            ])
        else:
            outputs[self.name].append(output.detach().cpu().numpy())

def save_features(outputs, path):
    outputs = process_outputs(outputs)
    print('Saving data:', path)
    #print(outputs['obss'].shape)
    for key in outputs:
        data = outputs[key]
        fullPath = os.path.join('feature_dir', path, key+'.h5')
        if not os.path.exists(os.path.dirname(fullPath)):
            os.makedirs(os.path.dirname(fullPath))

        #print('saving data shape:{} \nfor key:{} \nat path:{}'.format(data.shape, key, fullPath))
        f = h5.File(fullPath, 'w')
        f.create_dataset('obj_arr', data=data)
        f.close()

def process_outputs(outputs):
    dataDict = {}
    for key in outputs:
        data = np.array(outputs[key]).astype('float16')
        #print(key, data.shape)
        dataDict[key] = data

    return dataDict

def clear(outputs):
    for key in outputs:
        outputs[key] = []

def apply_extractor(model):
    model_layers = flatten(get_layers(model))
    for i, L in enumerate(model_layers):
        name = 'features_'+str(i)+'_'+str(L)
        extractor = Extractor(name)
        L.register_forward_hook(extractor.extract)
        print('applied forward hook to extract features from:{}'.format(name))


# Define agent

model_dir = utils.get_model_dir(args.model)
agent = utils.Agent(args.env, env.observation_space, model_dir, args.argmax, args.procs)
apply_extractor(agent.acmodel)
print("CUDA available: {}\n".format(torch.cuda.is_available()))

# Initialize logs

logs = {"num_frames_per_episode": [], "return_per_episode": []}

# Run the agent

start_time = time.time()

obss = env.reset()
log_done_counter = 0
log_episode_return = torch.zeros(args.procs, device=agent.device)
log_episode_num_frames = torch.zeros(args.procs, device=agent.device)

while log_done_counter < args.episodes:
    obss_0 = np.array([obss[-1]['image']])
    actions = agent.get_actions(obss)
    obss, rewards, dones, _ = env.step(actions)
    agent.analyze_feedbacks(rewards, dones)
    #import pdb; pdb.set_trace()
    obss_imgs = np.array([obs['image'] for obs in obss])
    obss_imgs = np.concatenate([obss_0, obss_imgs])
    outputs['obss'].append(obss_imgs)
    outputs['rewards'].append(rewards)
    outputs['dones'].append(dones)

    log_episode_return += torch.tensor(rewards, device=agent.device, dtype=torch.float)
    log_episode_num_frames += torch.ones(args.procs, device=agent.device)

    for i, done in enumerate(dones):
        if done:
            log_done_counter += 1
            ep_return = log_episode_return[i].item()
            ep_frames = log_episode_num_frames[i].item()
            logs["return_per_episode"].append(ep_return)
            logs["num_frames_per_episode"].append(ep_frames)

            save_template = '{model}/env_{env}-ep_{i}-return_{ep_return}-frames_{ep_frames}'.format(
                model=args.model,
                env=args.env,
                i=log_done_counter,
                ep_return=ep_return,
                ep_frames=ep_frames
            )

            save_features(outputs, save_template)
            clear(outputs)

    mask = 1 - torch.tensor(dones, device=agent.device, dtype=torch.float)
    log_episode_return *= mask
    log_episode_num_frames *= mask




end_time = time.time()


# Print logs

num_frames = sum(logs["num_frames_per_episode"])
fps = num_frames/(end_time - start_time)
duration = int(end_time - start_time)
return_per_episode = utils.synthesize(logs["return_per_episode"])
num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

print("F {} | FPS {:.0f} | D {} | R:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {}"
      .format(num_frames, fps, duration,
              *return_per_episode.values(),
              *num_frames_per_episode.values()))

# Print worst episodes

n = args.worst_episodes_to_show
if n > 0:
    print("\n{} worst episodes:".format(n))

    indexes = sorted(range(len(logs["return_per_episode"])), key=lambda k: logs["return_per_episode"][k])
    for i in indexes[:n]:
        print("- episode {}: R={}, F={}".format(i, logs["return_per_episode"][i], logs["num_frames_per_episode"][i]))


