import gym
import numpy as np
import time
import pandas as pd
from openpyxl import load_workbook
import json

from tabulate import tabulate
import seaborn
import matplotlib.pyplot as plt

np.random.seed(0)

def to_json(fn, data):
    with open(fn, 'w') as f:
        json.dump(data, f)
        
def from_json(fn):
    with open(fn) as f:
        return (json.load(f))
    
def run_policy(env, policy):
    problem = gym.make(env)
    problem.seed(2019)
    #problem = env
    rewards = []
    av_reward = []
    ts = []
    outcomes = []
    
    mean_reward = np.nan
    for i_episode in range(1,5001):
        total_rewards = 0
        observation = problem.reset()
        for t in range(problem._max_episode_steps):
            #problem.render()
            action = policy[observation]
            observation, reward, done, info = problem.step(action)
            total_rewards += reward
            if done:
                if (t+1) >= problem._max_episode_steps or reward == 0:
                     outcomes.append(0)
                     #print("LOSE")
                else:
                     outcomes.append(1)
                     #print("WIN")
                 #print("Episode finished after {} timesteps".format(t+1))
                ts.append(t+1)
                av_reward.append(mean_reward)
                break
        rewards.append(total_rewards)
        if i_episode % 50 == 0:
            mean_reward = np.mean(rewards[i_episode-50:])            
            print (np.mean(rewards[i_episode-50:]))
            #print('Current avg_reward: %f' % np.mean(av_reward))
    #env.monitor.close()
    #print (np.mean(av_reward))
    return [ts, outcomes, rewards, av_reward]

#episode_offset = 10000
tol = 1e-4
def run_learner(env, alpha = 0.1, gamma = 0.9, radr = 0.99, rar = 1):
    rewards = []
    iterations = []
    timesteps = []
    deltas = []
    results = {}

    scores, iters, ts, diffs = 0, 0, 0, 0
    
    Q = np.zeros((env.nS, env.nA))
    action = np.random.randint(0, env.nA)
    d_old=[0]
    d_cur=[]
    converge = []
    
    for i_episode in range(100000):
        t = time.clock()     
        t_reward = 0

        a = 0            
        s = env.reset()
        Q_old = Q.copy()
        for i in range(env._max_episode_steps):
            s_prime, r, done, info = env.step(a)          
            t_reward += r
            if done:
                #Q[s, a] = (1.0 - alpha/(i_episode+1)) * Q[s, a] + ((alpha/(i_episode+1)) * (r))
                Q[s, a] = (1.0 - alpha) * Q[s, a] + ((alpha) * (r))
                rar = rar * radr

                #if reward==0:
                #    print ("dead!")
                #if reward==1:
                #    print("goal!"
                break  
            else:
                a_max = np.argmax(Q[s_prime])
                #Q[s, a] = (1.0 - alpha/(i_episode+1)) * Q[s, a] + ((alpha/(i_episode+1)) * (r + gamma * Q[s_prime, a_max]))
                Q[s, a] = (1.0 - alpha) * Q[s, a] + ((alpha) * (r + gamma * Q[s_prime, a_max]))
                   
                # get next action
                if np.random.random() < rar:
                   action = np.random.randint(0, env.nA)
                else:
                   action = np.argmax(Q[s_prime])
                        
                s = s_prime
                a = action            
        #alpha *= 0.9995
        scores += t_reward
        iters += i
        ts += time.clock()-t
        d = np.sum(abs(Q-Q_old))
        diffs += d        
        
        if ((i_episode+1) % 100) == 0:
            #print (scores/100)
            rewards.append(scores/100)
            iterations.append(iters/100)
            timesteps.append(ts/100)
            #deltas.append(1/(1+diffs/100))
            deltas.append(diffs)
            scores, iters, ts, diffs = 0, 0, 0, 0
            #print (i_episode+1, "of", 100000, "episodes done...")
        
        d_cur.append(d)
        
        if (np.sum(Q) != 0):
            if np.max(abs(np.asarray(d_cur[i_episode-5:])-np.asarray(d_old[i_episode-5:]))) <= tol:
                #print ("seems to have converged after", i_episode, "episodes!","...keep going until the end...")
                converge.append(i_episode+1)
                #break
            
        d_old.append(d)
            
    
    results['iterations'] = iterations
    results['timesteps'] = timesteps
    results['deltas'] = deltas
    results['rewards'] = rewards
    
    return (results, Q, converge)
         
# FROZEN LAKE LARGE
env = gym.make('FrozenLake8x8-v0')#('Taxi-v2')#,is_slippery=True)
env.seed = (2019)

# Parameters
alphas = [0.01, 0.1, 0.5, 0.9]
gammas = [0.5, 0.7, 0.8, 0.9, 0.99, 0.995]
#episodes = 10000

epsilons = [0.99, 0.999, 0.9995, 0.9999, 0.99999]

#output
path1 = './OUTPUT/frozenlake-ql_Results.xlsx'
path2 = './OUTPUT/frozenlake-ql_Q_policies_full.json'
path3 = './OUTPUT/frozenlake-ql_Performance.xlsx'
path4 = './OUTPUT/frozenlake-ql_Convergence.xlsx'

dfq = pd.DataFrame()
dfs = pd.DataFrame()
dfc = pd.DataFrame()
score = {}
data = {}
convergence = {}
for alpha in alphas:
    g = {}
    for gamma in gammas:
        e = {}
        for epsilon in epsilons:
            results, Q, converge = run_learner(env, alpha = alpha, gamma = gamma, radr = epsilon)
            print ("alpha=",alpha,"gamma=",gamma,"epsilon=",epsilon, "mean reward=", np.mean(results['rewards']))
            
            #output1 - convergence data

            results['alpha'] = alpha
            results['gamma'] = gamma
            results['epsilon'] = epsilon
            df = pd.DataFrame(results)
            if dfq.empty:
                dfq = df.copy()
            else:
                dfq = dfq.append(df)
dfq.to_excel('./OUTPUT/frozenlake-ql_test.xlsx', sheet_name='qlearner')

            if len(converge) != 0:
                convergence['alpha'] = [alpha]
                convergence['gamma'] = [gamma]
                convergence['epsilon'] = [epsilon]
                convergence['num_converge'] = [len(converge)]
                convergence['min_iter'] = [np.min(converge)]
                convergence['max_iter'] =[ np.max(converge)]
                convergence['median_iter'] = [int(np.median(converge))]
                df = pd.DataFrame(convergence)
                if dfc.empty:
                    dfc = df.copy()
                else:
                    dfc = dfc.append(df)
            
            #output2 - policies and values
            e[epsilon] = {"Q":Q.tolist(), "V":np.max(Q,axis=1).tolist()}
            g[gamma] = e
            data[alpha] = g
            
            #output3 - performance
            policy = np.argmax(Q,axis=1)
            res = run_policy('FrozenLake8x8-v0',policy)
            score['alpha'] = alpha
            score['gamma'] = gamma
            score['epsilon'] = epsilon
            score['timesteps'] = res[0]
            score['outcomes'] = res[1]
            score['rewards'] = res[2]
            score['av_reward'] = res[3]
            df = pd.DataFrame(score)
            if dfs.empty:
                dfs = df.copy()
            else:
                dfs = dfs.append(df)


write to files              
dfq.to_excel(path1, sheet_name='qlearner')
to_json(path2, data)
dfs.to_excel(path3, sheet_name='qlearner')
dfs.to_excel(path4, sheet_name='qlearner')           
              
                
