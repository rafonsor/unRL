# Artificial (General) Intelligence
The path forward:
  - [The Alberta Plan for AI Research](https://arxiv.org/pdf/2208.11173.pdf)
  - [Reward-respecting subtasks for model-based reinforcement learning](https://www.sciencedirect.com/science/article/pii/S0004370223001479#br0380)
  - FM/LLM-powered RL Agents

# Reinforcement Learning Algorithms

## Model-Free

- [x] A2C: [Advantage Actor-Critic](unrl/algos/actor_critic.py)
- [x] ACER: [Sample Efficient Actor-Critic with Experience Replay](https://arxiv.org/pdf/1611.01224.pdf)
- [x] ACKTR: [Actor Critic using Kronecker-Factored Trust Region](https://arxiv.org/abs/1708.05144)
- [ ] AQT: [Action Q-Transformer](https://arxiv.org/pdf/2306.13879.pdf)
- [x] DQN: [Deep Q-Network](unrl/algos/dqn.py)
  - [x] DDQN: [Double Deep Q-Network](unrl/algos/dqn.py)
  - [x] DuelingDQN: [Dueling Deep Q-Network](unrl/algos/dqn.py)
  - [ ] h-DQN: [Hierarchical-DQN: Integrating Temporal Abstraction and Intrinsic Motivation](https://proceedings.neurips.cc/paper_files/paper/2016/file/f442d33fa06832082290ad8544a8da27-Paper.pdf)
  - [x] PER-DQN: [Double Deep Q-Network with Prioritized Experience Replay](unrl/algos/dqn.py)
  - [ ] Rainbow DQN: [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/pdf/1710.02298)
- [x] DDPG: [Deep Deterministic Policy Gradient](unrl/algos/ddpg.py)
  - [x] TD3: [Twin-Delayed Deep Deterministic Policy Gradient](unrl/algos/ddpg.py)
- [ ] FuNs: [Feudal Networks for Hierarchical Reinforcement Learning](http://proceedings.mlr.press/v70/vezhnevets17a.html)
- [ ] GAE: [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
- [ ] GAIL: [Generative Adversarial Imitation Learning](https://arxiv.org/pdf/1606.03476.pdf)
- [ ] GCL: [Guided Cost Learning: Deep Inverse Optimal Control via Policy Optimization](https://proceedings.mlr.press/v48/finn16.html)
- [ ] HER: [Hindsight Experience Replay](https://arxiv.org/pdf/1707.01495.pdf)
- [ ] IMPALA: [Importance weighted Actor-Learner Architectures](http://proceedings.mlr.press/v80/espeholt18a.html)
- [x] NAF: [Normalised Advantage Functions](https://arxiv.org/pdf/1603.00748.pdf)
- [ ] NEC: [Neural Episodic Control](https://proceedings.mlr.press/v70/pritzel17a/pritzel17a.pdf)
- [ ] OK: [The option keyboard: Combining skills in reinforcement learning](https://proceedings.neurips.cc/paper_files/paper/2019/hash/251c5ffd6b62cc21c446c963c76cf214-Abstract.html)
- [ ] Option-Critic: [The Option-Critic Architecture](https://ojs.aaai.org/index.php/AAAI/article/view/10916)
- [x] PPO: [Proximal Policy Optimization](unrl/algos/policy_gradient.py)
  - [ ] Continual PPO: [Loss of Plasticity in Deep Continual Learning](https://arxiv.org/pdf/2306.13812.pdf)
    - In Appendix E
    - Paper unveils "continual backpropagation": proposes tracking Utility of activation units to guide parameter re-initialisation
  - [x] TRPO: [Trust-Region Policy Optimization](unrl/algos/policy_gradient.py)
- [ ] Q-Transformer: [Scalable Offline Reinforcement Learning via Autoregressive Q-Functions](https://arxiv.org/pdf/2309.10150)
- [x] REINFORCE: [REINFORCE](unrl/algos/policy_gradient.py)
  - [x] Baseline: [REINFORCE with State-Value Baseline](unrl/algos/policy_gradient.py)
- [x] SAC: [Soft Actor-Critic](unrl/algos/ddpg.py)


## Model-Based

- [ ] World Model: [Recurrent World Models Facilitate Policy Evolution](https://proceedings.neurips.cc/paper/2018/hash/2de5d16682c3c35007e4e92982f1a2ba-Abstract.html)
- [ ] AlphaZero: [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/pdf/1712.01815.pdf)
  - AlphaGo Zero [Mastering the game of go without human knowledge](https://www.nature.com/articles/nature24270)
- [ ] DreamerV3: [Mastering Diverse Domains through World Models](https://arxiv.org/pdf/2301.04104.pdf)
- [ ] I2A: [Imagination-augmented agents for deep reinforcement learning](https://proceedings.neurips.cc/paper/2017/hash/9e82757e9a1c12cb710ad680db11f6f1-Abstract.html)
- [x] ICM: [Curiosity-driven Exploration by Self-supervised Prediction](https://proceedings.mlr.press/v70/pathak17a.html)
- [ ] PETS: [Probabilistic Ensembles with Trajectory Sampling](https://proceedings.neurips.cc/paper_files/paper/2018/hash/3de568f8597b94bda53149c7d7f5958c-Abstract.html)
- [ ] SAVE: [Search with Amortized Value Estimates](https://arxiv.org/abs/1912.02807)
