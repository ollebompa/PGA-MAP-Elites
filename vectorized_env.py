import numpy as np
import cloudpickle
import pickle
from multiprocessing import Process, Queue, Event, Pipe
import copy
import time
import torch



def parallel_critic(replay_fn,
                    critic_fn,
                    trans_in_queue,
                    remote,
                    global_sync,
                    close_processes,
                    nr_of_steps,
                    batch_size,
                    nr_of_steps_act,
                    random_init):

    '''
    Function that runs the  processes for the critic training
    Parameters:
        replay_fn: function that inililises the replay buffer
        critic_fn: function that inililises the critc
        trans_in_queue: que to recive trasitions
        remote: pipe to recive species archive
        global_sync: event to trigger synch
        nr_of_steps: nr of steps to train critic per generation
        batch_size: batch size for trining critic
    '''
    # inililise replay buffer and critic
    replay_buffer = replay_fn.x()
    critic = critic_fn.x()
    archive = False
    waiting = False
    # start loop for process
    while True:
        try:
            if close_processes.is_set():
                print("Close Critic Process")
                remote.send((critic, replay_buffer))
                time.sleep(10)
                break
            # collect new tranitions
            while trans_in_queue.qsize() > 0:
                try:
                    idx, transitions = trans_in_queue.get_nowait()
                    replay_buffer.add(transitions)
                except:
                    pass
            # get new archive
            if remote.poll():
                archive = remote.recv()

            # start critic training
            if replay_buffer.additions > random_init * 0.9 and archive and not waiting: # hack as well
                t1 = time.time()
                # train critic
                critic_loss = critic.train(archive, replay_buffer, nr_of_steps, batch_size=batch_size)
                train_time = time.time() - t1
                waiting = True
            # synch
            if global_sync.is_set():
                out_actors = []
                for actor in critic.actors:
                    a = copy.deepcopy(actor)
                    for param in a.parameters():
                        param.requires_grad = False
                    out_actors.append(a)
                states = replay_buffer.sample_state(batch_size, nr_of_steps_act)
                remote.send((critic.critic, out_actors, states, critic_loss.detach(), train_time))
                global_sync.clear()
                waiting = False

        except KeyboardInterrupt:
            break
        


def parallel_worker(process_id,
                    env_fn_wrapper,
                    eval_in_queue,
                    eval_out_queue,
                    trans_out_queue,
                    close_processes,
                    remote,
                    master_seed):
    
    '''
    Function that runs the paralell processes for the evaluation
    Parameters:
        process_id (int): ID of the process so it can be identified
        env_fn_wrapper : function that when called starts a new environment
        eval_in_queue (Queue object): queue for incoming actors
        eval_out_queue (Queue object): queue for outgoing actors
        trans_out_queue (Queue object): queue for outgoing transitions
    '''
    # start environmet simulation
    env = env_fn_wrapper.x()
    # begin process loop
    while True:
        try:
            # get a new actor to evalaute
            try:
                idx, actor, evaluation_id, eval_mode = eval_in_queue.get_nowait()
                env.seed(int((master_seed + 100) * evaluation_id))
                state = env.reset()
                done = False
                # eval loop
                while not done:
                    action = actor.select_action(np.array(state)) 
                    next_state, reward, done, _ = env.step(action)
                    done_bool = float(done) if env.T < env._max_episode_steps else 0
                    if env.T == 1:
                        state_array = state
                        action_array = action
                        next_state_array = next_state
                        reward_array = reward
                        done_bool_array = done_bool
                    else:
                        state_array = np.vstack((state, state_array))
                        action_array = np.vstack((action, action_array))
                        next_state_array = np.vstack((next_state, next_state_array))
                        reward_array = np.vstack((reward, reward_array))
                        done_bool_array = np.vstack((done_bool, done_bool_array))
                    state = next_state
                eval_out_queue.put((idx, (env.tot_reward, env.desc, env.alive, env.T)))
                if not eval_mode:
                    trans_out_queue.put((idx, (state_array, action_array, next_state_array, reward_array, done_bool_array)))
            except:
                pass
            if close_processes.is_set():
                print(f"Close Eval Process nr. {process_id}")
                remote.send((process_id, env.np_random.get_state()))
                env.close()
                time.sleep(10)
                break
        except KeyboardInterrupt:
            env.close()
            break



class ParallelEnv(object):
    def __init__(self, env_fns, replay_fn, critic_fn, nr_of_steps, batch_size, nr_of_steps_act, random_init, seed):
        """
        A class for paralell evaluation.
        """
        self.n_processes = len(env_fns)
        self.eval_in_queue = Queue()
        self.eval_out_queue = Queue()
        self.trans_out_queue = Queue()
        self.remotes, self.locals = zip(*[Pipe() for _ in range(self.n_processes + 1)])
        self.global_sync = Event()
        self.close_processes = Event()

        
        self.steps = None
        self.nr_of_steps = nr_of_steps
        self.batch_size = batch_size
        self.seed=seed
        self.evaluation_id=0

        self.processes = [Process(target=parallel_worker, 
                                  args=(process_id,
                                        CloudpickleWrapper(env_fn),
                                        self.eval_in_queue,
                                        self.eval_out_queue,
                                        self.trans_out_queue,
                                        self.close_processes,
                                        self.remotes[process_id],
                                        self.seed)) for process_id, env_fn in enumerate(env_fns)]

        for p in self.processes:
            p.daemon = True
            p.start()

        self.critic_process = Process(target=parallel_critic, 
                                        args=(CloudpickleWrapper(replay_fn),
                                              CloudpickleWrapper(critic_fn),
                                              self.trans_out_queue,
                                              self.remotes[-1],
                                              self.global_sync,
                                              self.close_processes,
                                              nr_of_steps,
                                              batch_size,
                                              nr_of_steps_act,
                                              random_init))

        self.critic_process.daemon = True
        self.critic_process.start()



    def eval_policy(self, actors, eval_mode=False):
        self.steps = 0
        results = [None] * len(actors)
        for idx, actor in enumerate(actors):
            self.evaluation_id += 1
            self.eval_in_queue.put((idx, actor, self.evaluation_id, eval_mode))
        for _ in range(len(actors)):
            idx, result = self.eval_out_queue.get()
            self.steps += result[3]
            results[idx] = result
        return results


    def update_archive(self, archive):
        self.locals[-1].send(archive)


    def get_critic(self):
        self.global_sync.set()
        critic, actor, states, critic_loss, time = self.locals[-1].recv()
        print(f"Train Time: {time}")
        print(f"Critic Loss: {critic_loss}")
        return critic, actor, states, time


    def close(self):
        self.close_processes.set()
        rng_states = []
        for local in self.locals[0:-1]:
            rng_states.append(local.recv())
        critic, replay_buffer = self.locals[-1].recv()
        for p in self.processes:
            p.terminate()
        self.critic_process.terminate()

        return critic, replay_buffer, [x[1] for x in sorted(rng_states, key=lambda element: element[0])]



class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_env.py#L190
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        self.x = pickle.loads(ob)
