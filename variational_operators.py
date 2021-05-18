'''
Copyright 2019, INRIA
SBX and ido_dd and polynomilal mutauion variation operators based on pymap_elites framework
https://github.com/resibots/pymap_elites/blob/master/map_elites/
pymap_elites main contributor(s):
    Jean-Baptiste Mouret, jean-baptiste.mouret@inria.fr
    Eloise Dalin , eloise.dalin@inria.fr
    Pierre Desreumaux , pierre.desreumaux@inria.fr
Modified by Olle Nilsson: olle.nilsson19@imperial.ac.uk
'''


import copy
import numpy as np
import torch
from multiprocessing import Process, Queue
from functools import partial
import heapq
from operator import itemgetter

from vectorized_env import CloudpickleWrapper


def parallel_worker(process_id,
                    actors_train_in_queue,
                    actors_train_out_queue,
                    learning_rate,
                    actor_fn_wrapper
                    ):

    '''
    Function that runs the parallel processes for the variation operator
    Parameters:
            process_id (int): ID of the process so it can be identified
            actors_train_in_queue (Queue object): queue for incoming actors
            actors_train_out_queue (Queue object): queue for outgoing actors
            learning_rate (float): learning rate for gradient variation
            actor_fn_wrapper : function that when called retuns a new actor network
    '''
    actor_fn = actor_fn_wrapper.x
    # Start process loop
    while True:
        try:
            # get new actor and critic etc.
            n, actor_x, critic, states, nr_of_steps = actors_train_in_queue.get()
            actor_z = copy.deepcopy(actor_x)
            actor_z.type = "grad"
            actor_z.parent_1_id = actor_x.id
            actor_z.parent_2_id = None
            # Enable grad
            for param in actor_z.parameters():
                param.requires_grad = True
            optimizer = torch.optim.Adam(actor_z.parameters(), lr=learning_rate)
            # gradient decent loop
            for i in range(nr_of_steps):
                state = states[i]
                actor_loss = -critic.Q1(state, actor_z(state)).mean() 
                optimizer.zero_grad()
                actor_loss.backward()
                optimizer.step()
            # Disable grad so can sent across proceeses
            for param in actor_z.parameters():
                param.requires_grad = False
            actors_train_out_queue.put((n, actor_z))

        except KeyboardInterrupt:
            break


class VariationalOperator(object):
    """
    A class for applying the variation operator in parallel.
    """
    def __init__(self,
                actor_fn,
                num_cpu = False,
                gradient_op = True,
                crossover_op = "iso_dd",
                mutation_op = None,
                learning_rate = 3e-4, 
                max_gene = False,
                min_gene = False,
                mutation_rate = 0.05,
                crossover_rate = 0.75,
                eta_m = 5.0,
                eta_c = 10.0,
                sigma = 0.1,
                max_uniform = 0.1,
                iso_sigma = 0.005,
                line_sigma = 0.05):
        
        self.actor_fn = actor_fn
        self.gradient_op = gradient_op
    
        if crossover_op in ["sbx", "iso_dd"]:
            self.crossover_op = getattr(self, crossover_op)
        else:
            self.crossover_op = False

        if mutation_op in ["polynomial_mutation", "gaussian_mutation", "uniform_mutation"]:
            self.mutation_op = getattr(self, mutation_op)
        else:
            self.mutation_op = False

        print(f"Mutation operator: {self.mutation_op}")
        print(f"Crossover operator: {self.crossover_op}")
        
        self.learning_rate = learning_rate
        self.max = max_gene
        self.min = min_gene
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.eta_m = eta_m
        self.eta_c = eta_c
        self.sigma = sigma
        self.max_uniform = max_uniform
        self.iso_sigma = iso_sigma
        self.line_sigma = line_sigma

        self.n_processes = num_cpu
        self.actors_train_in_queue = Queue()
        self.actors_train_out_queue = Queue()
        # Setup parallel processes
        self.processes = [Process(target=parallel_worker, 
                                    args=(process_id,
                                            self.actors_train_in_queue,
                                            self.actors_train_out_queue,
                                            self.learning_rate,
                                            CloudpickleWrapper(self.actor_fn))) for process_id in range(self.n_processes)]
        # Start parallel processes
        for p in self.processes:
            p.daemon = True
            p.start()



    def close(self):
        '''
        Close parallel processes
        '''
        for p in self.processes:
            p.terminate()



    def __call__(self,
                archive,
                batch_size,
                proportion_evo,
                critic=False,
                states=False,
                train_batch_size=False,
                nr_of_steps_act=False):
        '''
        the variation operator object is called to apply the varation
 
        Parameters:
            archive (dict): main archive
            batch_size (int): how many actors to sample per generation
            proportion_evo (float): proportion of GA variation
            critic: Critic to use in gradient variation
            states: states (tansitions) to use in policy gradient update 
            train_batch_size (int): batch size for policy gradient update
            nr_of_steps_act (int): nr of gradient steps to take per actor
        '''

        
        keys = list(archive.keys())
        actors_z = []
        # sample form archive
        if self.mutation_op and not self.crossover_op:
            actors_x_evo = []
            rand_evo = np.random.randint(len(keys), size=int(batch_size * proportion_evo))
            for n in range(0, len(rand_evo)):
                actors_x_evo += [archive[keys[rand_evo[n]]]]

        elif self.crossover_op:
            actors_x_evo = []
            actors_y_evo = []
            rand_evo_1 = np.random.randint(len(keys), size=int(batch_size * proportion_evo))
            rand_evo_2 = np.random.randint(len(keys), size=int(batch_size * proportion_evo))
            for n in range(0, len(rand_evo_1)):
                actors_x_evo += [archive[keys[rand_evo_1[n]]]]
                actors_y_evo += [archive[keys[rand_evo_2[n]]]]

        # apply GA variation
        for n in range(len(actors_x_evo)):
            if self.crossover_op:
                if self.mutation_op:
                    actors_z += [self.evo(actors_x_evo[n].x, actors_y_evo[n].x, self.crossover_op, self.mutation_op)]
                else:
                    actors_z += [self.evo(actors_x_evo[n].x, actors_y_evo[n].x, self.crossover_op)]
            elif self.mutation_op:
                actors_z += [self.evo(actors_x_evo[n].x, False, False, self.mutation_op)]

        # sample form archive
        if self.gradient_op:
            actors_x_grad = []
            rand_grad = np.random.randint(len(keys), size=(batch_size - int(batch_size * proportion_evo)))
            for n in range(0, len(rand_grad)):
                actors_x_grad += [archive[keys[rand_grad[n]]]]
            # apply PG variation
            actors_z_grad = [None] * len(actors_x_grad)
            for n in range(len(actors_x_grad)):
                self.actors_train_in_queue.put((n,
                                                actors_x_grad[n].x,
                                                critic,
                                                states,
                                                nr_of_steps_act))

            for _ in range(len(actors_x_grad)):
                n, actor_z = self.actors_train_out_queue.get()
                actors_z_grad[n] = actor_z
            actors_z += actors_z_grad
        return actors_z


    def evo(self, actor_x, actor_y=False, crossover_op=False, mutation_op=False):
        actor_z = copy.deepcopy(actor_x)
        actor_z.optimizer = None
        actor_z.type = "evo"
        if crossover_op:
            actor_z.parent_1_id = actor_x.id
            actor_z.parent_2_id = actor_y.id
            actor_z_state_dict = self.crossover(actor_x.state_dict(), actor_y.state_dict(), crossover_op)
            if mutation_op:
                actor_z_state_dict = self.mutation(actor_z_state_dict, mutation_op)
        elif mutation_op:
            actor_z.parent_1_id = actor_x.id
            actor_z.parent_2_id = None
            actor_z_state_dict = self.mutation(actor_x.state_dict(), mutation_op)
        actor_z.load_state_dict(actor_z_state_dict)
        return actor_z



    def crossover(self, actor_x_state_dict, actor_y_state_dict, crossover_op):
        actor_z_state_dict = copy.deepcopy(actor_x_state_dict)
        for tensor in actor_x_state_dict:
            if "weight" or "bias" in tensor:
                actor_z_state_dict[tensor] = crossover_op(actor_x_state_dict[tensor], actor_y_state_dict[tensor])
        return actor_z_state_dict



    def mutation(self, actor_x_state_dict, mutation_op):
        y = copy.deepcopy(actor_x_state_dict)
        for tensor in actor_x_state_dict:
            if "weight" or "bias" in tensor:
                y[tensor] = mutation_op(actor_x_state_dict[tensor])
        return y


    def iso_dd(self, x, y):
        '''
        Iso+Line
        Ref:
        Vassiliades V, Mouret JB. Discovering the elite hypervolume by leveraging interspecies correlation.
        GECCO 2018
        '''
        a = torch.zeros_like(x).normal_(mean=0, std=self.iso_sigma)
        b = np.random.normal(0, self.line_sigma)
        z = x.clone() + a + b * (y - x)

        if not self.max and not self.min:
            return z
        else:
            return torch.clamp(z, self.min, self.max)



    def sbx(self, x, y):
        if not self.max and not self.min:
            return self.__sbx_unbounded(x, y)
        else:
            return self.__sbx_bounded(x, y)


    def __sbx_unbounded(self, x, y):
        '''
        SBX (cf Deb 2001, p 113) Simulated Binary Crossover
        Unbounded version
        A large value ef eta gives a higher probablitity for
        creating a `near-parent' solutions and a small value allows
        distant solutions to be selected as offspring.
        '''
        z = x.clone()
        c = torch.rand_like(z)
        index = torch.where(c < self.crossover_rate)
        r1 = torch.rand(index[0].shape)
        r2 = torch.rand(index[0].shape)

        if len(z.shape) == 1:
            diff = torch.abs(x[index[0]] - y[index[0]])
            x1 = torch.min(x[index[0]], y[index[0]])
            x2 = torch.max(x[index[0]], y[index[0]])
            z_idx = z[index[0]]
        else:
            diff = torch.abs(x[index[0], index[1]] - y[index[0], index[1]])
            x1 = torch.min(x[index[0], index[1]], y[index[0], index[1]])
            x2 = torch.max(x[index[0], index[1]], y[index[0], index[1]])
            z_idx = z[index[0], index[1]]

        beta_q = torch.where(r1 <= 0.5, (2.0 * r1) ** (1.0 / (self.eta_c + 1)), (1.0 / (2.0 * (1.0 - r1))) ** (1.0 / (self.eta_c + 1)))

        c1 = 0.5 * (x1 + x2 - beta_q * (x2 - x1))
        c2 = 0.5 * (x1 + x2 + beta_q * (x2 - x1))

        z_mut = torch.where(diff > 1e-15, torch.where(r2 <= 0.5, c2, c1), z_idx)

        if len(y.shape) == 1:
            z[index[0]] = z_mut
        else:
            z[index[0], index[1]] = z_mut
        return z


    def __sbx_bounded(self, x, y):
        '''
        SBX (cf Deb 2001, p 113) Simulated Binary Crossover
        A large value ef eta gives a higher probablitity for
        creating a `near-parent' solutions and a small value allows
        distant solutions to be selected as offspring.
        '''
        z = x.clone()
        c = torch.rand_like(z)
        index = torch.where(c < self.crossover_rate)
        r1 = torch.rand(index[0].shape)
        r2 = torch.rand(index[0].shape)

        if len(z.shape) == 1:
            diff = torch.abs(x[index[0]] - y[index[0]])
            x1 = torch.min(x[index[0]], y[index[0]])
            x2 = torch.max(x[index[0]], y[index[0]])
            z_idx = z[index[0]]
        else:
            diff = torch.abs(x[index[0], index[1]] - y[index[0], index[1]])
            x1 = torch.min(x[index[0], index[1]], y[index[0], index[1]])
            x2 = torch.max(x[index[0], index[1]], y[index[0], index[1]])
            z_idx = z[index[0], index[1]]


        beta = 1.0 + (2.0 * (x1 - self.min) / (x2 - x1))
        alpha = 2.0 - beta ** - (self.eta_c + 1)
        beta_q = torch.where(r1 <= (1.0 / alpha), (r1 * alpha) ** (1.0 / (self.eta_c + 1)), (1.0 / (2.0 - r1 * alpha)) ** (1.0 / (self.eta_c + 1)))

        c1 = 0.5 * (x1 + x2 - beta_q * (x2 - x1))

        beta = 1.0 + (2.0 * (self.max - x2) / (x2 - x1))
        alpha = 2.0 - beta ** - (self.eta_c + 1)

        beta_q = torch.where(r1 <= (1.0 / alpha), (r1 * alpha) ** (1.0 / (self.eta_c + 1)), (1.0 / (2.0 - r1 * alpha)) ** (1.0 / (self.eta_c + 1)))
        c2 = 0.5 * (x1 + x2 + beta_q * (x2 - x1))

        c1 = torch.clamp(c1, self.min, self.max)
        c2 = torch.clamp(c2, self.min, self.max)

        z_mut = torch.where(diff > 1e-15, torch.where(r2 <= 0.5, c2, c1), z_idx)

        if len(y.shape) == 1:
            z[index[0]] = z_mut
        else:
            z[index[0], index[1]] = z_mut
        return z


    def polynomial_mutation(self, x):
        '''
        Cf Deb 2001, p 124 ; param: eta_m
        '''
        y = x.clone()
        m = torch.rand_like(y)
        index = torch.where(m < self.mutation_rate)
        r = torch.rand(index[0].shape)
        delta = torch.where(r < 0.5,\
            (2 * r) ** (1.0 / (self.eta_m + 1.0)) -1.0,\
                    1.0 - ((2.0 * (1.0 - r)) ** (1.0 / (self.eta_m + 1.0))))
        if len(y.shape) == 1:
            y[index[0]] += delta
        else:
            y[index[0], index[1]] += delta

        if not self.max and not self.min:
            return y
        else:
            return torch.clamp(y, self.min, self.max)



    def gaussian_mutation(self, x):
        y = x.clone()
        m = torch.rand_like(y)
        index = torch.where(m < self.mutation_rate)
        delta = torch.zeros(index[0].shape).normal_(mean=0, std=self.sigma)
        if len(y.shape) == 1:
            y[index[0]] += delta
        else:
            y[index[0], index[1]] += delta

        if not self.max and not self.min:
            return y
        else:
            return torch.clamp(y, self.min, self.max)



    def uniform_mutation(self, x):
        y = x.clone()
        m = torch.rand_like(y)
        index = torch.where(m < self.mutation_rate)
        delta = torch.zeros(index[0].shape).uniform_(-self.max_uniform, self.max_uniform)
        if len(y.shape) == 1:
            y[index[0]] += delta
        else:
            y[index[0], index[1]] += delta

        if not self.max and not self.min:
                return y
        else:
            return torch.clamp(y, self.min, self.max)




if __name__ == "__main__":
    actor_x = Actor(5, 5, 1)
    actor_y = Actor(5, 5, 1)
    var = VariationalOperator(mutation_op=False)
    actor_z = var(actor_x, actor_y)
    print(actor_x.state_dict()["l1.bias"])
    print(actor_y.state_dict()["l1.bias"])
    print(actor_z.state_dict()["l1.bias"])
