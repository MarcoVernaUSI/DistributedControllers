from enum import Enum
from random import shuffle
from typing import Sequence, Tuple, TypeVar, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

State = TypeVar('State')
Sensing = TypeVar('Sensing')
Control = TypeVar('Control')
Communication = TypeVar('Communication')
MState = Sequence[State]
MSensing = Sequence[Sensing]
MControl = Sequence[Control]

Dynamic = Callable[[Sequence[State], Sequence[Control]], MState]
Sensor = Callable[[MState], MSensing]
ControlOutput = Tuple[Sequence[Control], Sequence[Communication]]
Controller = Callable[[Sequence[State], Sequence[Sensing]], ControlOutput]

class Sync(Enum):
    random = 1
    sequential = 2
    sync = 3


class SNet(nn.Module):
    def __init__(self):
        super(SNet, self).__init__()
        self.l1 = nn.Linear(4, 10)
        self.l2 = nn.Linear(10, 2)

    def forward(self, input):
        ys = F.torch.tanh(self.l1(input))
        return self.l2(ys)


class S2Net(nn.Module):
    def __init__(self):
        super(S2Net, self).__init__()
        self.l1 = nn.Linear(4, 10)
        self.l2 = nn.Linear(10, 3)

    def forward(self, input):
        ys = F.torch.tanh(self.l1(input))
        return self.l2(ys)



class SNetLeft(nn.Module):
    def __init__(self):
        super(SNetLeft, self).__init__()
        self.l1 = nn.Linear(2, 10)
        self.l2 = nn.Linear(10, 2)

    def forward(self, input):
        ys = F.torch.tanh(self.l1(input))
        return self.l2(ys)


def input_from(ss, comm, i):
    return torch.cat((ss[i], comm[i:i+1], comm[i+2:i+3]), 0)

def input_from2(ss, comm, i):
    return torch.cat((ss[i], comm[(2*i):(2*i)+1], comm[(2*i)+3:(2*i)+4]), 0)


def input_from_left(ss, comm, i):
    return torch.cat((ss[i][:1], comm[i:i+1]), 0)


def init_comm(N):
    return Variable(torch.Tensor([0] * (N + 2)))


class ComNet(nn.Module):
    def __init__(self, N, sync = Sync.sequential, module = SNet,
                 input_fn=input_from):
        super(ComNet, self).__init__()
        self.single_net = module()
        self.N = N
        self.sync = sync
        self.input_fn = input_fn

    def step(self, xs, comm, sync):
        if sync == Sync.sync:
            input = torch.stack([self.input_fn(xs, comm, i) for i in range(self.N)], 0)
            output = self.single_net(input)
            control = output[:, 0]
            comm[1:-1] = output[:, 1]
        else:
            indices = list(range(self.N))
            if sync == Sync.random:
                shuffle(indices)
            cs = []
            for i in indices:
                output = self.single_net(self.input_fn(xs, comm, i))
                comm[i+1] = output[1]
                cs.append(output[:1])
            control = torch.cat(cs, 0)
        return control

        # prende in input il sensing [s1,s2] e la comunicazione ricevuta [c1,c2]
    def single_step(self, xs, comm):
        input = torch.cat((torch.from_numpy(xs).float(), torch.from_numpy(comm).float()), 0)
        output = self.single_net(input)
        control = output[0]
        next_comm = output[1]
    #    print(next_comm)
        return control, next_comm

    def forward(self, runs):
        rs = []
        for run in runs:
            comm = init_comm(self.N)
            controls = []
            for xs in run:
                controls.append(self.step(xs, comm, self.sync))
            rs.append(torch.stack(controls))
        return torch.stack(rs)

    def controller(self, sync = Sync.sequential):
        N = self.N
        comm = init_comm(N)

        def f(sensing):
            with torch.no_grad():
                sensing = torch.FloatTensor(sensing)                
                control = self.step(sensing, comm, sync=sync).numpy()
                return control, comm[1:-1].clone().numpy().flatten()
        return f



class Com2Net(nn.Module):
    def __init__(self, N, sync= Sync.sequential, module = S2Net,
                 input_fn=input_from2):
        super(Com2Net, self).__init__()
        self.single_net = module()
        self.N = N
        self.sync = sync
        self.input_fn = input_fn

    def step(self, xs, comm, sync):
        # il sync va aggiornato ancora con la doppia comunicazione
        if sync == Sync.sync:
            input = torch.stack([self.input_fn(xs, comm, i) for i in range(self.N)], 0)
            output = self.single_net(input)            
            control = output[:, 0]
            comm[1:-1] = output[:, 1]
        else:
            indices = list(range(self.N))
            if sync == Sync.random:
                shuffle(indices)
            cs = []
            for i in indices:
                output = self.single_net(self.input_fn(xs, comm, i))
                comm[(2*i)+1:(2*i)+3] = output[1:]
                cs.append(output[:1])
            control = torch.cat(cs, 0)
        return control

    def forward(self, runs):
        rs = []
        for run in runs:
            comm = init_comm(self.N*2)
            controls = []
            for xs in run:
                controls.append(self.step(xs, comm, self.sync))
            rs.append(torch.stack(controls))
        return torch.stack(rs)

    def controller(self, sync = Sync.sequential):
        N = self.N
        comm = init_comm(N*2)

        def f(sensing):
            with torch.no_grad():
                sensing = torch.FloatTensor(sensing)                
                control = self.step(sensing, comm, sync=sync).numpy()
                return control, comm[1:-1].clone().numpy().flatten()
        return f
