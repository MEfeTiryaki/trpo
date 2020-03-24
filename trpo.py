import numpy as np

import torch
from torch.autograd import Variable
from utils import *


def getSurrogateloss(model,states,actions,advantages,logProbabilityOld):
    log_prob = model.getLogProbabilityDensity(states,Variable(actions))
    action_loss = -advantages.squeeze() * torch.exp(log_prob - Variable(logProbabilityOld))
    return action_loss.mean()

def FisherVectorProduct(v , model, states, actions,logProbabilityOld,damping):
    kl = model.meanKlDivergence(states, actions,logProbabilityOld)
    # print("kl\n" ,kl)
    grads = torch.autograd.grad(kl, model.parameters()
                    ,retain_graph=True, create_graph=True)
    flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])
    kl_v = (flat_grad_kl * v).sum()
    grads = torch.autograd.grad(kl_v, model.parameters())
    flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1)
                                    for grad in grads]).data
    return flat_grad_grad_kl + v * damping

def trpo_step(model, states, actions, advantages,logProbabilityOld, max_kl, damping):

    get_loss = lambda x: getSurrogateloss(x,
                                    states,
                                    actions,
                                    advantages,
                                    logProbabilityOld)
    loss = get_loss(model)
    grads = torch.autograd.grad(loss, model.parameters())
    loss_grad = torch.cat([grad.view(-1) for grad in grads])

    Fvp = lambda v: FisherVectorProduct(v,
                                        model,
                                        states,
                                        actions,
                                        logProbabilityOld,
                                        damping)

    stepdir = conjugate_gradients(Fvp, -loss_grad, 10)
    shs = 0.5 * (stepdir * Fvp(stepdir)).sum(0, keepdim=True)
    lm = torch.sqrt(shs / max_kl)
    fullstep = stepdir / lm[0]
    neggdotstepdir = (-loss_grad * stepdir).sum(0, keepdim=True)
    prev_params = get_flat_params_from(model)
    success, new_params = linesearch(model, get_loss, prev_params, fullstep,
                                     neggdotstepdir / lm[0])
    set_flat_params_to(model, new_params)
    return loss

def conjugate_gradients(Avp, b, nsteps, residual_tol=1e-10):
    x = torch.zeros(b.size())
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for i in range(nsteps):
        _Avp = Avp(p)
        alpha = rdotr / torch.dot(p, _Avp)
        x += alpha * p
        r -= alpha * _Avp
        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break

    return x

def linesearch(model,
               f,
               x,
               fullstep,
               expected_improve_rate,
               max_backtracks=10,
               accept_ratio=.1):
    fval = f(model).data
    # print("\tfval before", fval.item())
    for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
        xnew = x + stepfrac * fullstep
        set_flat_params_to(model, xnew)
        newfval = f(model).data
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        # print("\ta : %6.4e /e : %6.4e /r : %6.4e "%(actual_improve.item(), expected_improve.item(), ratio.item()))

        if ratio.item() > accept_ratio and actual_improve.item() > 0:
            # print("\tfval after", newfval.item())
            return True, xnew
    return False, x
