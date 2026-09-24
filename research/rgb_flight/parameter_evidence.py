"""Parameter and gradient receipts from real optimizer work."""
import hashlib
import torch


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


class ParameterEvidence:
    def __init__(self,modules):
        self.parameters={root+'.'+name:p for root,module in modules.items() for name,p in module.named_parameters()}
        self.before={name:digest(p) for name,p in self.parameters.items()}
        self.gradients={}

    def observe_gradients(self):
        for name,p in self.parameters.items():
            if p.grad is not None:
                finite=bool(torch.isfinite(p.grad).all())
                if not finite:raise RuntimeError('Nonfinite gradient: '+name)
                self.gradients[name]=float(p.grad.detach().float().norm())

    def finish(self,require_update=True):
        groups={}
        for name,p in self.parameters.items():
            group='.'.join(name.split('.')[:2])
            row=groups.setdefault(group,dict(parameters=0,trainable=0,gradient_tensors=0,
                                            nonzero_gradient_tensors=0,changed_tensors=0,frozen_changed=0))
            changed=digest(p)!=self.before[name]
            row['parameters']+=1;row['trainable']+=int(p.requires_grad)
            row['gradient_tensors']+=int(name in self.gradients)
            row['nonzero_gradient_tensors']+=int(self.gradients.get(name,0)>0)
            row['changed_tensors']+=int(changed);row['frozen_changed']+=int(changed and not p.requires_grad)
        if any(r['frozen_changed'] for r in groups.values()):raise RuntimeError('Frozen parameter changed')
        if require_update and not any(r['nonzero_gradient_tensors'] and r['changed_tensors'] for r in groups.values()):
            raise RuntimeError('Optimizer did not change parameters with nonzero gradients')
        return groups
