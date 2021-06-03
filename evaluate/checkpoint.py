import os

import torch


def save_checkpoint(net, clf, critic, epoch, args, script_name, acc):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'clf': clf.state_dict(),
        'critic': critic.state_dict(),
        'epoch': epoch,
        'args': vars(args),
        'script': script_name,
        'acc': acc
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    destination = os.path.join('./checkpoint', args.filename + "-epoch%d-acc%.2f" % (epoch+1, acc))
    torch.save(state, destination)
