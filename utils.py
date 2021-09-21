import os
from datetime import datetime, timezone, timedelta
import torch
import warnings
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score


def result_matrics(nn_output, ground_truth):
    nn_output_label = {}
    accuracy = {}
    for cat in nn_output:
        _, tmp_predicted = nn_output[cat].cpu().max(1)
        tmp_groundtruth = ground_truth[cat].cpu()
        nn_output_label[cat] = [tmp_predicted, tmp_groundtruth]

    # ignore warning
    with warnings.catch_warnings():
         warnings.simplefilter('ignore')
         for cat in nn_output_label:
             accuracy[cat] = accuracy_score(y_true=nn_output_label[cat][1].numpy(),
                                            y_pred=nn_output_label[cat][0].numpy())
             
    return accuracy


def checkpoint_save(model, name, epoch):
    f = os.path.join(name, 'checkpoint-{:06d}.pth'.format(epoch))
    torch.save(model.state_dict(), f)
    print('Saved checkpoint:', f)

    return f


def checkpoint_load(model, name):
    print('Restoring checkpoint: {}'.format(name))
    model.load_state_dict(torch.load(name, map_location='cpu'))
    epoch = int(os.path.splitext(os.path.basename(name))[0].split('-')[1])
    return epoch


def get_cur_time():
    timezone_offset = 8.0
    tzinfo = timezone(timedelta(hours=timezone_offset))
    return datetime.strftime(datetime.now(tzinfo), '%Y-%m-%d_%H-%M')