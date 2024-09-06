import matplotlib.pyplot as plt
import numpy as np

from config import JOINTS_LIST


def get_predictions_dict():
    return {'total': {'all': 0, 'correct': 0},
            'head': {'all': 0, 'correct': 0},
            'sho': {'all': 0, 'correct': 0},
            'elb': {'all': 0, 'correct': 0},
            'wri': {'all': 0, 'correct': 0},
            'hip': {'all': 0, 'correct': 0},
            'knee': {'all': 0, 'correct': 0},
            'ank': {'all': 0, 'correct': 0},
            }


def pck(pred, gt, threshold, predictions):
    """
    PCK: An estimation is considered correct if it lies within α·max(h, w) from the true position,
    where h and w are the height and width of the bounding box
    """
    a = 0.2
    for i in range(pred.shape[0] - 1):  # For each joint

        predictions['total']['all'] += 1
        predictions[JOINTS_LIST[i]]['all'] += 1

        if np.max(gt[i, :, :]) == 0:  # If joint was not specified in ground truth file

            if np.max(pred[i, :, :]) == 0:
                predictions['total']['correct'] += 1
                predictions[JOINTS_LIST[i]]['correct'] += 1

        else:
            # Returns the indices of elements in an array where the given condition is satisfied.
            p_x, p_y = np.where(pred[i, :, :] == np.max(pred[i, :, :]))
            gt_x, gt_y = np.where(gt[i, :, :] == np.max(gt[i, :, :]))
            # distance between two points in 2d space
            dis = np.sqrt(pow((p_x[0] - gt_x[0]), 2) + pow((p_y[0] - gt_y[0]), 2))
            if dis < (a * threshold):
                predictions['total']['correct'] += 1
                predictions[JOINTS_LIST[i]]['correct'] += 1

    return predictions


def compute_metric(pred_maps, gt_maps, maxbbox_list, temporal):
    predictions = get_predictions_dict()

    for s in range(gt_maps.shape[0]):  # For each seq in batch
        for t in range(temporal):  # For each frame
            gt = gt_maps[s, t, :, :, :]
            pred = np.array(pred_maps[t][s, :, :, :].cpu().data)
            maxbbox = maxbbox_list[s, t]
            predictions = pck(pred, gt, maxbbox, predictions)

    return predictions


def get_acc(predictions):
    acc = {}
    for key in predictions:
        acc[key] = (predictions[key]['correct'] / predictions[key]['all'])
    return acc


def plot(loss_list, metric_list, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (22, 8))
    fig.subplots_adjust(wspace = .2)
    plotLoss(ax1, loss_list, title)
    plotMetric(ax2, metric_list, title)
    plt.show()


def plotLoss(ax, loss_list, title):
    ax.plot(loss_list[:, 0], label = "Train loss")
    ax.plot(loss_list[:, 1], label = "Validation loss")
    ax.set_title("Loss Curves - " + title, fontsize = 12)
    ax.set_ylabel("Loss", fontsize = 10)
    ax.set_xlabel("Epoch", fontsize = 10)
    ax.legend(prop = {'size': 10})


def plotMetric(ax, metric_list, title):
    head_acc = [x['head'] for x in metric_list]
    sho_acc = [x['sho'] for x in metric_list]
    elb_acc = [x['elb'] for x in metric_list]
    wri_acc = [x['wri'] for x in metric_list]
    hip_acc = [x['hip'] for x in metric_list]
    knee_acc = [x['knee'] for x in metric_list]
    ank_acc = [x['ank'] for x in metric_list]
    total_acc = [x['total'] for x in metric_list]

    ax.plot(total_acc[:], label = "Total")
    ax.plot(head_acc[:], label = "Head")
    ax.plot(sho_acc[:], label = "Shoulders")
    ax.plot(elb_acc[:], label = "Elbows")
    ax.plot(wri_acc[:], label = "Wrists")
    ax.plot(hip_acc[:], label = "Hips")
    ax.plot(knee_acc[:], label = "knees")
    ax.plot(ank_acc[:], label = "Ankles")

    ax.set_title("PCK accuracy - " + title, fontsize = 12)
    ax.set_ylabel("Score", fontsize = 10)
    ax.set_xlabel("Epoch", fontsize = 10)
    ax.legend(prop = {'size': 10})
