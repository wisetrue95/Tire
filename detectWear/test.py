import argparse
import torch
import utils.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
import torch.nn as nn
import matplotlib.pyplot as plt


def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    val_data_loader = config.init_obj('val_data_loader', module_data)
    attention_val_data_loader = config.init_obj('attention_val_data_loader', module_data)

    # build model architecture
    model = module_arch.resnet101()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs+1, 3)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))
    correct = 0
    with torch.no_grad():
        for batch_idx, (val, attention) in enumerate(zip(val_data_loader, attention_val_data_loader)):

            data, target = val[0].to(device), val[1].to(device)
            attention = attention[0].to(device)
            output = model(data, attention)

            #
            # save sample images, or do something with output here
            #
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            # print(len(test_loader.dataset))

            classtire = ['danger', 'safety', 'warning']


            for j in range(val_data_loader.batch_size):
                if (pred[j] == 0):  # visualize danger class
                    plt.title(' GT:{} - Pred: {} '.format(classtire[target[j]], classtire[pred[j]]))
                    plt.imshow(data[j].cpu().data.squeeze().permute(1, 2, 0))
                    plt.show()

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size

    n_samples = len(val_data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='config.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
