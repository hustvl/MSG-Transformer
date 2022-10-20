import argparse
import time

from configs import get_config
from build import build_model
import sys
sys.path.append("..")
from logger import create_logger
from timm_ms import accuracy, AverageMeter

import mindspore as ms
import mindspore.nn as nn
from mindvision.classification.dataset import ImageNet


def parse_option():
    parser = argparse.ArgumentParser('MSG-Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def validate(config, data_loader, model):
    criterion = nn.CrossEntropyLoss()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):

        # compute output
        output = model.predict(images)

        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        loss_meter.update(loss.asnumpy().item(), target.shape[0])
        acc1_meter.update(acc1.asnumpy().item(), target.shape[0])
        acc5_meter.update(acc5.asnumpy().item(), target.shape[0])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if idx % config.PRINT_FREQ == 0:
            logger.info(
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


def main(config):

    download_train = ImageNet(path="./data/imagenet", split="train")
    download_test = ImageNet(path="./data/imagenet", split="infer")
    dataset_train = download_train.run()
    dataset_test = download_test.run()

    model = build_model(config)
    model = ms.Model(model)

    #val
    acc1, acc5, loss = validate(config, dataset_test, model)


if __name__ == '__main__':
    _, config = parse_option()
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="GPU")
    logger = create_logger(output_dir=config.OUTPUT, name=f"{config.MODEL.NAME}")
    main(config)

