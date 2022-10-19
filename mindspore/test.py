import mindspore as ms
import mindspore.nn as nn
import argparse
import numpy as np

from configs import get_config
from build import build_model
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

def main(config):

    download_train = ImageNet(path="./data/imagenet", split="train")
    download_test = ImageNet(path="./data/imagenet", split="infer")
    dataset_train = download_train.run()
    dataset_test = download_test.run()

    model = build_model(config)
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    net_opt = nn.Momentum(model.trainable_params(), learning_rate=0.01, momentum=0.9)

    network = ms.Model(model, loss_fn=net_loss, optimizer=net_opt)

    #inference
    ds_test = dataset_test.create_dict_iterator()
    for data in ds_test:
        output = network.predict(data["image"])
        print(output)


if __name__ == '__main__':
    _, config = parse_option()
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="GPU")

    main(config)

