"""Implementation of prototypical networks for Omniglot."""

import argparse
import os

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F  # pylint: disable=unused-import
from torch.utils import tensorboard

import omniglot
import util  # pylint: disable=unused-import
import neptune.new as neptune
from tqdm import tqdm

NUM_INPUT_CHANNELS = 1
NUM_HIDDEN_CHANNELS = 64
KERNEL_SIZE = 3
NUM_CONV_LAYERS = 4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SUMMARY_INTERVAL = 10
SAVE_INTERVAL = 500
PRINT_INTERVAL = 10
VAL_INTERVAL = PRINT_INTERVAL * 5
NUM_TEST_TASKS = 600
print('Using device: ',DEVICE)

class ProtoNetNetwork(nn.Module):
    """Container for ProtoNet weights and image-to-latent computation."""

    def __init__(self):
        """Inits ProtoNetNetwork.

        The network consists of four convolutional blocks, each comprising a
        convolution layer, a batch normalization layer, ReLU activation, and 2x2
        max pooling for downsampling. There is an additional flattening
        operation at the end.

        Note that unlike conventional use, batch normalization is always done
        with batch statistics, regardless of whether we are training or
        evaluating. This technically makes meta-learning transductive, as
        opposed to inductive.
        """
        super().__init__()
        layers = []
        in_channels = NUM_INPUT_CHANNELS
        for _ in range(NUM_CONV_LAYERS):
            layers.append(
                nn.Conv2d(
                    in_channels,
                    NUM_HIDDEN_CHANNELS,
                    (KERNEL_SIZE, KERNEL_SIZE),
                    padding='same'
                )
            )
            layers.append(nn.BatchNorm2d(NUM_HIDDEN_CHANNELS))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            in_channels = NUM_HIDDEN_CHANNELS
        layers.append(nn.Flatten())
        self._layers = nn.Sequential(*layers)
        self.to(DEVICE)

    def forward(self, images):
        """Computes the latent representation of a batch of images.

        Args:
            images (Tensor): batch of Omniglot images
                shape (num_images, channels, height, width)

        Returns:
            a Tensor containing a batch of latent representations
                shape (num_images, latents)
        """
        return self._layers(images)


class ProtoNet:
    """Trains and assesses a prototypical network."""

    def __init__(self, learning_rate, log_dir, tensorboard_writer=None, neptune_run=None):
        """Inits ProtoNet.

        Args:
            learning_rate (float): learning rate for the Adam optimizer
            log_dir (str): path to logging directory
        """

        self._network = ProtoNetNetwork()
        self._optimizer = torch.optim.Adam(
            self._network.parameters(),
            lr=learning_rate
        )
        self._log_dir = log_dir

        self.writer = tensorboard_writer
        self.neptune_run = neptune_run

        self._start_train_step = 0

    def _step(self, task_batch):
        """Computes ProtoNet mean loss (and accuracy) on a batch of tasks.

        Args:
            task_batch (tuple[Tensor, Tensor, Tensor, Tensor]):
                batch of tasks from an Omniglot DataLoader

        Returns:
            a Tensor containing mean ProtoNet loss over the batch
                shape ()
            mean support set accuracy over the batch as a float
            mean query set accuracy over the batch as a float
        """
        loss_batch = []
        accuracy_support_batch = []
        accuracy_query_batch = []
        for task in task_batch:
            images_support, labels_support, images_query, labels_query = task
            images_support = images_support.to(DEVICE)
            labels_support = labels_support.to(DEVICE)
            images_query = images_query.to(DEVICE)
            labels_query = labels_query.to(DEVICE)
            
            # batch sizes and number of classes
            batch_size_support = images_support.shape[0]
            batch_size_query = images_query.shape[0]
            n_classes = torch.max(labels_support)+1
            n_shots = torch.div(batch_size_support,n_classes, rounding_mode='trunc')

            # Support and query encodings
            encodings_support = self._network(images_support).reshape(batch_size_support, 1, -1)
            encodings_query = self._network(images_query).reshape(batch_size_query, 1, -1)
            # encodings_support = images_support.reshape(batch_size_support, 1, 784)
            # encodings_query = images_query.reshape(batch_size_query, 1, 784)
            enc_size = encodings_support.shape[-1]

            # Compute Prototypes
            indices = torch.argsort(labels_support).reshape(n_classes,n_shots)
            im = encodings_support.reshape(batch_size_support,enc_size)[indices]
            prototypes =  torch.mean(im,dim=1).reshape(1,n_classes,enc_size)

            # Compute distances to prototypes
            distances_query = torch.linalg.norm(prototypes-encodings_query, dim=-1)  # shape = (batch_size_query, n_classes)
            distances_support = torch.linalg.norm(prototypes-encodings_support, dim=-1)  # shape = (batch_size_support, n_classes)

            # Compute logits
            logits_query = -1*distances_query**2
            logits_support = -1*distances_support**2

            # Compute loss and accuracies
            single_batch_loss = F.cross_entropy(logits_query, labels_query)
            single_batch_support_acc = util.score(logits_support.detach(), labels_support)
            single_batch_query_acc = util.score(logits_query.detach(), labels_query)

            # Populate values of loss and accuracies
            loss_batch.append(single_batch_loss)
            accuracy_support_batch.append(single_batch_support_acc)
            accuracy_query_batch.append(single_batch_query_acc)

        return (
            torch.mean(torch.stack(loss_batch)),
            np.mean(accuracy_support_batch),
            np.mean(accuracy_query_batch)
        )

    def train(self, dataloader_train, dataloader_val):
        """Train the ProtoNet.

        Consumes dataloader_train to optimize weights of ProtoNetNetwork
        while periodically validating on dataloader_val, logging metrics, and
        saving checkpoints.

        Args:
            dataloader_train (DataLoader): loader for train tasks
            dataloader_val (DataLoader): loader for validation tasks
        """
        print(f'Starting training at iteration {self._start_train_step}.')
        for i_step, task_batch in tqdm(enumerate(
                dataloader_train,
                start=self._start_train_step
        ), position=0, leave=True):
            self._optimizer.zero_grad()
            loss, accuracy_support, accuracy_query = self._step(task_batch)
            loss.backward()
            self._optimizer.step()

            if i_step % PRINT_INTERVAL == 0:
                if self.writer:
                    self.writer.add_scalar('loss/train', loss.item(), i_step)
                    self.writer.add_scalar(
                        'train_accuracy/support',
                        accuracy_support.item(),
                        i_step
                    )
                    self.writer.add_scalar(
                        'train_accuracy/query',
                        accuracy_query.item(),
                        i_step
                    )
                if self.neptune_run:
                    self.neptune_run['train/loss_support'].log(loss.item(), step=i_step)
                    self.neptune_run['train/acc_support'].log(accuracy_support.item(), step=i_step)
                    self.neptune_run['train/acc_query'].log(accuracy_query.item(), step=i_step)

            if i_step % VAL_INTERVAL == 0:
                with torch.no_grad():
                    losses, accuracies_support, accuracies_query = [], [], []
                    for val_task_batch in dataloader_val:
                        loss, accuracy_support, accuracy_query = (
                            self._step(val_task_batch)
                        )
                        losses.append(loss.item())
                        accuracies_support.append(accuracy_support)
                        accuracies_query.append(accuracy_query)
                    loss = np.mean(losses)
                    accuracy_support = np.mean(accuracies_support)
                    accuracy_query = np.mean(accuracies_query)
                if self.writer:
                    self.writer.add_scalar('loss/val', loss, i_step)
                    self.writer.add_scalar(
                        'val_accuracy/support',
                        accuracy_support,
                        i_step
                    )
                    self.writer.add_scalar(
                        'val_accuracy/query',
                        accuracy_query,
                        i_step
                    )
                if self.neptune_run:
                    self.neptune_run['val/loss_support'].log(loss.item(), step=i_step)
                    self.neptune_run['val/acc_support'].log(accuracy_support.item(), step=i_step)
                    self.neptune_run['val/acc_query'].log(accuracy_query.item(), step=i_step)

            if i_step % SAVE_INTERVAL == 0:
                self._save(i_step)
      
        self._save('final save')

    def test(self, dataloader_test):
        """Evaluate the ProtoNet on test tasks.

        Args:
            dataloader_test (DataLoader): loader for test tasks
        """
        accuracies = []
        for task_batch in dataloader_test:
            accuracies.append(self._step(task_batch)[2])
        mean = np.mean(accuracies)
        std = np.std(accuracies)
        mean_95_confidence_interval = 1.96 * std / np.sqrt(NUM_TEST_TASKS)
        print(
            f'Accuracy over {NUM_TEST_TASKS} test tasks: '
            f'mean {mean:.3f}, '
            f'95% confidence interval {mean_95_confidence_interval:.3f}'
        )
        if self.neptune_run:
            self.neptune_run['test_acc'].log(mean)
            self.neptune_run['test_acc_std'].log(std)
            self.neptune_run['test_acc_95_conf_int'].log(mean_95_confidence_interval)

    def load(self, checkpoint_step):
        """Loads a checkpoint.

        Args:
            checkpoint_step (int): iteration of checkpoint to load

        Raises:
            ValueError: if checkpoint for checkpoint_step is not found
        """
        target_path = (
            f'{os.path.join(self._log_dir, "state")}'
            f'{checkpoint_step}.pt'
        )
        if os.path.isfile(target_path):
            state = torch.load(target_path)
            self._network.load_state_dict(state['network_state_dict'])
            self._optimizer.load_state_dict(state['optimizer_state_dict'])
            self._start_train_step = checkpoint_step + 1
            print(f'Loaded checkpoint iteration {checkpoint_step}.')
        else:
            raise ValueError(
                f'No checkpoint for iteration {checkpoint_step} found.'
            )

    def _save(self, checkpoint_step):
        """Saves network and optimizer state_dicts as a checkpoint.

        Args:
            checkpoint_step (int): iteration to label checkpoint with
        """
        torch.save(
            dict(network_state_dict=self._network.state_dict(),
                 optimizer_state_dict=self._optimizer.state_dict()),
            f'{os.path.join(self._log_dir, "state")}{checkpoint_step}.pt'
        )
        print('Saved checkpoint.')


def main(args):
    tensorboard_writer = None
    neptune_run = None
    log_dir = args.log_dir
    if log_dir is None:
        log_dir = f'./logs/protonet/omniglot.way:{args.num_way}.support:{args.num_support}.query:{args.num_query}.lr:{args.learning_rate}.batch_size:{args.batch_size}'  # pylint: disable=line-too-long
    print(f'log_dir: {log_dir}')
    os.makedirs(log_dir, exist_ok=True)
    if args.tensorboard:
        tensorboard_writer = tensorboard.SummaryWriter(log_dir=log_dir)
    
    if args.neptune:
        api_token = ''
        neptune_run = neptune.init(project='kareem-elsawah/meta-learning', api_token=api_token, source_files=["*.py"])
    
    protonet = ProtoNet(args.learning_rate, log_dir, tensorboard_writer=tensorboard_writer, neptune_run=neptune_run)
    
    if args.checkpoint_step > -1:
        protonet.load(args.checkpoint_step)
    else:
        print('Checkpoint loading skipped.')

    if not args.test:
        num_training_tasks = args.batch_size * (args.num_train_iterations -
                                                args.checkpoint_step - 1)
        print(
            f'Training on tasks with composition '
            f'num_way={args.num_way}, '
            f'num_support={args.num_support}, '
            f'num_query={args.num_query}'
        )
        dataloader_train = omniglot.get_omniglot_dataloader(
            'train',
            args.batch_size,
            args.num_way,
            args.num_support,
            args.num_query,
            num_training_tasks
        )
        dataloader_val = omniglot.get_omniglot_dataloader(
            'val',
            args.batch_size,
            args.num_way,
            args.num_support,
            args.num_query,
            args.batch_size * 4
        )
        protonet.train(
            dataloader_train,
            dataloader_val
        )

        dataloader_test = omniglot.get_omniglot_dataloader(
            'test',
            1,
            args.num_way,
            args.num_support,
            args.num_query,
            NUM_TEST_TASKS
        )
        protonet.test(dataloader_test)
    else:
        print(
            f'Testing on tasks with composition '
            f'num_way={args.num_way}, '
            f'num_support={args.num_support}, '
            f'num_query={args.num_query}'
        )
        dataloader_test = omniglot.get_omniglot_dataloader(
            'test',
            1,
            args.num_way,
            args.num_support,
            args.num_query,
            NUM_TEST_TASKS
        )
        protonet.test(dataloader_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train a ProtoNet!')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='directory to save to or load from')
    parser.add_argument('--num_way', type=int, default=5,
                        help='number of classes in a task')
    parser.add_argument('--num_support', type=int, default=1,
                        help='number of support examples per class in a task')
    parser.add_argument('--num_query', type=int, default=15,
                        help='number of query examples per class in a task')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate for the network')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='number of tasks per outer-loop update')
    parser.add_argument('--num_train_iterations', type=int, default=3000,
                        help='number of outer-loop updates to train for')
    parser.add_argument('--test', default=False, action='store_true',
                        help='train-and-test or only-test')
    parser.add_argument('--checkpoint_step', type=int, default=-1,
                        help=('checkpoint iteration to load for resuming '
                              'training, or for evaluation (-1 is ignored)'))
    parser.add_argument('--tensorboard', default=False, action='store_true',
                        help='Log using tensorboard')
    parser.add_argument('--neptune', default=False, action='store_true',
                        help='Log using Neptune')

    main_args = parser.parse_args()
    main(main_args)
