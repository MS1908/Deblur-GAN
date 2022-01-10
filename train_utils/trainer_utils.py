import logging
import tqdm
import torch.optim as optim
from utils.metrics_logger import MetricLogger
from utils.metrics import MetricPipeline
from train_utils.adversarial_trainer_utils import *
from utils.schedulers import WarmRestart, LinearDecay
from utils.losses import loss_factory
from model.network_factory import gan_factory


class Trainer:

    def __init__(self, config, train, val):
        self.config = config
        self.train_dataset = train
        self.val_dataset = val
        self.adv_lambda = config['model']['adv_lambda']
        self.metric_logger = MetricLogger('logs/' + config['experiment_desc'])
        self.warmup_epochs = config['warmup_num']

    def train(self):
        self._init_params()
        for epoch in range(0, self.config['num_epochs']):
            if (epoch == self.warmup_epochs) and not (self.warmup_epochs == 0):
                self.netG.module.unfreeze()
                self.optimizer_G = self._get_optim(self.netG.parameters())
                self.scheduler_G = self._get_scheduler(self.optimizer_G)
            self._run_epoch(epoch)
            self._validate(epoch)
            self.scheduler_G.step()
            self.scheduler_D.step()

            if self.metric_logger.update_best_model():
                torch.save({
                    'model': self.netG.state_dict()
                }, 'weights/best_{}.h5'.format(self.config['experiment_desc']))

            torch.save({
                'model': self.netG.state_dict()
            }, 'weights/last_{}.h5'.format(self.config['experiment_desc']))

            print(self.metric_logger.loss_message())
            logging.debug("Experiment Name: %s, Epoch: %d, Loss: %s" % (self.config['experiment_desc'],
                                                                        epoch, self.metric_logger.loss_message()))

    def _run_epoch(self, epoch):
        self.metric_logger.clear()
        for param_group in self.optimizer_G.param_groups:
            lr = param_group['lr']

        epoch_size = self.config.get('train_batches_per_epoch') or len(self.train_dataset)
        tq = tqdm.tqdm(self.train_dataset, total=epoch_size)
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
        i = 0
        for data in tq:
            inputs, targets = MetricPipeline().get_input(data)
            outputs = self.netG(inputs)
            loss_D = self._update_d(outputs, targets)
            self.optimizer_G.zero_grad()
            loss_content = self.criterionG(outputs, targets)
            loss_adv = self.adv_trainer.loss_g(outputs, targets)
            loss_G = loss_content + self.adv_lambda * loss_adv
            loss_G.backward()
            self.optimizer_G.step()
            self.metric_logger.add_losses(loss_G.item(), loss_content.item(), loss_D)
            curr_psnr, curr_ssim, img_for_vis = MetricPipeline().get_images_and_metrics(inputs, outputs, targets)
            self.metric_logger.add_metrics(curr_psnr, curr_ssim)
            tq.set_postfix(loss=self.metric_logger.loss_message())
            if not i:
                self.metric_logger.add_image(img_for_vis, tag='train')
            i += 1
            if i > epoch_size:
                break
        tq.close()
        self.metric_logger.write_to_tensorboard(epoch)

    def _validate(self, epoch):
        self.metric_logger.clear()
        epoch_size = self.config.get('val_batches_per_epoch') or len(self.val_dataset)
        tq = tqdm.tqdm(self.val_dataset, total=epoch_size)
        tq.set_description('Validation')
        i = 0
        for data in tq:
            inputs, targets = MetricPipeline().get_input(data)
            with torch.no_grad():
                outputs = self.netG(inputs)
                loss_content = self.criterionG(outputs, targets)
                loss_adv = self.adv_trainer.loss_g(outputs, targets)
            loss_G = loss_content + self.adv_lambda * loss_adv
            self.metric_logger.add_losses(loss_G.item(), loss_content.item())
            curr_psnr, curr_ssim, img_for_vis = MetricPipeline().get_images_and_metrics(inputs, outputs, targets)
            self.metric_logger.add_metrics(curr_psnr, curr_ssim)
            if not i:
                self.metric_logger.add_image(img_for_vis, tag='val')
            i += 1
            if i > epoch_size:
                break
        tq.close()
        self.metric_logger.write_to_tensorboard(epoch, validation=True)

    def _update_d(self, outputs, targets):
        if self.config['model']['d_name'] == 'no_gan':
            return 0
        self.optimizer_D.zero_grad()
        loss_D = self.adv_lambda * self.adv_trainer.loss_d(outputs, targets)
        loss_D.backward(retain_graph=True)
        self.optimizer_D.step()
        return loss_D.item()

    def _get_optim(self, params):
        if self.config['optimizer']['name'] == 'adam':
            optimizer = optim.Adam(params, lr=self.config['optimizer']['lr'])
        elif self.config['optimizer']['name'] == 'sgd':
            optimizer = optim.SGD(params, lr=self.config['optimizer']['lr'])
        elif self.config['optimizer']['name'] == 'adadelta':
            optimizer = optim.Adadelta(params, lr=self.config['optimizer']['lr'])
        else:
            raise ValueError("Optimizer [%s] not recognized." % self.config['optimizer']['name'])
        return optimizer

    def _get_scheduler(self, optimizer):
        if self.config['scheduler']['name'] == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                             mode='min',
                                                             patience=self.config['scheduler']['patience'],
                                                             factor=self.config['scheduler']['factor'],
                                                             min_lr=self.config['scheduler']['min_lr'])
        elif self.config['optimizer']['name'] == 'sgdr':
            scheduler = WarmRestart(optimizer)
        elif self.config['scheduler']['name'] == 'linear':
            scheduler = LinearDecay(optimizer,
                                    min_lr=self.config['scheduler']['min_lr'],
                                    num_epochs=self.config['num_epochs'],
                                    start_epoch=self.config['scheduler']['start_epoch'])
        else:
            raise ValueError("Scheduler [%s] not recognized." % self.config['scheduler']['name'])
        return scheduler

    @staticmethod
    def _get_adversarial_trainer(d_name, net_d, criterion_d):
        if d_name == 'no_gan':
            return GANFactory.create_model('NoGAN')
        elif d_name == 'patch_gan' or d_name == 'multi_scale':
            return GANFactory.create_model('SingleGAN', net_d, criterion_d)
        elif d_name == 'double_gan':
            return GANFactory.create_model('DoubleGAN', net_d, criterion_d)
        else:
            raise ValueError("Discriminator Network [%s] not recognized." % d_name)

    def _init_params(self):
        self.criterionG, criterionD = loss_factory(self.config['model']['content_loss'],
                                                   self.config['model']['disc_loss'])
        self.netG, netD = gan_factory(self.config['model'])
        self.netG.cuda()
        self.adv_trainer = self._get_adversarial_trainer(self.config['model']['d_name'], netD, criterionD)
        self.optimizer_G = self._get_optim(filter(lambda p: p.requires_grad, self.netG.parameters()))
        self.optimizer_D = self._get_optim(self.adv_trainer.get_params())
        self.scheduler_G = self._get_scheduler(self.optimizer_G)
        self.scheduler_D = self._get_scheduler(self.optimizer_D)
