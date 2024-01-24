# @hairymax

import os
import wandb
import torch
from torch import optim
from torch.nn import CrossEntropyLoss, MSELoss
from tqdm import tqdm
from tensorboardX import SummaryWriter

from src.NN import MultiFTNet
from src.dataset_loader import get_train_valid_loader, get_train_loader, get_valid_loader
from datetime import datetime


class TrainMain:
    def __init__(self, conf, conf_val):
        self.conf = conf
        self.conf_val = conf_val
        self.step = 0
        self.val_step = 0
        self.start_epoch = 0
        # self.train_loader, self.valid_loader = get_train_valid_loader(self.conf)
        self.train_loader = get_train_loader(self.conf)
        self.valid_loader = get_valid_loader(self.conf_val)
        
        self.board_train_every = len(self.train_loader) // conf.board_loss_per_epoch
        self.board_valid_every = len(self.valid_loader) // conf.board_loss_per_epoch
        
        self.best_val_acc = 0

    def train_model(self):
        wandb.init(project='minivision-fas-modified', config=self.conf)
        self._init_model_param()
        self._train_stage()

    def _init_model_param(self):
        self.cls_criterion = CrossEntropyLoss()
        self.ft_criterion = MSELoss()
        self.model = self._define_network()
        self.optimizer = optim.SGD(self.model.module.parameters(),
                                   lr=self.conf.lr,
                                   weight_decay=5e-4,
                                   momentum=self.conf.momentum)

        self.schedule_lr = optim.lr_scheduler.MultiStepLR(
            self.optimizer, self.conf.milestones, self.conf.gamma, - 1)

        print("lr: ", self.conf.lr)
        print("epochs: ", self.conf.epochs)
        print("milestones: ", self.conf.milestones)

    def _train_stage(self):
        run_loss = 0.
        run_acc = 0.
        run_loss_cls = 0.
        run_loss_ft = 0.

        is_first = True

        print('Board train loss every {} steps'.format(self.board_train_every))

        for e in range(self.start_epoch, self.conf.epochs):
            if is_first:
                self.writer = SummaryWriter(self.conf.log_path)
                is_first = False

            print('Epoch {} started. lr: {}'.format(e, self.schedule_lr.get_last_lr()))
            self.model.train()

            for sample, ft_sample, labels in tqdm(iter(self.train_loader)):
                imgs = [sample, ft_sample]

                loss, acc, loss_cls, loss_ft = self._train_batch_data(imgs, labels)
                run_loss += loss
                run_acc += acc
                run_loss_cls += loss_cls
                run_loss_ft += loss_ft

                self.step += 1

                if self.step % self.board_train_every == 0 and self.step != 0:
                    avg_loss = run_loss / self.board_train_every
                    avg_acc = run_acc / self.board_train_every
                    avg_loss_cls = run_loss_cls / self.board_train_every
                    avg_loss_ft = run_loss_ft / self.board_train_every

                    self.writer.add_scalar('Loss/train', avg_loss, self.step)
                    self.writer.add_scalar('Acc/train', avg_acc, self.step)
                    self.writer.add_scalar('Loss_cls/train', avg_loss_cls, self.step)
                    self.writer.add_scalar('Loss_ft/train', avg_loss_ft, self.step)

                    # Log metrics to wandb
                    wandb.log({"Loss/train": avg_loss,
                            "Acc/train": avg_acc,
                            "Loss_cls/train": avg_loss_cls,
                            "Loss_ft/train": avg_loss_ft,
                            "Learning_rate": self.optimizer.param_groups[0]['lr']},
                            step=self.step)

                    run_loss = 0.
                    run_acc = 0.
                    run_loss_cls = 0.
                    run_loss_ft = 0.

            self.schedule_lr.step()
            
            # Validation
            self.model.eval()
            val_acc_total = 0.0
            val_loss_cls_total = 0.0
            print('Validation on {} batches.'.format(len(self.valid_loader)))

            for sample, labels in tqdm(iter(self.valid_loader)):
                with torch.no_grad():
                    acc, loss_cls = self._valid_batch_data(sample, labels)
                val_acc_total += acc
                val_loss_cls_total += loss_cls

            avg_val_acc = val_acc_total / len(self.valid_loader)
            avg_val_loss_cls = val_loss_cls_total / len(self.valid_loader)

            # Log average validation accuracy and loss to wandb and TensorBoard
            wandb.log({"Avg_Acc/valid": avg_val_acc, "Avg_Loss_cls/valid": avg_val_loss_cls}, step=self.step)
            self.writer.add_scalar('Acc/valid', avg_val_acc, self.step)
            self.writer.add_scalar('Loss_cls/valid', avg_val_loss_cls, self.step)

            # Check if this is the best model
            if avg_val_acc > self.best_val_acc:
                self.best_val_acc = avg_val_acc
                self._save_best_model(e)
                wandb.run.summary["best_val_acc"] = avg_val_acc
                wandb.run.summary["best_epoch"] = e

            self._save_state('epoch-{}'.format(e))

        self.writer.close()

    def _train_batch_data(self, imgs, labels):
        self.optimizer.zero_grad()
        labels = labels.to(self.conf.device)
        embeddings, feature_map = self.model.forward(imgs[0].to(self.conf.device))

        loss_cls = self.cls_criterion(embeddings, labels)
        loss_fea = self.ft_criterion(feature_map, imgs[1].to(self.conf.device))

        loss = 0.5*loss_cls + 0.5*loss_fea
        acc = self._get_accuracy(embeddings, labels)[0]
        loss.backward()
        self.optimizer.step()
        return loss.item(), acc, loss_cls.item(), loss_fea.item()


    def _valid_batch_data(self, img, labels):
        labels = labels.to(self.conf.device)
        embeddings = self.model.forward(img.to(self.conf.device))

        loss_cls = self.cls_criterion(embeddings, labels)
        acc = self._get_accuracy(embeddings, labels)[0]

        return acc, loss_cls.item()

    def _define_network(self):
        param = {
            'num_classes': self.conf.num_classes,
            'img_channel': self.conf.input_channel,
            'embedding_size': self.conf.embedding_size,
            'conv6_kernel': self.conf.kernel_size}

        model = MultiFTNet(**param).to(self.conf.device)
        model = torch.nn.DataParallel(model)#self.conf.devices)
        model.to(self.conf.device)
        return model

    def _get_accuracy(self, output, target, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        ret = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(dim=0, keepdim=True)
            ret.append(correct_k.mul_(1. / batch_size))
        return ret

    def _save_state(self, stage):
        save_path = self.conf.model_path
        job_name = self.conf.job_name
        time_stamp = (str(datetime.now())[:-10]).replace(' ', '-').replace(':', '-')
        torch.save(self.model.state_dict(), save_path + '/' +
                   ('{}_{}_{}.pth'.format(time_stamp, job_name, stage)))
        
    def _save_best_model(self, epoch):
        save_path = self.conf.model_path
        job_name = self.conf.job_name
        best_model_path = os.path.join(save_path, f'{job_name}_best_model.pth')
        torch.save(self.model.state_dict(), best_model_path)
        print(f"Best model (epoch {epoch}, val accuracy: {self.best_val_acc}) saved at {best_model_path}")
        wandb.log({"best_val_acc": self.best_val_acc}, step=epoch)
