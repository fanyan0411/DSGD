import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import IncrementalNet
from utils.inc_net import CosineIncrementalNet
from utils.toolkit import target2onehot, tensor2numpy
from utils.data_sampler import TwoStreamBatchSampler
import os, json
from time import time
from utils.distill_loss import matching_loss
EPSILON = 1e-8

init_epoch = 200
init_lr = 0.1
init_milestones = [60, 120, 170]
init_lr_decay = 0.1
init_weight_decay = 0.0005


epochs = 170
lrate = 0.1
milestones = [80, 120]
lrate_decay = 0.1
batch_size = 128
weight_decay = 2e-4
num_workers = 8
T = 2


class iCaRL(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, False)
        self.threshold = 0.95
        self.batch_size = batch_size
        
        # Task setting
        self.labeled_batch_size = int(self.batch_size/2) 
        self.label_num = args["label_num"]
        self.dataset_name = args["dataset"]
        self.init_cls = args["init_cls"]
        self.full_supervise = args["full_supervise"]
        self.total_exp = int(args["label_size"]/self.init_cls)

        # Semi-supervised loss
        self.usp_weight = args["usp_weight"]
        self.uce_loss = torch.nn.CrossEntropyLoss(reduction='none') #FY
        self.insert_pse_progressive = args['insert_pse_progressive']
        self.insert_pse = args['insert_pse']
        self.pse_weight = args["pse_weight"] # weight of previous pseudo labels of weak augmentation view in uloss

        # Sub-graph distillation loss
        self.rw_alpha = args["rw_alpha"]
        self.match_weight = args["match_weight"]
        self.rw_T = args["rw_T"]
        self.gamma_ml = args["gamma_ml"]

        # Origin knowledge distillation
        self.kd_onlylabel = args['kd_onlylabel']
        

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))

        if not self.args["resume"]:
            if not os.path.exists('./checkpoints'):
                os.makedirs('./checkpoints') #_initwsplab
            checkpoint_name = "./checkpoints/{}_{}_{}_{}_{}_{}_{}.pkl".format(self.args["dataset"], self.args["model_name"],self.args["init_cls"],self.args["increment"],init_epoch,self.args["label_num"],self._cur_task)
            if self._cur_task == 0 or self.args["save_all_resume"]:
                if not os.path.exists(checkpoint_name):
                    self.save_checkpoint("./checkpoints/{}_{}_{}_{}_{}_{}".format(self.args["dataset"], self.args["model_name"],self.args["init_cls"],self.args["increment"],init_epoch,self.args["label_num"]))
            

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        train_dataset, idxes = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),
        )

        ## Annotation index 
        labeled_idxs_onehot = train_dataset.lab_index_task # 
        
        labeled_idxs = np.where(labeled_idxs_onehot>0)[0]
        unlabeled_idxs = np.where(labeled_idxs_onehot==0)[0]

        batch_sampler = TwoStreamBatchSampler(
                unlabeled_idxs, labeled_idxs, self.batch_size, self.labeled_batch_size)

        
        self.train_loader = DataLoader(
            train_dataset, batch_sampler=batch_sampler, num_workers=num_workers, pin_memory=True
        ) 

        test_dataset, _ = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class[0], self.samples_per_class[1])
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        if self.args["resume"] and self._cur_task == 0:
            checkpoint_name = "./checkpoints/{}_{}_{}_{}_200_{}_{}.pkl".format(self.args["dataset"], self.args["model_name"],self.args["init_cls"],self.args["increment"],self.args["label_num"],self._cur_task)
            if os.path.isfile(checkpoint_name):
                self._network.module.load_state_dict(torch.load(checkpoint_name)["model_state_dict"])
            else:
                print(checkpoint_name, "is none")
                self.args["resume"] = False
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)

        if self._cur_task == 0:
            optimizer = optim.SGD(
                self._network.parameters(),
                momentum=0.9,
                lr=init_lr,
                weight_decay=init_weight_decay,
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=init_milestones, gamma=init_lr_decay
            )
            if not self.args["resume"]:
                self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            optimizer = optim.SGD(
                self._network.parameters(),
                lr=lrate,
                momentum=0.9,
                weight_decay=weight_decay,
            )  # 1e-5
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=milestones, gamma=lrate_decay
            )
            self._update_representation(train_loader, test_loader, optimizer, scheduler)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(init_epoch))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses, losses_clf, ulosses = 0.0, 0.0, 0.0
            correct, total = 0, 0
            for i, (_, inputs, inputs_s, targets, pse_targets, lab_index_task) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                inputs_s = inputs_s.to(self._device)
                lab_index_task = lab_index_task.to(self._device)
                logits = self._network(inputs)["logits"]

                if not self.full_supervise: 
                    targets = targets * lab_index_task + torch.ones_like(targets) * -100 * (1-lab_index_task) # 20230608
                

                loss_clf = F.cross_entropy(logits, targets)
                with torch.no_grad():
                    logits_w = F.softmax(logits.clone(), 1)
                    wprobs, wpslab = logits_w.max(1)
                    wpslab = (targets>-100) * targets + wpslab * (targets==-100)   #20240417
                mask = wprobs.ge(self.threshold).float()
                logits_s = self._network(inputs_s)["logits"]
                uloss  = torch.mean(mask * self.uce_loss(logits_s, wpslab))  # unsupervised loss, 表示无标记样本的损失
                uloss *= self.usp_weight
                loss = loss_clf + uloss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                losses_clf += loss_clf.item()
                ulosses += uloss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 5 == 0:
                test_acc,_ = self._compute_accuracy(self._network, test_loader)   # 20230613 FY
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, uloss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    ulosses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    losses / len(train_loader),
                    train_acc,
                )

            prog_bar.set_description(info)

            logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):

        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            losses_clf = 0.0
            losses_kd = 0.0
            ulosses = 0.0
            correct, total = 0, 0
            total_supervise = 0
            losses_match_logits = 0
            for bi, (_, inputs, inputs_s, targets, pse_targets, lab_index_task) in enumerate(train_loader):
                
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                inputs_s = inputs_s.to(self._device)
                logits = self._network(inputs)["logits"]
                lab_index_task = lab_index_task.to(self._device) 
                pse_targets = pse_targets.to(self._device)

                
                if not self.full_supervise == True:
                    targets = targets * lab_index_task + torch.ones_like(targets) * -100 * (1-lab_index_task) # 20230608
                targets_withpse = targets * lab_index_task + pse_targets * (1-lab_index_task) # 和上面的相比多了伪标签
                loss_clf = F.cross_entropy(logits, targets)

                if self.kd_onlylabel == True:
                    kd_onlylabel = lab_index_task
                else:
                    kd_onlylabel = None
                loss_kd =  _KD_loss(
                    logits[:, : self._known_classes],
                    self._old_network(inputs)["logits"],
                    T,
                    lab_index_task=kd_onlylabel,
                    )

                with torch.no_grad():
                    wprobs, wpslab = F.softmax(logits.clone(), 1).max(1)
                    wpslab_target = (targets>-100) * targets + wpslab * (targets==-100)
                    wpslab_pse = (targets_withpse>-100) * targets_withpse + wpslab * (targets_withpse==-100)  # 20230608 targets #
                mask = ((wprobs.ge(self.threshold).float() + (targets_withpse>-100)) > 0).float()  #targets_withpse=-100表示新任务的无标记样本和旧任务的无伪标记样本
                logits_s = self._network(inputs_s)["logits"]

                def sigmoid_growth(x):
                    '''
                    L / (1 + np.exp(-k*(x - x0)))
                    # 设置S形生长函数的参数
                    L = 1  # 饱和值
                    k = 1  # 增长率
                    x0 = 5   # 达到饱和值一半的时间或相关变量的值
                    '''
                    return 1 / (1 + np.exp(-1*(x - int(self.total_exp * 0.5))))
                
                if self.insert_pse_progressive:
                    if self.insert_pse == 'threshold':
                        if self._cur_task>4:
                            self.pse_weight = 0.5  # 如果采用逐渐加伪标签的方式，并且采用的是截断函数。
                    if self.insert_pse == 'logitstic':
                        self.pse_weight = sigmoid_growth(self._cur_task)

                uloss  = self.usp_weight * ((1-self.pse_weight) * torch.mean(mask * self.uce_loss(logits_s, wpslab_target)) + self.pse_weight * torch.mean(mask * self.uce_loss(logits_s, wpslab_pse)))
                
                if self.match_weight>0:
                    loss_match = matching_loss(self._old_network(inputs)["logits"], logits, self._known_classes, targets_withpse, targets, rw_alpha=self.rw_alpha, T=self.rw_T, gamma=self.gamma_ml)
                    loss_match_logits = self.match_weight * loss_match['matching_loss_seed']
                else:
                    loss_match_logits = torch.tensor(0)
                
                loss = loss_clf + loss_kd + uloss + loss_match_logits

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                losses_clf += loss_clf.item()
                losses_kd += loss_kd.item()
                ulosses += uloss.item()
                losses_match_logits += loss_match_logits.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if (epoch+1) % 5 == 0 or epoch==0:
                test_acc, cnn_accy = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_kd {:.3f}, Loss_u {:.3f}, Loss_match {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    losses_kd / len(train_loader),
                    ulosses / len(train_loader),
                    losses_match_logits / len(train_loader),
                    train_acc,
                    test_acc,
                )
                logging.info(cnn_accy["grouped"])
                logging.info(info)
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_kd {:.3f}, Loss_u {:.3f}, Loss_m {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    losses_kd / len(train_loader),
                    ulosses / len(train_loader),
                    losses_match_logits / len(train_loader),
                    train_acc
                )
                logging.info(info)
            prog_bar.set_description(info)

        logging.info('training time of each epoch: {:.3f}s'.format((time() - prog_bar.start_t)/(epoch+1)))

            


def _KD_loss(pred, soft, T, kd_index=None):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    if kd_index != None:
        return -1 * torch.einsum('nm,n->nm', torch.mul(soft, pred), kd_index).sum()/kd_index.sum()
    else:
        return -1 * torch.mul(soft, pred).sum() / pred.shape[0]
