import logging, os, json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.data import iCIFAR10, iCIFAR100, iImageNet100, iImageNet1000
import torch

class DataManager(object):
    def __init__(self, dataset_name, shuffle, seed, init_cls, increment, args):
        self.dataset_name = dataset_name
        self._setup_data(dataset_name, shuffle, seed)
        assert init_cls <= len(self._class_order), "No enough classes."
        self._increments = [init_cls]
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0:
            self._increments.append(offset)
        self.label_num = args["label_num"]

    @property
    def nb_tasks(self):
        return len(self._increments)

    def get_task_size(self, task):
        return self._increments[task]  #[10, 10, 10, 10, 10, 10, 10, 10, 10, 10]

    def get_total_classnum(self):
        return len(self._class_order)

    def get_dataset(
        self, indices, source, mode, appendent=None, ret_data=False, m_rate=None
    ):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
            trsf_s = transforms.Compose([*self._train_trsf_s, *self._common_trsf])
        elif mode == "flip":
            trsf = transforms.Compose(
                [
                    *self._test_trsf,
                    transforms.RandomHorizontalFlip(p=1.0),
                    *self._common_trsf,
                ]
            )
            trsf_s = trsf
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
            trsf_s = trsf
        else:
            raise ValueError("Unknown mode {}.".format(mode))
        
        ## 确定标记信息
        if mode != "test" or len(y) != 10000:
            fkeys_path = './data'
            dataset_name = self.dataset_name + '_labelindex'
            destination_name = 'label_map_count_' + self.label_num + '_index_0' 

            result_path = os.path.join(fkeys_path, dataset_name, destination_name)
            with open(result_path, "r") as f:
                label_index_value = json.load(f)['values']
            
            if 'imagenet' in self.dataset_name:
                x_idx = list(map(lambda i: x[i].split('/')[-1].split('.')[0], range(x.shape[0])))
                label_index_value = [i for i, num in enumerate(x_idx) if num in label_index_value]
                self.label_index_value = label_index_value
            else:
                label_index_value = list(map(int,label_index_value))
            label_index = np.zeros_like(y)
            label_index[label_index_value] = 1
        else:
            label_index = np.ones_like(y)
        ## 确定标记信息  end
        data, targets, pse_targets, lab_index_task, task_idxes = [], [], [], [], []
        for idx in indices:
            if m_rate is None:
                class_data, class_targets, class_label_index, idxes = self._select(
                    x, y, label_index, low_range=idx, high_range=idx + 1
                ) #FY   class_label_index 表示该样本是否有标签, idxes 表示该类别的样本id
            else:
                class_data, class_targets, class_label_index = self._select_rmm(
                    x, y, label_index, low_range=idx, high_range=idx + 1, m_rate=m_rate
                )
            data.append(class_data)
            targets.append(class_targets)
            pse_targets.append(np.ones_like(class_targets) * -100)
            lab_index_task.append(class_label_index)  #FY
            task_idxes.append(idxes)  
        
        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets, appendent_pse_targets, appendent_targets_lab_idx = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)
            pse_targets.append(appendent_pse_targets)
            lab_index_task.append(appendent_targets_lab_idx)  # FY
            #task_idxes.append(np.ones_like(appendent_targets)) #FY 不知道为什么这么要这么写 0603
            task_idxes.append(np.ones_like(appendent_targets)) ## 这里需要表示该样本是否有标签，以给后面batch中标记样本的分配提供便利
        '''
        if appendent is not None and len(appendent) != 0 and 1>0:
            appendent_data, appendent_targets, appendent_pse_targets, appendent_targets_lab_idx = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)
            pse_targets.append(appendent_pse_targets)
            lab_index_task.append(appendent_targets_lab_idx)  # FY
            #task_idxes.append(np.ones_like(appendent_targets)) #FY 不知道为什么这么要这么写 0603
            task_idxes.append(np.ones_like(appendent_targets)) ## 这里需要表示该样本是否有标签，以给后面batch中标记样本的分配提供便利
        
        if appendent is not None and len(appendent) != 0 and 1>0:
            appendent_data, appendent_targets, appendent_pse_targets, appendent_targets_lab_idx = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)
            pse_targets.append(appendent_pse_targets)
            lab_index_task.append(appendent_targets_lab_idx)  # FY
            #task_idxes.append(np.ones_like(appendent_targets)) #FY 不知道为什么这么要这么写 0603
            task_idxes.append(np.ones_like(appendent_targets)) ## 这里需要表示该样本是否有标签，以给后面batch中标记样本的分配提供便利
        '''
        data, targets = np.concatenate(data), np.concatenate(targets)
        pse_targets = np.concatenate(pse_targets)
        lab_index_task = np.concatenate(lab_index_task)
        task_idxes = np.concatenate(task_idxes)
        
        if ret_data:
            return data, targets, DummyDataset(data, targets, pse_targets, lab_index_task, trsf, trsf_s, self.use_path)
        else:
            #print('here1', self.use_path)
            return DummyDataset(data, targets, pse_targets, lab_index_task, trsf, trsf_s, self.use_path), task_idxes  #FY

    def get_dataset_with_split(
        self, indices, source, mode, appendent=None, val_samples_per_class=0
    ):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        train_data, train_targets = [], []
        val_data, val_targets = [], []
        for idx in indices:
            class_data, class_targets = self._select(
                x, y, low_range=idx, high_range=idx + 1
            )
            val_indx = np.random.choice(
                len(class_data), val_samples_per_class, replace=False
            )
            train_indx = list(set(np.arange(len(class_data))) - set(val_indx))
            val_data.append(class_data[val_indx])
            val_targets.append(class_targets[val_indx])
            train_data.append(class_data[train_indx])
            train_targets.append(class_targets[train_indx])

        if appendent is not None:
            appendent_data, appendent_targets = appendent
            for idx in range(0, int(np.max(appendent_targets)) + 1):
                append_data, append_targets = self._select(
                    appendent_data, appendent_targets, low_range=idx, high_range=idx + 1
                )
                val_indx = np.random.choice(
                    len(append_data), val_samples_per_class, replace=False
                )
                train_indx = list(set(np.arange(len(append_data))) - set(val_indx))
                val_data.append(append_data[val_indx])
                val_targets.append(append_targets[val_indx])
                train_data.append(append_data[train_indx])
                train_targets.append(append_targets[train_indx])

        train_data, train_targets = np.concatenate(train_data), np.concatenate(
            train_targets
        )
        val_data, val_targets = np.concatenate(val_data), np.concatenate(val_targets)

        return DummyDataset(
            train_data, train_targets, trsf, self.use_path
        ), DummyDataset(val_data, val_targets, trsf, self.use_path)

    def _setup_data(self, dataset_name, shuffle, seed):
        idata = _get_idata(dataset_name)
        idata.download_data()

        # Data
        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        self._test_data, self._test_targets = idata.test_data, idata.test_targets
        self.use_path = idata.use_path

        # Transforms
        self._train_trsf = idata.train_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf
        self._train_trsf_s = idata.train_trsf_s

        # Order
        order = [i for i in range(len(np.unique(self._train_targets)))]
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = idata.class_order
        self._class_order = order
        logging.info(self._class_order)

        # Map indices
        self._train_targets = _map_new_class_index(
            self._train_targets, self._class_order
        )
        self._test_targets = _map_new_class_index(self._test_targets, self._class_order)

    def _select(self, x, y,label_index, low_range, high_range):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[idxes], y[idxes], label_index[idxes], idxes

    def _select_rmm(self, x, y, label_index, low_range, high_range, m_rate):
        assert m_rate is not None
        if m_rate != 0:
            idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
            selected_idxes = np.random.randint(
                0, len(idxes), size=int((1 - m_rate) * len(idxes))
            )
            new_idxes = idxes[selected_idxes]
            new_idxes = np.sort(new_idxes)
        else:
            new_idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[new_idxes], y[new_idxes], label_index[new_idxes]

    def getlen(self, index):
        y = self._train_targets
        return np.sum(np.where(y == index))


class DummyDataset(Dataset):
    def __init__(self, images, labels, pse_labels, lab_index_task, trsf, trsf_s, use_path=False):
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        self.lab_index_task = lab_index_task
        self.pse_labels = pse_labels
        self.trsf = trsf
        self.use_path = use_path
        self.trsf_s = trsf_s

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        #print(self.use_path)
        if self.use_path:
            image_w = self.trsf(pil_loader(self.images[idx]))
            image_s = self.trsf_s(pil_loader(self.images[idx]))  # 20230704 for imagenet
        else:
            
            image_w = self.trsf(Image.fromarray(self.images[idx]))
            image_s = self.trsf_s(Image.fromarray(self.images[idx]))
        label = self.labels[idx]
        pse_labels = self.pse_labels[idx]
        lab_index_task = self.lab_index_task[idx]
        

        return idx, image_w, image_s, label, pse_labels, lab_index_task


def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))


def _get_idata(dataset_name):
    name = dataset_name.lower()
    if name == "cifar10":
        return iCIFAR10()
    elif name == "cifar100":
        return iCIFAR100()
    elif name == "imagenet1000":
        return iImageNet1000()
    elif name == "imagenet100":
        return iImageNet100()
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))


def pil_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def accimage_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    accimage is an accelerated Image loader and preprocessor leveraging Intel IPP.
    accimage is available on conda-forge.
    """
    import accimage

    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)
