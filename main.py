import os
import random
import argparse
import matplotlib
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from scipy.stats import spearmanr
from dataset import get_data_dict
from dataset import SurgeryFeatureDataset
from model import UnifiedSkillNet, CodingPredictor
from utils import load_config_file, set_random_seed

matplotlib.use('Agg')


class Trainer:
    def __init__(self, input_dim_list, embedding_dim_list, instance_norm_flags,
                 middle_dim_list, middle_dim_other, num_targets,
                 num_layers_attend, num_layers_assess, heavy_assess_head,
                 num_layers_predict, contrastive_window, contrastive_step,
                 pretrained_model_path=None):

        self.contrastive_window = contrastive_window
        self.contrastive_step = contrastive_step

        self.model = UnifiedSkillNet(input_dim_list, embedding_dim_list, instance_norm_flags,
                                     middle_dim_list, middle_dim_other, num_targets,
                                     num_layers_attend, num_layers_assess, heavy_assess_head)

        if pretrained_model_path:
            self.model.load_state_dict(torch.load(pretrained_model_path))

        self.num_targets = num_targets
        self.num_feature_types = len(input_dim_list)

        self.coding_predictor = CodingPredictor(
            num_feature_types=self.num_feature_types,
            middle_dim_list=middle_dim_list,
            num_layers=num_layers_predict)

        print(self.model)
        print(self.coding_predictor)

    def train(self, train_train_dataset, train_test_dataset, test_test_dataset,
              num_epochs, batch_size, learning_rate, weight_decay,
              contrastive_loss_weights, fast_test,
              device, result_dir, log_freq=10):

        mse_criterion = nn.MSELoss(reduction='none')

        train_train_loader = torch.utils.data.DataLoader(
            train_train_dataset, batch_size=1, shuffle=True, num_workers=4)  # batch_size=1, must

        if result_dir:
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            logger = SummaryWriter(result_dir)

        self.model.to(device)
        self.coding_predictor.to(device)

        optimizer = optim.Adam(
            list(self.model.parameters()) + list(self.coding_predictor.parameters()),
            lr=learning_rate, weight_decay=weight_decay
        )
        optimizer.zero_grad()

        step = 1

        for epoch in range(num_epochs):

            self.model.train()
            self.coding_predictor.train()

            for _, data in enumerate(train_train_loader):

                feature_list, score, video = data
                feature_list = [i.to(device) for i in feature_list]
                score = score.to(device).float()

                # feature_list: [torch.Size([1, 14, 3312]), torch.Size([1, 2048, 3312])] 
                # score: torch.Size([1, target_num])

                codings = self.model.get_codings(feature_list)
                codings = [i if i.shape[1] != 1 else None for i in codings]  # workaround

                pred_codings = self.coding_predictor(codings)

                contrastive_loss = 0

                for i in range(len(codings)):
                    if codings[i] is not None and contrastive_loss_weights[i] != 0:
                        temp_ctr_loss = self.get_contrastive_loss(codings[i].detach(), pred_codings[i])
                        temp_ctr_loss = temp_ctr_loss * contrastive_loss_weights[i]
                        contrastive_loss += temp_ctr_loss

                        if result_dir:
                            logger.add_scalar('Train-Loss-CON-P{}'.format(i + 1), temp_ctr_loss.item(), step)

                output = self.model(feature_list) # [TO BE IMPROVED] efficiency

                mse_loss = 0
                for i in range(output[1].shape[2]):
                    mse_loss += torch.mean(mse_criterion(output[1][:, :, i], score))

                if result_dir:
                    logger.add_scalar('Train-Loss-MSE', mse_loss.item(), step)

                total_loss = mse_loss + contrastive_loss

                if step % (10 * batch_size) == 1:
                    print('Step {} - Total Loss {}'.format(step, total_loss.item()))

                total_loss /= batch_size

                total_loss.backward()

                if step % batch_size == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                # real_step = step // batch_size

                step += 1

            if epoch % log_freq == 0:

                if result_dir:
                    torch.save(self.coding_predictor.state_dict(),
                               '{}/epoch-{}.predictor'.format(result_dir, epoch))
                    torch.save(self.model.state_dict(),
                               '{}/epoch-{}.model'.format(result_dir, epoch))
                    torch.save(optimizer.state_dict(),
                               '{}/epoch-{}.opt'.format(result_dir, epoch))

                train_mse, train_srocc, train_preds = self.test(train_test_dataset,
                                                                fast_test, device, result_dir=result_dir,
                                                                model_path=None)
                test_mse, test_srocc, test_preds = self.test(test_test_dataset,
                                                             fast_test, device, result_dir=result_dir, model_path=None)

                if result_dir:
                    for t_i in range(train_mse.shape[0]):
                        for p_i in range(train_mse.shape[1]):
                            logger.add_scalar('Train-MSE-T{}P{}'.format(t_i, p_i), train_mse[t_i, p_i], epoch)
                            logger.add_scalar('Test-MSE-T{}P{}'.format(t_i, p_i), test_mse[t_i, p_i], epoch)
                            logger.add_scalar('Train-SROCC-T{}P{}'.format(t_i, p_i), train_srocc[t_i, p_i], epoch)
                            logger.add_scalar('Test-SROCC-T{}P{}'.format(t_i, p_i), test_srocc[t_i, p_i], epoch)

                    np.save(os.path.join(result_dir,
                                         'train_preds-epoch{}.npy'.format(epoch)), train_preds)
                    np.save(os.path.join(result_dir,
                                         'test_preds-epoch{}.npy'.format(epoch)), test_preds)

                print('Epoch {} - Train-MSE {}'.format(epoch, train_mse))
                print('Epoch {} - Test-MSE {}'.format(epoch, test_mse))
                print('Epoch {} - Train-SROCC {}'.format(epoch, train_srocc))
                print('Epoch {} - Test-SROCC {}'.format(epoch, test_srocc))

        if result_dir:
            logger.close()

    def get_contrastive_loss(self, coding, pred_coding):

        criterion = nn.CrossEntropyLoss()

        coding = coding.squeeze(0).T  # Becareful, T x F
        pred_coding = pred_coding.squeeze(0).T

        window = self.contrastive_window
        step = self.contrastive_step
        offsets = [i for i in range(-window, window + 1, step)]
        assert (1 in offsets)

        seq_len = pred_coding.shape[0] - 2 * window - 1
        assert (seq_len > 0)

        similarities = []
        for offset in offsets:  # [TO BE IMPROVED] effiency
            similarities.append(torch.bmm(
                pred_coding[window:window + seq_len].unsqueeze(1),
                coding[window + offset:window + offset + seq_len].unsqueeze(2)  # offset=1 pos, other neg
            ))

        similarities = torch.cat(similarities, dim=1).squeeze(2)  # T x offsets

        target = torch.ones((similarities.shape[0],),
                            dtype=torch.long, device=similarities.device) * offsets.index(1)

        return criterion(similarities, target)

    def test(self, test_dataset, fast_test, device, result_dir=None, model_path=None):

        assert (test_dataset.mode == 'test')

        self.model.eval()
        self.model.to(device)

        if model_path:
            self.model.load_state_dict(torch.load(model_path))

        all_gts = {}
        all_preds = {}

        with torch.no_grad():

            for video_idx in range(len(test_dataset)):

                feature_list, score, video = test_dataset[video_idx]
                # feature_list: [[torch.Size([1, 14, 3312])], [torch.Size([10, 2048, 3312])]]

                num_feature_types = len(feature_list)
                num_temporal_augs = len(feature_list[0])
                assert (len(set([len(i) for i in feature_list])) == 1)

                all_aug_pred = []

                for temporal_rid in range(num_temporal_augs):

                    t_feature_list = [i[temporal_rid] for i in feature_list]
                    # t_feature_list: [torch.Size([1, 14, 3312]), torch.Size([10, 2048, 3312])]

                    num_spatial_augs = [i.shape[0] for i in t_feature_list]

                    # All combination of spatial_rids
                    spatial_rid_combinations = np.array(np.meshgrid(
                        *[np.arange(i) for i in num_spatial_augs])).T.reshape(-1, num_feature_types)

                    if fast_test:
                        spatial_rid_combinations = [random.choice(spatial_rid_combinations)]

                    for spatial_rids in spatial_rid_combinations:
                        s_feature_list = [t_feature_list[i][spatial_rids[i]]
                                          for i in range(num_feature_types)]

                        # s_feature_list: [torch.Size([14, 3967]), torch.Size([2048, 3967])]

                        s_feature_list = [i.unsqueeze(0).to(device) for i in s_feature_list]

                        output = self.model(s_feature_list)

                        all_aug_pred.append(np.concatenate([
                            np.expand_dims(output[0].squeeze(0).cpu().numpy(), 1),  # (target_num, 1)
                            output[1].squeeze(0).cpu().numpy()], 1  # (target_num, path_num)
                        ))

                all_aug_pred = np.array(all_aug_pred)  # (aug_num, target_num, path_num+1)

                #                 print('Prediction: ', video)
                #                 print('Mean: ', all_aug_pred.mean(0))
                #                 print('Std: ', all_aug_pred.std(0))

                all_gts[video] = score  # (target_num)
                all_preds[video] = all_aug_pred.mean(0)  # (target_num, path_num+1)

        video_list = [i for i in all_gts.keys()]

        t1 = np.expand_dims(np.array([all_gts[i] for i in video_list]), 2)  # (video_num, target_num, 1)
        t2 = np.array([all_preds[i] for i in video_list])  # (video_num, target_num, path_num+1)

        mse = ((t1 - t2) ** 2).mean(0)  # (target_num, path_num+1)

        srocc = np.zeros_like(mse)
        for i in range(mse.shape[0]):
            for j in range(mse.shape[1]):
                srocc[i, j] = spearmanr(t1[:, i, 0], t2[:, i, j])[0]

        self.model.train()

        return mse, srocc, all_preds


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config', type=str)
    args = parser.parse_args()
    all_params = load_config_file(args.config)
    locals().update(all_params)

    print(args.config)
    print(all_params)

    if random_seed:
        set_random_seed(random_seed)

    train_data_dict = get_data_dict(
        video_dir=video_dir,
        label_dir=label_dir,
        feature_dir_list=train_feature_dir_list,
        video_list=train_video_list,
        score_key_list=score_key_list,
        score_range_list=score_range_list,
        new_sample_rate=new_sample_rate,
        old_sample_rate=old_sample_rate,
        frame_rate=frame_rate,
        temporal_aug=temporal_aug
    )

    test_data_dict = get_data_dict(
        video_dir=video_dir,
        label_dir=label_dir,
        feature_dir_list=test_feature_dir_list,
        video_list=test_video_list,
        score_key_list=score_key_list,
        score_range_list=score_range_list,
        new_sample_rate=new_sample_rate,
        old_sample_rate=old_sample_rate,
        frame_rate=frame_rate,
        temporal_aug=temporal_aug
    )

    num_targets = len(score_key_list)

    train_train_dataset = SurgeryFeatureDataset(train_data_dict, mode='train')
    train_test_dataset = SurgeryFeatureDataset(train_data_dict, mode='test')
    test_test_dataset = SurgeryFeatureDataset(test_data_dict, mode='test')

    if not os.path.exists('result'):
        os.makedirs('result')

    heavy_assess_head = [i != 0 for i in contrastive_loss_weights]

    trainer = Trainer(input_dim_list, embedding_dim_list, instance_norm_flags,
                      middle_dim_list, middle_dim_other, num_targets,
                      num_layers_attend, num_layers_assess, heavy_assess_head,
                      num_layers_predict, contrastive_window, contrastive_step,
                      pretrained_model_path=None)

    trainer.train(train_train_dataset, train_test_dataset, test_test_dataset,
                  num_epochs, batch_size, learning_rate, weight_decay,
                  contrastive_loss_weights, fast_test,
                  device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                  result_dir=os.path.join('result', naming),
                  log_freq=log_freq
                  )
