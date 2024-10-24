from Algorithm.Algorithm import Algorithm
import torch
from utils import calculate_RMSE,calculate_R2, regression_loss

class CorrelationLearning(Algorithm):
    def __init__(self, opt):
        Algorithm.__init__(self, opt)
        self.keep_best_model_metric_name = 'R2'
        self.max_best = True

    def allocate_tensors(self):
        self.tensor = {}
        self.tensor['feature'] = torch.FloatTensor()
        self.tensor['target'] = torch.FloatTensor()
        self.tensor['target_mask'] = torch.LongTensor()

    def set_tensors(self, batch):
        feature, target, target_mask = batch
        self.tensor['feature'].resize_(feature.size()).copy_(feature)
        self.tensor['target'].resize_(target.size()).copy_(target)
        self.tensor['target_mask'].resize_(target_mask.size()).copy_(target_mask)


    def train_step(self, batch):
        return self.process_batch(batch, do_train=True)

    def evaluation_step(self, batch):
        return self.process_batch(batch, do_train=False)

    def process_batch(self, batch, do_train=True):

        self.set_tensors(batch)
        feature = self.tensor['feature']  # [batchsize, feature_dim]
        target = self.tensor['target']     # [batchsize, num_tasks]
        target_mask = self.tensor['target_mask']  # [batchsize, num_tasks]


        CorrelationNet = self.learners['CorrelationNet']

        if do_train:
            self.optimizers['CorrelationNet'].zero_grad()

        record = {}

        # forward
        pred = CorrelationNet(feature)     # (batchsize, num_task)
        # print(pred)


        if do_train:
            loss = regression_loss(pred=pred, target=target, target_mask=target_mask)

            # loss.backward(retain_graph=True)
            loss.backward()
            self.optimizers['CorrelationNet'].step()
            record['MSE'] = loss.cpu().item()


        if not do_train:
            _, RMSE_avg = calculate_RMSE(pred=pred, target=target, target_mask=target_mask)
            _, R2_avg = calculate_R2(pred=pred, target=target, target_mask=target_mask)
            record['RMSE'] = RMSE_avg.cpu().item()
            record['R2'] = R2_avg.cpu().item()

        return record
