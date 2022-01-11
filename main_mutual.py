from load_data import Data
import numpy as np
import torch
import time
from collections import defaultdict
from model import *
from torch.optim.lr_scheduler import ExponentialLR
import argparse
from matplotlib import pyplot as plt
    
class Experiment:

    def __init__(self, learning_rate=0.0005, ent_vec_dim=200, rel_vec_dim=200, 
                 num_iterations=500, batch_size=128, decay_rate=0., cuda=False, 
                 input_dropout=0.3, hidden_dropout1=0.4, hidden_dropout2=0.5,
                 label_smoothing=0.):
        self.learning_rate = learning_rate
        self.ent_vec_dim = ent_vec_dim
        self.rel_vec_dim = rel_vec_dim
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.decay_rate = decay_rate
        self.label_smoothing = label_smoothing
        self.cuda = cuda
        self.kwargs = {"input_dropout": input_dropout, "hidden_dropout1": hidden_dropout1,
                       "hidden_dropout2": hidden_dropout2}
        
    def get_data_idxs(self, data):
        data_idxs = [(self.entity_idxs[data[i][0]], self.relation_idxs[data[i][1]], \
                      self.entity_idxs[data[i][2]]) for i in range(len(data))]
        return data_idxs
    
    def get_er_vocab(self, data):
        er_vocab = defaultdict(list)
        for triple in data:
            er_vocab[(triple[0], triple[1])].append(triple[2])
        return er_vocab

    def get_batch(self, er_vocab, er_vocab_pairs, idx):
        batch = er_vocab_pairs[idx:idx+self.batch_size]
        targets = np.zeros((len(batch), len(d.entities)))
        ## 对应的源实体关系 对应的目标实体为1
        for idx, pair in enumerate(batch):
            targets[idx, er_vocab[pair]] = 1.
        targets = torch.FloatTensor(targets)
        if self.cuda:
            targets = targets.cuda()
        return np.array(batch), targets

    def save_model(self, model, save_dir):
        print("Saving Model")
        torch.save(model.state_dict(), (save_dir + "{}.pth").format("best_teacher_model"))
        print("Done saving Model")

    def evaluate(self, model_1, model_2, data):
        hits = []
        ranks = []
        for i in range(10):
            hits.append([])
        # test_data_idx : list[tuple[int, int, int]], get data triples indices
        test_data_idxs = self.get_data_idxs(data)
        # er_vocab : defaultdict(list)
        # first get global data triples, then build a dict like er_vocab[head, relation)].append(tail)
        er_vocab = self.get_er_vocab(self.get_data_idxs(d.data))

        print("Number of data points: %d" % len(test_data_idxs))
        # 
        for i in range(0, len(test_data_idxs), self.batch_size):
            # take a batch size triples
            data_batch, _ = self.get_batch(er_vocab, test_data_idxs, i)
            e1_idx = torch.tensor(data_batch[:,0])
            r_idx = torch.tensor(data_batch[:,1])
            e2_idx = torch.tensor(data_batch[:,2])
            if self.cuda:
                e1_idx = e1_idx.cuda()
                r_idx = r_idx.cuda()
                e2_idx = e2_idx.cuda()
            # get predict score [batchsize, entity_type_size]
            model_1.eval()
            model_2.eval()
            predictions_1 = model_1.forward(e1_idx, r_idx)
            predictions_2 = model_2.forward(e1_idx, r_idx)
            predictions = (predictions_1+predictions_2)/2

            for j in range(data_batch.shape[0]):
                # filter exist correct triple, set the corresponding scores as 0, 
                # and Retain the predicted score for the triple to be evaluated
                filt = er_vocab[(data_batch[j][0], data_batch[j][1])]
                target_value = predictions[j,e2_idx[j]].item()
                predictions[j, filt] = 0.0
                predictions[j, e2_idx[j]] = target_value
            # sort the scores
            sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)

            sort_idxs = sort_idxs.cpu().numpy()
            for j in range(data_batch.shape[0]):
                # find the rank of positive triple
                rank = np.where(sort_idxs[j]==e2_idx[j].item())[0][0]
                ranks.append(rank+1)
                # here could be changed to 'for hits_level in [1,3,5,10]'  
                for hits_level in range(10):
                    if rank <= hits_level:
                        hits[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)

        print('Hits @10: {0}'.format(np.mean(hits[9])))
        print('Hits @3: {0}'.format(np.mean(hits[2])))
        print('Hits @1: {0}'.format(np.mean(hits[0])))
        print('Mean rank: {0}'.format(np.mean(ranks)))
        print('Mean reciprocal rank: {0}'.format(np.mean(1./np.array(ranks))))
        MRR = np.mean(1./np.array(ranks))
        return MRR

    def train_and_eval(self,args):
        print("Training the TuckER model...")
        self.entity_idxs = {d.entities[i]:i for i in range(len(d.entities))}
        self.relation_idxs = {d.relations[i]:i for i in range(len(d.relations))}

        train_data_idxs = self.get_data_idxs(d.train_data)
        print("Number of training data points: %d" % len(train_data_idxs))

        valid_res, test_res = 0., 0. 
        best_epoch = 0
        model_1 = TuckER(d, self.ent_vec_dim, self.rel_vec_dim, **self.kwargs)
        model_2 = TuckER(d, self.ent_vec_dim, self.rel_vec_dim, **self.kwargs)
        # model = MLP_1(d, self.ent_vec_dim, self.rel_vec_dim, **self.kwargs)
        if self.cuda:
            model_1.cuda()
            model_2.cuda()
        model_1.init()
        model_2.init()
        opt_1 = torch.optim.Adam(model_1.parameters(), lr=self.learning_rate)
        opt_2 = torch.optim.Adam(model_2.parameters(), lr=self.learning_rate)

        if self.decay_rate:
            scheduler_1 = ExponentialLR(opt_1, self.decay_rate)
            scheduler_2 = ExponentialLR(opt_2, self.decay_rate)

        er_vocab = self.get_er_vocab(train_data_idxs) # {(head,rel):[tail]}
        er_vocab_pairs = list(er_vocab.keys()) # head&rel

        print("Starting training...")
        for it in range(1, self.num_iterations+1):
            start_train = time.time()
            model_1.train()
            model_2.train()    
            losses = []
            np.random.shuffle(er_vocab_pairs)
            for j in range(0, len(er_vocab_pairs), self.batch_size):
                ## 这里怎么有点一直在重复取的感觉，哦没有,它是间距batch_size
                data_batch, targets = self.get_batch(er_vocab, er_vocab_pairs, j)
                opt_1.zero_grad()
                opt_2.zero_grad()
                e1_idx = torch.tensor(data_batch[:,0])
                r_idx = torch.tensor(data_batch[:,1])  
                if self.cuda:
                    e1_idx = e1_idx.cuda()
                    r_idx = r_idx.cuda()
                predictions_1 = model_1.forward(e1_idx, r_idx)
                predictions_2 = model_2.forward(e1_idx, r_idx)
                if self.label_smoothing:
                    targets = ((1.0-self.label_smoothing)*targets) + (1.0/targets.size(1))           
                loss = model_1.loss(predictions_1, targets) + model_2.loss(predictions_2, targets) + torch.mean((predictions_1-predictions_2)**2)
                loss.backward()
                opt_1.step()
                opt_2.step()

                losses.append(loss.item())
            if self.decay_rate:
                scheduler_1.step()
                scheduler_2.step()
            print(it)
            print(time.time()-start_train)    
            print(np.mean(losses))
            # exit()
            model_1.eval()
            model_2.eval()
            with torch.no_grad():
                print("Validation:")
                valid_MRR = self.evaluate(model_1, model_2, d.valid_data)
                valid_res = max(valid_res, valid_MRR)
                if not it%2:
                    print("Test:")
                    start_test = time.time()
                    test_MRR = self.evaluate(model_1, model_2, d.test_data)
                    if test_MRR > test_res:
                        test_res = test_MRR
                        best_epoch = it
                        # self.save_model(model, save_dir)
                        # if test_res > args.best_result:
                        #     self.save_model(model, save_dir)
                    print(time.time()-start_test)
        print("Best result : valid {0} test {1}".format(valid_res,test_res))
        print("Best epoch is {0}".format(best_epoch))

    def pred_plot(self, save_dir, data):
        self.entity_idxs = {d.entities[i]:i for i in range(len(d.entities))}
        self.relation_idxs = {d.relations[i]:i for i in range(len(d.relations))}

        train_data_idxs = self.get_data_idxs(d.train_data)
        model = MLP_2(d, self.ent_vec_dim, self.rel_vec_dim, **self.kwargs)
        model.load_state_dict(torch.load('{}/best_model.pth'.format(save_dir)), strict=False)
        if self.cuda:
            model.cuda()
        test_data_idxs = self.get_data_idxs(data)
        er_vocab = self.get_er_vocab(self.get_data_idxs(d.data))

        print("Number of data points: %d" % len(test_data_idxs))
        
        for i in range(0, len(test_data_idxs), self.batch_size):
            data_batch, _ = self.get_batch(er_vocab, test_data_idxs, i)
            e1_idx = torch.tensor(data_batch[:,0])
            r_idx = torch.tensor(data_batch[:,1])
            e2_idx = torch.tensor(data_batch[:,2])
            if self.cuda:
                e1_idx = e1_idx.cuda()
                r_idx = r_idx.cuda()
                e2_idx = e2_idx.cuda()
            predictions = model.forward(e1_idx, r_idx)

            for j in range(data_batch.shape[0]):
                filt = er_vocab[(data_batch[j][0], data_batch[j][1])]
                target_value = predictions[j,e2_idx[j]].item()
                predictions[j, filt] = 0.0
                predictions[j, e2_idx[j]] = target_value

            sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)
            sort_idxs = sort_idxs.cpu().numpy()
            sort_values = sort_values.detach().cpu().numpy()
            for j in range(predictions.shape[0]):
                rank = np.where(sort_idxs[j]==e2_idx[j].item())[0][0]
            # if j % 100 == 0:
                fig = plt.figure(figsize=(4,3))
                plt.title("Rank : {}".format(rank))
                plt.scatter(np.arange(len(sort_values[j])),sort_values[j])
                plt.scatter(rank, sort_values[j][rank], c="red")
                # plt.show()
                plt.savefig("pic/{}.png".format(j))
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="FB15k-237", nargs="?",
                    help="Which dataset to use: FB15k, FB15k-237, WN18 or WN18RR.")
    parser.add_argument("--num_iterations", type=int, default=10, nargs="?",
                    help="Number of iterations.")
    parser.add_argument("--batch_size", type=int, default=128, nargs="?",
                    help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.0005, nargs="?",
                    help="Learning rate.")
    parser.add_argument("--dr", type=float, default=1.0, nargs="?",
                    help="Decay rate.")
    parser.add_argument("--edim", type=int, default=200, nargs="?",
                    help="Entity embedding dimensionality.")
    parser.add_argument("--rdim", type=int, default=200, nargs="?",
                    help="Relation embedding dimensionality.")
    parser.add_argument("--cuda", type=bool, default=True, nargs="?",
                    help="Whether to use cuda (GPU) or not (CPU).")
    parser.add_argument("--input_dropout", type=float, default=0.3, nargs="?",
                    help="Input layer dropout.")
    parser.add_argument("--hidden_dropout1", type=float, default=0.4, nargs="?",
                    help="Dropout after the first hidden layer.")
    parser.add_argument("--hidden_dropout2", type=float, default=0.5, nargs="?",
                    help="Dropout after the second hidden layer.")
    parser.add_argument("--label_smoothing", type=float, default=0.1, nargs="?",
                    help="Amount of label smoothing.")
    parser.add_argument("--reverse", type=bool, default=True, nargs="?",
                    help="Whether to use reverse.")
    parser.add_argument("--best_result", type=float, default=0.4, nargs="?",
                    help="If larger than best result, then save.")

    args = parser.parse_args()
    dataset = args.dataset
    print(args)
    print("*"*20)
    save_dir = "checkpoint/%s/" % dataset
    data_dir = "data/%s/" % dataset
    torch.backends.cudnn.deterministic = True 
    seed = 20
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed) 
    d = Data(data_dir=data_dir, reverse=args.reverse)
    # d.load_rev_data(data_dir)
    d.dis_entities()
    experiment = Experiment(num_iterations=args.num_iterations, batch_size=args.batch_size, learning_rate=args.lr, 
                            decay_rate=args.dr, ent_vec_dim=args.edim, rel_vec_dim=args.rdim, cuda=args.cuda,
                            input_dropout=args.input_dropout, hidden_dropout1=args.hidden_dropout1, 
                            hidden_dropout2=args.hidden_dropout2, label_smoothing=args.label_smoothing)
    experiment.train_and_eval(args)
    print(dataset)
    # experiment.pred_plot(save_dir, d.test_data)
