import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler, random_split
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, MNIST
import numpy as np
import os
import urllib.request
import sklearn.datasets
from torchvision import models
from dataset_and_model import load_data, load_model

############################
# Data preparation
############################
def create_neibordataset(dataset, n_pairs, seed=None):
    # only work on indices
    if seed is not None:
        np.random.seed(42+seed)
    indices = list(range(len(dataset)))
    print("----Creating neighboring dataset----")
    print(f"Total number of samples: {len(indices)}")
    print(f"Number of pairs: {n_pairs}")
    removed_indices = np.random.choice(indices, n_pairs, replace=False)
    neighbordataset_1 = [i for i in indices if i not in removed_indices]
    datasets_dict = {0: neighbordataset_1.copy()}
    # randomly replace 1 elements from neighbordataset_1 for "n_pairs" times
    replace_indices = np.random.choice(neighbordataset_1, n_pairs, replace=False)
    for i in range(n_pairs):
        new_subset = neighbordataset_1.copy()
        # replace
        idx = new_subset.index(replace_indices[i])
        print(f"Pair {i+1}: Replacing index {replace_indices[i]} with {removed_indices[i]}")
        new_subset[idx] = removed_indices[i]
        datasets_dict[i+1] = new_subset
    return datasets_dict

def distribute_data(n_workers, neighbor_datasets_dict, heter=True, seed=None):
    # neighbor_dataset should be: dict {0:[], 1:[], ...}
    if seed is not None:
        np.random.seed(42+seed)
    print("----Distributing data----")
    distributed = {}
    combined_indices = None
    for key, subset in neighbor_datasets_dict.items():
        if combined_indices is None:
            combined_indices = np.arange(len(subset))
            np.random.shuffle(combined_indices)
        shuffled_subset = [subset[i] for i in combined_indices]
        chunk_size = len(combined_indices) // n_workers
        distributed_subset = {}
        for worker_id in range(n_workers):
            start_idx = worker_id * chunk_size
            end_idx = (worker_id + 1) * chunk_size if worker_id != n_workers - 1 else len(combined_indices)
            if not heter:
                distributed_subset[worker_id] = shuffled_subset
            else:
                distributed_subset[worker_id] = [shuffled_subset[i] for i in range(start_idx, end_idx)]
        distributed[key] = distributed_subset
    return distributed  # {0: {worker_id: ...}, 1: {...}, ...}

class DistributedSampler(Sampler):
    def __init__(self, indices, worker_id=None, seed=0):
        # super(DistributedSampler, self).__init__(indices)
        self.indices = indices
        self.worker_id = worker_id
        self.seed = seed

    def set_seed(self, seed):
        self.seed = seed

    def update_seed(self):
        self.seed += 1

    def __iter__(self):
        seed = int(self.worker_id) * 10000 + int(self.seed)
        indices = torch.tensor(self.indices)
        self.shuffled_indices = indices[torch.randperm(len(indices), generator=torch.Generator().manual_seed(seed))]
        return iter(self.shuffled_indices.tolist())

    def __len__(self):
        return len(self.indices)
    
class TestSampler(Sampler):
    def __init__(self, length, max_length, seed):
        # super(TestSampler, self).__init__(length)
        self.length = length
        self.max_length = max_length
        self.seed = seed
        self.indices = torch.randperm(self.length, generator=torch.Generator().manual_seed(self.seed))[:self.max_length]

    def __iter__(self):
        return iter(self.indices.tolist())

    def __len__(self):
        return len(self.indices)

############################
# Information logging
############################
class Logger:
    def __init__(self, log_dir, model_name='', dataset_name='', loss_name='', q=0, lr=0.01, delay=0, bs=1):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.filename = f"log_{model_name}_{dataset_name}_{loss_name}_q{q}_lr{lr:.0e}_bs{bs}_delay{delay}.txt"
        self.file_path = os.path.join(log_dir, self.filename)

    def update(self, iteration, delay, train_loss, train_acc, test_loss, test_acc, generalization_gap, stability):
        with open(self.file_path, 'a') as f:
            f.write(f"Iteration: {iteration}, Delay: {delay}, Train Loss: {train_loss},Train ACC: {train_acc}, Test Loss: {test_loss}, Test ACC: {test_acc}, Generalization Gap: {generalization_gap}, Stability: {stability}\n")
    def savelog(self):
        pass

class DataRecorder:
    def __init__(self, rec_dir='', model_name='', dataset_name='', loss_name='', q=0, lr=0.01, delay=0, bs=1):
        self.rec_dir = rec_dir
        if not os.path.exists(rec_dir):
            os.makedirs(rec_dir)
        self.filename = f"{model_name}_{dataset_name}_{loss_name}_q{q}_lr{lr:.0e}_bs{bs}_delay{delay}.pt"
        self.data = {'iteration': [],
                     'delay': [],
                     'train_loss': [],
                     'train_acc': [],
                     'test_loss': [],
                     'test_acc': [],
                     'generalization_gap': [],
                     'stability': []}

    def update(self, iteration, delay, train_loss, train_acc, test_loss, test_acc, generalization_gap, stability):
        self.data['iteration'].append(iteration)
        self.data['delay'].append(delay)
        self.data['train_loss'].append(train_loss)
        self.data['train_acc'].append(train_acc)
        self.data['test_loss'].append(test_loss)
        self.data['test_acc'].append(test_acc)
        self.data['generalization_gap'].append(generalization_gap)
        self.data['stability'].append(stability)

    def save(self):
        filepath = os.path.join(self.rec_dir, self.filename)
        torch.save(self.data, filepath)
        print(f"Data saved to {filepath}")

class ModelCheckpoint:
    def __init__(self, checkpoint_dir, model_name='', dataset_name='', loss_name='', q=0, lr=0.01, delay=0, bs=1):
        self.checkpoint_dir =  checkpoint_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.model_name=model_name
        self.dataset_name=dataset_name
        self.loss_name=loss_name
        self.lr=lr
        self.delay=delay
        self.bs=bs
        self.q=q

    def save(self, model, iteration):
        self.filename= f"{self.model_name}_iter{iteration}_{self.dataset_name}_{self.loss_name}_q{self.q}_lr{self.lr:.0e}_bs{self.bs}_delay{self.delay}.pt"
        self.filepath = os.path.join(self.checkpoint_dir, self.filename)
        torch.save(model.state_dict(), self.filepath)
        print(f"Model saved to {self.filepath} at iteration {iteration}")

############################
# Server class
############################
class Server:
    def __init__(self, device='cuda', train_type='fixed', n_pairs=1, num_workers=1, batch_size=1,
                  dataset_name=None, dataset_path=None, model_name=None, loss_name='mse', q=0, lr=0.01, 
                  iterations=0, evaluation_time=0, random_seed=0, log_dir=None, checkpoint_dir=None, rec_dir=None):
        # train_type = 'fixed' or 'random'
        self.device = device
        self.train_type = train_type
        self.n_pairs = n_pairs  # number of pairs of neighboring datasets
        self.num_workers = num_workers
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.model_name = model_name
        self.loss_name = loss_name
        self.q = q
        self.lr=lr
        self.iterations = iterations
        self.evaluation_time = evaluation_time
        self.batch_size = batch_size
        self.logger = Logger(log_dir, model_name=model_name, dataset_name=dataset_name,
                            loss_name=loss_name, q=self.q, lr=lr, delay=num_workers-1, bs=batch_size)
        self.checkpoint = ModelCheckpoint(checkpoint_dir, model_name=model_name, dataset_name=dataset_name,
                                          loss_name=loss_name, q=self.q, lr=lr, delay=num_workers-1, bs=batch_size)
        self.datarecorder = DataRecorder(rec_dir=rec_dir, model_name=model_name, dataset_name=dataset_name,
                                          loss_name=loss_name, q=self.q, lr=lr, delay=num_workers-1, bs=batch_size)
        self.iteration = 1 #iteration starts from 1
        self.delay = 0 # delay factor: if train_type ='fixed', the delay factor = num_workers - 1
        self.random_seed=random_seed

        # Load datasets and models
        self.train_dataset = load_data(dataset_name, dataset_path, 'train')
        self.test_dataset = load_data(dataset_name, dataset_path, 'test')
        self.neighbor_datasets = create_neibordataset(self.train_dataset, self.n_pairs, seed=self.random_seed)
        self.models = {i: load_model(model_name, loss_name, q=self.q) for i in range(self.n_pairs+1)}  #create n_"pairs+1" models
        self.optimizers = {i: optim.SGD(self.models[i].parameters(), lr=self.lr) for i in range(self.n_pairs+1)}

        # Set device
        if self.device == 'cuda':
            if torch.cuda.is_available():
                for i in range(self.n_pairs+1):
                    self.models[i].to(torch.device(self.device))
            else:
                raise RuntimeError("CUDA is not available. Please set device to 'cpu'.")
        elif self.device == 'cpu':
            for i in range(self.n_pairs+1):
                self.models[i].to(torch.device(self.device))
        else:
            raise ValueError("Unsupported device. Use 'cuda' or 'cpu'.")    

        # Distribute data to workers
        # Distributed_datasets is a turple with (dict1, dict2) 
        if self.train_type == 'fixed':
            self.current_worker_id = 0
            self.distributed_datasets = distribute_data(num_workers, self.neighbor_datasets, heter=True, seed=self.random_seed)
        elif self.train_type == 'random':
            self.distributed_datasets = distribute_data(num_workers, self.neighbor_datasets, heter=True, seed=self.random_seed)
        self.workers = [Worker(worker_id, bs=self.batch_size, distributed_datasets=self.distributed_datasets, n_pairs=self.n_pairs, server=self) for worker_id in range(num_workers)]

    def train(self):
        for worker in self.workers:
            worker.update_models()
            worker.gradient_calculate()
        while self.iteration <= self.iterations:
            if self.train_type == 'fixed':
                self.step_fixeddelay()
            elif self.train_type == 'random':
                self.step()
            else:
                raise ValueError("Unsupported training type. Use 'fixed' or 'random'.")
            # if self.iteration % 1000 == 0:
            #     print(f"Iteration: {self.iteration}, Current Worker ID: {self.current_worker_id}, Delay: {self.delay}")
            if self.iteration % self.evaluation_time == 0 or self.iteration == self.iterations or self.iteration == 1 or self.iteration in [100, 200, 300, 400, 500, 600, 700, 800, 900]:
                self.evaluation()
            self.iteration += 1

    def step_fixeddelay(self):
        selected_worker = self.workers[self.current_worker_id]
        if self.current_worker_id < self.num_workers - 1:
            self.current_worker_id += 1
        else:   
            self.current_worker_id = 0
        
        iteration_last = selected_worker.iteration_last
        self.delay = self.iteration - iteration_last - 1

        # Update model parameters using gradients from the worker
        upd = [True] * (self.n_pairs + 1)
        for n in range(self.n_pairs + 1):
            self.optimizers[n].zero_grad(set_to_none=True)
            for Sparam, Wparam in zip(self.models[n].parameters(), selected_worker.models[n].parameters()):
                if Wparam.grad is None:
                    upd[n] = False
                    continue
                Sparam.grad = Wparam.grad.detach().clone()
            if upd[n]:
                self.optimizers[n].step()

        # Send updated models back to the worker
        selected_worker.update_models()
        selected_worker.gradient_calculate()


    def evaluation(self):
        # slow evaluation
        train_loss1, train_acc = self.evaluate(self.models[0], self.train_dataset)
        # if len(self.train_dataset) <= 10*len(self.test_dataset):
        #     test_sampler = TestSampler(len(self.test_dataset), max_length=len(self.train_dataset), seed=self.iteration+self.random_seed)
        # else:
        #     test_sampler = None
        test_loss1, test_acc = self.evaluate(self.models[0], self.test_dataset)

        stability = self.stability_calculate()
        print(f"Iteration: {self.iteration}, Delay: {self.delay:.0f}, Train Loss: {train_loss1:.6f}, Train Acc: {train_acc:.6f}, Test Loss: {test_loss1:.6f}, Test Acc: {test_acc:.6f}, Generalization Error: {test_loss1 - train_loss1:.6f}, Stability: {stability:.6f}")
        self.logger.update(self.iteration, self.delay, train_loss1, train_acc, test_loss1, test_acc, test_loss1 - train_loss1, stability)
        self.datarecorder.update(self.iteration, self.delay, train_loss1, train_acc, test_loss1, test_acc, test_loss1 - train_loss1, stability)
        if self.iteration == self.iterations:
            self.datarecorder.save()
            self.checkpoint.save(self.models[0], self.iteration)

    def evaluate(self, model, dataset, sampler=None):
        if sampler is None:
            dataloader = DataLoader(dataset, batch_size=1000, shuffle=False)
        else:
            dataloader = DataLoader(dataset, batch_size=1000, sampler=sampler)

        total_loss = 0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs, loss = model(inputs, labels)
                total_loss += loss.item() * inputs.size(0)
                # accuracy
                preds = torch.argmax(outputs, dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

        avg_loss = total_loss / total_samples
        acc = total_correct / total_samples
        return avg_loss, acc

    def stability_calculate(self):
        diff_norm = 0
        params1 = dict(self.models[0].named_parameters())
        for n in range(self.n_pairs):
            params2 = dict(self.models[n+1].named_parameters())
            diff_norm = diff_norm + sum(torch.norm(params1[name] - params2[name]).item() ** 2 for name in params1.keys()) ** 0.5
        return diff_norm / self.n_pairs

############################
# Worker class
############################
class Worker:
    def __init__(self, worker_id, bs, distributed_datasets, n_pairs, server):
        self.worker_id = worker_id
        self.active = True
        self.n_pairs = n_pairs
        self.iteration_last = 0
        self.batch_size = bs
        self.server = server
        self.device = server.device
        self.models = {i: load_model(server.model_name, server.loss_name) for i in range(self.n_pairs+1)}
        for i in range(self.n_pairs+1):
            self.models[i].to(self.device)
        self.ini_seed = server.random_seed
        self.samplers = {
                i: DistributedSampler(distributed_datasets[i][worker_id], worker_id=self.worker_id, seed=self.ini_seed) for i in range(self.n_pairs+1)
            }
        self.dataloaders = {
                i: DataLoader(server.train_dataset, batch_size=self.batch_size, sampler=self.samplers[i]) for i in range(self.n_pairs+1)
            }
        self.generators = {
                i: enumerate(self.dataloaders[i]) for i in range(self.n_pairs+1)
            }

    def next_data(self, model_name):
        try:
            batch_idx , (data, labels) = next(self.generators[model_name])
        except StopIteration:
            self.samplers[model_name].update_seed()
            self.dataloaders[model_name] = DataLoader(self.server.train_dataset, batch_size=self.batch_size, sampler=self.samplers[model_name])
            self.generators[model_name] = enumerate(self.dataloaders[model_name])
            batch_idx , (data, labels) = next(self.generators[model_name])
        # batch_indices = self.dataloaders[model_name].batch_sampler.sampler.shuffled_indices[
        #     batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size
        # ]
        # print(f"Worker {self.worker_id}, Iteration {self.server.iteration}, Model {model_name}, Sample IDs: {batch_indices}")

        return data.to(self.device), labels.to(self.device)

    def gradient_calculate(self):
        for i in range(self.n_pairs+1):
            model = self.models[i]
            model.zero_grad()
            inputs, labels = self.next_data(i)
            _ , loss = model(inputs, labels)
            loss.backward()
        self.iteration_last = self.server.iteration
        self.active = False

    def update_models(self):
        for i in range(self.n_pairs+1):
            self.models[i].load_state_dict(self.server.models[i].state_dict())
        self.active = True

############################
# Main function
############################

if __name__ == "__main__":
    """
    Usage:
      Unified argparse entry point for all previous standalone main() examples.

    Collected typical experiment configurations (reproducible via CLI):
      1. GISETTE (linear_gisette, MSE)
         python AsynchronousSGD.py --dataset gisette --model linear_gisette --loss mse --lr 2e-5 --iterations 6000 --eval-interval 60 --n-pairs 10 --batch-size 16 --num-workers-list 41,31,21,11,1 --repeats 5 --seed-base 0 --seed-step 10 --device cpu
      2. RCV1 (linear_rcv1, MSE)
         python AsynchronousSGD.py --dataset rcv1 --model linear_rcv1 --loss mse --lr 0.015 --iterations 30000 --eval-interval 300 --n-pairs 10 --batch-size 16 --num-workers-list 121 --repeats 5 --seed-base 0 --seed-step 10 --device cpu
      3. MNIST (fcnet_mnist, CE)
         python AsynchronousSGD.py --dataset mnist --model fcnet_mnist --loss ce --lr 1e-2 --iterations 40000 --eval-interval 800 --n-pairs 3 --batch-size 16 --num-workers-list 1,11,21,31,41,51 --repeats 5 --seed-base 1 --seed-step 10
      4. IJCNN (linear_ijcnn, MSE)
         python AsynchronousSGD.py --dataset ijcnn --model linear_ijcnn --loss mse --lr 2e-2 --iterations 40000 --eval-interval 400 --n-pairs 10 --batch-size 16 --num-workers-list 1,21,41,61,81 --repeats 5 --seed-base 2 --seed-step 12 --device cpu
      5. CIFAR10 (resnet18_cifar10, CE)
         python AsynchronousSGD.py --dataset cifar10 --model resnet18_cifar10 --loss ce --lr 0.01 --iterations 60000 --eval-interval 1200 --n-pairs 1 --batch-size 16 --num-workers-list 1,11,21,31,41 --repeats 5 --seed-base 0 --seed-step 10 --device cuda

    Main arguments:
      --dataset            One of {rcv1, gisette, ijcnn, w1a, mnist, cifar10}
      --model              Model name matching the dataset
      --loss               mse | hingeloss | ce
      --n-pairs            Number of neighboring dataset pairs (creates n_pairs+1 models)
      --num-workers-list   Comma-separated list of worker counts
      --repeats            Number of repeated runs (creates r1..rK)
      --seed-base          Base random seed
      --seed-step          Seed increment per repeat
      --eval-interval      Evaluation interval (iterations)
      --device             cpu | cuda
      --train-type         fixed | random (scheduling strategy)

    Notes:
      1. Each repeat uses seed = seed_base + (repeat_index - 1) * seed_step.
      2. For multiple worker settings in one run, models are retrained sequentially.
      3. Output folders:
         logs/          text metric logs
         checkpoints/   final model (only when last iteration reached)
         records/       tensor (.pt) metric arrays
      4. Stability metric = average L2 distance across parameters between model 0 and each neighboring model.
    """
    import argparse, gc, torch, numpy as np, os

    parser = argparse.ArgumentParser(description="Asynchronous SGD Stability Experiment")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["rcv1", "gisette", "ijcnn", "w1a", "mnist", "cifar10", "a1a", "a2a", "a3a", "a4a", "a5a", "a6a", "a7a"],
                        help="dataset name")
    parser.add_argument("--dataset-path", type=str, default="./data", help="dataset root directory")
    parser.add_argument("--model", type=str, required=True,
                        help="model name, e.g. linear_rcv1 / linear_gisette / fcnet_mnist / linear_ijcnn / linear_w1a / resnet18_cifar10")
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "hingeloss", "ce"], help="loss function type")
    parser.add_argument("--q", type=float, default=1.5, help="hinge loss q")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--lr-list", type=str, default="1",
                        help="comma-separated list of learning rates, e.g. 1,11,21")
    parser.add_argument("--iterations", type=int, default=6000, help="total iterations")
    parser.add_argument("--eval-interval", type=int, default=300, help="evaluation interval (iteration)")
    parser.add_argument("--batch-size", type=int, default=16, help="batch size")
    parser.add_argument("--n-pairs", type=int, default=1, help="number of neighboring dataset pairs (creates n_pairs+1 models)")
    parser.add_argument("--num-workers", type=int, default=1, help="number of workers")
    parser.add_argument("--num-workers-list", type=str, default="1",
                        help="comma-separated list of worker counts, e.g. 1,11,21")
    parser.add_argument("--train-type", type=str, default="fixed", choices=["fixed", "random"], help="training scheduling strategy")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help="device")
    parser.add_argument("--repeats", type=int, default=1, help="number of repeated runs (creates r1..rK directories)")
    parser.add_argument("--seed-base", type=int, default=0, help="initial random seed")
    parser.add_argument("--seed-step", type=int, default=10, help="seed increment per repeat")
    parser.add_argument("--log-root", type=str, default=".", help="log/model/record root directory")
    parser.add_argument("--save-prefix", type=str, default="", help="optional extra directory prefix, e.g. exp1_ (ignored if empty)")
    args = parser.parse_args()

    num_workers_list = [int(x) for x in args.num_workers_list.split(",") if x.strip()]
    lr_list = [float(x) for x in str(args.lr_list).split(",") if x.strip()]

    for rep in range(1, args.repeats + 1):
        current_seed = args.seed_base + (rep - 1) * args.seed_step
        torch.manual_seed(current_seed)
        np.random.seed(current_seed)

        sub_prefix = f"{args.save_prefix}" if args.save_prefix else ""
        base_dir = os.path.join(args.log_root, f"{sub_prefix}{args.dataset}", f"r{rep}")
        log_dir = os.path.join(base_dir, "logs")
        ckpt_dir = os.path.join(base_dir, "checkpoints")
        rec_dir = os.path.join(base_dir, "records")
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(rec_dir, exist_ok=True)

        for nworkers in num_workers_list:
        # for lr in lr_list:
            print(f"[Repeat {rep}/{args.repeats}] Workers={nworkers} Seed={current_seed}")
            # print(f"[Repeat {rep}/{args.repeats}] Workers={args.num_workers} Seed={current_seed} LR={lr}")
            server = Server(device=args.device,
                            train_type=args.train_type,
                            n_pairs=args.n_pairs,
                            num_workers=nworkers,   #args.num_workers,
                            batch_size=args.batch_size,
                            dataset_name=args.dataset,
                            dataset_path=args.dataset_path,
                            model_name=args.model,
                            loss_name=args.loss,
                            q=args.q,
                            lr=args.lr, #lr,
                            iterations=args.iterations,
                            evaluation_time=args.eval_interval,
                            random_seed=current_seed,
                            log_dir=log_dir,
                            checkpoint_dir=ckpt_dir,
                            rec_dir=rec_dir)
            server.train()
            del server
            gc.collect()
            if args.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
