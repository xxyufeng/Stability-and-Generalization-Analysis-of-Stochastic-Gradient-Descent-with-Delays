# Stability-and-Generalization-Analysis-of-Stochastic-Gradient-Descent-with-Delays
This repository is the implementation of the section numerical experiments of paper "Stability and Generalization Analysis of Stochastic Gradient Descent with Delays."

## Requirement
```
torch >= 2.5.1
torch >= 0.20.1
numpy >= 2.0.2
```

## Usage
Collected typical experiment configurations (reproducible via CLI):
1. MNIST (fcnet_mnist, CE)
   ```
   python AsynchronousSGD.py --dataset mnist --model fcnet_mnist --loss ce --lr 5e-3 
                             --iterations 20000 --eval-interval 200 --n-pairs 5 --batch-size 64
                             --num-workers-list 1,11,21,31 --repeats 5 --seed-base 1 --seed-step 10 --device cuda
   ```
2. CIFAR10 (resnet18_cifar10, CE)
   ```
   python AsynchronousSGD.py --dataset cifar10 --model resnet18_cifar10 --loss ce --lr 2e-2
                             --iterations 60000 --eval-interval 1200 --n-pairs 5 --batch-size 64
                             --num-workers-list 1,11,21,31 --repeats 5 --seed-base 0 --seed-step 10 --device cuda
   ```

## Main arguments
```
      --dataset            One of {rcv1, gisette, ijcnn, w1a, mnist, cifar10}
      --model              Model name matching the dataset
      --loss               mse | hingeloss | ce
      --n-pairs            Number of neighboring dataset pairs (creates n_pairs+1 models)
      --num-workers-list   Comma-separated list of worker counts (delay factor)
      --repeats            Number of repeated runs (creates r1..rK)
      --seed-base          Base random seed
      --seed-step          Seed increment per repeat
      --eval-interval      Evaluation interval (iterations)
      --device             cpu | cuda
      --train-type         fixed | random (scheduling strategy)
```

## Notes
1. Fixed delay = num_worker - 1
   
     Local workers communicate with server sequentially, and the server conduct model update once it receive the gradients calculated by local workers. Therefore, the delay is fixed as "num_worker -1".
   
2. Each repeat uses seed = seed_base + (repeat_index - 1) * seed_step.
3. Output folders:
   logs/          text metric logs
   checkpoints/   final model (only when last iteration reached)
   records/       tensor (.pt) metric arrays
4. Stability metric = average L2 distance across parameters between model 0 and each neighboring model.
 
