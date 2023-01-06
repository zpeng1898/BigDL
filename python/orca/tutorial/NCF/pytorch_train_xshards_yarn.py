#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Step 0: Import necessary libraries
import os
import argparse
import torch.nn as nn
import torch.optim as optim

from process_xshards import prepare_data
from pytorch_model import NCF

from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.learn.pytorch import Estimator
from bigdl.orca.learn.pytorch.callbacks.tensorboard import TensorBoardCallback
from bigdl.orca.learn.metrics import Accuracy, Precision, Recall

parser = argparse.ArgumentParser(description="PyTorch Example")
parser.add_argument("--remote_dir", type=str, help="The path to load data from remote resources")
parser.add_argument("--cluster_mode", type=str, default="yarn-client",
                    help="The cluster mode, such as local, yarn-client, yarn-cluster, "
                         "k8s-client, k8s-cluster, spark-submit or bigdl-submit.")
parser.add_argument("--backend", type=str, default="spark", help="ray or spark")
parser.add_argument("--callback", type=bool, default=True,
                    help="Whether to use TensorBoardCallback.")
parser.add_argument("--workers_per_node", type=int, default=1,
                    help="The number of PyTorch workers on each node.")
args = parser.parse_args()


# Step 1: Init Orca Context
if args.cluster_mode == "local":
    sc = init_orca_context(cluster_mode="local", memory="12g")
elif args.cluster_mode.startswith("yarn"):
    if args.cluster_mode == "yarn-client":
        sc = init_orca_context(cluster_mode="yarn-client", cores=4, memory="10g", num_nodes=2,
                               driver_cores=2, driver_memory="2g",
                               extra_python_lib="pytorch_model.py,process_xshards.py")
    elif args.cluster_mode == "yarn-cluster":
        sc = init_orca_context(cluster_mode="yarn-cluster", cores=4, memory="10g", num_nodes=2,
                               driver_cores=2, driver_memory="2g",
                               extra_python_lib="pytorch_model.py,process_xshards.py")
elif args.cluster_mode.startswith("k8s"):
    if args.cluster_mode == "k8s-client":
        conf = {
            "spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim"
            ".options.claimName": "nfsvolumeclaim",
            "spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim"
            ".mount.path": "/bigdl/nfsdata"
        }
        sc = init_orca_context(cluster_mode="k8s-client", num_nodes=2, cores=4, memory="10g",
                               driver_cores=2, driver_memory="2g",
                               master=os.environ.get("RUNTIME_SPARK_MASTER"),
                               container_image=os.environ.get("RUNTIME_K8S_SPARK_IMAGE"),
                               extra_python_lib="pytorch_model.py,process_xshards.py",
                               conf=conf)
    elif args.cluster_mode == "k8s-cluster":
        conf = {
            "spark.kubernetes.driver.volumes.persistentVolumeClaim.nfsvolumeclaim"
            ".options.claimName": "nfsvolumeclaim",
            "spark.kubernetes.driver.volumes.persistentVolumeClaim.nfsvolumeclaim"
            ".mount.path": "/bigdl/nfsdata",
            "spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim"
            ".options.claimName": "nfsvolumeclaim",
            "spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim"
            ".mount.path": "/bigdl/nfsdata",
            "spark.kubernetes.authenticate.driver.serviceAccountName": "spark",
            "spark.kubernetes.file.upload.path": "/bigdl/nfsdata/"
        }
        sc = init_orca_context(cluster_mode="k8s-cluster", num_nodes=2, cores=4, memory="10g",
                               driver_cores=2, driver_memory="2g",
                               master=os.environ.get("RUNTIME_SPARK_MASTER"),
                               container_image=os.environ.get("RUNTIME_K8S_SPARK_IMAGE"),
                               penv_archive="file:///bigdl/nfsdata/environment.tar.gz",
                               extra_python_lib="pytorch_model.py,process_xshards.py",
                               conf=conf)
elif args.cluster_mode == "bigdl-submit":
    sc = init_orca_context(cluster_mode="bigdl-submit")
elif args.cluster_mode == "spark-submit":
    sc = init_orca_context(cluster_mode="spark-submit")
else:
    print("init_orca_context failed. cluster_mode should be one of 'yarn-client', "
          "'yarn-cluster', 'k8s-client', 'k8s-cluster', 'bigdl-submit' or 'spark-submit', "
          "but got " + args.cluster_mode)


# Step 2: Read and process data using Orca XShards
train_data, test_data, user_num, item_num, sparse_feats_input_dims, num_dense_feats, \
    feature_cols, label_cols = prepare_data(args.remote_dir, num_ng=4)


# Step 3: Define the model, optimizer and loss
def model_creator(config):
    model = NCF(user_num=config["user_num"],
                item_num=config["item_num"],
                factor_num=config["factor_num"],
                num_layers=config["num_layers"],
                dropout=config["dropout"],
                model=config["model"],
                sparse_feats_input_dims=config["sparse_feats_input_dims"],
                sparse_feats_embed_dims=config["sparse_feats_embed_dims"],
                num_dense_feats=config["num_dense_feats"])
    model.train()
    return model


def optimizer_creator(model, config):
    return optim.Adam(model.parameters(), lr=config["lr"])

loss = nn.BCEWithLogitsLoss()


# Step 4: Distributed training with Orca PyTorch Estimator
callbacks = [TensorBoardCallback(log_dir="runs", freq=1000)] if args.callback else []

est = Estimator.from_torch(model=model_creator,
                           optimizer=optimizer_creator,
                           loss=loss,
                           metrics=[Accuracy(), Precision(), Recall()],
                           backend=args.backend,
                           use_tqdm=True,
                           workers_per_node=args.workers_per_node,
                           config={"user_num": user_num,
                                   "item_num": item_num,
                                   "factor_num": 16,
                                   "num_layers": 3,
                                   "dropout": 0.5,
                                   "lr": 0.001,
                                   "model": "NeuMF-end",
                                   "sparse_feats_input_dims": sparse_feats_input_dims,
                                   "sparse_feats_embed_dims": 8,
                                   "num_dense_feats": num_dense_feats})
est.fit(data=train_data, epochs=2,
        feature_cols=feature_cols,
        label_cols=label_cols,
        batch_size=10240,
        callbacks=callbacks)


# Step 5: Distributed evaluation of the trained model
result = est.evaluate(data=test_data,
                      feature_cols=feature_cols,
                      label_cols=label_cols,
                      batch_size=10240)
print("Evaluation results:")
for r in result:
    print("{}: {}".format(r, result[r]))


# Step 6: Save the trained pytorch model and processed data for resuming training or prediction
est.save("NCF_model")

train_path = os.path.join(args.remote_dir, "train_xshards")
test_path = os.path.join(args.remote_dir, "test_xshards")
train_data.save_pickle(train_path)
test_data.save_pickle(test_path)


# Step 7: Stop Orca Context when program finishes
stop_orca_context()