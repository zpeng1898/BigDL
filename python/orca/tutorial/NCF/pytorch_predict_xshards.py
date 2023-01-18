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
import json
import torch.nn as nn
import torch.optim as optim

from pytorch_model import NCF

from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.learn.pytorch import Estimator
from bigdl.orca.data import XShards


# Step 1: Init Orca Context
sc = init_orca_context(cluster_mode="local")


# Step 2: Load the processed data
data = XShards().load_pickle("test_processed_xshards")


# Step 3: Define the model
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
    return model


# Step 4: Create Orca PyTorch Estimator and load the model
backend = "spark"  # "ray" or "spark"
with open("config.json", "r") as f:
    config = json.load(f)

est = Estimator.from_torch(model=model_creator,
                           backend=backend,
                           config=config)
est.load("NCF_model")


# Step 5: Distributed inference of the loaded model
predictions = est.predict(data=data,
                          feature_cols=config["feature_cols"],
                          batch_size=10240)
print("Prediction results of the first 5 rows:")
print(predictions.head(n=5))


# Step 6: Save the prediction results
predictions.save_pickle("./test_predictions_xshards")


# Step 7: Stop Orca Context when program finishes
stop_orca_context()
