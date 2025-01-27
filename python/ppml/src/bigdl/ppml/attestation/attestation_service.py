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

import requests
import json
import base64
from collections import OrderedDict
use_secure_cert = False
headers = {"Content-Type":"application/json"}

def bigdl_attestation_service(base_url, app_id, api_key, quote, policy_id):
    payload = OrderedDict()
    payload["appID"] = app_id
    payload["apiKey"] = api_key
    payload["quote"] = base64.b64encode(quote).decode()
    if len(policy_id) > 0:
        payload["policyID"] = policy_id
    try:
        resp = requests.post(url="https://" + base_url + "/verifyQuote", data=json.dumps(payload), headers=headers, verify=use_secure_cert)
        resp_dict = json.loads(resp.text)
        result = resp_dict["result"]
    except (json.JSONDecodeError, KeyError):
        result = -1
    return result
