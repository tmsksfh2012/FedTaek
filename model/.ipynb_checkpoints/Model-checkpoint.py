import torch
from torch import nn
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import syft as sy
import numpy as np

from data.Data import DataAssetEntity, DataSetEntity, DataHelper


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = nn.functional.log_softmax(self.fc3(x), dim=1)
        return x

@sy.syft_function()
def local_update(data, labels, current_params):
    # """
    # 로컬 업데이트 함수: logistic regression 예제
    # """
    import numpy as np
    X = data.to_numpy()
    y = labels.to_numpy().ravel()

    W, b = current_params
    # forward
    logits = X @ W + b
    exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)

    num_samples = X.shape[0]
    y_onehot = np.zeros_like(probs)
    y_onehot[np.arange(num_samples), y] = 1

    grad_logits = (probs - y_onehot) / num_samples
    grad_W = X.T @ grad_logits
    grad_b = np.sum(grad_logits, axis=0)

    lr = 0.01
    W_new = W - lr * grad_W
    b_new = b - lr * grad_b

    return [W_new, b_new]

@sy.syft_function(
    output_policy=sy.SingleExecutionExactOutput()  # 혹은 sy.ExactOutput()
)
def aggregate_params(param_list: list):
    import numpy as np
    n = len(param_list)
    W_acc, b_acc = None
    for i, (W,b) in enumerate(param_list):
        if i == 0:
            W_acc = W.copy()
            b_acc = b.copy()
        else:
            W_acc += W
            b_acc += b
    W_avg = W_acc / n
    b_avg = b_acc / n
    return [W_avg, b_avg]

class ModelHelper:
    def __init__(self, data_helper: DataHelper):
        self.data_helper = data_helper
        
    def load_MNIST(self, num_samples=20000):
        import pandas as pd
        # load MNIST dataset
        BATCH_SIZE = 32

        mnist = fetch_openml('mnist_784', version=1)
        X_full = mnist.data.values.astype(np.float32)
        y_full = mnist.target.astype(np.int64).values
        # 단순히 0~1 스케일링
        X_full /= 255.0

        X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=10000, random_state=42)
        X_train = X_train[:20000]  # 데모용 일부 데이터만 사용
        y_train = y_train[:20000]

        # mock 데이터 생성: distribution을 비슷하게 유지하면서 무작위 perturbation
        X_mock = X_train + np.random.normal(0,0.05,X_train.shape)
        X_mock = np.clip(X_mock,0,1)
        y_mock = np.random.randint(0,10,size=y_train.shape)

        # pandas로 변환
        X_train_df = pd.DataFrame(X_train)
        y_train_df = pd.DataFrame(y_train, columns=["label"])
        X_mock_df = pd.DataFrame(X_mock)
        y_mock_df = pd.DataFrame(y_mock, columns=["label"])

        asset_X = DataAssetEntity(name="mnist_features", data=X_train_df, mock=X_mock_df)
        asset_y = DataAssetEntity(name="mnist_labels", data=y_train_df, mock=y_mock_df)
        mnist_dataset_entity = DataSetEntity(
                name="MNIST Dataset",
                description="MNIST digits dataset for federated learning demo",
                asset_list=[asset_X, asset_y]
        )

        self.data_helper.upload_dataset(dataset_entity=mnist_dataset_entity)
        
        # ds = self.data_helper.get_dataset_by_name(dataset_name = "MNIST Dataset")
        # self.f_X, self.f_y = ds.assets

    # def fl_training(self, admin, num_cl):
    #     W_init = np.zeros((784,10), dtype=np.float32)
    #     b_init = np.zeros((10,), dtype=np.float32)
    #     init_params = [W_init, b_init]

    #     # 데이터 파티션 나누기 (여기선 2개 클라이언트 가정)
    #     X_np = self.X_mock_df.to_numpy()
    #     num_samples = X_np.shape[0]
    #     indices = np.arange(num_samples)
    #     np.random.shuffle(indices)
    #     parts = np.array_split(indices, num_cl)

    #     jobs = []
    #     for part_idx in parts:
    #         # mock 기반: 실제로는 승인 후 data=f_X.data로도 가능
    #         # 여기선 data=f_X, labels=f_y를 그대로 넘기면 서버에서 mock/data 선택
    #         job = admin.code.local_update(
    #             data=self.f_X, 
    #             labels=self.f_y,
    #             current_params=init_params,
    #             blocking=False
    #         )
    #         jobs.append(job)

    #     client_results = []
    #     for job in jobs:
    #         client_results.append(job.wait().get())

    def fl_training(self, admin, num_cl):
        # f_X, f_y는 load_MNIST 이후 접근 가능하다고 가정
        ds = self.data_helper.get_dataset_by_name("MNIST Dataset")
        f_X, f_y = ds.assets

        # local_update, aggregate_params 함수를 서버에 제출
        admin.code.submit(local_update)
        admin.code.submit(aggregate_params)

        # 초기 파라미터
        W_init = np.zeros((784,10), dtype=np.float32)
        b_init = np.zeros((10,), dtype=np.float32)
        init_params = [W_init, b_init]

        # federated_training_round 함수 정의
        # 여기서 data와 labels를 num_cl개로 나누어 각 파티션마다 local_update job을 실행하고,
        # 결과를 aggregate_params job에 넘겨 평균 파라미터를 얻음.
        @sy.syft_function_single_use(data=f_X, labels=f_y)
        def federated_training_round(datasite, data, labels, init_params, num_cl: int):
            import numpy as np
            data_parts = np.array_split(data.to_numpy(), num_cl)
            label_parts = np.array_split(labels.to_numpy(), num_cl)

            job_results = []
            for i in range(num_cl):
                batch_data = data_parts[i]
                batch_labels = label_parts[i]
                # local_update job 실행
                batch_job = datasite.launch_job(local_update, data=batch_data, labels=batch_labels, current_params=init_params)
                job_results.append(batch_job.result)

            # 모든 부분 결과를 aggregate_params에 전달
            agg_job = datasite.launch_job(aggregate_params, param_list=job_results)
            return agg_job.result

        # federated_training_round 함수 서버에 제출
        admin.code.submit(federated_training_round)

        # federated_training_round 코드 요청
        # 승인이 필요하다면 이후 approve 후 실행 가능
        # request = admin.code.request_code_execution(federated_training_round, init_params=init_params, num_cl=num_cl)
        # 함수 실행 요청 (승인 필요)
        request = admin.code.request_code_execution(federated_training_round, reason="Federated training round")

        # 실제 실행(승인 이후)
        # blocking 실행 예제
        final_params = admin.code.federated_training_round(data=f_X, labels=f_y, init_params=init_params, num_cl=num_cl).get()

        return final_params