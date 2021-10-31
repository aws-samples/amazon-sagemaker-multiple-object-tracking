# Train and Deploy Multiple Object Tracking Model with Amazon SageMaker
---

[Multiple Object Tracking](https://motchallenge.net/) or MOT estimates a bounding box and ID for each pre-defined object in videos or consecutive frames, which has been used in tasks such as live sports, manufacturing, surveillance, and traffic monitoring. In the past, the high latency caused by the limitation of hardware and complexity of ML-based tracking algorithm is a major obstacle for its application in the industry. The state-of art algorithm [FairMOT](https://arxiv.org/abs/2004.01888) has reached the speed of about 30FPS on the [MOT challenge datasets](https://motchallenge.net/), which helps MOT find its way in many industrial scenarios.

This post shows how to train and deploy a state-of-art MOT algorithm [FairMOT](https://github.com/ifzhang/FairMOT) model with Amazon SageMaker.

<div align="center"><img width=600 src="./img/mot_sample.gif"></div>

## Prerequisites
- [Create an AWS account](https://aws.amazon.com/premiumsupport/knowledge-center/create-and-activate-aws-account/) or use the existing AWS account.
- Make sure that you have a minimum of one `ml.p3.16xlarge` instance for the Training Job. If it is the first time you train a machine learning model on `ml.p3.16xlarge`, you will need to [request a service quota increase for SageMaker training job]( https://docs.aws.amazon.com/sagemaker/latest/dg/regions-quotas.html).
- [Create a SageMaker Notebook instance](https://docs.aws.amazon.com/sagemaker/latest/dg/howitworks-create-ws.html). The default volume size is 5GB, you need to update the Volume Size to 100GB. For IAM role, choose the existing IAM role or create a new IAM role, attach the policy of AmazonSageMakerFullAccess and AmazonElasticContainerRegistryPublicFullAccess to the chosen IAM role.
- Make sure that you have a minimum of one `ml.p3.2xlarge` instance for Infenrece endpoint. If it is the first time you deploy a machine learning model on `ml.p3.2xlarge`, you will need to [request a service quota increase for SageMaker training job]( https://docs.aws.amazon.com/sagemaker/latest/dg/regions-quotas.html).
- Make sure that you have a minimum of one `ml.p3.2xlarge` instance for Processing jobs. If it is the first time you run a processing job on `ml.p3.2xlarge`, you will need to [request a service quota increase for SageMaker training job]( https://docs.aws.amazon.com/sagemaker/latest/dg/regions-quotas.html).
- The region `us-east-1` is recommended.

## Running Costs
- one SageMaker notebook on `ml.t3.medium` in us-east-1 region: `$0.05` per hour, ([250 hours free Tier usage per month for the first 2 months](https://aws.amazon.com/sagemaker/pricing/))
- one training job on `ml.p3.16xlarge` in us-east-1 region takes 3 hours, total training cost for each training job:`$85`
- one endpoint on `ml.p3.2xlarge` in us-east-1 region: `$3.825` per hour
- one SageMaker processing job on `ml.p3.2xlarge` in us-east-1 region: `$3.825`  per hour
- Assuming you run **one training job** with the default parameters, test the real time inference and batch inference with the default test data, and delete inference endpoint once finishing test, totally it costs less than `$95`.

## Training
---
To tune hyperparameters with Amazon SageMaker Hyperparameter Jobs, we modified the original training script to validate the model during training and set the validation loss as the objective metric. Currently our project only supports model training on a single instance.

Open [`fairmot-training.ipynb`](fairmot-training.ipynb) and run the cells step by step. It will take 3 hours to complete one training job. When performing hyperparameter tuning job, total running time will be about:
$$
\begin{align}
N_{traning}: \text{Maximum total number of training jobs} \\
N_{parallel}: \text{Maximum number of parallel training jobs} \\
T_{one\space training}: \text{time on one training job, 3 hours in this case} \\
T_{total}: \text{total running time} \\
T_{total} = ( N_{traning}\times T_{one\space training} ) / N_{parallel}
\end{align}
$$

## Serving
---
We provide two ways of deploying the trained model: real time inference with endpoint and batch inference.
- To deploy a real time endpoint, open [`fairmot-inference.ipynb`](fairmot-inference.ipynb) and run the cells step by step.
- To run batch inference, open [`fairmot-batch-inference.ipynb`](fairmot-batch-inference.ipynb) and run the cells step by step.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License
---
This library is licensed under the MIT-0 License. See the LICENSE file.
