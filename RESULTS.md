# Text Classification Run

本次运行使用 CPU 完成中文评论 8 分类任务。

## Training Configuration

![Training configuration](assets/training_config.png)

- Model: `uer/chinese_roberta_L-2_H-128`
- Device: `cpu`
- Epochs: `5`
- Batch size: `8`
- Number of labels: `8`

## Evaluation Result

![Evaluation result](assets/evaluation_result.png)

第 4 轮第 200 步取得最佳验证结果：precision `0.49`、recall `0.48`、F1 `0.45`。

最终评估结果为：precision `0.45`、recall `0.46`、F1 `0.42`。

最佳模型保存在 `checkpoints/comment_classify/model_best`。
