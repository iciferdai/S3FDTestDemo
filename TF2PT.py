import numpy as np
import torch

def convert_tensorflow_to_pytorch(tf_weight, weight_name):
    if 'weight' in weight_name:
        if len(tf_weight.shape) == 4:  # 卷积层权重
            return torch.from_numpy(tf_weight.transpose(3, 2, 0, 1)).float()
        elif len(tf_weight.shape) == 2:  # 全连接层权重
            return torch.from_numpy(tf_weight.transpose(1, 0)).float()
        else:  # 其他权重
            return torch.from_numpy(tf_weight).float()
    elif 'bias' in weight_name:
        # 对偏置进行形状调整
        return torch.from_numpy(tf_weight.squeeze()).float()
    else:
        # 对其他可能的权重（如 L2Norm 的权重）进行处理
        return torch.from_numpy(tf_weight.squeeze()).float()

class TensorFlowToPyTorchConverter:
    def __init__(self, tf_weights_path, output_path):
        self.tf_weights_path = tf_weights_path
        self.output_path = output_path

    def convert_and_save(self):
        # 加载 TensorFlow 权重文件
        tf_weights = np.load(self.tf_weights_path, allow_pickle=True)
        pytorch_weights = {}

        # 遍历 TensorFlow 权重字典
        for key in tf_weights:
            # 获取 TensorFlow 权重
            tf_weight = tf_weights[key]
            print(f"convert tf_weights key: {key}, weight shape: {tf_weight.shape}")
            # 转换权重
            pt_weight = convert_tensorflow_to_pytorch(tf_weight, key)

            # 提取层名称和权重类型
            layer_name, weight_type = key.split('/') if '/' in key else (key, None)
            weight_type = weight_type.split(':')[0] if ':' in weight_type else weight_type

            # 构造 PyTorch 权重字典的键名
            if 'weight' in weight_type:
                pt_key = f"{layer_name}.weight"
            elif 'bias' in weight_type:
                pt_key = f"{layer_name}.bias"
            else:
                continue  # 跳过无法识别的权重

            # 对 L2Norm 层的权重进行特殊处理
            if 'norm' in layer_name and 'weight' in weight_type:
                pt_weight = pt_weight.squeeze()  # 将四维权重转换为一维

            pytorch_weights[pt_key] = pt_weight

        # 保存转换后的权重到文件
        torch.save(pytorch_weights, self.output_path)
        print(f"Save to path: {self.output_path}")


if __name__ == "__main__":
    # 使用示例
    converter = TensorFlowToPyTorchConverter(
        tf_weights_path="./S3FD.npy",
        output_path="./S3FD_pytorch.pth"
    )
    converter.convert_and_save()