# S3FD-Demo
> S3FD源码、运行Demo、TF权重、TF转PT权重  
> S3FD Source Code, Runable Demo, TF Weights, TF-to-PT Weight Conversion Tool  

## 1. 背景 / Background

在构思并验证某Demo过程中，从某基于TF框架的开源项目中接触到S3FD人脸检测模型，该模型的表现符合预期
项目需要基于PT框架，因此启动了该模型的跨框架迁移适配 
> - 将该模型的源代码重构为符合PyTorch框架规范的实现形式  
> - 在运行验证阶段发现，因TF与PT框架的权重存储格式存在差异，加载权重时出现异常。因此顺便写了权重转换工具（仅本模型，非通用），从TF权重读取、解析改写并生成 PT格式权重  
> - 模型调用通过，然后实现了配套的推理调用及后处理，形成一个对输入图片做人脸做框选的基础Demo  

最终该Demo在项目思路迭代后选用了其他模型方案，此部分代码暂被搁置；近期在代码梳理过程中，发现该迁移适配工作具备一定复用价值，遂将其整理为独立的轻量化项目


## 2. 功能 / Functions
- 一个S3FD人脸检测模型（基于PT的模型源码+TF/PT权重） 
- 一个运行模型并进行后处理的Demo（输入图片、模型推理、置信度/最小像素/NMS过滤、框选标记、呈现） 
- 一个TF权重转PT的工具代码  

## 3. 效果 / Effect Demo

### Demo0：少量人物的普通图片
#### 图片基本信息
内容：盗梦空间宣传图； 人物：6； jpeg格式； 分辨率：1920×1077  
图片特点：人物少，分辨率适中，清晰度低，覆盖人种、性别、正/侧脸  
#### 执行结果参考：
![demo0_result](https://github.com/iciferdai/S3FDTestDemo/blob/main/images/demo0_result.PNG)

##### 调用传参：
```python
detected_faces = extractor.extract(input_image,  
                                   is_resize=True,  
                                   min_pixel_threshold=10,  
                                   input_threshold=0.3,  
                                   confidence_threshold=0.9,  
                                   max_num=10)
```
##### 检测日志：
```log
Detect face-0: 878,87,980,221.2; score: 0.9998749494552612
Detect face-1: 1578,89,1672,216.6; score: 0.9998646974563599
Detect face-2: 214,136,284,237.2; score: 0.9973050355911255
Detect face-3: 1458,160,1506,231.5; score: 0.9919811487197876
Detect face-4: 1257,197,1316,279.5; score: 0.9869749546051025
Detect face-5: 482,129,533,203.8; score: 0.9824582934379578
```
#####  说明&参考：
人物少，图片质量达标，从日志结果看置信度均在0.98以上

---

### Demo1：老旧照片多人脸
#### 图片基本信息
内容：大逃杀的剧情合照图； 人物：41； jpg格式； 分辨率：4000×2250  
图片特点：人物多，分辨率高，清晰度中，多人正脸，老旧照片样式  

#### 执行结果参考：
![demo1_result](https://github.com/iciferdai/S3FDTestDemo/blob/main/images/demo1_result.PNG)

##### 调用传参-方案1：
```python
detected_faces = extractor.extract(input_image,  
                                   is_resize=True,  
                                   min_pixel_threshold=10,  
                                   input_threshold=0.3,  
                                   confidence_threshold=0.7,  
                                   max_num=100)
```
#####  检测日志-方案1：
```log
Detect face-0: 3330,1195,3436,1343.5; score: 0.9981608986854553
Detect face-1: 1807,961,1912,1099.6; score: 0.9978493452072144
Detect face-2: 2316,1204,2422,1344.8; score: 0.9974572062492371
...(略)...
Detect face-38: 620,1152,715,1295.0; score: 0.8323475122451782
Detect face-39: 289,1194,405,1348.0; score: 0.8157237768173218
Detect face-40: 2515,945,2613,1094.6; score: 0.7439660429954529
```
##### 调用传参-方案2：
```python
detected_faces = extractor.extract(input_image,  
                                   is_resize=False,  
                                   min_pixel_threshold=10,  
                                   input_threshold=0.3,  
                                   confidence_threshold=0.9,  
                                   max_num=100)
```
#####  检测日志-方案2：
```
Detect face-0: 3117,920,3216,1050.9; score: 1.0
Detect face-1: 1947,1132,2054,1277.2; score: 1.0
Detect face-2: 3335,1201,3435,1333.0; score: 0.9999998807907104
...(略)...
Detect face-38: 1176,939,1263,1058.9; score: 0.9999327659606934
Detect face-39: 1543,593,1628,705.2; score: 0.9999322891235352
Detect face-40: 3046,507,3136,626.9; score: 0.9999322891235352
```
#####   说明&参考：
is_resize控制处理过程中是否对图片压缩处理，主要用于图片分辨率过大的场景，开启后对最长的边（长或宽）执行：`scale_to = 640 if d >= 1280 else d / 2`但最小到64的操作  
可以看到正常对图片进行压缩处理后，是可以保证有效的人脸检测的，方案1中置信度最低依然达到了0.74  
但如果不执行缩放，方案2中最低的置信度都达到了4个9，可见分辨率对模型检测结果的影响，但相反过高的分辨率会带来模型推理计算量及时延的提升，需综合考量  

---

### Demo2：大量各国人物的模糊图
#### 图片基本信息
内容：世界球员大合照； 人物：100； jpeg格式； 分辨率：1280×640  
图片特点：人物极多，分辨率低，非常模糊，大量多国/不同肤色人物  

#### 执行结果参考：
![demo2_result](https://github.com/iciferdai/S3FDTestDemo/blob/main/images/demo2_result.PNG)

##### 调用传参：
```python
detected_faces = extractor.extract(input_image,  
                                   is_resize=False,  
                                   min_pixel_threshold=8,  
                                   input_threshold=0.1,  
                                   confidence_threshold=0.25,  
                                   max_num=200)
```
#####  检测日志：
```log
Detect face-0: 772,291,790,316.3; score: 0.9966832995414734
Detect face-1: 507,203,524,226.1; score: 0.9958066940307617
Detect face-2: 633,207,649,230.1; score: 0.9956745505332947
...(略)...
Detect face-97: 319,289,336,316.5; score: 0.9145740866661072
Detect face-98: 829,165,847,190.3; score: 0.6895750164985657
Detect face-99: 1135,288,1151,308.9; score: 0.2718046009540558
```
#####   说明&参考：
由于分辨率低，并且图片模糊，采用resize尝试会发现，将置信度阈值降到极低(0.1-0.2)，仍然会有部分人脸无法检出，并且由于置信度低，导致部分区域被错误检测为人脸；而关闭resize，可以较好的检出人脸，所以图片质量较差时，不要开启resize  
观察日志可发现，最后2张人脸呈现置信度快速下跌，face-97置信度依旧>0.9，但face-98快速下跌至0.69，而face-99甚至跌倒了0.27，所以如果不追求100%召回，可以不用将置信度阈值设置到很低，依旧可以达到非常好的效果  

## 4. 后继 / Future Plans
无

##  结束 / End
>:loudspeaker: Notice：  
>本项目为个人学习与实验性项目  
> This is personal learning and experimental project  
