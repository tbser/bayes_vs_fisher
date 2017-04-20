# bayes_vs_fisher
PatternRecognition-homework

实验一
(1) 有两类样本（如鲈鱼和鲑鱼），每个样本有两个特征（如长度和亮度），每类有若干个（比如20个）样本点，假设每类样本点服从二维正态分布，自己随机给出具体数据，计算每类数据的均值点，并且把两个均值点连成一线段，用垂直平分该线段的直线作为分类边界。再根据该分类边界对一随机给出的样本判别类别。画出图形。

提示：
1．可以如下产生第一类数据：
   % x是第一类数据，每一行代表一个样本（两个特征）
   x1(:,1) = normrnd(10,4,20,1);生成一组（20个）服从正态分布的随机数。
                               ；参数意义：第一、二参数分别表示均值及均方差，
                               ；第三、四参数表示生成的是20行1列的向量
   x1(:,2) = normrnd(12,4,20,1);
2．可假设分类边界为 kx-y+b=0，根据垂直平分的条件计算出k和b。
3．如果新的样本点代入分类边界方程的值的符号和第一类样本均值代入分类边界方程的符号相同，则是判断为第一类。


(2) 根据贝叶斯公式，给出在类条件概率密度为正态分布时具体的判别函数表达式，用此判别函数设计分类器。数据随机生成，比如生成两类样本（如鲈鱼和鲑鱼），每个样本有两个特征（如长度和亮度），每类有若干个（比如20个）样本点，假设每类样本点服从二维正态分布，随机生成具体数据，然后估计每类的均值与协方差，在两类协方差相同的情况下求出分类边界。先验概率自己给定，比如都为0.5。如果可能，画出在两类协方差不相同的情况下的分类边界。画出图形。

实验二
（1）实现fisher线性分类器，画出分类面。
（2）将fisher和bayes进行比较。
