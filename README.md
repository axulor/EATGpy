#### 这是一篇论文的仿真复现程序
Lin, Z. Q., et al. (2020). "Evolutionary accumulated temptation game on small world networks." Physica a-Statistical Mechanics and Its Applications 553.

“小世界网络上的累积诱惑博弈”

##### 1. 理论知识

这是一个两人两策略博弈，收益矩阵为
$$
\left(\begin{array}{cc}1 & 0 \\ b_{i}^{t} & 0\end{array}\right)
$$

其中，
$$
b_{i}^{t+1}=\left\{\begin{array}{ll}b_{i}^{t} \times \bar{d}_{n_{i}, t}^{\alpha}, & s_{i}^{t}=C \\ b_{i}^{t} / \bar{d}_{n_{i}, t}^{\alpha}, & s_{i}^{t}=D\end{array}\right.
，
\bar{d}_{n_{i}, t}=\frac{1}{\left|\Omega_{i, t}^{c}\right|} \sum_{j \in \Omega_{i, t}^{c}} d_{j}
$$

诱惑参数将会随时间而变化，这会对稳态时合作者密度产生影响

##### 2. 仿真工作
- 参数设置  
  - WS小世界网络（节点数量为1000，平均度为4，断边重连率为0.5）
  - 初始一半为合作者，一半为背叛者，累积因子Alpha默认设置为 0.3
  - 初始诱惑参数为 1
- 仿真：平均诱惑的变化趋势
  - 记录多轮重复博弈后，网络中平均诱惑参数的变化趋势，并分为合作者的诱惑参数和背叛者的诱惑参数两部分
  - 文献中指出：平均诱惑参数总体加速上升，对合作者而言与总体增长趋势一致，对背叛者而言先呈下降趋势，再上升并维持在个位数值，之后长期趋于稳定 

##### 3. 文件类型
agent.py    智能体类

simulation.py  仿真类

main.py        仿真参数设置，设置好后直接运行此文件

test.py        测试程序，与该工作无关

score.csv          用于保存每一轮的数据：依次为节点下一轮诱惑参数，当前收益，当前策略

