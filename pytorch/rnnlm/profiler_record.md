## rnn

### base:
dataset : 6.684 ms  
dataload : 0.176 ms  
prepare : 767.281 ms  
train : 11505.280 ms  
test : 188.623 ms  
total : 12468.044 ms  

### non_blocking:
dataset : 6.722 ms  
dataload : 0.137 ms  
prepare : 767.180 ms  
train : 10862.478 ms  
test : 187.231 ms  
total : 11823.748 ms  

aten::to目测并没有变短  

### num_works=4:
dataset : 6.616 ms  
dataload : 0.159 ms  
prepare : 835.391 ms  
train : 10983.421 ms  
test : 337.802 ms  
total : 12163.389 ms  

出现了奇怪的现象。  
num_workers分别取值0,4,8，耗时不降反增。  
用官方测例实现一样的效果。  
观察官方测例(1_num_worker.py)的perf图，0时每次DataLoader耗时30+ms，4时反倒增加到100+ms,8时降低到5ms  
不理解为什么

### result:
模型准备耗时+66.9%  
模型训练耗时-37.0%  
显存：+138%  