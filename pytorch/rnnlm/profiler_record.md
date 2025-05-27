## transformer

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

### result:
模型准备耗时+66.9%  
模型训练耗时-37.0%  
显存：+138%  