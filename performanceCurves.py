import matplotlib.pyplot as plt
import numpy as np

################ LlaVa - NEXT
nan = np.nan
weightVec = [2, 3, 4, 6, 8, 16]

numQuantType = 3
numActQuant = 3
numWeightQuant = 6

markersize=12
linewidth = 3
colorVec = ['green' , 'blue' , 'red']           # denotes awq vs sq vs naive
lineStyleVec = ['solid' , 'dashed' , 'dotted']  # denotes activation quantization
markerVec = ['o', 's', 'v']                     # denotes activation quantization

act_names = ['fp16', 'int8', 'int4']
quant_names = ['_awq', '_sq', '_naive']

## CIDER
fp16_awq = [0, 1.0696, 1.1598, 1.1813, 1.1759, 1.1709]
int8_awq = [nan, 1.0907, 1.1532, 1.1721, 1.1755, 1.1755]
int4_awq = [nan, 0.0444, 0.0328, 0.0645, 0.0666, 0.0666]

fp16_sq = [nan, nan, nan, nan, 1.177, nan]
int8_sq = [nan, nan, nan, nan, 1.1563, nan]
int4_sq = [nan, nan, nan, nan, 0.0476, nan]

fp16_naive = [nan, nan, nan, nan, nan, nan]
int8_naive = [nan, nan, 1.1152, 1.1825, 1.1534, nan]
int4_naive = [nan, nan, nan, nan, 0.270, nan]

for ind_ActQuant in range(numActQuant):
    currLineStyle = lineStyleVec[ind_ActQuant]
    currActStr = act_names[ind_ActQuant]
    for ind_QuantType in range(numQuantType):
        currMarker = markerVec[ind_QuantType]
        currColor = colorVec[ind_QuantType]
        currStr = currActStr + quant_names[ind_QuantType]
        plt.plot(weightVec, eval(currStr), color=currColor, linestyle=currLineStyle, linewidth = linewidth,
            marker=currMarker, markerfacecolor=currColor, markersize=markersize)

plt.ylim(0,2)
plt.xlim(1,20)
plt.xlabel('Weight bit width')
plt.ylabel('CIDER')
plt.title('CIDER!')
plt.show()

## VQAv2
fp16_awq = [0, 0.7526, 0.762, 0.7658, 0.7652, 0.764]
int8_awq = [nan, 0.7502, 0.7652, 0.7654, 0.7666, 0.7666]
int4_awq = [nan, 0.1042, 0.209, 0.3244, 0.3333, 0.3333]

fp16_sq = [nan, nan, nan, nan, 0.7648, nan]
int8_sq = [nan, nan, nan, nan, 0.761, nan]
int4_sq = [nan, nan, nan, nan, 0.3048, nan]

fp16_naive = [nan, nan, nan, nan, nan, nan]
int8_naive = [nan, nan, 0.7630, 0.755, 0.768, nan]
int4_naive = [nan, nan, nan, nan, 0.606, nan]

for ind_ActQuant in range(numActQuant):
    currLineStyle = lineStyleVec[ind_ActQuant]
    currActStr = act_names[ind_ActQuant]
    for ind_QuantType in range(numQuantType):
        currMarker = markerVec[ind_QuantType]
        currColor = colorVec[ind_QuantType]
        currStr = currActStr + quant_names[ind_QuantType]
        plt.plot(weightVec, eval(currStr), color=currColor, linestyle=currLineStyle, linewidth = linewidth,
            marker=currMarker, markerfacecolor=currColor, markersize=markersize)

plt.ylim(0,2)
plt.xlim(1,20)
plt.xlabel('Weight bit width')
plt.ylabel('VQAv2')
plt.title('VQAv2!')
plt.show()

################ OPT
