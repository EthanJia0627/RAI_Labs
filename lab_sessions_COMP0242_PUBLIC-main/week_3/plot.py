from utils import *
import matplotlib.pyplot as plt
Q_table = [100,1000,10000]
R_table = [0.01,0.1,1]
Damping_table = [True,False]



# # Damping
# for Q_val in Q_table:
#     for R_val in R_table:
#         fig = []  # 用于存储每个关节的图
#         for i in range(7):
#             fig.append(plt.figure(figsize=(10, 8)))  # 初始化每个关节的图形对象
        
#         # 先绘制Damping为True和False的曲线
#         for Damping in Damping_table:
#             q_mes_all, qd_mes_all = get_data_with(Q_val, R_val, Damping)
#             Damping_str = "with Damping" if Damping else ""
            
#             for i in range(7):
#                 # 绘制位置曲线
#                 plt.figure(fig[i].number)  # 切换到对应关节的图形对象
#                 plt.subplot(2, 1, 1)
#                 plt.plot([q[i] for q in q_mes_all], label=f'Position - {Damping_str}')
#                 plt.title(f'Position Tracking for Joint {i+1} with Q:{Q_val}, R:{R_val}')
#                 plt.xlabel('Time steps')
#                 plt.ylabel('Position')
#                 plt.legend()

#                 # 绘制速度曲线
#                 plt.subplot(2, 1, 2)
#                 plt.plot([qd[i] for qd in qd_mes_all], label=f'Velocity - {Damping_str}')
#                 plt.title(f'Velocity Tracking for Joint {i+1} with Q:{Q_val}, R:{R_val}')
#                 plt.xlabel('Time steps')
#                 plt.ylabel('Velocity')
#                 plt.legend()

#         # 保存图像
#         for i in range(7):
#             fig[i].tight_layout()
#             figname = f'Joint_{i+1}_Q{Q_val}_R{R_val}_Damping_Comparison'
#             fig[i].savefig(savepath + '/Compare/Damping/' + figname + '.png')
#             plt.close(fig[i])  # 关闭图形对象，防止内存溢出


# # # Q

# for Damping in Damping_table:
#     Damping_str = "with Damping" if Damping else ""
#     for R_val in R_table:
#         fig = []  # 用于存储每个关节的图
#         for i in range(7):
#             fig.append(plt.figure(figsize=(10, 8)))  # 初始化每个关节的图形对象
#         for Q_val in Q_table:
            
        
#             q_mes_all,qd_mes_all = get_data_with(Q_val,R_val,Damping)
#             for i in range(7):
#                 plt.figure(fig[i].number) 
    
#                 # Position plot for joint i
#                 plt.subplot(2, 1, 1)
#                 plt.plot([q[i] for q in q_mes_all], label=f'Measured Position - Joint {i+1} Q: {Q_val}')
#                 plt.title(f'Q on Position Tracking for Joint {i+1} with R:{R_val}'+Damping_str)
#                 plt.xlabel('Time steps')
#                 plt.ylabel('Position')
#                 plt.legend()

                    
#                 # Velocity plot for joint i
#                 plt.subplot(2, 1, 2)
#                 plt.plot([qd[i] for qd in qd_mes_all], label=f'Measured Velocity - Joint {i+1} Q: {Q_val}')
#                 plt.title(f'Q on Velocity Tracking for Joint {i+1} with R:{R_val}'+Damping_str)
#                 plt.xlabel('Time steps')
#                 plt.ylabel('Velocity')
#                 plt.legend()
#         for i in range(7):
#             fig[i].tight_layout()
#             figname = f'Q Compare: Joint {i+1} with R:{R_val}'+Damping_str
#             plt.savefig(savepath+'/Compare/Q&R/Q/'+figname+'.png')
#             plt.close(fig[i])  # 关闭图形对象，防止内存溢出



# # # R

# for Damping in Damping_table:
#     Damping_str = "with Damping" if Damping else ""
#     for Q_val in Q_table:
#         fig = []  # 用于存储每个关节的图
#         for i in range(7):
#             fig.append(plt.figure(figsize=(10, 8)))  # 初始化每个关节的图形对象
        
#         for R_val in R_table:    
        
#             q_mes_all,qd_mes_all = get_data_with(Q_val,R_val,Damping)
#             for i in range(7):
#                 plt.figure(fig[i].number) 
    
#                 # Position plot for joint i
#                 plt.subplot(2, 1, 1)
#                 plt.plot([q[i] for q in q_mes_all], label=f'Measured Position - Joint {i+1} R: {R_val}')
#                 plt.title(f'R on Position Tracking for Joint {i+1} with Q:{Q_val}'+Damping_str)
#                 plt.xlabel('Time steps')
#                 plt.ylabel('Position')
#                 plt.legend()

                    
#                 # Velocity plot for joint i
#                 plt.subplot(2, 1, 2)
#                 plt.plot([qd[i] for qd in qd_mes_all], label=f'Measured Velocity - Joint {i+1} R: {R_val}')
#                 plt.title(f'R on Velocity Tracking for Joint {i+1} with Q:{Q_val}'+Damping_str)
#                 plt.xlabel('Time steps')
#                 plt.ylabel('Velocity')
#                 plt.legend()
#         for i in range(7):
#             fig[i].tight_layout()
#             figname = f'R Compare: Joint {i+1} with Q:{Q_val}'+Damping_str
#             plt.savefig(savepath+'/Compare/Q&R/R/'+figname+'.png')
#             plt.close(fig[i])  # 关闭图形对象，防止内存溢出
import matplotlib.pyplot as plt

Q_table = [100, 1000, 10000]  # Q值
R_table = [0.01, 0.1, 1]      # R值
Damping_table = [True, False]  # 有无Damping
color_table = ['b', 'g', 'r', 'c', 'm']  # 五种不同颜色对应不同Q/R比值

# 计算Q/R比值并生成颜色映射
QR_ratios = [(Q / R) for Q in Q_table for R in R_table]
QR_ratios_unique = sorted(set(QR_ratios))  # 确保唯一的Q/R比值
QR_to_color = {QR_ratios_unique[i]: color_table[i] for i in range(len(QR_ratios_unique))}

# 生成图像
for i in range(7):  # 对7个关节进行绘图
    for Damping in Damping_table:  # 对有无Damping分别绘制图像
        plt.figure(figsize=(10, 8))
        Damping_str = "with Damping" if Damping else ""
        
        # 用于标记每个Q/R比值是否已经绘制了legend
        legend_flags = {QR_ratio: False for QR_ratio in QR_ratios_unique}

        for Q_val in Q_table:
            for R_val in R_table:
                q_mes_all, qd_mes_all = get_data_with(Q_val, R_val, Damping)  # 获取数据
                QR_ratio = Q_val / R_val  # 计算Q/R比值
                color = QR_to_color[QR_ratio]  # 根据比值获取颜色

                # 如果该比值的legend还没有显示过，添加label；否则不添加
                label_str = f'Q/R={QR_ratio:.0f} ' + Damping_str if not legend_flags[QR_ratio] else None
                legend_flags[QR_ratio] = True  # 标记为已显示

                # 位置图
                plt.subplot(2, 1, 1)
                plt.plot([q[i] for q in q_mes_all], color=color, label=label_str)
                plt.title(f'Position Tracking for Joint {i+1} - Damping: {Damping_str}')
                plt.xlabel('Time steps')
                plt.ylabel('Position')
                plt.legend()
                
                # 速度图
                plt.subplot(2, 1, 2)
                plt.plot([qd[i] for qd in qd_mes_all], color=color, label=label_str)
                plt.title(f'Velocity Tracking for Joint {i+1} - Damping: {Damping_str}')
                plt.xlabel('Time steps')
                plt.ylabel('Velocity')
                plt.legend()
        
        plt.tight_layout()
        figname = f'Joint_{i+1}_{Damping_str}_Q:R_Compare'
        plt.savefig(savepath + '/Compare/Q:R/' + figname + '.png')
        plt.clf()  # 清理图像以准备下一个
