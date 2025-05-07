import matplotlib.pyplot as plt

# Data for Detection mAP vs Ego Distance
x = [10, 20, 30, 40, 50]
y_r50 = [0.6724, 0.5526, 0.3622, 0.1559, 0.0508]
y_r101 = [0.7060, 0.5748, 0.4021, 0.1907, 0.0701]
y_v99 = [0.6939, 0.6002, 0.4200, 0.1971, 0.0780]
labels = ['R50', 'R101', 'V99']
x_label = 'Ego Distance (m)'
y_label = 'Detection mAP'

# Define consistent colors
colors = {'R50': 'tab:blue', 'R101': 'tab:orange', 'V99': 'tab:green'}

# Plot Detection mAP
plt.figure(figsize=(8, 5))
plt.plot(x, y_r50, marker='o', linestyle='-', color=colors['R50'], label=labels[0])
plt.plot(x, y_r101, marker='o', linestyle='--', color=colors['R101'], label=labels[1])
plt.plot(x, y_v99, marker='o', linestyle='--', color=colors['V99'], label=labels[2])

plt.xlabel(x_label, fontsize=14)
plt.ylabel(y_label, fontsize=14)
plt.title('Detection Performance vs Ego Distance', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Data for NDS and AR vs Visibility Levels
visibility_levels = ['0-40', '40-60', '60-80', '80-100']
nds_r50 = [0.5032, 0.5006, 0.5117, 0.6990]
nds_r101 = [0.5032, 0.5006, 0.5131, 0.7072]
nds_v99 = [0.5036, 0.5005, 0.5163, 0.7116]
ar_r50 = [0.4040, 0.4555, 0.4505, 0.4680]
ar_r101 = [0.4325, 0.4600, 0.4760, 0.4745]
ar_v99 = [0.4130, 0.4530, 0.4565, 0.4735]

# Plot NDS and AR vs Visibility Levels
plt.figure(figsize=(8, 5))
plt.plot(visibility_levels, nds_r50, marker='o', linestyle='-', color=colors['R50'], label='NDS R50')
plt.plot(visibility_levels, nds_r101, marker='o', linestyle='--', color=colors['R101'], label='NDS R101')
plt.plot(visibility_levels, nds_v99, marker='o', linestyle='--', color=colors['V99'], label='NDS V99')
plt.plot(visibility_levels, ar_r50, marker='s', linestyle='-', color=colors['R50'], label='AR R50')
plt.plot(visibility_levels, ar_r101, marker='s', linestyle='--', color=colors['R101'], label='AR R101')
plt.plot(visibility_levels, ar_v99, marker='s', linestyle='--', color=colors['V99'], label='AR V99')

plt.xlabel('Visibility Levels (%)', fontsize=14)
plt.ylabel('Detection Metric', fontsize=14)
plt.title('Detection Performance vs Visibility', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.gca().invert_xaxis()
plt.show()
