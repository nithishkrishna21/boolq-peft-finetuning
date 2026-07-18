import matplotlib.pyplot as plt
import numpy as np

# Example data: replace these with your actual values
learning_rates = ['1e-4', '5e-4', '1e-3']
epochs = [7, 9]

# Dev accuracies for Model 1 and Model 2
model1_dev_accuracies = np.array([[0.7042813455657493, 0.6217125382262997, 0.6217125382262997],  # Model 1, dev accuracy, epoch 7
                                  [0.7110091743119266, 0.6446483180428134, 0.6217125382262997]])  # Model 1, dev accuracy, epoch 9

model2_dev_accuracies = np.array([[0.6217125382262997, 0.6217125382262997, 0.62171253822629979],  # Model 2, dev accuracy, epoch 7
                                  [0.6217125382262997, 0.6217125382262997, 0.6217125382262997]])  # Model 2, dev accuracy, epoch 9

# Test accuracies for Model 1 and Model 2
model1_test_accuracies = np.array([[0.7126839523475823, 0.6313945339873861, 0.6285914505956552],  # Model 1, test accuracy, epoch 7
                                   [0.7105816398037842, 0.641906096706377, 0.6341976173791171]])  # Model 1, test accuracy, epoch 9

model2_test_accuracies = np.array([[0.6222845129642607, 0.6089698668535389, 0.611072179397337],  # Model 2, test accuracy, epoch 7
                                   [0.6278906797477225, 0.6159775753328661,  0.6201822004204625]])  # Model 2, test accuracy, epoch 9

# Bar width
bar_width = 0.3

# Position of bars on x-axis
r1 = np.arange(len(learning_rates))
r2 = [x + bar_width for x in r1]

# Create two subplots: one for test accuracies, one for dev accuracies
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

# Plotting the dev accuracies
ax1.bar(r1, model1_dev_accuracies[0], color='b', width=bar_width, edgecolor='grey', label='Model 1 Epoch 7')
ax1.bar(r2, model1_dev_accuracies[1], color='c', width=bar_width, edgecolor='grey', label='Model 1 Epoch 9')

ax1.bar(r1, model2_dev_accuracies[0], color='r', width=bar_width, edgecolor='grey', label='Model 2 Epoch 7', alpha=0.5)
ax1.bar(r2, model2_dev_accuracies[1], color='orange', width=bar_width, edgecolor='grey', label='Model 2 Epoch 9', alpha=0.5)

# Adding labels for dev plot
ax1.set_xlabel('Learning Rate', fontweight='bold', fontsize=12)
ax1.set_ylabel('Dev Accuracy', fontweight='bold', fontsize=12)
ax1.set_xticks([r + bar_width / 2 for r in range(len(learning_rates))])
ax1.set_xticklabels(learning_rates)
ax1.set_title('Dev Accuracy Comparison')

# Add legend for dev plot
ax1.legend()

# Plotting the test accuracies
ax2.bar(r1, model1_test_accuracies[0], color='b', width=bar_width, edgecolor='grey', label='Model 1 Epoch 7')
ax2.bar(r2, model1_test_accuracies[1], color='c', width=bar_width, edgecolor='grey', label='Model 1 Epoch 9')

ax2.bar(r1, model2_test_accuracies[0], color='r', width=bar_width, edgecolor='grey', label='Model 2 Epoch 7', alpha=0.5)
ax2.bar(r2, model2_test_accuracies[1], color='orange', width=bar_width, edgecolor='grey', label='Model 2 Epoch 9', alpha=0.5)

# Adding labels for test plot
ax2.set_xlabel('Learning Rate', fontweight='bold', fontsize=12)
ax2.set_ylabel('Test Accuracy', fontweight='bold', fontsize=12)
ax2.set_xticks([r + bar_width / 2 for r in range(len(learning_rates))])
ax2.set_xticklabels(learning_rates)
ax2.set_title('Test Accuracy Comparison')

# Add legend for test plot
ax2.legend()

# Adjust layout to avoid overlap
plt.tight_layout()

# Show the plot
plt.show()
