import matplotlib.pyplot as plt
def plot_and_save(plot_list, label, save_path):
    plt.figure(figsize=(8, 6))
    plt.plot(plot_list, label=label)
    plt.xlabel('Epoch')
    plt.ylabel(label)
    plt.title(label)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()