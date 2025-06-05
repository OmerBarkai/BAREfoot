import matplotlib.pyplot as plt
import os
from datetime import datetime
import numpy as np
def publish_figure(Frame=False, *args):
    plt.rcParams['svg.fonttype'] = 'none'
    ax = plt.gca()
    ax.xaxis.label.set_fontsize(10)
    ax.xaxis.label.set_fontname('Arial')
    ax.xaxis.label.set_fontweight('normal')
    ax.yaxis.label.set_fontsize(10)
    ax.yaxis.label.set_fontname('Arial')
    ax.yaxis.label.set_fontweight('normal')
    ax.title.set_fontsize(6)
    ax.title.set_fontname('Arial')
    ax.title.set_fontweight('normal')
    ax.spines['top'].set_visible(Frame)
    ax.spines['right'].set_visible(Frame)
    # Set the tick label font size
    for tick in ax.get_xticklabels():
        tick.set_fontsize(8)  # Set the fontsize for tick labels
        tick.set_fontname('Arial')
        tick.set_fontweight('normal')
    for tick in ax.get_yticklabels():
        tick.set_fontsize(8)  # Set the fontsize for tick labels
        tick.set_fontname('Arial')
        tick.set_fontweight('normal')

    # Modify existing colorbar font properties
    for cb in ax.figure.get_axes():
        if cb != ax:  # Identify colorbar axes
            cb.yaxis.label.set_fontsize(10)
            cb.yaxis.label.set_fontname('Arial')
            cb.yaxis.label.set_fontweight('normal')

            for tick in cb.get_yticklabels():
                tick.set_fontsize(8)
                tick.set_fontname('Arial')
                tick.set_fontweight('normal')




def save_figs(file_format='png', folder='figures', dpi=300, transparent=True):

    # Check if the folder exists, and if not, create it
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Get a list of all open figures
    figures = [plt.figure(num) for num in plt.get_fignums()]

    for i, fig in enumerate(figures):
        # Specify the file name for each figure (e.g., figure_1.png, figure_2.png, ...)
        file_name = f'{folder}/PyFig_{i + 1}_{datetime.now().strftime("%y%m%d%H%M%S")}.{file_format}'

        # Save the figure
        fig.savefig(file_name, format=file_format, dpi=dpi, transparent=transparent)
        print(f'Saved {file_name}')


def heatmap(data, x_ticks=None, y_ticks=None, x_labels=None, y_labels=None, cmap='viridis', show_values=True):
    # Plotting the data
    fig, ax = plt.subplots()
    cax = ax.imshow(data, cmap=cmap)

    # Add colorbar for reference
    fig.colorbar(cax)

    # Setting default ticks if not provided
    if x_ticks is None:
        x_ticks = np.arange(data.shape[1])
    if y_ticks is None:
        y_ticks = np.arange(data.shape[0])

    # Setting custom ticks and labels if provided
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    if x_labels is not None and len(x_labels) == len(x_ticks):
        ax.set_xticklabels(x_labels)
    if y_labels is not None and len(y_labels) == len(y_ticks):
        ax.set_yticklabels(y_labels)

    if show_values:
        # Annotating each cell with the data value
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                text = ax.text(j, i, f'{data.iloc[i, j]:.2f}',
                               ha='center', va='center', color='white')

    # Showing the plot
    plt.show()

def topFeatures(model, n_topfeatures, plot=True, save=False, filename=''):
    # Feature importance
    if hasattr(model, 'feature_importances_'):  # if its XGB
        importances = model.feature_importances_[np.argsort(model.feature_importances_)]
        importances_headers = model.feature_names_in_[np.argsort(model.feature_importances_)]
    if hasattr(model, 'coef_'):  # if its sklearn
        model_imps = model.coef_[0]
        importances = np.abs(model_imps[np.argsort(np.abs(model_imps))])
        importances_headers = model.feature_names_in_[np.argsort(np.abs(model_imps))]


    plt.figure()
    plt.xticks(rotation=45, ha='right')
    colors = plt.cm.Blues(np.linspace(0.4, 1, len(importances[-n_topfeatures:])))  # Adjust the range for desired shade
    # Plot horizontal lines with varying shades of blue
    for i, (category, value) in enumerate(zip(importances_headers[-n_topfeatures:], importances[-n_topfeatures:])):
        plt.hlines(category, xmin=0, xmax=value, color=colors[i], linewidth=2.5)
        plt.plot(value, category, 'o', color=colors[i])
    plt.xticks(ticks=np.arange(0, importances.max(axis=0).max() * 1.2, 0.01))
    plt.xlabel('Feature Importance')
    plt.subplots_adjust(left=0.4)

    if save==True:
        if len(filename)<1:
            print('Not saved. Filename was no provided.')
        else:
            if filename[-4:]!='.png':
                filename = filename + '.png'
            plt.savefig(filename, format='png', dpi=300, transparent=False)

    if plot==False:
        plt.close()  # Close the figure to avoid showing it

    return importances, importances_headers

def plotConfusionMat(model, y_test, y_pred, cmap='Blues'):
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=model.classes_.astype(int), normalize='true', cmap='Blues')
    plt.title('(F1: ' + str(round(f1_score(y_test, y_pred), 2)) + ')', fontsize=10)
    for im in plt.gca().get_images():  # set clim manually within the image
        im.set_clim(vmin=0, vmax=1)
        im.figure.set_size_inches(3, 3)
    font = {'family': 'Arial', 'weight': 'normal', 'size': 8}
    publish_figure(Frame=True)
    plt.rc('font', **font)

from matplotlib.colors import LinearSegmentedColormap
def truncate_colormap(cmap_name, min_val=0.5, max_val=1.0, n=100):
    cmap = plt.get_cmap(cmap_name)
    new_cmap = LinearSegmentedColormap.from_list(f"trunc_{cmap_name}", cmap(np.linspace(min_val, max_val, n)))
    return new_cmap

def custom_cmap(name,from_color,to_color):
    import matplotlib.colors as mcolors
    return mcolors.LinearSegmentedColormap.from_list(name,[from_color,to_color])