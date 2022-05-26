from sklearn.decomposition import PCA
import os
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA


def pca_clustering_3D(x, y, save_dir=None):
    pca = PCA(n_components=3)  # trasforma in 3D
    emb_array = np.array(x).squeeze()  # np.array(x) trasforma la lista in array, .squeeze toglie dimensionalità inutili

    output = pca.fit_transform(emb_array) # trasforma tutto in 3D
    component_1, component_2, component_3 = output[:, 0], output[:, 1], output[:, 2]

    # fig = px.scatter_3d(x=component_1,
    #                     y=component_2,
    #                     z=component_3,
    #                     # symbol=y,
    #                     color=y,
    #                     title=save_dir[6:])
    fig = px.scatter_3d(x=component_1,
                        y=component_2,
                        z=component_3,
                        # symbol=y,
                        color=y,
                        title=save_dir[6:])

    if save_dir:
        filepath = open(f'./{save_dir}.html', "w+")
        fig.write_html(filepath)
        #fig.show()
        return filepath
    else:
        fig.show()

        return None
