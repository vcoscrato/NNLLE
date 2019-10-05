import numpy as np
from scipy.sparse import issparse

from matplotlib import pyplot as plt


def label_bar(rects, ax, labels=None):
    colors = ['blue', 'orange']
    for rect, color in zip(rects, colors):
        width = rect.get_width()
        rect.set_color('r')
        ax.annotate('{:3.2f}'.format(width),
                    xy=(rect.get_width() / 2, rect.get_y() - 0.2 + rect.get_height() / 2),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    size=30)


def simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)


class SparseMatrix(object):
    """
    Transformation to be used in a sklearn pipeline
    check if a array is sparse.
    # TODO: The NLS, LLS, NNPredict should accept sparse array
    """
    def __init__(self):
        pass

    def fit(self):
        return self

    @staticmethod
    def transform(x):
        if issparse(x):
            return x.toarray()
        return x


class ExplainText(object):
    def __init__(self, model, class_names, names_features):
        """
        :param model: NLS model;
        :param class_names: class names to be utilized in the plot;
        :param names_features: names of the features.
        """
        self.model = model
        self.class_names = class_names
        self.names_features = names_features

    def get_text_explanation(self, x_explain, document, num_features=10):
        """
        Get the explanation of text document.
        :param x_explain: document to be explained, should be vectorized;
        :param document: document in text format;
        :param num_features: number of features to produce the explanation.
        :return: betas values and words correspondent to the explanation.
        """
        explanation = self.model.get_thetas(x_pred=x_explain, net_scale=True)
        betas = explanation[2][0]
        words_from_text_indices = np.argwhere(x_explain[0] > 0).reshape(-1)

        # Prediction from the model
        prediction = self.model.predict(x_explain).reshape(-1)
        predict_proba = self.model.predict_proba(x_explain).reshape(-1)
        ind_pred_proba = np.argsort(predict_proba)[::-1]

        # col_betas = int(prediction)
        col_betas = ind_pred_proba[0]
        col_betas_neg = ind_pred_proba[1]

        betas_document = betas[words_from_text_indices, col_betas]
        betas_document_neg = betas[words_from_text_indices, col_betas_neg]

        betas_final = betas_document - betas_document_neg
        words_features_document = self.names_features[words_from_text_indices].reshape(-1)

        # Organize
        beta_0_abs = np.abs(betas_final)
        betas_rank_ind = np.flip(np.argsort(beta_0_abs))[:num_features]
        return dict(betas=betas_final[betas_rank_ind]
                    , betas_document=betas_document[betas_rank_ind]
                    , betas_document_neg=betas_document_neg[betas_rank_ind]
                    , words=words_features_document[betas_rank_ind]
                    , prediction=prediction
                    , prediction_proba=predict_proba
                    , document=document
                    )

    def explain_graphical(self, x_explain, document, num_features=10):
        exp = self.get_text_explanation(x_explain, document, num_features=num_features)
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        rects1 = axs[0].barh(self.class_names, exp['prediction_proba'])
        axs[0].set_xticks([])
        colors = ['blue', 'orange']
        for rect, color in zip(rects1, colors):
            rects1[0].set_color(color)
        axs[0].set_title('Prediction probabilities')
        simpleaxis(axs[0])
        label_bar(rects1, axs[0])
        names = exp['words']
        vals = exp['betas']
        vals = vals[::-1]
        names = names[::-1]
        colors = ['green' if x > 0 else 'red' for x in vals]
        pos = np.arange(len(vals))
        axs[1].barh(pos, vals, align='center', color=colors)
        axs[1].set_yticks(pos)
        axs[1].set_yticklabels(names)
        axs[2].set_title('Important Features')
        simpleaxis(axs[2])
        axs[2].set_xticks([])
        axs[2].set_yticks([])
        axs[2].text(0, 1, '\n' + exp['document'], style='italic', wrap=True, va='top')
        axs[2].set_title('Document to Explain')
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)
        return fig, axs
