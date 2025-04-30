import numpy as np

def calculate_word_level_accuracy(predictions, true_labels):
    correct_words = 0
    total_words = 0

    for preds, lbls in zip(predictions, true_labels):

        idxs = np.where(np.array(lbls) == 0)[0]
        arrays = np.split(lbls, idxs, axis=0)
        word_lbls = arrays
        word_lbls = [np.delete(x, np.where(x == 0)) for x in word_lbls]

        idxs = np.where(np.array(preds) == 0)[0]
        arrays = np.split(preds, idxs, axis=0)
        word_preds = arrays
        word_preds = [np.delete(x, np.where(x == 0)) for x in word_preds]
        for label, prediction in zip(word_preds, word_lbls):
            if label.shape[0] == 0 and prediction.shape[0] == 0:
                continue
            if list(label) == list(prediction):
                correct_words += 1
            total_words += 1

    return correct_words / total_words if total_words > 0 else 0.0