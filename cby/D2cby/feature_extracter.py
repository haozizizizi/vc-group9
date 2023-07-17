from skimage.feature import hog, daisy, canny
import glob

# def PCA():

def feature_HOG(X):
    X_feature = []
    for x in X:
        x_feature = hog(x, orientations=8, pixels_per_cell=(10, 10),
                    cells_per_block=(1, 1), visualize=False)
        X_feature.append(x_feature)
    return X_feature


def feature_Daisy(X):
    X_features = []
    for x in X:
        x_feature = daisy(x, step=4, radius=15, rings=3, histograms=8, orientations=8, normalization='l1', sigmas=None,
                          ring_radii=None, visualize=False)
        X_features.append(x_feature)

    X_features = [feature.flatten() for feature in X_features]
    return X_features
def feature_Canny(X):
    X_features = []
    for x in X:
        x_feature = canny(x)
        X_features.append(x_feature)

    X_features = [feature.flatten() for feature in X_features]
    return X_features


# def LBP():

