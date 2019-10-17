import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from matplotlib.pyplot import imshow

from sklearn.model_selection import train_test_split
from sklearn.ensemble import (GradientBoostingRegressor, 
                              GradientBoostingClassifier, 
                              AdaBoostClassifier,
                              RandomForestClassifier,
                             RandomForestRegressor)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from load_clean_data import *

def plot_simple(data, name):
    fig, axs = plt.subplots(1,10, figsize=(10,1))

    for thing, ax in zip(data, axs.flatten()):
        image = thing.reshape(28, 28)
        ax.imshow(image, cmap=cm.Greys)
        ax.axis('off')
        
    fig.savefig('/Users/AaronLee/Documents/GalvanizeDSI/DoodlePredictor/images/{}.png'.format(name), bbox_inches='tight')

def plot_simple_all(df1, df2, df3, df4):
    # plot true FACES
    face_true = df1[df1['recognized'] == True]['npy']
    plot_simple(face_true, 'row_true_faces')
    # plot false FACES
    face_false = df1[df1['recognized'] == False]['npy']
    plot_simple(face_false, 'row_false_faces')

    # plot true EYES
    eye_true = df2[df2['recognized'] == True]['npy']
    plot_simple(eye_true, 'row_true_eyes')
    # plot false EYES
    eye_false = df2[df2['recognized'] == False]['npy']
    plot_simple(eye_false, 'row_false_eyes')

    # plot true NOSE
    nose_true = df3[df3['recognized'] == True]['npy']
    plot_simple(nose_true, 'row_true_nose')
    # plot false NOSE
    nose_false = df3[df3['recognized'] == False]['npy']
    plot_simple(nose_false, 'row_false_nose')

    # plot true EAR
    ear_true = df4[df4['recognized'] == True]['npy']
    plot_simple(ear_true, 'row_true_ear')
    # plot false EAR
    ear_false = df4[df4['recognized'] == False]['npy']
    plot_simple(ear_false, 'row_false_ear')

def model_prep(df, df2):
    """
    df: Dataframe
    df2: Dataframe
    ------------
    returns
    X: Dataframe
    y: numpy array
    """
    # 0 = face | 1 = eye | 2 = nose | 3 = ear
    X = df2
    # y_dirty = df['word'].replace({'face': 0, 'eye': 1, 'mouth': 2, 'nose': 3})
    y_dirty = df['word'].replace({'face': 0, 'eye': 1, 'nose': 2, 'ear': 3})
    y = y_dirty.to_numpy()
    return X, y, y_dirty


def plot_var_comp(df, df2):
    X, y, y_dirty = model_prep(df, df2)
    # scale data
    scaler = MinMaxScaler(feature_range=[0,1])
    data_rescaled = scaler.fit_transform(X)
    pca = PCA().fit(data_rescaled)
    #Plotting the Cumulative Summation of the Explained Variance
    var = [0.9, 0.8, 0.6, 0.4]
    color = ['k', 'c', 'b', 'orange']
    plt.figure(figsize=(8,8))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Variance (%)') #for each component
    plt.title('Doodle Explained Variance')
    for i, j in zip(var, color):
        plt.axhline(y = i, color=j, linestyle='--', label = '{}% Explained Variance'.format(i))
    plt.legend(loc='best')
    plt.show()

def explainedVariance(percentage, images): 
    # percentage should be a decimal from 0 to 1 
    pca = PCA(percentage)
    pca.fit(images)
    components = pca.transform(images)
    approxOriginal = pca.inverse_transform(components)
    return approxOriginal

def plot_faces(df):
    comp = [180, 106, 44, 17, 4]
    var = [0.9, 0.8, 0.6, 0.4, 0.2]

    # Original Image (784 components)
    plt.figure(figsize=(20,6))
    plt.subplot(1, 6, 1)
    plt.imshow(df.npy[0].reshape(28,28),
                cmap = plt.cm.gray, interpolation='nearest',
                clim=(0, 255))
    plt.xlabel('784 Components', fontsize = 12)
    plt.title('Original Image', fontsize = 14)

    for i, j, k in zip(comp, var, range(2, 7)):
        plt.subplot(1, 6, k)
        plt.imshow(explainedVariance(j, df.npy[0].reshape(28, 28)),
                    cmap = plt.cm.gray, interpolation='nearest',
                    clim=(0, 255))
        plt.xlabel('{} Components'.format(i), fontsize = 12)
        plt.title('{}% of Explained Variance'.format(j), fontsize = 14)

def plot_2d_pca(df, df2):
    X, y, y_dirty = model_prep(df, df2)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    pca = PCA(2)
    p_comp = pca.fit_transform(X)
    p_df = pd.DataFrame(data = p_comp
                , columns = ['principal component 1', 'principal component 2'])
    final_df = pd.concat([p_df, y_dirty], axis=1)

    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)

    targets = [0, 1]

    ax.scatter(final_df['principal component 1'][:5000]
            , final_df['principal component 2'][:5000]
            , c = 'r'
            , s = 50
            , alpha=0.1
            , marker=('$0$'))
    ax.scatter(final_df['principal component 1'][5000:10000]
            , final_df['principal component 2'][5000:10000]
            , c = 'b'
            , s = 50
            , alpha=0.1
            , marker=('$1$'))
    ax.scatter(final_df['principal component 1'][10000:15000]
            , final_df['principal component 2'][10000:15000]
            , c = 'g'
            , s = 50
            , alpha=0.1
            , marker=('$2$'))
    ax.scatter(final_df['principal component 1'][15000:]
            , final_df['principal component 2'][15000:]
            , c = 'c'
            , s = 50
            , alpha=0.1
            , marker=('$3$'))
    ax.legend(['face', 'eye', 'nose', 'ear'])
    ax.grid()

    ax.figure.savefig('/Users/AaronLee/Documents/GalvanizeDSI/DoodlePredictor/images/two_comp_pca_1.png', 
    bbox_inches='tight')

def plot_3d_pca(df, df2):
    # %matplotlib notebook
    X, y, y_dirty = model_prep(df, df2)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    pca = PCA(3)
    p_comp = pca.fit_transform(X)
    p_df = pd.DataFrame(data = p_comp
                , columns = ['principal component 1', 'principal component 2', 'principal component 3'])
    final_df = pd.concat([p_df, y_dirty], axis=1)

    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection="3d")

    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_zlabel('Principal Component 3', fontsize = 15)
    ax.set_title('3 component PCA', fontsize = 20)

    ax.scatter3D(final_df['principal component 1'][:5000]
                , final_df['principal component 2'][:5000]
                , final_df['principal component 3'][:5000]
                , c = 'r'
                , s = 50
                , alpha=0.1
                , marker=('$0$'))
    ax.scatter(final_df['principal component 1'][5000:10000]
            , final_df['principal component 2'][5000:10000]
            , final_df['principal component 3'][5000:10000]
            , c = 'b'
            , s = 50
            , alpha=0.1
            , marker=('$1$'))
    ax.scatter(final_df['principal component 1'][10000:15000]
            , final_df['principal component 2'][10000:15000]
            , final_df['principal component 3'][10000:15000]
            , c = 'g'
            , s = 50
            , alpha=0.1
            , marker=('$2$'))
    ax.scatter(final_df['principal component 1'][15000:]
            , final_df['principal component 2'][15000:]
            , final_df['principal component 3'][15000:]
            , c = 'c'
            , s = 50
            , alpha=0.1
            , marker=('$3$'))

    ax.legend(['face', 'eye', 'nose', 'ear'])

    for angle in range(0, 360):
        ax.view_init(30, angle)
        plt.draw()
        plt.pause(.001)

    plt.show()

def prep_for_class(df, df2):
    X, y, y_dirty = model_prep(df, df2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                random_state=123, 
                                                test_size=0.2)
    pca = PCA(0.90)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    return X_train, X_test, y_train, y_test

def get_model_score(df, df2, model):
    X_train, X_test, y_train, y_test = prep_for_class(df, df2)
    clf = model()
    clf_func = clf.fit(X_train, y_train)

    y_pred = clf_func.predict(X_test)
    print (model)
    print(classification_report(y_test, y_pred))