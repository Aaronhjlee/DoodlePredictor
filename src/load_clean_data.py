import pandas as pd
import numpy as np
import ndjson

# load the npy and ndjson data
def load_data():
    faces = np.load('/Users/AaronLee/Documents/GalvanizeDSI/DoodlePredictor/data/face.npy')
    eyes = np.load('/Users/AaronLee/Documents/GalvanizeDSI/DoodlePredictor/data/eye.npy')
    noses = np.load('/Users/AaronLee/Documents/GalvanizeDSI/DoodlePredictor/data/nose.npy')
    ears = np.load('/Users/AaronLee/Documents/GalvanizeDSI/DoodlePredictor/data/ear.npy')
    
    # face
    with open('/Users/AaronLee/Documents/GalvanizeDSI/DoodlePredictor/data_simplified/full_simplified_face.ndjson') as f:
        temp = ndjson.load(f)
    face_df = pd.DataFrame.from_dict(temp)
    
    # eye
    with open('/Users/AaronLee/Documents/GalvanizeDSI/DoodlePredictor/data_simplified/full_simplified_eye.ndjson') as j:
        temp1 = ndjson.load(j)
    eye_df = pd.DataFrame.from_dict(temp1)

    # nose
    with open('/Users/AaronLee/Documents/GalvanizeDSI/DoodlePredictor/data_simplified/full_simplified_nose.ndjson') as l:
        temp3 = ndjson.load(l)
    nose_df = pd.DataFrame.from_dict(temp3)
    
    # ear
    with open('/Users/AaronLee/Documents/GalvanizeDSI/DoodlePredictor/data_simplified/full_simplified_ear.ndjson') as m:
        temp4 = ndjson.load(m)
    ear_df = pd.DataFrame.from_dict(temp4)
    
    print('loaded data')
    return faces, eyes, noses, ears, face_df, eye_df, nose_df, ear_df

# merge npy and ndjson data
def merge(df, npy):
    df['npy'] = list(npy)
    print('merged data')
    return df

# sampling only if it was recognized by the system
def sample_it(df, n):
    df_out = df[df['recognized'] == True].sample(n=n, replace=True)
    print('taken samples')
    return df_out

def clean_merge_it(df, df1):
    df = df.drop(['countrycode', 'drawing', 'key_id', 'recognized', 'timestamp'], axis=1)
    df1 = df1.drop(['countrycode', 'drawing', 'key_id', 'recognized', 'timestamp'], axis=1)
    print('clean and merged')
    return pd.concat([df, df1], join='inner', ignore_index=True)

def get_clean_data():
    # load the data
    faces, eyes, noses, ears, face_df, eye_df, nose_df, ear_df = load_data()

    # merge the npy with df
    face_df = merge(face_df, faces)
    eye_df = merge(eye_df, eyes)
    nose_df = merge(nose_df, noses)
    ear_df = merge(ear_df, ears)

    # sample subset of data
    sample_face = sample_it(face_df, 5000)
    sample_eye = sample_it(eye_df, 5000)
    sample_nose = sample_it(nose_df, 5000)
    sample_ear = sample_it(ear_df, 5000)

    # merged sample and cleaned
    df_a = clean_merge_it(sample_face, sample_eye)
    df_b = clean_merge_it(sample_nose, sample_ear)
    df = pd.concat([df_a, df_b], join='inner', ignore_index=True)

    # split npy to (20000,784)
    df2 = pd.DataFrame(df.npy.tolist())
    return df, df2, face_df, eye_df, nose_df, ear_df