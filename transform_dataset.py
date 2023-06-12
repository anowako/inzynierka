import numpy as np
import librosa
import pandas as pd
import multiprocessing as mp
import os
from tqdm import tqdm
from huggingface_hub import login
from datasets import load_dataset
import threading
import queue

def transform_func(audio, sr):
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=1024, hop_length=256)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=23)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    frame_energy = librosa.feature.rms(y=audio, frame_length=1024, hop_length=256)
    feature_vector = np.vstack((mfcc, mfcc_delta, mfcc_delta2, frame_energy))
    v_a, v_b = feature_vector.shape
    return np.reshape(feature_vector, (1, v_a, v_b))

def process_audio(q_in, q_out):
    age_filter = {'teens': 10,
        'twenties': 20,
        'thirties': 30,
        'fourties': 40,
        'fifties': 50,
        'sixties': 60,
        'seventies': 70,
        'eighties': 80,
        'nineties': 90,
        '': None}
    while True:
        data = q_in.get()
        if data is None:
            break
        name = data['audio']['path'].split('/')[-1][:-4]
        audio = data['audio']['array']
        age = age_filter[data['age']]
        gender = data['gender']
        permitted_age = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        permitted_gender = ['male', 'female']
        sr = data['audio']['sampling_rate']
        if age in permitted_age and gender in permitted_gender:
            label = f'{age}_{gender}'
            transformed_audio = transform_func(audio, sr)
            root_dir = f'/mnt/adamn/abc/inzynierka/data/transformed_cv_pl/{label}'
            output_path = os.path.join(root_dir, name + '.npy')
            try:
                if not os.path.exists(root_dir): os.makedirs(root_dir)
            except FileExistsError:
                pass
            result_dict = {'path': output_path, 'array': transformed_audio}
            q_out.put(result_dict)
        q_in.task_done()

def write_results(q_out, progress_bar):
    while True:
        result = q_out.get()
        if result is None:
            break
        
        np.save(result['path'], result['array'])
        progress_bar.update()
        q_out.task_done()

def process_dataset(dataset):
    total_files = len(dataset['train']) + len(dataset['test']) + len(dataset['validation'])
    progress_bar = tqdm(total=total_files, desc='Processing')

    q_in = queue.Queue()
    q_out = queue.Queue()

    num_threads = mp.cpu_count()
    threads = []

    for _ in range(5):
        write_thread = threading.Thread(target=write_results, args=(q_out, progress_bar))
        write_thread.start()
        threads.append(write_thread)

    for _ in range(num_threads - 5):
        thread = threading.Thread(target=process_audio, args=(q_in, q_out))
        thread.start()
        threads.append(thread)

    for data_name in dataset:
        if data_name in ['train', 'test', 'validation']:
            ds = dataset[data_name]
            for data in ds:
                q_in.put(data)

    for _ in range(num_threads - 5):
        q_in.put(None)

    q_in.join()

    for _ in range(5):
        q_out.put(None)
    for thread in threads:
        thread.join()

# Przetwarzanie datasetu
login(token='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')
dataset = load_dataset("mozilla-foundation/common_voice_13_0", "pl", cache_dir='/mnt/adamn/abc/inzynierka/data/CommonVoice')
process_dataset(dataset)