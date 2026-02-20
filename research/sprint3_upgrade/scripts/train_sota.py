
import os
import yaml
import argparse
import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from tqdm import tqdm
from pathlib import Path

# Import model definition (assuming it exists in scripts/models or define it here)
# For simplicity, we define the model here to match SOTA exactly
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Concatenate, Bidirectional, LSTM, Dropout, Flatten, Dense, RepeatVector, UpSampling1D
from tensorflow.keras.models import Model

def build_multiscale_cnn_bilstm_ae(input_shape, encoding_dim=16):
    inputs = Input(shape=input_shape)
    
    # Encoder
    # Multi-scale CNN
    conv1 = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(inputs)
    conv2 = Conv1D(filters=32, kernel_size=5, activation='relu', padding='same')(inputs)
    conv3 = Conv1D(filters=32, kernel_size=7, activation='relu', padding='same')(inputs)
    
    pool1 = MaxPooling1D(pool_size=2)(conv1)
    pool2 = MaxPooling1D(pool_size=2)(conv2)
    pool3 = MaxPooling1D(pool_size=2)(conv3)
    
    concat = Concatenate()([pool1, pool2, pool3])
    
    bilstm1 = Bidirectional(LSTM(64, return_sequences=True))(concat)
    dropout1 = Dropout(0.2)(bilstm1)
    
    flatten = Flatten()(dropout1)
    encoded = Dense(encoding_dim, activation='relu')(flatten)
    
    # Decoder
    repeat = RepeatVector(input_shape[0] // 2)(encoded) # Adjust for pooling
    
    bilstm2 = Bidirectional(LSTM(64, return_sequences=True))(repeat)
    dropout2 = Dropout(0.2)(bilstm2)
    
    upsample = UpSampling1D(size=2)(dropout2)
    
    decoded = Conv1D(filters=input_shape[1], kernel_size=3, activation='sigmoid', padding='same')(upsample)
    
    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return autoencoder

# Data Pipeline (Memory Efficient)
def npz_generator(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    for f in files:
        try:
            with np.load(f, allow_pickle=True) as data:
                X = data['X'] if 'X' in data else (data['x'] if 'x' in data else None)
                if X is not None:
                    for i in range(len(X)):
                        yield X[i], X[i] # AE Target = Input
        except: pass

def create_dataset(data_dir, batch_size=256, shuffle=True, input_shape=(10, 77)):
    dataset = tf.data.Dataset.from_generator(
        lambda: npz_generator(data_dir),
        output_signature=(
            tf.TensorSpec(shape=input_shape, dtype=tf.float32),
            tf.TensorSpec(shape=input_shape, dtype=tf.float32)
        )
    )
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def count_samples(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    total = 0
    for f in files:
        try:
            with np.load(f, allow_pickle=True) as data:
                key = 'X' if 'X' in data else 'x'
                if key in data:
                    total += data[key].shape[0]
        except: pass
    return total

def main():
    parser = argparse.ArgumentParser(description="Train SOTA Model (Sprint 3)")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    print(f"--- Training Experiment: {config['experiment_name']} ---")
    
    # Paths
    processed_dir = Path(config['paths']['processed_data'])
    train_dir = processed_dir / "train"
    test_dir = processed_dir / "test"
    output_dir = Path(config['paths']['output_dir'])
    model_dir = Path(config['paths']['model_save_dir'])
    
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Pipeline Setup
    batch_size = config['training']['batch_size']
    total_train = count_samples(train_dir)
    print(f"Total Train Samples: {total_train}")
    
    if total_train == 0:
        print("Error: No training data found.")
        return

    full_ds = create_dataset(train_dir, batch_size=batch_size, shuffle=True)
    
    val_size = int(total_train * config['training']['validation_split'])
    train_size = total_train - val_size
    
    train_steps = train_size // batch_size
    val_steps = val_size // batch_size
    
    train_ds = full_ds.take(train_steps)
    val_ds = full_ds.skip(train_steps).take(val_steps)
    
    # 2. Build Model
    # Determine input shape from first sample
    sample = next(iter(full_ds.take(1)))
    input_shape = sample[0].shape[1:] # (batch, seq, feat) -> (seq, feat)
    print(f"Detected Input Shape: {input_shape}")
    
    model = build_multiscale_cnn_bilstm_ae(input_shape, encoding_dim=config['model']['encoding_dim'])
    model.summary()
    
    # 3. Train
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=config['training']['patience'], 
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_dir / "best_model.h5"),
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        )
    ]
    
    history = model.fit(
        train_ds,
        epochs=config['training']['epochs'],
        steps_per_epoch=train_steps,
        validation_data=val_ds,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save History Plot
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss (MSE)')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(output_dir / "loss_history.png")
    print(f"History plot saved to {output_dir}/loss_history.png")
    
    # 4. Determine Threshold
    print("\nCalculating Threshold...")
    val_errors = []
    for batch_x, _ in tqdm(val_ds, desc="Val Batches", total=val_steps):
        recon = model.predict(batch_x, verbose=0)
        mse = np.mean(np.square(batch_x - recon), axis=(1, 2))
        val_errors.extend(mse)
        
    threshold = np.percentile(val_errors, config['thresholding']['percentile'])
    print(f"Threshold (Percentile {config['thresholding']['percentile']}): {threshold}")
    
    # 5. Evaluation
    print("\nEvaluating on Test Data...")
    
    # Custom generator for Test (X, y)
    def test_gen():
        files = sorted(glob.glob(os.path.join(test_dir, "*.npz")))
        for f in files:
            try:
                with np.load(f, allow_pickle=True) as data:
                    X = data['X'] if 'X' in data else (data['x'] if 'x' in data else None)
                    y = data['y'] if 'y' in data else (data['Y'] if 'Y' in data else None)
                    if X is not None:
                        for i in range(len(X)):
                            yield X[i], y[i]
            except: pass
            
    test_ds = tf.data.Dataset.from_generator(
        test_gen,
        output_signature=(
            tf.TensorSpec(shape=input_shape, dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    all_errors = []
    all_y = []
    
    for batch_x, batch_y in tqdm(test_ds, desc="Test Batches"):
        recon = model.predict(batch_x, verbose=0)
        mse = np.mean(np.square(batch_x - recon), axis=(1, 2))
        all_errors.extend(mse)
        all_y.extend(batch_y.numpy())
        
    y_pred = (np.array(all_errors) > threshold).astype(int)
    y_true = np.array(all_y)
    
    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"\n--- Evaluation Results ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Save Metrics
    results = {
        'threshold': float(threshold),
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1)
    }
    with open(output_dir / "metrics.yaml", 'w') as f:
        yaml.dump(results, f)

if __name__ == "__main__":
    main()
