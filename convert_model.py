"""
Model Conversion & Quantization Script
Converts the Keras 3 language classifier into an optimized LiteRT (.tflite) asset.
"""
import logging
import os

import tensorflow as tf

# Configure logging for pipeline visibility
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def convert_and_quantize(keras_model_path: str, tflite_export_path: str):
    if not os.path.exists(keras_model_path):
        logging.error(f"Model file not found: {keras_model_path}")
        return

    logging.info(f"Loading Keras model from {keras_model_path}...")
    
    # Custom object scope to catch and bypass unmapped Keras 3 serialization arguments
    with tf.keras.utils.custom_object_scope({'quantization_config': None}):
        try:
            model = tf.keras.models.load_model(keras_model_path)
        except Exception as e:
            logging.error(f"Failed to load Keras model: {e}")
            return

    logging.info("Initializing LiteRT Converter...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # FIX: Enable TensorFlow Select Ops & disable tensor list lowering for dynamic LSTM support
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, 
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter._experimental_lower_tensor_list_ops = False
    
    # Apply Dynamic Range Quantization (FP32 -> INT8)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    logging.info("Converting model. This may take a moment...")
    try:
        tflite_model = converter.convert()
    except Exception as e:
        logging.error(f"Conversion failed: {e}")
        return

    with open(tflite_export_path, "wb") as f:
        f.write(tflite_model)
    
    # Calculate and log compression metrics
    orig_size_mb = os.path.getsize(keras_model_path) / (1024 * 1024)
    new_size_mb = os.path.getsize(tflite_export_path) / (1024 * 1024)
    compression_ratio = (1 - (new_size_mb / orig_size_mb)) * 100
    
    logging.info("Conversion Successful!")
    logging.info(f"Original Size:  {orig_size_mb:.2f} MB")
    logging.info(f"Quantized Size: {new_size_mb:.2f} MB")
    logging.info(f"Compression:    {compression_ratio:.1f}% reduction")

if __name__ == "__main__":
    KERAS_PATH = "ultimate_african_ai.keras"
    TFLITE_PATH = "models/language_classifier.tflite"
    
    # Ensure export directory exists before writing
    os.makedirs(os.path.dirname(TFLITE_PATH), exist_ok=True)
    convert_and_quantize(KERAS_PATH, TFLITE_PATH)