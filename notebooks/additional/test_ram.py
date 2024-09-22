
from helper_functions import (
    convert_prefetchdataset_to_numpy_arrays,
    evaluate_prediction,
    predict_and_print_full_results,
    lite_model_from_file_predicts_dataset,
    find_decision_threshold,
    convert_bytes,
    get_file_size,
    evaluate_prediction_with_threshold
    )
import tensorflow as tf
import numpy as np
import tracemalloc

from create_spectrogram import (
    create_spectrograms_from_audio_dataset,
    extract_patches_from_audio_dataset 
)

# @profile(precision=4)
def run_full_int_q_tflite_model(tflite_file, indices, x_data):
  # Initialize the interpreter
  tracemalloc.start()

  interpreter = tf.lite.Interpreter(model_path=str(tflite_file))
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()[0]
  output_details = interpreter.get_output_details()[0]

#   predictions_prob = np.zeros((len(indices),), dtype=int)
#   predictions_class = np.zeros((len(indices),), dtype=int)
  for i, test_image_index in enumerate(indices):
    test_data_point = x_data[test_image_index]

    # Check if the input type is quantized, then rescale input data to uint8
    if input_details['dtype'] == np.uint8:
      input_scale, input_zero_point = input_details["quantization"]
      test_data_point = test_data_point / input_scale + input_zero_point

    test_data_point = np.expand_dims(test_data_point, axis=0).astype(input_details["dtype"])
    interpreter.set_tensor(input_details["index"], test_data_point)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage: {current / 10**6}MB; Peak: {peak / 10**6}MB")
    # predictions_class[i] = output.argmax()
    # predictions_prob[i] = output[1]

#   return predictions_class, predictions_prob

# @profile
def full_int_model_predict(tflite_file, x_data):
  indices = range(len(x_data))
  return run_full_int_q_tflite_model(tflite_file, indices, x_data)
#   return predictions

# @profile
def test():
    full_int_models = [
        # ("../spectrogram_models_from_notebooks/cnn/quantized/cnn_mel_spec_full_int_q.tflite", "CNN-Mel"), 
        ("../spectrogram_models_from_notebooks/squeezenet/squeezenet_spec_16kHz_full_int_q.tflite", "SqueezeNet-Mel"), 
        # ("../spectrogram_models_from_notebooks/bnn/hpo/bnn_mel_spec_full_int_q.tflite", "BNN-Mel"),
        # ('../spectrogram_models_from_notebooks/vit/quantized/vit_mel_spec_full_int_q.tflite', "Transformer-Mel"),
        # ("../time_series_models_from_notebooks/cnn/quantized/cnn_time_series_full_int_q.tflite", "CNN-Time"), 
        # ("../time_series_models_from_notebooks/squeezenet/squeezenet30%_time_series_16kHz_full_int_q.tflite", "SqueezeNet-Time"), 
        # ("../time_series_models_from_notebooks/bnn/hpo/bnn_time_ser_full_int_q.tflite", "BNN-Time"),
        # ("../time_series_models_from_notebooks/tiny_transformer/quantized/tiny_transformer_time_series_full_int_q.tflite", "Transformer-Time"),
            ]
    test_dataset = tf.keras.utils.audio_dataset_from_directory("../dataset/testing", labels='inferred', sampling_rate=16000, batch_size=32, shuffle=False, seed=3407)
    test_spectrogram_ds = create_spectrograms_from_audio_dataset(test_dataset, sample_rate = 16000) #.cache().prefetch(tf.data.AUTOTUNE)
    spec_x_test_np, spec_y_test_np = convert_prefetchdataset_to_numpy_arrays(test_spectrogram_ds)
    # pred_labels, y_pred_prob = full_int_model_predict(full_int_models[0][0], [spec_x_test_np[0]])
    full_int_model_predict(full_int_models[0][0], [spec_x_test_np[0]])
    # test_metrics_dict = evaluate_prediction_with_threshold(time_ser_y_test_np, y_pred_prob, model_format="tflite", optimal_threshold=threshold)

if __name__ == "__main__":
  test()