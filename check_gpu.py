import tensorflow as tf

print(f"Versi TensorFlow: {tf.__version__}")
gpu_devices = tf.config.list_physical_devices('GPU')

print("GPU tersedia:", gpu_devices)

if gpu_devices:
  details = tf.config.experimental.get_device_details(gpu_devices[0])
  print(f"✅ Selamat! GPU Anda terdeteksi: {details.get('device_name', 'Unknown')}")
else:
  print("❌ Gagal. TensorFlow tidak dapat menemukan GPU Anda.")