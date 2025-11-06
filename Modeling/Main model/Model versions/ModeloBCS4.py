# =============================================================
# Nombre del archivo: ModeloBCS4.py
# Autores: Mónica Monserrat Martínez Vásquez,
#          Bárbara Paola Alcántara Vega 
#
# Descripción:
#   entrenamiento de un conjunto (ensemble) de tres modelos EfficientNetB7
#   en precisión mixta FP16 con generación automática de un reporte PDF.
#   el sistema aplica estrategias de reducción de aprendizaje (cosine decay),
#   augmentación avanzada, cálculo de pesos balanceados por clase y guarda
#   los resultados y métricas finales en disco.
#
# Dependencias: os, numpy, tensorflow, matplotlib, sklearn, reportlab
# =============================================================

import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import json
from tensorflow.keras import layers, models, mixed_precision
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard, LearningRateScheduler
from tensorflow.keras.applications import EfficientNetB7
from sklearn.utils.class_weight import compute_class_weight
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import matplotlib
matplotlib.use('Agg')  # Para guardar sin mostrar


# Configuración de memoria de GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"Memory growth activado: {len(gpus)} GPU(s)")

# Política de precisión mixta
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print("Mixed precision: FP16")

# Configuración general
baseDir = r"C:\Users\Artur\OneDrive\Documents\DataSet MuMetrics"
imagesDir = os.path.join(baseDir, 'DS')
outputDir = os.path.join(baseDir, 'model_14gb_final')
checkpointDir = os.path.join(baseDir, 'Checkpoints_14gb_final')
reportDir = os.path.join(outputDir, 'report')
os.makedirs(outputDir, exist_ok=True)
os.makedirs(checkpointDir, exist_ok=True)
os.makedirs(reportDir, exist_ok=True)

imageSize = (448, 448)
batchSize = 12
valTestSplit = 0.30
seed = 42
autoTune = tf.data.AUTOTUNE

# Lectura de carpetas y clases
subdirs = sorted([d for d in os.listdir(imagesDir) if os.path.isdir(os.path.join(imagesDir, d))])
classNames = sorted(subdirs, key=lambda s: float(s))
bcsValues = [float(c) for c in classNames]
print("Clases BCS:", bcsValues)

# Carga de datos
trainDataset = tf.keras.utils.image_dataset_from_directory(
    imagesDir, label_mode='int', color_mode='rgb', batch_size=batchSize,
    image_size=imageSize, shuffle=True, seed=seed, validation_split=valTestSplit, subset='training'
)

valTestDataset = tf.keras.utils.image_dataset_from_directory(
    imagesDir, label_mode='int', color_mode='rgb', batch_size=batchSize,
    image_size=imageSize, shuffle=True, seed=seed, validation_split=valTestSplit, subset='validation'
)

# Remapeo de clases y pesos balanceados
labels_list = [label.numpy() for _, label in trainDataset.unbatch()]
unique_labels = np.unique(labels_list)
original_to_consecutive = {old: new for new, old in enumerate(sorted(unique_labels))}
remapped_labels = [original_to_consecutive[old] for old in labels_list]
class_weights = compute_class_weight('balanced', classes=np.arange(len(unique_labels)), y=remapped_labels)
class_weight_dict = {i: float(w) for i, w in enumerate(class_weights)}

@tf.function
def remap_labels(images, labels):
    """
    reasigna índices originales de clase a un rango consecutivo.

    args:
        images (tf.Tensor): batch de imágenes.
        labels (tf.Tensor): etiquetas de clase originales.

    returns:
        tuple(tf.Tensor, tf.Tensor)
    """
    old_indices = tf.constant(list(original_to_consecutive.keys()), dtype=tf.int32)
    new_indices = tf.constant(list(original_to_consecutive.values()), dtype=tf.int32)
    indices = tf.argmax(tf.equal(old_indices[None, :], labels[:, None]), axis=1)
    return images, tf.gather(new_indices, indices)

bcsLookupTable = tf.constant(bcsValues, dtype=tf.float16)
@tf.function
def putRegressionLabels(images, classIndices):
    """
    convierte índices consecutivos a etiquetas reales de BCS.

    args:
        images (tf.Tensor): batch de imágenes.
        classIndices (tf.Tensor): índices consecutivos.

    returns:
        tuple(tf.Tensor, tf.Tensor)
    """
    original_indices = tf.gather(tf.constant(list({v: k for k, v in original_to_consecutive.items()}.keys()), dtype=tf.int32), classIndices)
    labels = tf.gather(bcsLookupTable, original_indices)
    return images, tf.expand_dims(tf.cast(labels, tf.float16), axis=-1)

trainReg = trainDataset.map(remap_labels, autoTune).map(putRegressionLabels, autoTune)
valTestReg = valTestDataset.map(remap_labels, autoTune).map(putRegressionLabels, autoTune)

dataset_size = tf.data.experimental.cardinality(valTestDataset).numpy()
half = dataset_size // 2
valReg = valTestReg.take(half)
testReg = valTestReg.skip(half)

# Augmentación y pipeline final
augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.25),
    layers.RandomZoom(0.25),
    layers.RandomContrast(0.3),
    layers.RandomBrightness(0.3),
])

def augment_extreme(images, labels):
    """
    aplica augmentación agresiva y recorta valores válidos.
    """
    images = tf.cast(images, tf.float16)
    images = augmentation(images, training=True)
    return tf.clip_by_value(images, 0, 255), labels

@tf.function
def fix_label_shape(images, labels):
    """Asegura que la etiqueta tenga forma [batch, 1]."""
    return images, tf.reshape(labels, [tf.shape(labels)[0], 1])

trainDatasetFinal = (trainReg
    .shuffle(8 * batchSize, seed=seed, reshuffle_each_iteration=True)
    .map(augment_extreme, autoTune)
    .map(fix_label_shape, autoTune)
    .prefetch(autoTune))

valDataset = valReg.map(lambda x, y: (tf.clip_by_value(x, 0, 255), tf.reshape(y, [tf.shape(y)[0], 1])), autoTune).prefetch(autoTune)
testDataset = testReg.map(lambda x, y: (tf.clip_by_value(x, 0, 255), tf.reshape(y, [tf.shape(y)[0], 1])), autoTune).prefetch(autoTune)

# Scheduler de tasa de aprendizaje
num_train_images = tf.data.experimental.cardinality(trainDataset).numpy()
steps_per_epoch = num_train_images // batchSize
total_epochs = 200
total_steps = total_epochs * steps_per_epoch

class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Implementa un scheduler con calentamiento y decaimiento cosenoidal.
    """
    def __init__(self, initial_lr, warmup_steps, total_steps):
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup = step / self.warmup_steps * self.initial_lr
        cosine = 0.5 * self.initial_lr * (1 + tf.cos(3.14159 * (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)))
        return tf.where(step < self.warmup_steps, warmup, cosine)

lr_schedule = WarmupCosineDecay(1e-4, 5 * steps_per_epoch, total_steps)
lr_callback = LearningRateScheduler(lr_schedule)

# Construcción del modelo EfficientNetB7
def buildB7Model():
    """
    Crea un modelo EfficientNetB7 preentrenado con capa de regresión.
    """
    base = EfficientNetB7(include_top=False, weights='imagenet', input_shape=(*imageSize, 3))
    base.trainable = True
    inputs = layers.Input(shape=(*imageSize, 3))
    x = base(inputs, training=True)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='swish')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='linear', dtype='float32')(x)
    return models.Model(inputs, outputs)

# Entrenamiento de los tres modelos (ensemble)
models_list = []
histories = []

for i, seed_val in enumerate([42, 123, 999]):
    print(f"\n=== ENTRENANDO MODELO {i+1}/3 (seed={seed_val}) ===")
    tf.random.set_seed(seed_val)
    model = buildB7Model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.Huber(delta=0.5),
        metrics=['mae']
    )
    log_dir = os.path.join(outputDir, f"logs_model_{i}")
    callbacks = [
        ModelCheckpoint(os.path.join(checkpointDir, f"b7_{i}.weights.h5"), save_weights_only=True, save_best_only=True, monitor='val_mae', mode='min'),
        EarlyStopping(monitor='val_mae', patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_mae', factor=0.5, patience=6, min_lr=1e-7),
        TensorBoard(log_dir=log_dir),
        lr_callback
    ]
    history = model.fit(trainDatasetFinal, validation_data=valDataset, epochs=total_epochs,
                        callbacks=callbacks, class_weight=class_weight_dict, verbose=1)
    models_list.append(model)
    histories.append(history)

# Guardado de pesos y valores
print("\nGuardando pesos...")
for i, m in enumerate(models_list):
    m.save_weights(os.path.join(outputDir, f"MuuMetrics_B7_{i}.weights.h5"))

with open(os.path.join(outputDir, "bcs_values.json"), "w") as f:
    json.dump(bcsValues, f)

# Guardado de gráficos de desempeño
mae_plot_path = os.path.join(reportDir, "mae_comparison.png")
plt.figure(figsize=(14, 8))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
for i, history in enumerate(histories):
    plt.plot(history.history['mae'], label=f'Modelo {i+1} - Train MAE', color=colors[i], linestyle='-')
    plt.plot(history.history['val_mae'], label=f'Modelo {i+1} - Val MAE', color=colors[i], linestyle='--')
plt.title('MAE por Época - Ensemble EfficientNetB7')
plt.xlabel('Época')
plt.ylabel('MAE')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(mae_plot_path, dpi=300, bbox_inches='tight')
plt.close()

# Evaluación final en conjunto de prueba
print("\nEvaluación final en test...")
y_true, y_pred = [], []
for x, y in testDataset:
    preds = tf.reduce_mean([m(x, training=False) for m in models_list], axis=0)
    y_true.extend(y.numpy().flatten())
    y_pred.extend(preds.numpy().flatten())

y_true, y_pred = np.array(y_true), np.array(y_pred)
y_pred_rounded = np.round(y_pred * 4) / 4
mae = np.mean(np.abs(y_true - y_pred))
mae_r = np.mean(np.abs(y_true - y_pred_rounded))
acc = np.mean(np.abs(y_true - y_pred) <= 0.5) * 100

scatter_plot_path = os.path.join(reportDir, "prediction_scatter.png")
plt.figure(figsize=(10,8))
plt.scatter(y_true, y_pred, alpha=0.6, s=30, label=f'MAE: {mae:.3f}')
plt.scatter(y_true, y_pred_rounded, alpha=0.6, s=30, c='red', label=f'Redondeado: {mae_r:.3f}')
plt.plot([1,5],[1,5],'k--', lw=2)
plt.xlabel('BCS Real')
plt.ylabel('BCS Predicho')
plt.title('Predicción vs Real - Ensemble')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(scatter_plot_path, dpi=300, bbox_inches='tight')
plt.close()

# Generación de reporte PDF
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors  # ← ¡CRÍTICO!
    from reportlab.lib.units import inch

    pdf_path = os.path.join(outputDir, "MuuMetricsBCS_Report.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # Título
    story.append(Paragraph("MuuMetrics BCS - Reporte Final", styles['Title']))
    story.append(Spacer(1, 0.3*inch))

    # Métricas
    story.append(Paragraph("Resultados Finales", styles['Heading2']))
    data = [
        ["Métrica", "Valor"],
        ["MAE (raw)", f"{mae:.4f}"],
        ["MAE redondeado", f"{mae_r:.4f}"],
        ["% ±0.5 BCS", f"{acc:.1f}%"],
        ["Imágenes", str(num_train_images)],
        ["Batch", str(batchSize)],
        ["Tamaño", f"{imageSize[0]}x{imageSize[1]}"]
    ]
    t = Table(data, colWidths=[2.5*inch, 2*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 12),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND', (0,1), (-1,-1), colors.beige),
        ('GRID', (0,0), (-1,-1), 1, colors.black)
    ]))
    story.append(t)
    story.append(Spacer(1, 0.3*inch))

    # Gráficas
    if os.path.exists(mae_plot_path):
        story.append(Paragraph("MAE por Época (Entrenamiento y Validación)", styles['Heading2']))
        story.append(Image(mae_plot_path, width=5.5*inch, height=3.5*inch))
        story.append(Spacer(1, 0.2*inch))

    if os.path.exists(scatter_plot_path):
        story.append(Paragraph("Predicción vs Valor Real", styles['Heading2']))
        story.append(Image(scatter_plot_path, width=5.5*inch, height=4.5*inch))

    # Fecha
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph(f"Generado: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Italic']))

    doc.build(story)
    print(f"\nREPORTE PDF GENERADO: {pdf_path}")

except ImportError as e:
    print(f"Error: Falta dependencia → pip install reportlab")
    print(f"Detalles: {e}")
except Exception as e:
    print(f"Error al generar PDF: {e}")
    print("Continuando sin PDF...")