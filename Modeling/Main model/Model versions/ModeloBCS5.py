# =============================================================
# Nombre del archivo: ModeloBCS2.py
# Autores: Mónica Monserrat Martínez Vásquez,
#          Bárbara Paola Alcántara Vega 
#
# Descripción:
#   entrenamiento extendido del modelo MuuMetrics BCS utilizando un ensemble
#   de tres EfficientNetB7 con precisión mixta FP16. incluye cálculo de bias
#   y varianza mediante un callback personalizado, generación automática de
#   gráficas (loss, bias, varianza, mae) y creación de un reporte PDF final.
#
# Dependencias: os, numpy, tensorflow, sklearn, matplotlib, reportlab
# =============================================================

import os, datetime, json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, mixed_precision
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
                                        TensorBoard, LearningRateScheduler, Callback)
from tensorflow.keras.applications import EfficientNetB7
from sklearn.utils.class_weight import compute_class_weight
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle, PageBreak)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
import matplotlib
matplotlib.use('Agg')

# Configuración general y memoria GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"Memory growth activado: {len(gpus)} GPU(s)")

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print("Mixed precision: FP16")

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

# Carga de datos y preparación de clases
subdirs = sorted([d for d in os.listdir(imagesDir) if os.path.isdir(os.path.join(imagesDir, d))])
classNames = sorted(subdirs, key=lambda s: float(s))
bcsValues = [float(c) for c in classNames]

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
    Reasigna los índices originales a índices consecutivos.

    Args:
        images (tf.Tensor): lote de imágenes.
        labels (tf.Tensor): etiquetas enteras originales.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]:
            imágenes originales y nuevas etiquetas consecutivas.
    """
    old_indices = tf.constant(list(original_to_consecutive.keys()), dtype=tf.int32)
    new_indices = tf.constant(list(original_to_consecutive.values()), dtype=tf.int32)
    indices = tf.argmax(tf.equal(old_indices[None, :], labels[:, None]), axis=1)
    return images, tf.gather(new_indices, indices)

bcsLookupTable = tf.constant(bcsValues, dtype=tf.float16)
@tf.function
def putRegressionLabels(images, classIndices):
    """
    Convierte los índices de clase consecutivos a etiquetas numéricas de BCS.

    Args:
        images (tf.Tensor): lote de imágenes.
        classIndices (tf.Tensor): índices consecutivos de clase.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]:
            imágenes y etiquetas numéricas de BCS con forma [batch, 1].
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

# Augmentación y pipeline
augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.25),
    layers.RandomZoom(0.25),
    layers.RandomContrast(0.3),
    layers.RandomBrightness(0.3),
])

def augment_extreme(images, labels):
    """
    Aplica augmentación agresiva de datos.

    Args:
        images (tf.Tensor): lote de imágenes originales.
        labels (tf.Tensor): etiquetas reales de BCS.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]:
            imágenes aumentadas (float16) y etiquetas originales.
    """
    images = tf.cast(images, tf.float16)
    images = augmentation(images, training=True)
    return tf.clip_by_value(images, 0, 255), labels

@tf.function
def fix_label_shape(images, labels):
    """
    Asegura que las etiquetas tengan la forma [batch, 1].

    Args:
        images (tf.Tensor): lote de imágenes.
        labels (tf.Tensor): etiquetas reales.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]:
            imágenes sin cambio y etiquetas con forma corregida.
    """
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

    Args:
        initial_lr (float): tasa de aprendizaje inicial.
        warmup_steps (int): número de pasos de calentamiento.
        total_steps (int): total de pasos del entrenamiento.

    Returns:
        float: tasa de aprendizaje ajustada en función del paso actual.
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

# Modelo EfficientNetB7 base
def buildB7Model():
    """
    Construye un modelo EfficientNetB7 para regresión del BCS.

    Returns:
        tf.keras.Model: modelo listo para compilar y entrenar.
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

# Callback personalizado para bias y varianza
class BiasVarianceLogger(Callback):
    """
    Calcula y registra bias y varianza al final de cada época.

    Args:
        train_data (tf.data.Dataset): conjunto de entrenamiento.
        val_data (tf.data.Dataset): conjunto de validación.

    Attributes:
        train_bias (list): historial de bias en entrenamiento.
        val_bias (list): historial de bias en validación.
        train_var (list): historial de varianza en entrenamiento.
        val_var (list): historial de varianza en validación.
    """
    def __init__(self, train_data, val_data):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.train_bias, self.val_bias = [], []
        self.train_var, self.val_var = [], []

    def eval_ds(self, ds):
        """
        Evalúa bias y varianza sobre un dataset.

        Args:
            ds (tf.data.Dataset): dataset a evaluar.

        Returns:
            Tuple[float, float]: bias y varianza promedio.
        """
        y_true, y_pred = [], []
        for x, y in ds:
            preds = self.model(x, training=False)
            y_true.extend(y.numpy().flatten())
            y_pred.extend(preds.numpy().flatten())
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        bias = np.mean(y_pred - y_true)
        var = np.var(y_pred - y_true)
        return bias, var

    def on_epoch_end(self, epoch, logs=None):
        """
        Ejecuta el cálculo de bias y varianza al final de cada época.
        """
        logs = logs or {}
        b_train, v_train = self.eval_ds(self.train_data)
        b_val, v_val = self.eval_ds(self.val_data)
        self.train_bias.append(b_train)
        self.val_bias.append(b_val)
        self.train_var.append(v_train)
        self.val_var.append(v_val)
        logs['bias'] = b_train
        logs['val_bias'] = b_val
        logs['variance'] = v_train
        logs['val_variance'] = v_val
        print(f" — val_bias={b_val:.4f}, val_var={v_val:.4f}")

# Entrenamiento de 3 modelos
models_list, histories = [], []
for i, seed_val in enumerate([42, 123, 999]):
    print(f"\n=== ENTRENANDO MODELO {i+1}/3 (seed={seed_val}) ===")
    tf.random.set_seed(seed_val)
    model = buildB7Model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.Huber(delta=0.5),
        metrics=['mae']
    )
    bias_logger = BiasVarianceLogger(trainDatasetFinal, valDataset)
    callbacks = [
        ModelCheckpoint(os.path.join(checkpointDir, f"b7_{i}.weights.h5"), save_weights_only=True, save_best_only=True, monitor='val_mae', mode='min'),
        EarlyStopping(monitor='val_mae', patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_mae', factor=0.5, patience=6, min_lr=1e-7),
        TensorBoard(log_dir=os.path.join(outputDir, f"logs_model_{i}")),
        lr_callback,
        bias_logger
    ]
    history = model.fit(trainDatasetFinal, validation_data=valDataset, epochs=total_epochs,
                        callbacks=callbacks, class_weight=class_weight_dict, verbose=1)
    history.history['bias'] = bias_logger.train_bias
    history.history['val_bias'] = bias_logger.val_bias
    history.history['variance'] = bias_logger.train_var
    history.history['val_variance'] = bias_logger.val_var
    models_list.append(model)
    histories.append(history)

# Graficas
plot_paths = {}
def save_plot(metric, val_metric, title, fname):
    plt.figure(figsize=(10,6))
    for i, history in enumerate(histories):
        plt.plot(history.history[metric], label=f'Model {i+1} {metric}')
        plt.plot(history.history[val_metric], '--', label=f'Model {i+1} {val_metric}')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.legend()
    plt.grid(True, alpha=0.3)
    path = os.path.join(reportDir, fname)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    plot_paths[metric] = path

save_plot('loss', 'val_loss', 'Loss por Época', 'loss_plot.png')
save_plot('bias', 'val_bias', 'Bias por Época', 'bias_plot.png')
save_plot('variance', 'val_variance', 'Varianza por Época', 'variance_plot.png')

#Evaluacion final y pdf
print("\nEvaluando ensemble final...")
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

# Reporte PDF
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
    ["Batch", str(batchSize)],
    ["Tamaño", f"{imageSize[0]}x{imageSize[1]}"]
]
t = Table(data, colWidths=[2.5*inch, 2*inch])
t.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.grey),
    ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
    ('BACKGROUND', (0,1), (-1,-1), colors.beige),
    ('GRID', (0,0), (-1,-1), 1, colors.black)
]))
story.append(t)
story.append(Spacer(1, 0.3*inch))

# Gráficas
for title, path in [
    ("MAE por Época", plot_paths.get('mae', '')),
    ("Loss por Época", plot_paths['loss']),
    ("Bias por Época", plot_paths['bias']),
    ("Varianza por Época", plot_paths['variance'])
]:
    if path and os.path.exists(path):
        story.append(PageBreak())
        story.append(Paragraph(title, styles['Heading2']))
        story.append(Image(path, width=5.5*inch, height=3.5*inch))
        story.append(Spacer(1, 0.2*inch))

if os.path.exists(scatter_plot_path):
    story.append(PageBreak())
    story.append(Paragraph("Predicción vs Valor Real", styles['Heading2']))
    story.append(Image(scatter_plot_path, width=5.5*inch, height=4.5*inch))

# Fecha
story.append(Spacer(1, 0.5*inch))
story.append(Paragraph(f"Generado: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Italic']))

doc.build(story)
print(f"\nREPORTE PDF GENERADO: {pdf_path}")
